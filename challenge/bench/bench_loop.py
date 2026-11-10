#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Time one whole frame on the traced path, with one stopwatch.

    Date      2026-09-24
    Version   v0.1
    Measured  2026-09-24, output in ../data/loop_results_20260924_093510/

WHY THIS EXISTS
    The 10.889 ms on Figure 1 of an earlier draft is not a measurement. It is
    2.320 + 0.837 + 5.469 + 2.263, taking each term from whichever of the two
    existing harnesses measured it. That composition is defensible but it is
    not a stopwatch reading, and three things are wrong with leaving it there:

      - bench_stages.py drives YOLOv4Runner directly. bench_levers.py drives
        the same model through create_pipeline_from_config. Those are different
        code paths, and the plain rung differs by 3.4 ms between them
        (59.763 against 56.339), so the terms being added do not all come from
        the same surroundings.
      - bench_levers.py times the host-to-device transfer inside its model
        figure and bench_stages.py times it separately, so the composition has
        to silently drop a row to avoid counting it twice.
      - Neither harness ever fed the traced pipeline a real JPEG. Both reuse
        one prepared input tensor. A composition cannot tell you whether
        preparing a fresh frame every time costs more on the traced path than
        it did on the plain one.

    This harness fixes all three. One process, one device, one loop, one
    perf_counter spanning the entire frame, real JPEG bytes in and real boxes
    out, on the pipeline configured the fast way.

WHAT IT TIMES
    decode      cv2.imdecode of the JPEG bytes, as the server receives them
    prepare     resize, BGR to RGB, to torch tensor
    to_host     build the ttnn host tensor the pipeline wants from that torch
                tensor. bench_stages.py had no equivalent of this stage; its
                "upload" covered both this and the transfer.
    model       pipeline.enqueue(...).pop_all(), which contains the transfer,
                the traced replay and the read back
    boxes       post_processing: max-and-argmax, then overlap removal

    loop        t_end - t_start across all five, measured directly rather than
                summed. This is the number the composition was standing in for.
                It is reported alongside the sum of the stages; if the two
                disagree by more than a few tenths of a millisecond, the
                per-stage instrumentation is costing something and the loop
                figure is the one to trust.

THE CORRECTNESS CHECK THAT MATTERS HERE
    A captured trace replays a fixed sequence against a fixed input buffer. If
    the pipeline were to ignore the host tensor handed to it on each enqueue,
    every frame would return the boxes from whatever was uploaded at compile
    time, and the loop would look fast and be meaningless.

    So by default this runs with two different pictures in alternation and
    asserts that they produce different detections. If they do not, the run is
    marked unsound and no timing is reported. Pass --single-image to turn that
    off, at the cost of the check.

BEFORE YOU RUN IT, ONE THING TO CONFIRM
    This needs to build a fresh ttnn host tensor per frame from a torch tensor.
    YOLOv4PerformanceRunnerInfra exposes that, but the argument name and which
    method takes it differ between branches. The harness probes for it with
    inspect.signature and prints what it found, and stops with an explicit
    message if it finds nothing usable. Run the probe in run_loop.sh first,
    which prints the signatures without opening the device.

HOW TO RUN
        bash bench/run_loop.sh

    Or by hand, from the tt-metal root with python_env active and TT_METAL_HOME
    and PYTHONPATH set:

        python bench/bench_loop.py --iters 300
"""

import argparse
import contextlib
import inspect
import io
import json
import os
import platform
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone

import cv2
import numpy as np
import torch

import ttnn
from models.demos.yolov4.common import (
    YOLOV4_L1_SMALL_SIZE,
    image_to_tensor,
)
from models.demos.yolov4.post_processing import post_processing
from models.demos.yolov4.runner.performant_runner_infra import YOLOv4PerformanceRunnerInfra
from models.demos.yolov4.runner.pipeline_runner import YoloV4PipelineRunner
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config

try:
    from models.demos.utils.common_demo_utils import get_mesh_mappers
except ImportError:  # pragma: no cover - depends on the branch
    get_mesh_mappers = None

# From the device_params of test_e2e_performant.py. Copied, not guessed.
TRACE_REGION_SIZE = 6434816

STAGES = ["decode", "prepare", "to_host", "model", "boxes"]

# Names the infra might give the per-frame torch tensor argument, most likely
# first. Probed against the real signature at runtime.
TENSOR_ARG_NAMES = ("torch_input_tensor", "torch_input", "input_tensor", "tensor")

def summarise(samples_ms):
    """Nearest-rank percentiles, as in bench_stages.py and bench_levers.py."""
    s = sorted(samples_ms)
    n = len(s)

    def pct(p):
        return s[min(n - 1, max(0, int(round(p / 100.0 * n)) - 1))]

    return {
        "n": n,
        "mean": round(statistics.fmean(s), 3),
        "median": round(statistics.median(s), 3),
        "p95": round(pct(95), 3),
        "p99": round(pct(99), 3),
        "min": round(s[0], 3),
        "max": round(s[-1], 3),
        "stdev": round(statistics.stdev(s), 3) if n > 1 else 0.0,
    }

def to_tensor(jpeg_bytes, resolution):
    """JPEG bytes to the torch tensor the model wants. Same steps as the loop."""
    frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(frame, resolution)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return image_to_tensor(img)

def _first(x):
    return x[0] if isinstance(x, (tuple, list)) else x

def _set_stored_input(infra, tensor):
    """Overwrite the tensor attribute the infra keeps its own frame in.

    Measured 2026-09-24: setup_dram_sharded_input(torch_input_tensor=...) takes
    the argument and builds from its own stored tensor anyway, so two different
    pictures produced byte-identical host tensors. Writing the attribute is the
    route that works when the argument does not.

    The attribute is named here rather than found by scanning dir(). A scan
    that matches nothing writes nothing and says nothing, which is how a run
    ends up timing one picture repeatedly. Raising is caught by the caller,
    which then tries the next candidate.
    """
    cur = getattr(infra, "torch_input_tensor", None)
    if not isinstance(cur, torch.Tensor):
        raise RuntimeError(
            "infra.torch_input_tensor is not a tensor. The attribute this "
            "harness writes may have been renamed upstream; check "
            "models/demos/yolov4/runner/performant_runner_infra.py.")
    if cur.shape != tensor.shape:
        raise RuntimeError(
            f"infra.torch_input_tensor has shape {tuple(cur.shape)}, the "
            f"frame to write has {tuple(tensor.shape)}.")
    infra.torch_input_tensor = tensor
    return ["torch_input_tensor"]

def _snapshot_stored_inputs(infra):
    """Copy the infra's own frame before we overwrite it.

    infra.validate() compares the model's output against a reference tied to
    the tensor the infra was constructed with. If the run ends with one of our
    frames in that slot, the accuracy line becomes meaningless. Take a copy
    now and put it back before validating.

    Named rather than scanned, for the reason in _set_stored_input: a snapshot
    that silently captures nothing restores nothing, and validate() then runs
    against a contaminated tensor. That is what produced PCC 0.985 at 320 and
    -2.5e-05 at 640 on 2026-09-24.
    """
    cur = getattr(infra, "torch_input_tensor", None)
    if not isinstance(cur, torch.Tensor):
        raise RuntimeError(
            "infra.torch_input_tensor is not a tensor, so the frame cannot be "
            "saved and the accuracy check afterwards could not be trusted. "
            "Check models/demos/yolov4/runner/performant_runner_infra.py for "
            "a renamed attribute.")
    return {"torch_input_tensor": cur.clone()}

def _restore_stored_inputs(infra, snap):
    for attr, tensor in snap.items():
        setattr(infra, attr, tensor)

def candidate_builders(infra):
    """Every plausible way to hand this infra a fresh frame, best first."""
    cands = []
    dram = getattr(infra, "setup_dram_sharded_input", None)
    l1 = (getattr(infra, "setup_l1_sharded_input", None)
          or getattr(infra, "_setup_l1_sharded_input", None))

    if dram is not None:
        cands.append(("setup_dram_sharded_input(torch_input_tensor=...)",
                      lambda d, t, f=dram: _first(f(d, torch_input_tensor=t))))

        def dram_via_attr(d, t, f=dram):
            _set_stored_input(infra, t)
            return _first(f(d))
        cands.append(("set the infra's stored tensor, then "
                      "setup_dram_sharded_input(device)", dram_via_attr))

    if l1 is not None:
        nm = getattr(l1, "__name__", "setup_l1_sharded_input")
        cands.append((f"{nm}(device, tensor)",
                      lambda d, t, f=l1: _first(f(d, t))))

        def l1_via_attr(d, t, f=l1):
            _set_stored_input(infra, t)
            return _first(f(d))
        cands.append((f"set the infra's stored tensor, then {nm}(device)",
                      l1_via_attr))
    return cands

def select_builder(infra, device, probe_a, probe_b):
    """Pick a builder, and prove it before the run rather than after.

    Each candidate is given two different pictures. Its host tensor has to come
    out different, otherwise the frame data is being dropped before the card
    ever sees it and every timing taken afterwards would be of the same picture
    300 times over. The first three attempts at this harness all failed here
    and only found out at the end.
    """
    for label, fn in candidate_builders(infra):
        try:
            a = host_signature(fn(device, probe_a))
            b = host_signature(fn(device, probe_b))
        except Exception as exc:  # noqa: BLE001
            print(f"    no: {label}")
            print(f"        raised {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:120]}")
            continue
        if a is None or b is None:
            print(f"    no: {label} (host tensor could not be fingerprinted)")
            continue
        if a == b:
            print(f"    no: {label}")
            print(f"        ignores the tensor, {a} for both pictures")
            continue
        print(f"    yes: {label}")
        print(f"        {a} then {b}, so the frame reaches the host tensor")
        return label, fn
    return None, None

def preflight_liveness(pipeline, build, device, images, resolution):
    """Push two different pictures through and compare what comes back.

    This used to be done by alternating pictures inside the measured loop,
    which proved the point and wrecked the numbers: giraffe.jpg is 80 KB of
    JPEG and dinning.png is 457 KB of PNG, so the decode and prepare stages
    were averaging two populations. Measured 2026-09-24, decode came out at
    6.050 ms against 2.320 ms for the same work on one picture.

    Doing it here costs two frames and leaves the loop measuring one picture.
    """
    seen = []
    for path, jpeg_bytes in (images[0], images[-1]):
        host = build(device, to_tensor(jpeg_bytes, resolution))
        out = pipeline.enqueue([host]).pop_all()
        pair, err = unpack_output(out[0] if out else None)
        if pair is None:
            return False, f"could not read the output back: {err}", []
        seen.append((os.path.basename(path), output_signature(normalise_pair(pair))))
    if any(sig is None for _, sig in seen):
        return False, "output could not be fingerprinted", seen
    if seen[0][1] == seen[1][1]:
        return (False,
                "UNSOUND: two different pictures produced the same output. The "
                "traced pipeline is replaying against its own buffer and "
                "ignoring the tensor passed to enqueue.", seen)
    return (True,
            "passed: two different pictures produced different output, so the "
            "traced pipeline is reading the frame it is handed", seen)

def unpack_output(out):
    """Turn whatever pop_all() hands back into (boxes, confs) torch tensors.

    post_processing calls .numpy() on what it is given, and numpy has no
    bfloat16. The plain path handed it float32 because YOLOv4Runner.run
    converts; the pipeline path hands back bfloat16 straight from the device,
    so cast here. This changes nothing about what the card computed, only how
    it is held on the host.
    """
    if isinstance(out, (tuple, list)) and len(out) == 2:
        parts = list(out)
    else:
        return None, f"expected a pair, got {type(out).__name__}"
    converted = []
    for part in parts:
        if not isinstance(part, (np.ndarray, torch.Tensor)) and \
                type(part).__module__.startswith("ttnn"):
            part = ttnn.to_torch(part)
        if isinstance(part, torch.Tensor) and part.dtype != torch.float32:
            part = part.float()
        converted.append(part)
    return converted, None

def load_images(paths, resolution):
    """Read the JPEG bytes once. Decoding them is part of what we time."""
    loaded = []
    for p in paths:
        if not os.path.exists(p):
            sys.exit(f"image not found: {p}\nRun from the tt-metal root, or pass --image.")
        with open(p, "rb") as fh:
            loaded.append((p, fh.read()))
        print(f"image   : {p} ({len(loaded[-1][1])} bytes)")
    return loaded

def normalise_pair(pair):
    """Give post_processing the shapes YOLOv4Runner used to give it.

    Worked out from two failures on hardware, 2026-09-24, rather than guessed.

    The traced pipeline returns boxes as [1, 1, 6384, 4] and confs as
    [1, 6300, 80]. Two things are wrong with the boxes tensor.

      The anchor count is padded. 6300 is the real number for 320x320
      (3 anchors x (40^2 + 20^2 + 10^2)); 6384 is that rounded up for the
      sharded layout. tt-metal pads at the end of the sharded axis, so the
      first 6300 rows are the real ones.

      The axes are in the wrong order for post_processing, which expects
      [batch, N, 1, 4] and takes box_array[:, :, 0] itself before masking.
      Collapsing to [batch, N, 4] here, as a first attempt did, makes that
      internal slice produce a 2-D array and the next index fails with
      "array is 2-dimensional, but 3 were indexed".

    So: reshape to [batch, N_padded, 1, 4], then trim to the anchor count
    that confs reports. confs is not padded and is used as the authority.
    """
    boxes, confs = pair
    if not (isinstance(boxes, torch.Tensor) and isinstance(confs, torch.Tensor)):
        return [boxes, confs]
    n = confs.shape[1]
    boxes = boxes.reshape(boxes.shape[0], -1, 1, boxes.shape[-1])
    if boxes.shape[1] < n:
        raise ValueError(
            f"boxes has {boxes.shape[1]} anchors but confs has {n}. "
            f"Trimming assumes boxes is padded, not truncated.")
    boxes = boxes[:, :n, :, :]
    return [boxes, confs]

def host_signature(host_tensor):
    """Fingerprint the ttnn host tensor actually handed to the pipeline."""
    try:
        return round(float(ttnn.to_torch(host_tensor).float().abs().sum().item()), 3)
    except Exception:  # noqa: BLE001
        return None

def output_signature(pair):
    """A cheap fingerprint of one frame's raw model output.

    Taken from the tensor rather than from post_processing's return value,
    because the shape of that return varies by branch and a failure to parse
    it would produce a false UNSOUND verdict. Two different pictures give
    different sums; the same picture gives the same sum every time.
    """
    try:
        return round(float(pair[0].abs().sum().item()), 3)
    except Exception:  # noqa: BLE001 - never let the check crash the run
        return None

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", action="append", default=None,
                    help="repeatable. Two different pictures by default, so the "
                         "harness can prove the trace is seeing new pixels.")
    ap.add_argument("--single-image", action="store_true",
                    help="one picture only, which also skips the liveness check")
    ap.add_argument("--alternate", action="store_true",
                    help="measure across every --image rather than only the "
                         "first. Mixing pictures of different size and format "
                         "makes the decode and prepare stages an average of "
                         "two populations, so this is off by default.")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--resolution", type=int, default=320, choices=(320, 640))
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--cqs", type=int, default=1, choices=(1, 2),
                    help="1 is the best single-frame latency we measured; 2 is "
                         "what a loop with frames in flight would use")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--nms", type=float, default=0.4)
    ap.add_argument("--out", default=None, help="JSON output path")
    args = ap.parse_args()

    if args.image:
        paths = args.image
    elif args.single_image:
        paths = ["models/demos/yolov4/resources/giraffe.jpg"]
    else:
        paths = ["models/demos/yolov4/resources/giraffe.jpg",
                 "models/demos/yolov4/resources/dinning.png"]

    resolution = (args.resolution, args.resolution)
    images = load_images(paths, resolution)
    check_trace_is_live = len(images) > 1

    print(f"resolution : {args.resolution}x{args.resolution}")
    print(f"path       : trace ON, {args.cqs} command queue(s), one frame at a time")
    print(f"iterations : {args.iters} measured, {args.warmup} warm-up")
    print(f"trace check: {'on, before the loop' if check_trace_is_live else 'OFF'}")

    open_kwargs = {"device_id": 0,
                   "l1_small_size": YOLOV4_L1_SMALL_SIZE,
                   "trace_region_size": TRACE_REGION_SIZE}
    if args.cqs > 1:
        open_kwargs["num_command_queues"] = args.cqs

    record = {"ok": False}
    device = None
    pipeline = None
    t = {k: [] for k in STAGES}
    loop_samples = []
    fingerprints = {}

    try:
        print("opening device ...", flush=True)
        device = ttnn.open_device(**open_kwargs)

        if get_mesh_mappers is not None:
            inputs_mesh_mapper, _, output_mesh_composer = get_mesh_mappers(device)
        else:
            inputs_mesh_mapper, output_mesh_composer = None, None

        print("building model ...", flush=True)
        infra = YOLOv4PerformanceRunnerInfra(
            device, args.batch, ttnn.bfloat8_b, ttnn.bfloat8_b,
            model_location_generator=None,
            resolution=resolution,
            mesh_mapper=inputs_mesh_mapper,
            mesh_composer=output_mesh_composer,
        )

        stored_snapshot = _snapshot_stored_inputs(infra)
        if stored_snapshot:
            print(f"  infra frame attributes saved for restore: "
                  f"{sorted(stored_snapshot)}", flush=True)

        print("  choosing an input builder, and proving it with two "
              "different pictures:", flush=True)
        probe_a = to_tensor(images[0][1], resolution)
        probe_b = to_tensor(images[-1][1], resolution)
        if len(images) < 2:
            # Nothing to prove against. Take the first candidate that runs and
            # say so, rather than rejecting all of them for being identical.
            print("    only one picture supplied, so no builder can be proved. "
                  "Taking the first that runs.", flush=True)
            how, build_host_tensor = None, None
            for label, fn in candidate_builders(infra):
                try:
                    fn(device, probe_a)
                except Exception:  # noqa: BLE001
                    continue
                how, build_host_tensor = label + " (UNPROVEN)", fn
                break
        else:
            how, build_host_tensor = select_builder(infra, device,
                                                    probe_a, probe_b)
        if build_host_tensor is None:
            sys.exit(
                "\nSTOP: no way of handing this infra a fresh frame survives the\n"
                "two-picture check, so any timing taken would be of one picture\n"
                "repeated. Send the source and the harness can be adjusted:\n\n"
                "    sed -n '1,200p' "
                "models/demos/yolov4/runner/performant_runner_infra.py\n")
        print(f"  using {how}", flush=True)
        record["input_builder"] = how

        tt_inputs_host, dram_cfg, l1_cfg = infra.setup_dram_sharded_input(device)

        pipeline = create_pipeline_from_config(
            config=PipelineConfig(
                use_trace=True,
                num_command_queues=args.cqs,
                all_transfers_on_separate_command_queue=False,
            ),
            model=YoloV4PipelineRunner(infra),
            device=device,
            dram_input_memory_config=dram_cfg,
            l1_input_memory_config=l1_cfg,
        )

        print("compiling (slow on a cold kernel cache) ...", flush=True)
        t0 = time.perf_counter()
        pipeline.compile(tt_inputs_host)
        record["compile_s"] = round(time.perf_counter() - t0, 3)
        print(f"  compiled in {record['compile_s']} s", flush=True)

        if len(images) > 1:
            print("checking the pipeline reads the frame it is handed ...",
                  flush=True)
            live_ok, live_msg, live_seen = preflight_liveness(
                pipeline, build_host_tensor, device, images, resolution)
            for nm, sig in live_seen:
                print(f"    {nm:<16} output fingerprint {sig}", flush=True)
            print(f"    {live_msg}", flush=True)
        else:
            live_ok, live_msg, live_seen = (
                True, "not checked, only one picture supplied", [])
        record["trace_is_live"] = live_msg
        record["trace_is_live_ok"] = live_ok
        record["liveness_fingerprints"] = [
            {"image": nm, "output": sig} for nm, sig in live_seen]

        print(f"{args.warmup} warm-up frames ...", flush=True)
        for _ in range(args.warmup):
            pipeline.enqueue([tt_inputs_host]).pop_all()

        print(f"{args.iters} measured frames ...", flush=True)
        post_first_output = ""
        unpack_note = None
        post_ok = True

        measure_images = images if args.alternate else images[:1]
        record["measured_images"] = [p for p, _ in measure_images]
        print(f"measuring on: "
              f"{', '.join(os.path.basename(p) for p, _ in measure_images)}",
              flush=True)

        for i in range(args.iters):
            name, jpeg_bytes = measure_images[i % len(measure_images)]

            t_start = time.perf_counter()

            frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
            t1 = time.perf_counter()

            img = cv2.resize(frame, resolution)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = image_to_tensor(img)
            t2 = time.perf_counter()

            host_tensor = build_host_tensor(device, tensor)
            t3 = time.perf_counter()

            out = pipeline.enqueue([host_tensor]).pop_all()
            t4 = time.perf_counter()

            pair, err = unpack_output(out[0] if out else None)
            if err is not None and unpack_note is None:
                unpack_note = err
            if pair is not None:
                pair = normalise_pair(pair)
                if i == 0:
                    shapes = [tuple(t.shape) for t in pair
                              if isinstance(t, torch.Tensor)]
                    record["output_shapes"] = [list(sh) for sh in shapes]
                    print(f"    output shapes after normalising: {shapes}",
                          flush=True)
                if post_ok:
                    buf = io.StringIO()
                    try:
                        with contextlib.redirect_stdout(buf):
                            post_processing(None, args.conf, args.nms, pair)
                    except Exception as exc:  # noqa: BLE001
                        # Do not lose the whole run over the last stage. Record
                        # it, stop calling it, and say so in the report.
                        post_ok = False
                        record["post_processing_error"] = (
                            f"{type(exc).__name__}: {exc}".splitlines()[0][:300])
                        print(f"    post_processing failed on frame {i}, "
                              f"skipping it for the rest of the run:",
                              flush=True)
                        print(f"      {record['post_processing_error']}",
                              flush=True)
                    if not post_first_output:
                        post_first_output = buf.getvalue().strip()
            t_end = time.perf_counter()

            # Everything below is outside the measured window on purpose.
            # Fingerprinting costs a full tensor reduction and would land in
            # the boxes stage if it sat above this line, which it did on the
            # first two attempts.
            if args.alternate and pair is not None and i < 4 * len(images):
                fingerprints.setdefault(name, []).append({
                    "input": round(float(tensor.float().abs().sum().item()), 3),
                    "host": host_signature(host_tensor),
                    "output": output_signature(pair),
                })

            t["decode"].append((t1 - t_start) * 1000.0)
            t["prepare"].append((t2 - t1) * 1000.0)
            t["to_host"].append((t3 - t2) * 1000.0)
            t["model"].append((t4 - t3) * 1000.0)
            t["boxes"].append((t_end - t4) * 1000.0)
            loop_samples.append((t_end - t_start) * 1000.0)

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{args.iters}", flush=True)

        record["unpack_note"] = unpack_note
        record["post_processing_ran"] = post_ok
        record["post_processing_internal_split_first_frame"] = post_first_output

        # Accuracy, the same gate the other two harnesses use.
        #
        # Restoring the attribute is not enough. tt_inputs_host was built after
        # select_builder had already left the infra holding our second picture,
        # so validating with it compared a dining room against a reference
        # computed for the infra's own frame. That is what produced PCC 0.985
        # at 320 and -2.5e-05 at 640 on 2026-09-24. Put the frame back AND
        # rebuild the host tensor from it before validating.
        _restore_stored_inputs(infra, stored_snapshot)
        try:
            tt_inputs_ref, _, _ = infra.setup_dram_sharded_input(device)
            last = pipeline.enqueue([tt_inputs_ref]).pop_all()
            infra.validate(last[0] if last else None)
            record["validated"] = "passed"
        except Exception as exc:  # noqa: BLE001
            record["validated"] = f"FAILED: {exc}".splitlines()[0][:300]

        record["ok"] = True

    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {exc}".splitlines()[0][:400]
        record["traceback"] = traceback.format_exc()[-1500:]
        print(f"FAILED: {record['error']}", flush=True)
    finally:
        if pipeline is not None:
            with contextlib.suppress(Exception):
                pipeline.cleanup()
        if device is not None:
            with contextlib.suppress(Exception):
                ttnn.close_device(device)

    if not record["ok"]:
        print(json.dumps(record, indent=2))
        sys.exit(1)

    # ---- the trace-is-live verdict, before any timing is reported ----------
    trace_verdict = record.get("trace_is_live", "not checked")
    sound = bool(record.get("trace_is_live_ok", True))

    result = {
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "images": [p for p, _ in images],
        "resolution": args.resolution,
        "batch": args.batch,
        "command_queues": args.cqs,
        "iterations": args.iters,
        "warmup_discarded": args.warmup,
        "conf_thresh": args.conf,
        "nms_thresh": args.nms,
        "path": (f"traced pipeline via create_pipeline_from_config, "
                 f"{args.cqs} command queue(s), one frame at a time, "
                 f"fresh JPEG decoded and uploaded every frame"),
        "stages_ms": {k: summarise(t[k]) for k in STAGES},
        "whole_loop_ms": summarise(loop_samples),
        **record,
    }
    stage_sum = sum(result["stages_ms"][k]["mean"] for k in STAGES)
    loop_mean = result["whole_loop_ms"]["mean"]
    result["stage_sum_mean_ms"] = round(stage_sum, 3)
    result["instrumentation_overhead_ms"] = round(loop_mean - stage_sum, 3)
    result["implied_fps_mean"] = round(1000.0 / loop_mean, 1) if loop_mean else None
    worst = result["whole_loop_ms"]["max"]
    result["implied_fps_worst_frame"] = round(1000.0 / worst, 1) if worst else None

    out = args.out or (f"bench_loop_{args.resolution}_"
                       f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)

    print()
    print("=" * 70)
    print(f"  one whole frame, traced path, {args.resolution}x{args.resolution}, "
          f"{args.iters} frames")
    print("=" * 70)
    print(f"  trace check : {trace_verdict}")
    print(f"  accuracy    : {record['validated']}")
    if record.get("liveness_fingerprints"):
        print()
        print("  liveness, two pictures through the pipeline before the loop:")
        for row in record["liveness_fingerprints"]:
            print(f"    {row['image']:<16} output fingerprint {row['output']}")
    if fingerprints:
        print()
        print("  per-frame fingerprints while alternating "
              "(input -> host tensor -> output):")
        for img, rows in fingerprints.items():
            for r in rows[:2]:
                print(f"    {os.path.basename(img):<16} "
                      f"{r['input']}  ->  {r['host']}  ->  {r['output']}")
    if not record.get("post_processing_ran", True):
        print()
        print("  WARNING: post_processing did not run. The boxes stage below is")
        print("  near zero and the LOOP total EXCLUDES turning the model output")
        print("  into rectangles, so it is not comparable with bench_stages.py")
        print("  or with the 10.889 ms composed figure. The cause was:")
        print(f"    {record.get('post_processing_error')}")
        print(f"  Output shapes seen: {record.get('output_shapes')}")
    print()
    if not sound:
        print("  TIMINGS SUPPRESSED. Fix the trace check before reading numbers.")
        print(f"  written to {os.path.abspath(out)}")
        sys.exit(2)

    if len(result.get("measured_images", [])) > 1:
        print("  NOTE: --alternate mixes pictures of different size and format,")
        print("  so decode, prepare and boxes are averages over two populations")
        print("  and are not comparable with bench_stages.py.")
        print()
    print(f"  {'stage':<10}{'mean':>9}{'median':>9}{'p99':>9}{'max':>9}   {'share':>6}")
    for k in STAGES:
        s = result["stages_ms"][k]
        share = 100.0 * s["mean"] / loop_mean if loop_mean else 0.0
        print(f"  {k:<10}{s['mean']:>9.3f}{s['median']:>9.3f}"
              f"{s['p99']:>9.3f}{s['max']:>9.3f}   {share:>5.1f}%")
    lp = result["whole_loop_ms"]
    print("  " + "-" * 66)
    print(f"  {'LOOP':<10}{lp['mean']:>9.3f}{lp['median']:>9.3f}"
          f"{lp['p99']:>9.3f}{lp['max']:>9.3f}   measured, not summed")
    print(f"  {'(sum)':<10}{stage_sum:>9.3f}      instrumentation costs "
          f"{result['instrumentation_overhead_ms']:+.3f} ms")
    print()
    print(f"  mean        {result['implied_fps_mean']} frames a second")
    print(f"  worst frame {result['implied_fps_worst_frame']} frames a second")
    print()
    print("  Against the composed figure this replaces, 10.889 ms from")
    print("  2.320 + 0.837 + 5.469 + 2.263. If this run lands close to it, the")
    print("  composition was fair and Figure 1 keeps its shape with a measured")
    print("  number in place of an assembled one. If it lands well above, the")
    print("  extra is the cost of preparing a fresh frame on the traced path,")
    print("  which neither existing harness could see.")
    print()
    print(f"  written to {os.path.abspath(out)}")

if __name__ == "__main__":
    main()
