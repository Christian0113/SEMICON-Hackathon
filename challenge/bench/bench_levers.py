#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Split the gap between the plain path and the traced path into its causes.

    Date      2026-09-24
    Version   v0.1
    Measured  2026-10-26 on a Wormhole n150, board 010001851180b3c4. At 320 the
              trace_2cq rung came out at 5.448 ms a frame against the published
              5.917 ms, 8% faster, with every configuration matching the
              reference output, so the ladder reproduces the figure it is
              checked against. At 640 the plain rung raises TT_FATAL and the
              three traced rungs run but fail their numerical check at
              PCC 0.768; see the resolution note in ../README.md.

WHY THIS EXISTS
    bench_stages.py measured the model at 59.763 ms a frame at 320 on the plain
    path. tests/perf/test_e2e_performant.py measures the same model at 5.917 ms.
    Nobody had measured where the factor of ten comes from. This does.

WHAT IS A LEVER HERE, AND WHAT IS NOT
    Reading models/demos/yolov4/runner/performant_runner_infra.py, the act_dtype
    and weight_dtype arguments are stored and then used only in log messages.
    The model is built by create_yolov4_model_parameters and TtYOLOv4 with no
    dtype argument, the input is hardcoded to bfloat16 in
    _setup_l1_sharded_input, and the convolution weights are pinned to
    bfloat8_b in common.py:create_conv2d_config regardless. So "bfloat8_b" in
    the perf test's parametrize is a label on the run rather than a setting of
    it, and there is no dtype lever to measure in this tree. Two levers remain:
    trace, and the second command queue.

THE LADDER
    plain                no trace, 1 command queue   the baseline
    trace                trace,    1 command queue   isolates trace
    trace_2cq            trace,    2 command queues  isolates the second queue
    trace_2cq_pipelined  as above, all frames enqueued before any is collected

    The first three are timed one frame at a time, enqueue then collect, which
    is how test_e2e_performant.py drives it, so trace_2cq should land near
    5.917 ms. If it does not, this harness is wrong and no other number in the
    run is worth reading. That check matters more than the split itself.

    The fourth exists because a second command queue is meant to let the host
    move the next frame's pixels while the card works on the current one. One
    frame at a time gives it nothing to overlap; enqueueing the whole batch
    does. The difference between the third and the fourth is the difference
    between latency and throughput, which is worth knowing separately for a
    webcam demo that can keep frames in flight.

WHAT IS HELD CONSTANT
    Same device, same weights, same giraffe.jpg, same resolution, same
    iteration count. Every configuration goes through create_pipeline_from_config,
    so the baseline is not YOLOv4Runner on a different code path but the same
    pipeline with its features switched off. That is what makes the deltas
    attributable to the levers rather than to the surrounding code.

    Every configuration validates its first measured output against the PyTorch
    reference through test_infra.validate(). A trace replaying a stale buffer
    would otherwise look fast and be wrong.

HOW TO RUN
    bash bench/run_levers.sh

    Or by hand, from the tt-metal root with python_env active and TT_METAL_HOME
    and PYTHONPATH set:

        python bench/bench_levers.py --iters 300

    Each configuration reopens the device, because trace_region_size and
    num_command_queues are device-open parameters. Expect a few minutes.
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone

import ttnn
from models.demos.yolov4.common import YOLOV4_L1_SMALL_SIZE
from models.demos.yolov4.runner.performant_runner_infra import YOLOv4PerformanceRunnerInfra
from models.demos.yolov4.runner.pipeline_runner import YoloV4PipelineRunner
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config

# Optional on a single device. The infra accepts None for both, so if this
# helper has moved or been renamed the bench still runs rather than failing at
# import time on something incidental.
try:
    from models.demos.utils.common_demo_utils import get_mesh_mappers
except ImportError:  # pragma: no cover - depends on the branch
    get_mesh_mappers = None

# From the device_params of test_e2e_performant.py. Copied, not guessed.
TRACE_REGION_SIZE = 6434816

# The published figure this harness has to reproduce: that perf test at
# 320x320, batch 1, single device.
REFERENCE_MS = 5.917

# name, use_trace, num_command_queues, enqueue_all_before_collecting
CONFIGS = [
    ("plain", False, 1, False),
    ("trace", True, 1, False),
    ("trace_2cq", True, 2, False),
    ("trace_2cq_pipelined", True, 2, True),
]

def summarise(samples_ms):
    """Nearest-rank percentiles, which is honest for the n we run here."""
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

def run_one(name, use_trace, num_cqs, pipelined, args):
    """One configuration, start to finish, on its own freshly opened device."""
    resolution = (args.resolution, args.resolution)
    record = {
        "name": name,
        "use_trace": use_trace,
        "num_command_queues": num_cqs,
        "enqueue_all_before_collecting": pipelined,
        "ok": False,
    }

    open_kwargs = {"device_id": 0, "l1_small_size": YOLOV4_L1_SMALL_SIZE}
    if use_trace:
        open_kwargs["trace_region_size"] = TRACE_REGION_SIZE
    if num_cqs > 1:
        open_kwargs["num_command_queues"] = num_cqs
    record["device_open_kwargs"] = dict(open_kwargs)

    print(f"\n{'=' * 68}")
    print(f"  {name}: trace={use_trace}, command queues={num_cqs}, "
          f"pipelined={pipelined}")
    print(f"{'=' * 68}", flush=True)

    device = None
    pipeline = None
    try:
        print("  opening device ...", flush=True)
        device = ttnn.open_device(**open_kwargs)

        if get_mesh_mappers is not None:
            inputs_mesh_mapper, _, output_mesh_composer = get_mesh_mappers(device)
        else:
            inputs_mesh_mapper, output_mesh_composer = None, None

        print("  building model ...", flush=True)
        infra = YOLOv4PerformanceRunnerInfra(
            device,
            args.batch,
            args.act_dtype,
            args.weight_dtype,
            model_location_generator=None,
            resolution=resolution,
            mesh_mapper=inputs_mesh_mapper,
            mesh_composer=output_mesh_composer,
        )

        tt_inputs_host, dram_cfg, l1_cfg = infra.setup_dram_sharded_input(device)

        pipeline = create_pipeline_from_config(
            config=PipelineConfig(
                use_trace=use_trace,
                num_command_queues=num_cqs,
                all_transfers_on_separate_command_queue=False,
            ),
            model=YoloV4PipelineRunner(infra),
            device=device,
            dram_input_memory_config=dram_cfg,
            l1_input_memory_config=l1_cfg,
        )

        print("  compiling (slow on a cold kernel cache) ...", flush=True)
        t0 = time.perf_counter()
        pipeline.compile(tt_inputs_host)
        record["compile_s"] = round(time.perf_counter() - t0, 3)
        print(f"  compiled in {record['compile_s']} s", flush=True)

        print(f"  {args.warmup} warm-up frames ...", flush=True)
        for _ in range(args.warmup):
            pipeline.enqueue([tt_inputs_host]).pop_all()

        print(f"  {args.iters} measured frames ...", flush=True)
        first_output = None

        if pipelined:
            # Throughput mode: hand the pipeline everything, then collect. Only
            # a total is meaningful, there are no per-frame boundaries to time.
            host_inputs = [tt_inputs_host] * args.iters
            t0 = time.perf_counter()
            outputs = pipeline.enqueue(host_inputs).pop_all()
            total_ms = (time.perf_counter() - t0) * 1000.0
            record["total_ms"] = round(total_ms, 3)
            record["per_frame_ms"] = round(total_ms / args.iters, 3)
            record["stats"] = None
            if outputs:
                first_output = outputs[0]
        else:
            # Latency mode: one frame in, one frame out, as the perf test does.
            samples = []
            for i in range(args.iters):
                t0 = time.perf_counter()
                out = pipeline.enqueue([tt_inputs_host]).pop_all()
                samples.append((time.perf_counter() - t0) * 1000.0)
                if i == 0 and out:
                    first_output = out[0]
                if (i + 1) % 100 == 0:
                    print(f"    {i + 1}/{args.iters}", flush=True)
            record["stats"] = summarise(samples)
            record["per_frame_ms"] = record["stats"]["mean"]
            record["total_ms"] = round(sum(samples), 3)

        record["fps"] = round(1000.0 / record["per_frame_ms"], 1)

        # Correctness. A trace replaying a stale buffer would look fast and be
        # wrong, so this is not optional.
        if first_output is None:
            record["validated"] = "SKIPPED: pipeline returned no output to check"
        else:
            try:
                infra.validate(first_output)
                record["validated"] = "passed"
            except Exception as exc:  # noqa: BLE001 - want the message, not a stop
                record["validated"] = f"FAILED: {exc}".splitlines()[0][:300]

        record["ok"] = True
        print(f"  {record['per_frame_ms']} ms a frame, {record['fps']} fps, "
              f"validation {record['validated']}", flush=True)

    except Exception as exc:  # noqa: BLE001 - one bad config must not end the run
        record["error"] = f"{type(exc).__name__}: {exc}".splitlines()[0][:400]
        record["traceback"] = traceback.format_exc()[-1500:]
        print(f"  FAILED: {record['error']}", flush=True)

    finally:
        if pipeline is not None:
            try:
                pipeline.cleanup()
            except Exception:  # noqa: BLE001
                pass
        if device is not None:
            try:
                ttnn.close_device(device)
            except Exception:  # noqa: BLE001
                pass

    return record

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--resolution", type=int, default=320, choices=(320, 640))
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--only", default=None,
                    help="comma-separated config names, to rerun just one")
    ap.add_argument("--out", default=None, help="JSON output path")
    args = ap.parse_args()

    # Inert in this tree, see the module docstring. Kept as a variable so that
    # if the dtype is ever wired through, the lever is already here.
    args.act_dtype = ttnn.bfloat8_b
    args.weight_dtype = ttnn.bfloat8_b

    wanted = None if args.only is None else {s.strip() for s in args.only.split(",")}
    configs = [c for c in CONFIGS if wanted is None or c[0] in wanted]
    if not configs:
        sys.exit(f"--only matched nothing. Names: {', '.join(c[0] for c in CONFIGS)}")

    print(f"resolution : {args.resolution}x{args.resolution}")
    print(f"batch      : {args.batch}")
    print(f"iterations : {args.iters} measured, {args.warmup} warm-up")
    print(f"reference  : {REFERENCE_MS} ms from "
          f"models/demos/yolov4/tests/perf/test_e2e_performant.py")

    results = [run_one(*c, args) for c in configs]
    by_name = {r["name"]: r for r in results if r.get("ok")}

    print(f"\n{'=' * 68}")
    print(f"  model time by configuration, {args.resolution}x{args.resolution}, "
          f"{args.iters} frames")
    print(f"{'=' * 68}")
    print(f"  {'configuration':<22}{'ms/frame':>10}{'fps':>8}{'median':>9}"
          f"{'p99':>9}  validation")
    for r in results:
        if not r.get("ok"):
            print(f"  {r['name']:<22}{'FAILED':>10}   {r.get('error', '')[:40]}")
            continue
        st = r.get("stats") or {}
        med = f"{st['median']:.2f}" if st else "-"
        p99 = f"{st['p99']:.2f}" if st else "-"
        print(f"  {r['name']:<22}{r['per_frame_ms']:>10.3f}{r['fps']:>8.1f}"
              f"{med:>9}{p99:>9}  {r.get('validated', '-')}")

    # The split. Levers are applied cumulatively, so each line is what that
    # step is worth given the ones before it.
    have_span = ("plain" in by_name and "trace_2cq" in by_name
                 and by_name["plain"]["per_frame_ms"] != by_name["trace_2cq"]["per_frame_ms"])
    span = (by_name["plain"]["per_frame_ms"] - by_name["trace_2cq"]["per_frame_ms"]
            if have_span else None)

    print("\n  what each step is worth, applied in this order:")

    def step(a, b, label):
        ra, rb = by_name.get(a), by_name.get(b)
        if not ra or not rb:
            print(f"    {label:<34} not measured, {a} or {b} did not run")
            return
        d = ra["per_frame_ms"] - rb["per_frame_ms"]
        if span:
            print(f"    {label:<34} {d:8.3f} ms   ({100.0 * d / span:.0f}% of the gap)")
        else:
            print(f"    {label:<34} {d:8.3f} ms")

    step("plain", "trace", "capture and replay the graph")
    step("trace", "trace_2cq", "add the second command queue")

    # Anything that failed its numerical check is producing wrong output, so
    # its timings describe a model nobody would ship. Say that before the split,
    # because a reader who sees only milliseconds will compare them anyway.
    failed = [r["name"] for r in results
              if r.get("ok") and str(r.get("validated", "")).startswith("FAILED")]
    if failed:
        print(f"\n  WRONG OUTPUT: {', '.join(failed)} did not match the reference "
              f"output.")
        print( "    The timings above are real but they describe a model that is "
               "not computing")
        print( "    the right answer, so they cannot be compared against a "
               "configuration that is.")

    # REFERENCE_MS is the perf test's figure at 320 and only at 320. Comparing a
    # 640 run against it once reported "DISAGREES ... treat the split as
    # unsound", which blamed the harness for a resolution it was never measured
    # at.
    if "trace_2cq" in by_name and args.resolution == 320:
        got = by_name["trace_2cq"]["per_frame_ms"]
        off = 100.0 * (got - REFERENCE_MS) / REFERENCE_MS
        verdict = ("harness agrees with the published figure" if abs(off) < 15
                   else "DISAGREES with the published figure; treat the split as unsound")
        print(f"\n  trace_2cq measured {got:.3f} ms against the published "
              f"{REFERENCE_MS} ms ({off:+.0f}%)")
        print(f"  {verdict}")
    elif "trace_2cq" in by_name:
        print(f"\n  trace_2cq measured "
              f"{by_name['trace_2cq']['per_frame_ms']:.3f} ms. The published "
              f"{REFERENCE_MS} ms is a 320x320 figure, so there is nothing to "
              f"check this against at {args.resolution}.")

    if "trace_2cq" in by_name and "trace_2cq_pipelined" in by_name:
        lat = by_name["trace_2cq"]["per_frame_ms"]
        thr = by_name["trace_2cq_pipelined"]["per_frame_ms"]
        print(f"\n  one frame at a time {lat:.3f} ms a frame, whole batch "
              f"enqueued {thr:.3f} ms a frame")
        print(f"  the second queue is worth a further {lat - thr:.3f} ms a frame "
              f"once there is something to overlap")

    out = args.out or (f"bench_levers_{args.resolution}_"
                       f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    payload = {
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "resolution": args.resolution,
        "batch": args.batch,
        "iterations": args.iters,
        "warmup_discarded": args.warmup,
        "reference_ms": REFERENCE_MS,
        "reference_source": "models/demos/yolov4/tests/perf/test_e2e_performant.py",
        "dtype_note": ("act_dtype and weight_dtype are inert in this tree: used only "
                       "in log messages by YOLOv4PerformanceRunnerInfra. Activations "
                       "are hardcoded bfloat16 in _setup_l1_sharded_input, weights "
                       "hardcoded bfloat8_b in common.py:create_conv2d_config."),
        "configurations": results,
    }
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  written to {os.path.abspath(out)}")

if __name__ == "__main__":
    main()
