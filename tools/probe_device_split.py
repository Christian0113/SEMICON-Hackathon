#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Step 1 of docs/GUIDE_SPEED_ACCURACY_MEMORY.md (WP-1): measure before cutting.

One card run, about 3 minutes. Every phase is wrapped so a failure in one does
not lose the others, and the JSON is rewritten after every phase.

  A  Split the per-frame CQ0 chain of the milestone-8 pipeline:
       trace_only        execute_trace back to back, no read
       trace_plus_read   + the two non-blocking D2H copies (boxes, confs)
       production        executor.execute() (H2D on CQ1, reshard, trace) + read
     read = trace_plus_read - trace_only  -> the prize for WP-4 (read on CQ1)

  C  Traced cost of the box decode that WP-3 moves to the host:
       epilogue_supplied  genboxes x3 + output epilogue (yolov4.py:67-95), as a trace
       epilogue_raw       the WP-3 replacement: 3x (to L1, untilize) + concat, as a trace
     prize = epilogue_supplied - epilogue_raw
     Also saves the raw head tensors and the supplied boxes/confs of the same
     frame to --out-dir, so WP-3's float32 decode can be checked on a laptop.

  B  ttnn.max / ttnn.argmax over the 80 classes of a [1,6300,80] tensor
     (untraced, so it includes dispatch: an upper bound). Decides whether the
     trunc_confs route is admissible.

  D  Facts the next steps need: which copy-into-preallocated APIs exist,
     conv double-buffer defaults, head output dtypes/layouts.

Run on the instance (nothing else using the card):

    source ~/tt-metal/python_env/bin/activate
    export TT_METAL_HOME=~/tt-metal PYTHONPATH=~/tt-metal
    python ~/SEMICON-Hackathon/tools/probe_device_split.py \
        --out-dir ~/SEMICON-Hackathon/experiments/probes 2>&1 | tee /tmp/probe.log
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import torch
import ttnn

from models.demos.yolov4.common import YOLOV4_L1_SMALL_SIZE
from models.demos.yolov4.runner.performant_runner_infra import YOLOv4PerformanceRunnerInfra
from models.demos.yolov4.runner.pipeline_runner import YoloV4PipelineRunner
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config

try:
    from models.demos.utils.common_demo_utils import get_mesh_mappers
except Exception:
    get_mesh_mappers = None

TRACE_REGION_SIZE = 6434816
RESOLUTION = (320, 320)

RESULT: dict = {}
OUT_JSON = None


def save():
    with open(OUT_JSON, "w") as f:
        json.dump(RESULT, f, indent=2, default=str)


def phase(name):
    def deco(fn):
        def run(*a, **k):
            print(f"\n==== phase {name} ====", flush=True)
            t0 = time.perf_counter()
            try:
                out = fn(*a, **k)
                RESULT.setdefault("phase_status", {})[name] = "ok"
                return out
            except Exception as e:  # keep going, record why
                RESULT.setdefault("phase_status", {})[name] = f"FAILED: {type(e).__name__}: {e}"[:400]
                RESULT.setdefault("tracebacks", {})[name] = traceback.format_exc()[-2500:]
                print(f"  phase {name} FAILED: {type(e).__name__}: {e}", flush=True)
                return None
            finally:
                RESULT.setdefault("phase_seconds", {})[name] = round(time.perf_counter() - t0, 1)
                save()
        return run
    return deco


def timed(device, fn, iters, warmup):
    """Mean ms per call of fn, with fn only enqueueing work; one sync at the end."""
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) * 1000.0 / iters


def nbytes(t):
    try:
        return int(t.volume()) * (2 if t.dtype == ttnn.bfloat16 else 4)
    except Exception:
        return None


def describe(t):
    d = {"shape": list(t.shape), "dtype": str(t.dtype), "layout": str(t.layout)}
    try:
        mc = t.memory_config()
        d["memory"] = f"{mc.memory_layout} / {mc.buffer_type}"
    except Exception:
        pass
    return d


# --------------------------------------------------------------------------- A
@phase("A_cq0_split")
def phase_a(device, pipeline, seed_host, iters, warmup):
    ex = pipeline.executor
    pipeline.preallocate_output_tensors_on_host(2)
    slot = pipeline.output_tensors[0]
    dev_out = ex.output_tensor
    read_cq = ex.get_read_cq()
    RESULT["A_output_tensors"] = [describe(t) for t in dev_out]
    RESULT["A_output_bytes"] = [nbytes(t) for t in dev_out]

    def trace_only():
        ttnn.execute_trace(device, ex.trace_id, cq_id=0, blocking=False)

    def read(which):
        for i in which:
            ttnn.copy_device_to_host_tensor(dev_out[i], slot[i], blocking=False, cq_id=read_cq)

    def trace_read_all():
        trace_only(); read((0, 1))

    def trace_read_boxes():
        trace_only(); read((0,))

    def production():
        next(iter(ex.execute([seed_host])))
        read((0, 1))

    r = {}
    r["trace_only_ms"] = timed(device, trace_only, iters, warmup)
    r["trace_plus_read_ms"] = timed(device, trace_read_all, iters, warmup)
    r["trace_plus_boxes_read_ms"] = timed(device, trace_read_boxes, iters, warmup)
    r["production_ms"] = timed(device, production, iters, warmup)
    r["read_ms"] = r["trace_plus_read_ms"] - r["trace_only_ms"]
    r["confs_read_ms"] = r["trace_plus_read_ms"] - r["trace_plus_boxes_read_ms"]
    r["h2d_reshard_wait_ms"] = r["production_ms"] - r["trace_plus_read_ms"]
    total = sum(b for b in RESULT["A_output_bytes"] if b)
    if r["read_ms"] > 0:
        r["read_gb_per_s"] = total / (r["read_ms"] * 1e-3) / 1e9
    RESULT["A"] = {k: round(v, 4) for k, v in r.items()}
    for k, v in RESULT["A"].items():
        print(f"  {k:<28} {v:8.4f}")

    # keep the supplied outputs of this frame for the laptop parity check of WP-3
    ttnn.synchronize_device(device)
    production()
    ttnn.synchronize_device(device)
    return [ttnn.to_torch(slot[0]).float().numpy(), ttnn.to_torch(slot[1]).float().numpy()]


# --------------------------------------------------------------------------- C
class _StopAtHead(Exception):
    pass


def head_outputs(device, infra, dram_cfg, l1_cfg, seed_host):
    """Run the model untraced up to the head and return (x4, x5, x6) on device."""
    model = infra.ttnn_yolov4_model
    real_head = model.head
    box = {}

    class Grab:
        def __call__(self, inputs):
            box["out"] = real_head(inputs)
            raise _StopAtHead()

    dram = ttnn.allocate_tensor_on_device(seed_host.shape, seed_host.dtype, seed_host.layout, device, dram_cfg)
    ttnn.copy_host_to_device_tensor(seed_host, dram, cq_id=0)
    x = ttnn.reshard(dram, l1_cfg)
    model.head = Grab()
    try:
        model(x)
    except _StopAtHead:
        pass
    finally:
        model.head = real_head
    ttnn.deallocate(dram)
    return box["out"]


def epilogue_supplied(device, model, x4, x5, x6):
    """yolov4.py:67-95 at 320x320, verbatim."""
    b0 = model.boxes_confs_0(device, x4)
    b1 = model.boxes_confs_1(device, x5)
    b2 = model.boxes_confs_2(device, x6)
    c1 = ttnn.to_layout(b0[1], ttnn.ROW_MAJOR_LAYOUT)
    c2 = ttnn.to_layout(b1[1], ttnn.ROW_MAJOR_LAYOUT)
    c3 = ttnn.to_layout(b2[1], ttnn.ROW_MAJOR_LAYOUT)
    confs = ttnn.concat([c1, c2, c3], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
    o1 = ttnn.to_layout(b0[0], ttnn.ROW_MAJOR_LAYOUT)
    o2 = ttnn.to_layout(b1[0], ttnn.ROW_MAJOR_LAYOUT)
    o3 = ttnn.to_layout(b2[0], ttnn.ROW_MAJOR_LAYOUT)
    o1 = ttnn.reshape(o1, (1, 4, 1, 4800))
    o2 = ttnn.reshape(o2, (1, 4, 1, 1200))
    o3 = ttnn.pad(o3, ((0, 0), (0, 0), (0, 0), (0, 28)), 0)
    o3 = ttnn.reshape(o3, (1, 4, 1, 384))
    o1 = ttnn.permute(o1, (0, 2, 3, 1))
    o2 = ttnn.permute(o2, (0, 2, 3, 1))
    o3 = ttnn.permute(o3, (0, 2, 3, 1))
    boxes = ttnn.concat([o1, o2, o3], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
    return boxes, confs


def epilogue_raw(x4, x5, x6, concat=True):
    """The WP-3 replacement: hand the raw head logits to the host."""
    outs = []
    for t in (x4, x5, x6):
        t = ttnn.to_memory_config(t, ttnn.L1_MEMORY_CONFIG)
        t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        outs.append(t)
    if not concat:
        return outs
    return ttnn.concat(outs, dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)


def trace_cost(device, fn, iters, warmup, label):
    out = fn()                                   # compile + allocate
    ttnn.synchronize_device(device)
    _free(out)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ms = timed(device, lambda: ttnn.execute_trace(device, tid, cq_id=0, blocking=False), iters, warmup)
    print(f"  {label:<28} {ms:8.4f} ms (traced)")
    return ms, tid, out


def _free(out):
    if isinstance(out, (list, tuple)):
        for o in out:
            _free(o)
    elif isinstance(out, ttnn.Tensor) and out.is_allocated():
        ttnn.deallocate(out, force=True)


@phase("C_box_decode")
def phase_c(device, infra, dram_cfg, l1_cfg, seed_host, iters, warmup, out_dir):
    x4, x5, x6 = head_outputs(device, infra, dram_cfg, l1_cfg, seed_host)
    RESULT["C_head_outputs"] = [describe(t) for t in (x4, x5, x6)]
    np.savez_compressed(os.path.join(out_dir, "head_raw_giraffe.npz"),
                        x4=ttnn.to_torch(x4).float().numpy(),
                        x5=ttnn.to_torch(x5).float().numpy(),
                        x6=ttnn.to_torch(x6).float().numpy())
    model = infra.ttnn_yolov4_model
    r = {}
    ms, tid, _ = trace_cost(device, lambda: epilogue_supplied(device, model, x4, x5, x6),
                            iters, warmup, "epilogue_supplied")
    r["epilogue_supplied_ms"] = ms
    ttnn.release_trace(device, tid)
    try:
        ms, tid, out = trace_cost(device, lambda: epilogue_raw(x4, x5, x6), iters, warmup, "epilogue_raw")
        RESULT["C_raw_output"] = describe(out)
        r["epilogue_raw_ms"] = ms
    except Exception as e:
        print(f"  concat variant failed ({e}); timing without the concat")
        RESULT["C_raw_concat_error"] = str(e)[:300]
        ms, tid, out = trace_cost(device, lambda: epilogue_raw(x4, x5, x6, concat=False),
                                  iters, warmup, "epilogue_raw_no_concat")
        RESULT["C_raw_output"] = [describe(t) for t in out]
        r["epilogue_raw_ms"] = ms
    ttnn.release_trace(device, tid)
    r["prize_ms"] = r["epilogue_supplied_ms"] - r["epilogue_raw_ms"]
    RESULT["C"] = {k: round(v, 4) for k, v in r.items()}
    print(f"  {'prize_ms (WP-3)':<28} {RESULT['C']['prize_ms']:8.4f}")


# --------------------------------------------------------------------------- B
@phase("B_argmax")
def phase_b(device, iters):
    t = ttnn.from_torch(torch.rand(1, 6300, 80), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    r = {}
    for name, fn in (("max", lambda: ttnn.max(t, dim=-1, keepdim=True)),
                     ("argmax", lambda: ttnn.argmax(t, dim=-1, keepdim=True))):
        try:
            outs = []
            ms = timed(device, lambda: outs.append(fn()), iters, 5)
            for o in outs:
                _free(o)
            r[f"{name}_untraced_ms"] = round(ms, 4)
            print(f"  {name + '_untraced_ms':<28} {ms:8.4f}")
        except Exception as e:
            r[f"{name}_error"] = f"{type(e).__name__}: {e}"[:300]
            print(f"  {name} failed: {e}")
    RESULT["B"] = r


# --------------------------------------------------------------------------- D
@phase("D_api_facts")
def phase_d():
    facts = {}
    for name in ("copy", "assign", "to_memory_config", "reshard", "interleaved_to_sharded",
                 "allocate_tensor_on_host", "event_synchronize", "record_event", "wait_for_event"):
        fn = getattr(ttnn, name, None)
        if fn is None:
            facts[name] = "MISSING"
            continue
        doc = (getattr(fn, "__doc__", "") or "").strip().splitlines()
        sig = ""
        try:
            sig = str(inspect.signature(fn))
        except Exception:
            pass
        facts[name] = {"signature": sig[:300], "doc_head": " | ".join(doc[:6])[:600],
                       "mentions_output_tensor": "output_tensor" in (fn.__doc__ or "") + sig}
    try:
        from models.tt_cnn.tt import builder
        f = {k: v.default for k, v in builder.Conv2dConfiguration.__dataclass_fields__.items()
             if "double_buffer" in k or k in ("deallocate_activation", "reallocate_halo_output")}
        facts["Conv2dConfiguration_defaults"] = {k: str(v) for k, v in f.items()}
    except Exception as e:
        facts["Conv2dConfiguration_defaults"] = f"error: {e}"
    RESULT["D"] = facts
    for k, v in facts.items():
        print(f"  {k}: {v if isinstance(v, str) else v.get('signature') or v.get('mentions_output_tensor', v)}")


# --------------------------------------------------------------------------- main
def main():
    global OUT_JSON
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out-dir", default=os.path.expanduser("~/SEMICON-Hackathon/experiments/probes"))
    a = ap.parse_args()
    out_dir = os.path.abspath(os.path.expanduser(a.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_JSON = os.path.join(out_dir, f"probe_device_split_{stamp}.json")

    tt_home = os.environ.get("TT_METAL_HOME") or os.path.expanduser("~/tt-metal")
    os.chdir(tt_home)                          # the model loads its weights by relative path
    RESULT.update({"measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "host": platform.node(), "iters": a.iters, "warmup": a.warmup,
                   "resolution": list(RESOLUTION), "tt_metal_home": tt_home})
    try:
        import subprocess
        RESULT["tt_metal_commit"] = subprocess.run(["git", "-C", tt_home, "rev-parse", "--short", "HEAD"],
                                                   capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    save()

    phase_d()
    device = ttnn.open_device(device_id=0, l1_small_size=YOLOV4_L1_SMALL_SIZE,
                              trace_region_size=2 * TRACE_REGION_SIZE, num_command_queues=2)
    try:
        in_map = out_comp = None
        if get_mesh_mappers is not None:
            in_map, _, out_comp = get_mesh_mappers(device)
        infra = YOLOv4PerformanceRunnerInfra(device, 1, ttnn.bfloat8_b, ttnn.bfloat8_b,
                                             model_location_generator=None, resolution=RESOLUTION,
                                             mesh_mapper=in_map, mesh_composer=out_comp)
        seed_host, dram_cfg, l1_cfg = infra.setup_dram_sharded_input(device)
        pipeline = create_pipeline_from_config(
            config=PipelineConfig(use_trace=True, num_command_queues=2,
                                  all_transfers_on_separate_command_queue=False),
            model=YoloV4PipelineRunner(infra), device=device,
            dram_input_memory_config=dram_cfg, l1_input_memory_config=l1_cfg)
        pipeline.compile(seed_host)

        supplied = phase_a(device, pipeline, seed_host, a.iters, a.warmup)
        if supplied is not None:
            np.savez_compressed(os.path.join(out_dir, "supplied_out_giraffe.npz"),
                                boxes=supplied[0], confs=supplied[1])

        # Phase C needs L1 that the main trace's persistent tensors occupy: release them first.
        ex = pipeline.executor
        pipeline.cleanup()
        for name in ("output_tensor", "l1_input_tensor", "dram_input_tensor"):
            _free(getattr(ex, name, None))
        ttnn.synchronize_device(device)

        phase_c(device, infra, dram_cfg, l1_cfg, seed_host, a.iters, a.warmup, out_dir)
        phase_b(device, max(50, a.iters // 3))
    finally:
        save()
        ttnn.close_device(device)

    print("\n==== summary ====")
    A, C, B = RESULT.get("A", {}), RESULT.get("C", {}), RESULT.get("B", {})
    print(f"  per-frame production chain   {A.get('production_ms', float('nan')):.3f} ms")
    print(f"    trace only                 {A.get('trace_only_ms', float('nan')):.3f} ms")
    print(f"    D2H read on CQ0 (WP-4)     {A.get('read_ms', float('nan')):.3f} ms")
    print(f"  genboxes+epilogue (traced)   {C.get('epilogue_supplied_ms', float('nan')):.3f} ms")
    print(f"  raw-head replacement         {C.get('epilogue_raw_ms', float('nan')):.3f} ms")
    print(f"  WP-3 device prize            {C.get('prize_ms', float('nan')):.3f} ms")
    print(f"  argmax / max untraced        {B.get('argmax_untraced_ms')} / {B.get('max_untraced_ms')} ms")
    print(f"  phase status                 {RESULT.get('phase_status')}")
    print(f"\n  written to {OUT_JSON}")


if __name__ == "__main__":
    sys.exit(main())
