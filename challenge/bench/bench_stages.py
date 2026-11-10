#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Measure where one frame's time actually goes, on the machine with the card.

    Date      2026-09-23
    Version   v0.2  -- run on hardware 2026-09-23; closing note
              updated 2026-09-24 once bench_levers.py had split
              the model stage

WHY THIS EXISTS
    A frame time read off the webcam demonstration's on-screen panel is not a
    measurement, for three reasons that this script removes one by one:

      1. It is never written to a file, so nothing can be traced back to a
         machine, a build or a day.
      2. Its stopwatch starts AFTER cv2.imdecode, so it excludes the JPEG
         decode that the figure claims to include.
      3. Its stopwatch starts BEFORE the device lock, and the browser keeps up
         to four frames in flight, so each reading includes waiting for up to
         three other frames. It measures a queue, not a frame.

    This script has no browser, no network, no threads and no lock. One frame at
    a time, each stage timed separately with perf_counter, N iterations, and the
    result written to JSON so it can be cited.

WHAT IT TIMES, IN ORDER
    decode     cv2.imdecode of the JPEG bytes, as the server receives them
    prepare    resize to the model input, BGR to RGB, to torch tensor
    upload     host to device, including a synchronise so the cost lands here
               rather than being absorbed by the next stage
    model      YOLOv4Runner.run, ending in the read back to host
    boxes      post_processing: max-and-argmax, then overlap removal

    The five add up to the honest per-frame cost on the machine. Where the
    model stage's cost actually goes, and how much of it can be taken off, is
    yours to find out. Copy this file's shape when you do: one thing at a
    time, N iterations, warm-up discarded, result written to a file.

WHAT IT DELIBERATELY DOES NOT MEASURE
    Anything involving the laptop, the browser or the network. In a car the
    camera is on the same board as the chip, so that time would not exist and
    it has no business in the challenge's headline number.

HOW TO RUN
    Use bench/run_bench.sh, which sets the environment for you. To run it by
    hand, from the tt-metal root with its python_env active:

        python bench/bench_stages.py --iters 300

    First run compiles kernels and takes a few minutes. Later runs are quick.
"""

import argparse
import contextlib
import io
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import torch

import ttnn
from models.demos.yolov4.common import (
    YOLOV4_L1_SMALL_SIZE,
    image_to_tensor,
    load_torch_model,
)
from models.demos.yolov4.post_processing import post_processing
from models.demos.yolov4.runner.runner import YOLOv4Runner
from models.demos.yolov4.tt.model_preprocessing import create_yolov4_model_parameters

STAGES = ["decode", "prepare", "upload", "model", "boxes"]

def find_sync():
    """ttnn has moved this call around between releases, so probe for it.

    Without a synchronise after the upload, ttnn may return before the transfer
    has finished and the cost turns up in the model stage instead. If no
    synchronise can be found the run still works, but the upload and model
    figures should then be read as one number.
    """
    for name in ("synchronize_device", "device_synchronize", "synchronize"):
        fn = getattr(ttnn, name, None)
        if callable(fn):
            return name, fn
    dev = getattr(ttnn, "device", None)
    if dev is not None:
        for name in ("synchronize_device", "Synchronize"):
            fn = getattr(dev, name, None)
            if callable(fn):
                return f"device.{name}", fn
    return None, None

def to_device(device, tensor):
    """Input preparation copied from tests/pcc/test_ttnn_yolov4.py.

    The channel count is padded from 3 to 16 for the sharded memory config.
    """
    n, c, h, w = tensor.shape
    if c == 3:
        c = 16
    mem = ttnn.create_sharded_memory_config(
        [n, c, h, w], ttnn.CoreGrid(x=8, y=8), ttnn.ShardStrategy.HEIGHT
    )
    tt_in = ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    return tt_in.to(device, mem)

def summarise(samples_ms):
    s = sorted(samples_ms)
    n = len(s)

    def pct(p):
        # nearest-rank, which is honest for the small n we run here
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

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", default="models/demos/yolov4/resources/giraffe.jpg",
                    help="a real JPEG, so the decode cost is realistic")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20,
                    help="discarded; the first frame compiles kernels")
    ap.add_argument("--resolution", type=int, default=320, choices=(320, 640))
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--nms", type=float, default=0.4)
    ap.add_argument("--out", default=None, help="JSON output path")
    args = ap.parse_args()

    if not os.path.exists(args.image):
        sys.exit(f"image not found: {args.image}\n"
                 f"Run this from the tt-metal root, or pass --image.")

    with open(args.image, "rb") as fh:
        jpeg_bytes = fh.read()
    print(f"image   : {args.image} ({len(jpeg_bytes)} bytes)")

    resolution = (args.resolution, args.resolution)
    sync_name, sync = find_sync()
    print(f"sync    : {sync_name or 'NOT FOUND, upload and model are one number'}")

    print("opening device ...", flush=True)
    device = ttnn.open_device(device_id=0, l1_small_size=YOLOV4_L1_SMALL_SIZE)
    try:
        print("loading torch weights ...", flush=True)
        torch_model = load_torch_model(None)
        dummy = torch.zeros(1, 3, resolution[1], resolution[0])
        print("building ttnn parameters ...", flush=True)
        parameters = create_yolov4_model_parameters(torch_model, dummy, resolution, device)
        runner = YOLOv4Runner(device, parameters, resolution)

        t = {k: [] for k in STAGES}
        post_first_output = ""

        total_iters = args.warmup + args.iters
        print(f"running {args.warmup} warm-up + {args.iters} measured frames "
              f"at {args.resolution}x{args.resolution} ...", flush=True)

        for i in range(total_iters):
            measured = i >= args.warmup

            t0 = time.perf_counter()
            frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
            t1 = time.perf_counter()

            img = cv2.resize(frame, resolution)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = image_to_tensor(img)
            t2 = time.perf_counter()

            tt_in = to_device(device, tensor)
            if sync is not None:
                try:
                    sync(device)
                except TypeError:
                    sync()
            t3 = time.perf_counter()

            boxes, confs = runner.run(tt_in)
            t4 = time.perf_counter()

            # post_processing prints its own argmax/NMS split on every call.
            # Keep the first one, discard the rest so the terminal stays usable.
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                post_processing(None, args.conf, args.nms, [boxes, confs])
            t5 = time.perf_counter()
            if measured and not post_first_output:
                post_first_output = buf.getvalue().strip()

            if measured:
                t["decode"].append((t1 - t0) * 1000.0)
                t["prepare"].append((t2 - t1) * 1000.0)
                t["upload"].append((t3 - t2) * 1000.0)
                t["model"].append((t4 - t3) * 1000.0)
                t["boxes"].append((t5 - t4) * 1000.0)

            if measured and (i - args.warmup + 1) % 50 == 0:
                done = i - args.warmup + 1
                print(f"  {done}/{args.iters}", flush=True)
    finally:
        ttnn.close_device(device)

    per_frame = [sum(t[k][j] for k in STAGES) for j in range(args.iters)]

    result = {
        "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "image": args.image,
        "resolution": args.resolution,
        "iterations": args.iters,
        "warmup_discarded": args.warmup,
        "conf_thresh": args.conf,
        "nms_thresh": args.nms,
        "synchronise_used": sync_name,
        "path": "the detector as supplied, unmodified",
        "stages_ms": {k: summarise(t[k]) for k in STAGES},
        "per_frame_total_ms": summarise(per_frame),
        "post_processing_internal_split_first_frame": post_first_output,
    }
    mean_total = result["per_frame_total_ms"]["mean"]
    result["implied_fps_mean"] = round(1000.0 / mean_total, 1) if mean_total else None
    worst = result["per_frame_total_ms"]["max"]
    result["implied_fps_worst_frame"] = round(1000.0 / worst, 1) if worst else None

    out = args.out or (f"bench_stages_{args.resolution}_"
                       f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)

    print()
    print("=" * 66)
    print(f"  per frame, {args.resolution}x{args.resolution}, plain path, "
          f"{args.iters} frames")
    print("=" * 66)
    print(f"  {'stage':<10}{'mean':>9}{'median':>9}{'p99':>9}{'max':>9}   "
          f"{'share':>6}")
    for k in STAGES:
        s = result["stages_ms"][k]
        share = 100.0 * s["mean"] / mean_total if mean_total else 0.0
        print(f"  {k:<10}{s['mean']:>9.3f}{s['median']:>9.3f}"
              f"{s['p99']:>9.3f}{s['max']:>9.3f}   {share:>5.1f}%")
    tot = result["per_frame_total_ms"]
    print("  " + "-" * 62)
    print(f"  {'TOTAL':<10}{tot['mean']:>9.3f}{tot['median']:>9.3f}"
          f"{tot['p99']:>9.3f}{tot['max']:>9.3f}")
    print()
    print(f"  mean        {result['implied_fps_mean']} frames a second")
    print(f"  worst frame {result['implied_fps_worst_frame']} frames a second")
    print()
    if post_first_output:
        print("  post_processing reported internally on the first measured frame:")
        for line in post_first_output.splitlines():
            print(f"    {line}")
        print()
    print(f"  written to {out}")
    print()
    print("  The model row is the whole of the detector on the path as")
    print("  supplied. It is not the limit of this card. Before you attack")
    print("  it, try the same run at a different resolution and see how much")
    print("  the model row actually moves. What that tells you is the point.")

if __name__ == "__main__":
    main()
