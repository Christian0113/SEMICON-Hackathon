#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Does the batch-2 trace compile, and does its per-frame split match batch 1?

    Lever    L04 (branch L04-batch2-genboxes)
    Date     2026-11-11

WHY THIS EXISTS
    L04 runs two images through one trace. The host then has to carve each
    image's 6300 rows back out of a batched device output laid out
    (head, anchor, image, position). That index math was verified off-card
    against a torch emulation of the device's op sequence -- but an emulation
    only proves the index math matches MY MODEL of the card, not the card.
    If ttnn's reshape or permute orders tile-padded rows differently than the
    emulation assumed, batch 2 produces plausible-looking detections that are
    silently wrong, and only a small mAP drift would ever show it.

    This probe settles that question directly, and in ~40 s per side instead
    of a 10-repeat evaluate.py run:

      1. it runs N real photographs through the submission at BATCH = 1,
         capturing the exact (boxes, confs) tensors that reach
         fast_detections -- i.e. milestone 8's own post-processing input;
      2. the same N photographs at BATCH = 2, capturing the same tensors
         after the batched split;
      3. compares them for BIT equality, per photograph.

    Bit equality is the right bar, not PCC: both sides read the same bf16
    device values through the same float16 conversion, so the only thing that
    can differ is WHICH rows land where. Any mismatch is an ordering bug.

    It also fails loudly and early if the batch-2 trace does not compile,
    which is L04's other open risk, and prints the device output shapes so a
    shape surprise is visible rather than inferred.

HOW TO RUN (on the instance, from ~/challenge)
    python bench/probe_batch2.py --batch 1 --out /tmp/b1.npz --n 8
    python bench/probe_batch2.py --batch 2 --out /tmp/b2.npz --n 8
    python bench/probe_batch2.py --compare /tmp/b1.npz /tmp/b2.npz

    Each side opens the card itself, so they must be separate processes:
    trace_region_size and num_command_queues are fixed at device-open time.
    --n must be even for a clean batch-2 pairing (an odd tail duplicates the
    last frame, which is correct but makes the probe's own report noisier).

    A PASS here does not say batch 2 is FASTER -- only that it is correct.
    The speed verdict still comes from a same-session 10-repeat evaluate.py
    A/B, per the L03 measurement protocol.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHALLENGE = os.path.dirname(HERE)
HARNESS = os.path.join(CHALLENGE, "harness")


def load_submission(path):
    """Same loader evaluate.py uses, so the module sees the same sys.path."""
    sys.path.insert(0, HARNESS)
    spec = importlib.util.spec_from_file_location("team_submission", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["team_submission"] = mod
    spec.loader.exec_module(mod)
    return mod


def open_card(sub):
    """Open the device exactly as harness/evaluate.py does."""
    import ttnn
    try:
        from models.demos.yolov4.common import YOLOV4_L1_SMALL_SIZE as L1
    except Exception:
        L1 = 10960
    opts = dict(getattr(sub, "DEVICE_OPTIONS", None) or {})
    opts.pop("device_id", None)
    opts.pop("l1_small_size", None)
    device = ttnn.open_device(device_id=0, l1_small_size=L1, **opts)
    print(f"  card opened (l1_small_size={L1}, {opts})")
    return device


def collect_by_path(mod_path, batch, paths):
    """Run `paths` through the submission at `batch`, returning
    {photograph: (boxes, confs)} as fast_detections would have received them.

    The pairing of tensors to photographs comes from the submission itself,
    not from guessing thread completion order: fast_detections is patched to
    return its OWN inputs as that photograph's "detections", so run() hands
    back {path: (boxes, confs)} through its existing path/future bookkeeping.
    """
    sub = load_submission(mod_path)
    sub.BATCH = batch
    if not hasattr(sub, "_l04_genboxes_call"):
        raise SystemExit(f"{mod_path} has no _l04_genboxes_call: not the L04 file")

    real_fast_detections = sub.fast_detections

    def passthrough(boxes, confs, W, H, *a, **k):
        return (np.asarray(boxes, dtype=np.float16).copy(),
                np.asarray(confs, dtype=np.float16).copy())

    device = open_card(sub)
    tt_home = os.environ.get("TT_METAL_HOME") or os.path.expanduser("~/tt-metal")
    if os.path.isdir(tt_home):
        os.chdir(tt_home)

    s = sub.Submission()
    out = {}
    try:
        t0 = time.perf_counter()
        s.setup(device)
        print(f"  setup {time.perf_counter() - t0:.1f}s  (batch={batch})")
        print(f"  [probe] batch knob in the loaded module: {sub.BATCH}")

        n_det = sum(len(v) for v in s.run(paths).values())
        print(f"  warm pass: {n_det} detections over {len(paths)} photographs")

        try:
            import ttnn
            shapes = [tuple(ttnn.to_torch(t).shape) for t in s.slots[0]]
            print(f"  device output shapes: boxes {shapes[0]}, confs {shapes[1]}")
        except Exception as e:
            print(f"  (could not read output shapes: {type(e).__name__}: {e})")

        sub.fast_detections = passthrough
        res = s.run(paths)
        sub.fast_detections = real_fast_detections
        for p, v in res.items():
            if isinstance(v, tuple):          # a captured (boxes, confs)
                out[p] = v
            # else: empty-gated photograph, skipped on both sides
    finally:
        try:
            s.teardown()
        finally:
            import ttnn
            ttnn.close_device(device)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", default=os.path.join(CHALLENGE, "my_submission.py"))
    ap.add_argument("--images", default=os.path.join(CHALLENGE, "practice", "images"))
    ap.add_argument("--n", type=int, default=8, help="photographs to probe (even)")
    ap.add_argument("--batch", type=int, choices=(1, 2), help="collect this side")
    ap.add_argument("--out", help="where to save this side's tensors (.npz)")
    ap.add_argument("--compare", nargs=2, metavar=("B1", "B2"),
                    help="compare two saved sides instead of running the card")
    a = ap.parse_args()

    if a.compare:
        z1, z2 = np.load(a.compare[0]), np.load(a.compare[1])
        keys1 = {k[:-7] for k in z1.files if k.endswith("__boxes")}
        keys2 = {k[:-7] for k in z2.files if k.endswith("__boxes")}
        if keys1 != keys2:
            print(f"FAIL: different photographs captured "
                  f"({len(keys1)} vs {len(keys2)}); only in first: "
                  f"{sorted(keys1 - keys2)[:3]}, only in second: "
                  f"{sorted(keys2 - keys1)[:3]}")
            return 1
        bad = 0
        for k in sorted(keys1):
            for what in ("boxes", "confs"):
                x, y = z1[f"{k}__{what}"], z2[f"{k}__{what}"]
                if x.shape != y.shape:
                    print(f"FAIL {k} {what}: shape {x.shape} vs {y.shape}")
                    bad += 1
                elif not np.array_equal(x, y):
                    d = (x.astype(np.float32) != y.astype(np.float32))
                    rows = np.flatnonzero(d.reshape(d.shape[1], -1).any(axis=1))
                    print(f"FAIL {k} {what}: {int(d.sum())} values differ, "
                          f"{rows.size} rows, first row {int(rows[0])}")
                    bad += 1
        if bad:
            print(f"\n{bad} mismatch(es): the batched split does NOT reproduce "
                  f"batch-1 row order. Fix _frame_rows before trusting any "
                  f"batch-2 timing.")
            return 1
        print(f"PASS: all {len(keys1)} photographs bit-identical in boxes and "
              f"confs. The batch-2 split reproduces milestone-8 row order on "
              f"the card.")
        return 0

    if a.batch is None or not a.out:
        ap.error("give --batch and --out, or --compare")
    if a.n % 2:
        ap.error("--n must be even so batch 2 pairs cleanly")

    names = sorted(os.listdir(a.images))[:a.n]
    paths = [os.path.join(a.images, n) for n in names]
    print(f"  {len(paths)} photographs from {a.images}")

    res = collect_by_path(a.submission, a.batch, paths)
    print(f"  captured {len(res)} photographs (others were empty-gated)")
    flat = {}
    for p, (b, c) in res.items():
        k = os.path.basename(p)
        flat[f"{k}__boxes"] = b
        flat[f"{k}__confs"] = c
    np.savez_compressed(a.out, **flat)
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
