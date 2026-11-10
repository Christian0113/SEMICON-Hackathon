#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Microbenchmark for YOLOv4 Post-Processing & NMS.

Isolates the CPU detection post-processing stage from the Tenstorrent hardware,
verifying 100% numerical parity and benchmarking speedup between:
  1. Baseline post_processing() from models.demos.yolov4.post_processing
  2. fast_post_process_v1() (Vectorized PyTorch + Batched C++ NMS)

Usage:
    python bench_postprocess.py [--iters 300] [--conf 0.001] [--nms 0.4]
"""
from __future__ import annotations
import argparse
import sys
import time
import numpy as np
import torch

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


# ==============================================================================
# 1. Baseline Reference Implementation (from models/demos/yolov4/post_processing.py)
# ==============================================================================

def baseline_nms_cpu(boxes, confs, nms_thresh=0.5):
    """Exact baseline pure-Python NMS loop."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]
        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        over = inter / (areas[idx_self] + areas[idx_other] - inter)
        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)


def baseline_post_processing(output, conf_thresh, nms_thresh, suppress_print=True):
    """Exact baseline post-processing logic (NumPy + Python 80-class loop)."""
    box_array = output[0]
    confs = output[1].float()

    if isinstance(box_array, torch.Tensor):
        box_array = box_array.cpu().detach().numpy()
    if isinstance(confs, torch.Tensor):
        confs = confs.cpu().detach().numpy()

    num_classes = confs.shape[2]
    box_array = box_array[:, :, 0]  # [batch, num, 4]

    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    bboxes_batch = []
    for i in range(box_array.shape[0]):
        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes = []
        for j in range(num_classes):
            cls_argwhere = (l_max_id == j)
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            if len(ll_max_conf) == 0:
                continue

            keep = baseline_nms_cpu(ll_box_array, ll_max_conf, nms_thresh)
            if keep.size > 0:
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    bboxes.append([
                        float(ll_box_array[k, 0]),
                        float(ll_box_array[k, 1]),
                        float(ll_box_array[k, 2]),
                        float(ll_box_array[k, 3]),
                        float(ll_max_conf[k]),
                        float(ll_max_conf[k]),
                        int(ll_max_id[k]),
                    ])
        bboxes_batch.append(bboxes)

    return bboxes_batch


# ==============================================================================
# 2. Optimized Vectorized Implementation (Idea 1: 100% Parity)
# ==============================================================================

def _pure_torch_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
    """Vectorized pure-PyTorch NMS fallback if torchvision is unavailable."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i.item())
        if order.numel() == 1:
            break
        other = order[1:]
        xx1 = torch.maximum(x1[i], x1[other])
        yy1 = torch.maximum(y1[i], y1[other])
        xx2 = torch.minimum(x2[i], x2[other])
        yy2 = torch.minimum(y2[i], y2[other])

        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h
        iou = inter / (areas[i] + areas[other] - inter)

        mask = iou <= iou_thresh
        order = other[mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def fast_post_process_v1(output, conf_thresh: float, nms_thresh: float) -> list[list]:
    """Vectorized PyTorch post-processing with 100% parity against baseline.
    
    Eliminates the 80-class Python loop and pure-Python NMS while preserving
    exact single-label (max + argmax) semantics.
    """
    box_array, confs = output[0], output[1]

    # Ensure tensors are PyTorch Float32 on CPU
    if not isinstance(box_array, torch.Tensor):
        box_array = torch.from_numpy(box_array)
    if not isinstance(confs, torch.Tensor):
        confs = torch.from_numpy(confs)

    box_array = box_array.float()
    confs = confs.float()

    if box_array.dim() == 4:
        box_array = box_array[:, :, 0]  # [batch, num, 4]

    batch_size = box_array.shape[0]
    # Single-label: Max confidence and corresponding class ID across all classes
    max_conf, max_id = torch.max(confs, dim=-1)  # [batch, num]

    bboxes_batch = []
    for b in range(batch_size):
        # 1. Vectorized threshold mask
        mask = max_conf[b] > conf_thresh
        if not mask.any():
            bboxes_batch.append([])
            continue

        cand_boxes = box_array[b, mask]      # [M, 4]
        cand_scores = max_conf[b, mask]      # [M]
        cand_classes = max_id[b, mask]       # [M]

        # 2. Per-class batched NMS
        if HAS_TORCHVISION:
            # torchvision.ops.batched_nms offsets boxes by class to do multi-class NMS in C++
            keep = torchvision.ops.batched_nms(cand_boxes, cand_scores, cand_classes, nms_thresh)
        else:
            # Pure PyTorch per-class fallback
            unique_classes = torch.unique(cand_classes)
            keep_list = []
            for cls in unique_classes:
                cls_idx = (cand_classes == cls).nonzero(as_tuple=True)[0]
                cls_keep = _pure_torch_nms(cand_boxes[cls_idx], cand_scores[cls_idx], nms_thresh)
                keep_list.append(cls_idx[cls_keep])
            keep = torch.cat(keep_list) if keep_list else torch.empty(0, dtype=torch.long)

        if keep.numel() == 0:
            bboxes_batch.append([])
            continue

        final_boxes = cand_boxes[keep]
        final_scores = cand_scores[keep]
        final_classes = cand_classes[keep]

        # 3. Format output
        bboxes = []
        for k in range(final_boxes.shape[0]):
            bboxes.append([
                float(final_boxes[k, 0]),
                float(final_boxes[k, 1]),
                float(final_boxes[k, 2]),
                float(final_boxes[k, 3]),
                float(final_scores[k]),
                float(final_scores[k]),
                int(final_classes[k]),
            ])
        bboxes_batch.append(bboxes)

    return bboxes_batch


# ==============================================================================
# 3. Benchmark & Correctness Verification Harness
# ==============================================================================

def generate_synthetic_output(num_anchors=6300, num_classes=80, seed=42):
    """Generate realistic YOLOv4 raw output tensors."""
    torch.manual_seed(seed)
    # Boxes: normalized (x1, y1, x2, y2) in [0, 1]
    cx = torch.rand(1, num_anchors, 1, 1)
    cy = torch.rand(1, num_anchors, 1, 1)
    w = torch.rand(1, num_anchors, 1, 1) * 0.3 + 0.02
    h = torch.rand(1, num_anchors, 1, 1) * 0.3 + 0.02
    x1 = torch.clamp(cx - w / 2, 0.0, 1.0)
    y1 = torch.clamp(cy - h / 2, 0.0, 1.0)
    x2 = torch.clamp(cx + w / 2, 0.0, 1.0)
    y2 = torch.clamp(cy + h / 2, 0.0, 1.0)
    boxes = torch.cat([x1, y1, x2, y2], dim=-1)

    # Confs: sigmoid outputs in [0, 1] with heavy low-confidence tail
    confs = torch.sigmoid(torch.randn(1, num_anchors, num_classes) * 2.0 - 4.0)
    return boxes, confs


def compare_detections(det1: list[list], det2: list[list], atol=1e-5) -> tuple[bool, str]:
    """Assert 100% numerical parity between two detection lists."""
    if len(det1) != len(det2):
        return False, f"Batch size mismatch: {len(det1)} vs {len(det2)}"

    for b in range(len(det1)):
        d1, d2 = det1[b], det2[b]
        if len(d1) != len(d2):
            return False, f"Batch {b}: Detection count mismatch: {len(d1)} vs {len(d2)}"

        # Sort both by (class_id, score descending, x1) for deterministic comparison
        s1 = sorted(d1, key=lambda x: (x[6], -x[4], x[0]))
        s2 = sorted(d2, key=lambda x: (x[6], -x[4], x[0]))

        for i, (box1, box2) in enumerate(zip(s1, s2)):
            # Compare class ID
            if box1[6] != box2[6]:
                return False, f"Detection {i}: Class ID mismatch: {box1[6]} vs {box2[6]}"
            # Compare score
            if abs(box1[4] - box2[4]) > atol:
                return False, f"Detection {i}: Score mismatch: {box1[4]:.6f} vs {box2[4]:.6f}"
            # Compare coordinates (x1, y1, x2, y2)
            for c in range(4):
                if abs(box1[c] - box2[c]) > atol:
                    return False, f"Detection {i}: Coord {c} mismatch: {box1[c]:.6f} vs {box2[c]:.6f}"

    return True, "100% Identical"


def benchmark_fn(fn, args, iters=300, warmup=20):
    """Run warm-up and timed benchmark loop."""
    for _ in range(warmup):
        fn(*args)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    return {
        "mean": float(np.mean(times)),
        "median": float(np.median(times)),
        "p95": float(np.percentile(times, 95)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
    }


def main():
    parser = argparse.ArgumentParser(description="Post-processing benchmark and verification")
    parser.add_argument("--iters", type=int, default=300, help="Number of benchmark iterations")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.4, help="NMS IoU threshold")
    args = parser.parse_args()

    print("=" * 70)
    print("  YOLOv4 Post-Processing & NMS Optimization: Idea 1 Benchmark")
    print(f"  PyTorch: {torch.__version__} | torchvision: {HAS_TORCHVISION}")
    print(f"  Iterations: {args.iters} | CONF_THRESH: {args.conf} | NMS_THRESH: {args.nms}")
    print("=" * 70)

    # 1. Generate Input Tensors
    boxes, confs = generate_synthetic_output(num_anchors=6300, num_classes=80)
    output = [boxes, confs]

    # 2. Check Numerical Parity
    print("\n[1/3] Verifying Numerical Parity...")
    det_base = baseline_post_processing(output, args.conf, args.nms)
    det_opt = fast_post_process_v1(output, args.conf, args.nms)

    passed, reason = compare_detections(det_base, det_opt)
    n_dets = len(det_base[0]) if det_base else 0
    if not passed:
        print(f"  FAILED: {reason}")
        sys.exit(1)
    else:
        print(f"  PASSED: Both versions output {n_dets} identical boxes ({reason})")

    # 3. Benchmark Baseline
    print(f"\n[2/3] Benchmarking Baseline post_processing ({args.iters} runs)...")
    res_base = benchmark_fn(baseline_post_processing, (output, args.conf, args.nms), iters=args.iters)
    print(f"  Baseline Mean:   {res_base['mean']:.3f} ms / frame  ({1000.0 / res_base['mean']:.1f} FPS)")
    print(f"  Baseline Median: {res_base['median']:.3f} ms / frame")

    # 4. Benchmark Vectorized Idea 1
    print(f"\n[3/3] Benchmarking fast_post_process_v1 ({args.iters} runs)...")
    res_opt = benchmark_fn(fast_post_process_v1, (output, args.conf, args.nms), iters=args.iters)
    print(f"  Optimized Mean:   {res_opt['mean']:.3f} ms / frame  ({1000.0 / res_opt['mean']:.1f} FPS)")
    print(f"  Optimized Median: {res_opt['median']:.3f} ms / frame")

    # 5. Summary
    speedup = res_base['mean'] / res_opt['mean']
    time_saved = res_base['mean'] - res_opt['mean']

    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY:")
    print(f"  Baseline Latency:   {res_base['mean']:.3f} ms")
    print(f"  Optimized Latency:  {res_opt['mean']:.3f} ms")
    print(f"  Time Saved:         {time_saved:.3f} ms per frame")
    print(f"  Speedup:            {speedup:.1f}x faster")
    print("=" * 70)


if __name__ == "__main__":
    main()
