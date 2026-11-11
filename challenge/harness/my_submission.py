# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Team Submission, milestone 9 (claude_fix step 2): host-side float32 box decode.

Milestone 9 change, on top of milestone 8 below (HOST_DECODE switch):
  The trace now ends at the three detection-head convolutions. The box decode
  that used to run on the card (genboxes x3 + output reshaping, about 230 tiny
  slice/concat/tilize ops per frame, all in bf16) is done on the host in
  float32 inside the post-processing workers. The card does less work per frame
  and the box coordinates lose no precision to bf16. At setup the host decode
  is checked against the float32 PyTorch reference (the same PCC test the model
  ships with); if the check fails or anything raises, setup falls back to the
  milestone-8 path automatically.

Milestone 8 description follows.

Builds on milestone 7 (my_submission.py, 7.36 ms/frame). Post-processing is
unchanged (same fast_detections, same thresholds, same gates), so mAP and the
detection count must match milestone 7 exactly. Only the scheduling of host and
card work changes.

Why milestone 7 stops at ~7.4 ms while the card can do ~5.45 ms/frame
(bench_levers trace_2cq_pipelined):

  Pipeline.enqueue(32 frames) issues 32 host->card writes from the main thread.
  Each write waits (on the card) for the previous frame's reshard, so the
  command queue fills and enqueue() itself blocks until most of the chunk has
  run. The "overlapped" NMS of the previous chunk therefore mostly runs AFTER
  the card is nearly idle, and pop_all() = synchronize_device() then drains the
  card completely at every chunk boundary. Card and CPU take turns instead of
  working at the same time.

What this version does instead:

  1. Per-frame streaming with a bounded in-flight window (IN_FLIGHT frames).
     The main thread keeps exactly IN_FLIGHT frames queued on the card. It never
     calls synchronize_device(); it waits on a per-frame event
     (ttnn.event_synchronize) recorded right after that frame's output read,
     so the card always has the next frames queued while the host works on
     an old one.
  2. A ring of RING preallocated host output buffers. Output reads go
     straight into pinned slots (copy_device_to_host_tensor, non-blocking),
     with no host allocation per frame.
  3. NMS moved off the main thread into POST_WORKERS threads. torch and
     torchvision release the GIL, so several frames are post-processed at once
     and the main thread only enqueues, waits on events and converts.
  4. Input built directly as bfloat16 via a 256-entry lookup table that ttnn
     itself generated at setup, so the values are bit-identical to
     image_to_tensor() + from_torch(), but with half the memory traffic and no
     float32 intermediate. BGR->RGB is folded into the same pass. A parity
     check at setup turns this off automatically if the result differs.
  5. torch.set_num_threads(1): we parallelise across frames, so torch's
     intra-op pool (96 threads on this host) only causes oversubscription.

Set STREAMING = False to fall back to the milestone-7 chunked path for A/B.
"""
from __future__ import annotations

import gc
import math
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

cv2.setNumThreads(1)

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

try:
    import ttnn
    from models.demos.yolov4.common import get_model_result, image_to_tensor
    from models.demos.yolov4.runner.performant_runner_infra import YOLOv4PerformanceRunnerInfra
    from models.demos.yolov4.runner.pipeline_runner import YoloV4PipelineRunner
    from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config
    HAS_TTNN = True
except ImportError:
    HAS_TTNN = False

try:
    from models.demos.utils.common_demo_utils import get_mesh_mappers
except Exception:
    get_mesh_mappers = None

import submission_api as api

RESOLUTION = (320, 320)   # supervisors: stay at 320. (320: 3.88 ms / 0.4656; 512: 4.80 / 0.6007 in my_submission_v2.py)
# Square, a multiple of 32, and (side*side/64) a multiple of 32 for the input sharding:
# 320, 384, 448, 512, 576 or 640. Sizes other than 320/640 use the 640 conv settings,
# need the host box decode, and are checked against the PyTorch reference at setup.
CUSTOM_RES = RESOLUTION not in ((320, 320), (640, 640))
CONF_THRESH = float(os.environ.get("SUB_CONF", 0.001))   # scoring floor; raising it trims NMS work, costs recall
NMS_THRESH = 0.4
EMPTY_THRESH = 0.05
HOST_SALIENCY_THRESH = 3.0

_OBJ_GATE = math.log(CONF_THRESH / (1.0 - CONF_THRESH)) - 0.05
_EMPTY_GATE = math.log(EMPTY_THRESH / (1.0 - EMPTY_THRESH)) if EMPTY_THRESH > 0.0 else -999.0

TRACE_REGION_SIZE = 6434816

# ---- scheduling knobs (tune on the card; see docs/IDEA_8_STREAMING_PIPELINE.md)
STREAMING = True     # False -> milestone-7 chunked path
IN_FLIGHT = int(os.environ.get("SUB_IN_FLIGHT", 4))        # frames queued on the card before the host waits on the oldest
RING = int(os.environ.get("SUB_RING", 48))            # preallocated host output slots; 48 gives large buffer for workers
LOOKAHEAD = 64       # how many frames ahead decoding runs
DECODERS = int(os.environ.get("SUB_DECODERS", 4))   # 4 JPEG decode threads; eliminates jpeg-wait (~0.24 ms)
POST_WORKERS = int(os.environ.get("SUB_POST_WORKERS", 2))  # 2 post workers (optimal under container CPU quota)
TORCH_THREADS = 1    # intra-op threads per torch call
FAST_INPUT = True    # bf16 LUT input path (auto-disabled if parity check fails)
CHUNK = 32           # only used when STREAMING = False
HOST_DECODE = True   # milestone 9: box decode on the host in float32 (falls back automatically)
CARD_REDUCE = os.environ.get("SUB_CARD_REDUCE", "0") == "1"   # class max/argmax on the card; measured slower (4.18 vs 3.76 ms)
DECODE_MIN_PCC = (0.99, 0.98)   # boxes, confs: host decode vs the card's own decode of the same frame
SWITCH_INTERVAL = float(os.environ.get("SUB_SWITCH_INTERVAL", 0.005))   # GIL hand-over interval during run(); Python default 0.005
STATS = True         # one timing line per run()

_COCO_IDS = [api.to_coco_id(i) for i in range(80)]


# ----------------------------------------------------------------------------
# Post-processing: identical to my_submission.py (milestone 7).
# ----------------------------------------------------------------------------
def _pure_torch_nms(boxes, scores, iou_thresh):
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
        inter = torch.clamp(xx2 - xx1, min=0.0) * torch.clamp(yy2 - yy1, min=0.0)
        iou = inter / (areas[i] + areas[other] - inter)
        order = other[iou <= iou_thresh]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def _class_nms(boxes, scores, classes, iou_thresh):
    if HAS_TORCHVISION:
        return torchvision.ops.batched_nms(boxes, scores, classes, iou_thresh)
    keep_list = []
    for cls in torch.unique(classes):
        idx = (classes == cls).nonzero(as_tuple=True)[0]
        keep_list.append(idx[_pure_torch_nms(boxes[idx], scores[idx], iou_thresh)])
    return torch.cat(keep_list) if keep_list else torch.empty(0, dtype=torch.long)


def fast_detections(boxes, confs, W, H, conf_thresh=CONF_THRESH, nms_thresh=NMS_THRESH):
    boxes = torch.as_tensor(boxes)
    confs = torch.as_tensor(confs)
    b = boxes.reshape(boxes.shape[0], boxes.shape[1], 4)[0].float()
    c = confs[0].float()

    max_conf, max_id = c.max(dim=1)
    if EMPTY_THRESH > 0.0 and float(max_conf.max()) < EMPTY_THRESH:
        return []

    m = max_conf > conf_thresh
    if not bool(m.any()):
        return []
    return _finish(b[m], max_conf[m], max_id[m], W, H, nms_thresh)


def _finish(cb, cs, ci, W, H, nms_thresh=NMS_THRESH):
    """Candidates (normalised boxes, max class score, class id) -> COCO detections."""
    keep = _class_nms(cb, cs, ci, nms_thresh)
    if keep.numel() == 0:
        return []

    scale = torch.tensor([W, H, W, H], dtype=cb.dtype, device=cb.device)
    px = cb[keep] * scale
    px = torch.minimum(px.clamp(min=0.0), scale)
    valid = (px[:, 2] > px[:, 0]) & (px[:, 3] > px[:, 1])
    if not bool(valid.any()):
        return []
    px = px[valid]
    s = cs[keep][valid].double()
    ids = ci[keep][valid].tolist()

    xywh = torch.stack((px[:, 0], px[:, 1], px[:, 2] - px[:, 0], px[:, 3] - px[:, 1]), 1).tolist()
    scores = (s * s).tolist()
    return [{"category_id": _COCO_IDS[k], "bbox": bb, "score": sc}
            for k, bb, sc in zip(ids, xywh, scores)]


# ----------------------------------------------------------------------------
# Milestone 9: host-side float32 decode of the raw head outputs.
# Same math as models/demos/yolov4/tt/genboxes.py and the PyTorch reference
# (post_processing.YoloLayer._yolo_forward_dynamic), in float32.
# ----------------------------------------------------------------------------
_ANCHORS = (  # anchor width, height in grid cells, per anchor, per scale (genboxes.py)
    torch.tensor([[1.5, 2.0], [2.375, 4.5], [5.0, 3.5]]),                  # 40x40, stride 8
    torch.tensor([[2.25, 4.6875], [4.75, 3.4375], [4.5, 9.125]]),          # 20x20, stride 16
    torch.tensor([[4.4375, 3.4375], [6.0, 7.59375], [14.34375, 12.53125]]),  # 10x10, stride 32
)
_GRIDS = tuple(RESOLUTION[0] // st for st in (8, 16, 32))     # (40, 20, 10) at 320, (80, 40, 20) at 640


class _Tables:
    """Per-row constants for the three scales stacked into one tensor
    (rows: S1*S1 of the finest grid, then S2*S2, then S3*S3)."""

    def __init__(self, grids):
        self.grids = grids
        self.rows = sum(S * S for S in grids)
        self.grid_xy = []
        rs, rgx, rgy = [], [], []
        for S in grids:
            i = torch.arange(S * S)
            gx, gy = (i % S).float(), torch.div(i, S, rounding_mode="floor").float()
            self.grid_xy.append((gx.reshape(-1, 1), gy.reshape(-1, 1)))   # row i -> (x = i % S, y = i // S)
            rs.append(torch.full((S * S,), float(S)))
            rgx.append(gx)
            rgy.append(gy)
        self.rs, self.rgx, self.rgy = torch.cat(rs), torch.cat(rgx), torch.cat(rgy)
        self.raw = torch.cat([(_ANCHORS[g][:, 0:1] / S).expand(3, S * S) for g, S in enumerate(grids)], 1)
        self.rah = torch.cat([(_ANCHORS[g][:, 1:2] / S).expand(3, S * S) for g, S in enumerate(grids)], 1)
        # position of (anchor, row) in the card / reference order (scale, then anchor, then row)
        self.rkey = torch.cat([torch.arange(3).view(3, 1) * (S * S) + torch.arange(S * S).view(1, -1)
                               + 3 * sum(T * T for T in grids[:g]) for g, S in enumerate(grids)], 1)


_TABLES = {}


def _tables(grids=_GRIDS):
    tb = _TABLES.get(grids)
    if tb is None:
        tb = _TABLES[grids] = _Tables(grids)
    return tb


def decode_heads(raws, grids=_GRIDS):
    """raws: three float32 tensors, one per scale, [HW, >=255] (channel a*85 + k).

    Returns boxes [1, N, 4] (normalised x1, y1, x2, y2) and confs [1, N, 80],
    anchor-major within each scale, finest scale first: the same order
    as the card's output and the reference.
    """
    tbl = _tables(grids)
    boxes, confs = [], []
    for g, (t, S) in enumerate(zip(raws, grids)):
        hw = S * S
        t = t[:hw, :255].reshape(hw, 3, 85)
        gx, gy = tbl.grid_xy[g]
        bx = (torch.sigmoid(t[..., 0]) + gx) / S
        by = (torch.sigmoid(t[..., 1]) + gy) / S
        bw = torch.exp(t[..., 2]) * (_ANCHORS[g][:, 0] / S)
        bh = torch.exp(t[..., 3]) * (_ANCHORS[g][:, 1] / S)
        x1 = bx - bw * 0.5
        y1 = by - bh * 0.5
        b = torch.stack((x1, y1, x1 + bw, y1 + bh), -1)                  # [hw, 3, 4]
        c = torch.sigmoid(t[..., 4:5]) * torch.sigmoid(t[..., 5:85])      # [hw, 3, 80]
        boxes.append(b.transpose(0, 1).reshape(-1, 4))
        confs.append(c.transpose(0, 1).reshape(-1, 80))
    return torch.cat(boxes).unsqueeze(0), torch.cat(confs).unsqueeze(0)


def _raw_to_float(host_struct):
    """Host slot (three ttnn host tensors) -> three fresh [HW, C] tensors
    (copies, so the ring slot can be reused as soon as this returns)."""
    out = []
    for h in host_struct:
        t = ttnn.to_torch(h)
        out.append(t.reshape(-1, t.shape[-1]).clone())
    return out


def _slot_detections(host_struct, W, H):
    """Worker job: read head tensor(s) straight out of the host slot and decode them."""
    if isinstance(host_struct, (list, tuple)):
        raws = [ttnn.to_torch(h).reshape(-1, h.shape[-1]) for h in host_struct]
        return detections_from_heads(raws, W, H)
    t = ttnn.to_torch(host_struct)
    return detections_from_heads(t.reshape(-1, t.shape[-1]), W, H)


def detections_full(raws, W, H, grids=_GRIDS):
    """Reference host path: decode all N x 80 scores, then fast_detections."""
    boxes, confs = decode_heads([r.float() for r in raws], grids)
    return fast_detections(boxes, confs, W, H)


def detections_from_heads(raws, W, H, conf_thresh=CONF_THRESH, grids=_GRIDS):
    """Fast host decode of detection head outputs.

    1. Empty-frame shortcut: if max objectness logit < logit(EMPTY_THRESH),
       returns [] instantly with zero class or sigmoid computation.
    2. Direct zero-allocation view [3, rows, 85].
    3. Direct bfloat16 max over 80 classes (eliminates 2MB float32 allocation).
    4. Candidate selection: conf = sigmoid(obj) * sigmoid(cls_max) > conf_thresh.
    5. Box coordinates decoded in float32 for candidate anchors, sorted in reference order.
    """
    tbl = _tables(grids)
    if isinstance(raws, (list, tuple)):
        raw_rows = torch.cat([r[:S * S, :255] for r, S in zip(raws, grids)])
    else:
        raw_rows = raws[:tbl.rows, :255]

    t = raw_rows.reshape(tbl.rows, 3, 85).transpose(0, 1)

    obj = t[..., 4]
    if EMPTY_THRESH > 0.0 and float(obj.max()) < _EMPTY_GATE:
        return []

    cand_ml, cand_mi = t[..., 5:85].max(dim=-1)
    cand_conf = torch.sigmoid(obj.float()) * torch.sigmoid(cand_ml.float())

    if EMPTY_THRESH > 0.0 and float(cand_conf.max()) < EMPTY_THRESH:
        return []

    keep = cand_conf > conf_thresh
    if not bool(keep.any()):
        return []

    a, r = keep.nonzero(as_tuple=True)
    conf = cand_conf[a, r]
    mi = cand_mi[a, r]

    o = tbl.rkey[a, r].argsort()
    a, r = a[o], r[o]
    conf, mi = conf[o], mi[o]

    tb = t[a, r, :4].float()
    S = tbl.rs[r]
    bx = (torch.sigmoid(tb[:, 0]) + tbl.rgx[r]) / S
    by = (torch.sigmoid(tb[:, 1]) + tbl.rgy[r]) / S
    bw = torch.exp(tb[:, 2]) * tbl.raw[a, r]
    bh = torch.exp(tb[:, 3]) * tbl.rah[a, r]
    x1 = bx - bw * 0.5
    y1 = by - bh * 0.5
    return _finish(torch.stack((x1, y1, x1 + bw, y1 + bh), -1), conf, mi, W, H)


def _pcc(a, b):
    a = torch.as_tensor(a).detach().double().flatten()
    b = torch.as_tensor(b).detach().double().flatten()
    return float(torch.corrcoef(torch.stack((a, b)))[0, 1])


class YoloV4HeadsRunner:
    """Like YoloV4PipelineRunner, but the trace stops after the detection head
    and hands back the raw head tensors (ROW_MAJOR bf16, L1 interleaved).
    When concat=True, concatenates the 3 heads along dim 2 into one [1, 1, 2100, 256]
    tensor so only 1 D2H PCIe transfer and 1 ttnn.to_torch call occur per frame.
    The first block is yolov4.py TtYOLOv4.__call__ verbatim (320x320)."""

    def __init__(self, infra, concat=False):
        self.infra = infra
        self.concat = concat

    def __call__(self, x):
        m = self.infra.ttnn_yolov4_model
        N, C, H, W = x.shape
        if C < 16:
            nchw = ttnn.pad(x, ((0, 0), (0, 16 - C), (0, 0), (0, 0)), value=0.0)
        else:
            nchw = x
        nhwc = ttnn.permute(nchw, (0, 2, 3, 1))
        ttnn.deallocate(nchw)
        ttnn.deallocate(x)
        nhwc = ttnn.reallocate(nhwc)
        x = ttnn.reshape(nhwc, [1, 1, nhwc.shape[0] * nhwc.shape[1] * nhwc.shape[2], nhwc.shape[-1]])
        d1 = m.down1(x)
        d2 = m.down2(d1)
        ttnn.deallocate(d1)
        d3 = m.down3(d2)
        ttnn.deallocate(d2)
        d4 = m.down4(d3)
        d5 = m.down5(d4)
        x20, x13, x6 = m.neck([d5, d4, d3])
        heads = m.head([x20, x13, x6])
        outs = []
        for t in heads:                                   # same two ops genboxes starts with
            t = ttnn.to_memory_config(t, ttnn.L1_MEMORY_CONFIG)
            outs.append(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT))
        if self.concat:
            return ttnn.concat(outs, dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
        return outs


# ----------------------------------------------------------------------------
# Milestone 10: per-anchor class reduction on the card ("offload to the ASIC").
# For each head tensor [1,1,HW,256] (ROW_MAJOR bf16): permute to [1,1,256,HW],
# slice per anchor the 5 box/objectness rows and the 80 class rows, pad the
# class block to 96 rows with -1e4 and tilize, then on the card:
#     m   = max over the class rows                      (exact, bf16 max)
#     cid = max over rows of (cls == m) * (81 - class_index) (exact; ties -> first, like np.argmax)
# The host receives per anchor 5 logits + m + cid (~100 KB per frame instead of
# the 1 MB head tensors) and does sigmoid / box decode / NMS in float32 exactly
# as before. 16 ops per scale, none of them bf16 coordinate arithmetic.
# ----------------------------------------------------------------------------
def _ceil32(n):
    return (n + 31) // 32 * 32


def _build_reduce_ctx(device, grids=_GRIDS):
    """Index tensors (81 - class_index on rows 0..79, 0 on the 16 pad rows) for
    each scale, [1,3,96,HWp] TILE bf16 in DRAM. Allocated before the trace.
    The card takes the max, so the FIRST class among ties wins (as np.argmax)."""
    idx = []
    for S in grids:
        HWp = _ceil32(S * S)
        t = torch.zeros(1, 3, 96, HWp)
        t[:, :, :80, :] = (81 - torch.arange(0, 80, dtype=torch.float32)).view(1, 1, 80, 1)
        idx.append(ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG))
    return {"idx": idx, "eq": None, "pad": None}


def _max_rows(x, dim=2):
    try:
        return ttnn.max(x, dim=dim, keepdim=True)
    except TypeError:
        return ttnn.max(x, dim=dim)


def _eq_bcast_h(a, b, mode):
    """mask = (a == b) with b broadcast over dim 2; three API variants."""
    if mode == "eq":
        return ttnn.eq(a, b)
    if mode == "sub_eqz":
        return ttnn.eqz(ttnn.subtract(a, b))
    if mode == "bcast":
        return ttnn.eqz(ttnn.bcast(a, b, ttnn.BcastOpMath.SUB, ttnn.BcastOpDim.H))
    raise ValueError(mode)


def _tilize_pad(cl, shape, mode):
    if mode == "val_padding":
        return ttnn.tilize_with_val_padding(cl, shape, -1e4, memory_config=ttnn.L1_MEMORY_CONFIG)
    if mode == "pad_then_tilize":
        pad_h, pad_w = shape[2] - cl.shape[2], shape[3] - cl.shape[3]
        cl = ttnn.pad(cl, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), value=-1e4)
        return ttnn.to_layout(cl, ttnn.TILE_LAYOUT)
    raise ValueError(mode)


def _reduce_scale(t, HW, idx_dev, ctx):
    x = ttnn.permute(t, (0, 1, 3, 2))                                  # [1,1,256,HW]
    bo = [ttnn.slice(x, [0, 0, 85 * a, 0], [1, 1, 85 * a + 5, HW]) for a in range(3)]
    cl = [ttnn.slice(x, [0, 0, 85 * a + 5, 0], [1, 1, 85 * a + 85, HW]) for a in range(3)]
    bo = ttnn.concat(bo, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)   # [1,3,5,HW]  tx ty tw th obj
    cl = ttnn.concat(cl, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)   # [1,3,80,HW]
    HWp = _ceil32(HW)
    cl = _tilize_pad(cl, [1, 3, 96, HWp], ctx["pad"])
    m = _max_rows(cl)                                                  # [1,3,1,HWp]
    if len(m.shape) == 3:
        m = ttnn.reshape(m, (1, 3, 1, HWp))
    mask = _eq_bcast_h(cl, m, ctx["eq"])
    sel = ttnn.multiply(mask, idx_dev)
    cid = _max_rows(sel)
    if len(cid.shape) == 3:
        cid = ttnn.reshape(cid, (1, 3, 1, HWp))
    m = ttnn.to_layout(m, ttnn.ROW_MAJOR_LAYOUT)
    cid = ttnn.to_layout(cid, ttnn.ROW_MAJOR_LAYOUT)
    return [bo, m, cid]


def _reduce_heads(outs, ctx, grids=_GRIDS):
    parts = [_reduce_scale(t, S * S, idx, ctx) for t, S, idx in zip(outs, grids, ctx["idx"])]
    if not ctx.get("concat"):
        return parts
    # One output tensor [1,3,7,sum(W)] (rows: tx ty tw th obj | max | class code),
    # so the host makes ONE ttnn.to_torch call per frame (each call costs ~0.3 ms
    # under GIL contention, whatever its size).
    per = []
    for bo, m, cid in parts:
        wm = m.shape[-1]
        if bo.shape[-1] != wm:
            bo = ttnn.pad(bo, ((0, 0), (0, 0), (0, 0), (0, wm - bo.shape[-1])), value=0.0)
        per.append(ttnn.concat([bo, m, cid], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG))
    return ttnn.concat(per, dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)


_RED_WIDTHS = []      # per-scale width inside the concatenated tensor (set at setup)
_READ_CALLS = [0]


class YoloV4ReducedRunner(YoloV4HeadsRunner):
    def __init__(self, infra, ctx):
        super().__init__(infra)
        self.ctx = ctx

    def __call__(self, x):
        outs = super().__call__(x)
        red = _reduce_heads(outs, self.ctx)
        _free(outs)
        return red


def _reduced_to_torch(parts, grids=_GRIDS):
    """Host slot ([[bo, m, cid] x 3], or one concatenated tensor) -> per scale
    (bo [3,5,hw] f32, m [3,hw], cid [3,hw])."""
    out = []
    if not isinstance(parts, (list, tuple)):
        t = ttnn.to_torch(parts).float()                     # [1,3,7,sum(W)]
        _READ_CALLS[0] += 1
        off = 0
        for w, S in zip(_RED_WIDTHS, grids):
            hw = S * S
            seg = t[0, :, :, off:off + w]
            out.append((seg[:, :5, :hw], seg[:, 5, :hw], seg[:, 6, :hw]))
            off += w
        return out
    for (bo, m, cid), S in zip(parts, grids):
        _READ_CALLS[0] += 3
        hw = S * S
        bo = ttnn.to_torch(bo).float().reshape(3, 5, -1)[:, :, :hw]
        m = ttnn.to_torch(m).float().reshape(3, -1)[:, :hw]
        cid = ttnn.to_torch(cid).float().reshape(3, -1)[:, :hw]
        out.append((bo, m, cid))
    return out


def detections_from_reduced(parts, W, H, conf_thresh=CONF_THRESH, grids=_GRIDS):
    """Same result as detections_from_heads, from the card-reduced output."""
    t0 = time.perf_counter()
    scales = _reduced_to_torch(parts, grids)
    if STATS:
        _READ_TIMES.append(time.perf_counter() - t0)
    tbl = _tables(grids)
    A, R, T, CF, CI = [], [], [], [], []
    off = 0
    top = 0.0
    for (bo, m, cid), S in zip(scales, grids):
        hw = S * S
        conf = torch.sigmoid(bo[:, 4]) * torch.sigmoid(m)                 # [3,hw]
        top = max(top, float(conf.max()))
        a, i = (conf > conf_thresh).nonzero(as_tuple=True)
        if a.numel():
            A.append(a)
            R.append(i + off)
            T.append(bo[a, :, i][:, :4])
            CF.append(conf[a, i])
            CI.append(81 - cid[a, i].long())
        off += hw
    if EMPTY_THRESH > 0.0 and top < EMPTY_THRESH:
        return []
    if not A:
        return []
    a, r, tb, conf, mi = torch.cat(A), torch.cat(R), torch.cat(T), torch.cat(CF), torch.cat(CI)
    mi = mi.clamp(0, 79)
    o = tbl.rkey[a, r].argsort()
    a, r, tb, conf, mi = a[o], r[o], tb[o], conf[o], mi[o]
    S = tbl.rs[r]
    bx = (torch.sigmoid(tb[:, 0]) + tbl.rgx[r]) / S
    by = (torch.sigmoid(tb[:, 1]) + tbl.rgy[r]) / S
    bw = torch.exp(tb[:, 2]) * tbl.raw[a, r]
    bh = torch.exp(tb[:, 3]) * tbl.rah[a, r]
    x1 = bx - bw * 0.5
    y1 = by - bh * 0.5
    return _finish(torch.stack((x1, y1, x1 + bw, y1 + bh), -1), conf, mi, W, H)


_READ_TIMES = []


def _free(struct):
    if isinstance(struct, (list, tuple)):
        for s in struct:
            _free(s)
    elif struct is not None and struct.is_allocated():
        ttnn.deallocate(struct, force=True)


def _copy_struct_to_host(dev, host, cq_id):
    """Non-blocking card->host copy of a (possibly nested) output structure."""
    if isinstance(dev, (list, tuple)):
        for d, h in zip(dev, host):
            _copy_struct_to_host(d, h, cq_id)
    else:
        ttnn.copy_device_to_host_tensor(dev, host, blocking=False, cq_id=cq_id)


# ----------------------------------------------------------------------------
class Submission(api.Submission):
    DEVICE_OPTIONS = {
        "trace_region_size": TRACE_REGION_SIZE,
        "num_command_queues": 2,
    }

    def setup(self, device):
        gc.disable()
        if device is None or not HAS_TTNN:
            raise RuntimeError("This submission requires a Tenstorrent device.")
        if TORCH_THREADS:
            torch.set_num_threads(TORCH_THREADS)
        self.device = device
        self.resolution = RESOLUTION

        if get_mesh_mappers is not None:
            self.in_mapper, _, self.out_composer = get_mesh_mappers(device)
        else:
            self.in_mapper = self.out_composer = None

        if CUSTOM_RES:
            self._enable_custom_resolution()
        self.infra = YOLOv4PerformanceRunnerInfra(
            device, 1, ttnn.bfloat8_b, ttnn.bfloat8_b,
            model_location_generator=None,
            resolution=self.resolution,
            mesh_mapper=self.in_mapper,
            mesh_composer=self.out_composer,
        )
        self.seed_host, dram_cfg, l1_cfg = self.infra.setup_dram_sharded_input(device)

        self.host_decode = False
        if HOST_DECODE:
            try:
                self.host_decode = self._check_host_decode(dram_cfg, l1_cfg)
            except Exception as e:  # never let an optimisation break setup
                print(f"  [submission] host decode check raised {type(e).__name__}: {e}")
                self.host_decode = False
        if not self.host_decode:
            self.card_reduce = False
        print(f"  [submission] box decode on {'HOST (float32)' if self.host_decode else 'CARD (as supplied)'}"
              f"{' + class max/argmax on CARD' if getattr(self, 'card_reduce', False) else ''}")
        if CUSTOM_RES and not self.host_decode:
            raise RuntimeError(f"RESOLUTION {RESOLUTION} needs the host box decode, which failed (see above); "
                               f"use (320, 320) or (640, 640)")

        def build(runner):
            p = create_pipeline_from_config(
                config=PipelineConfig(use_trace=True, num_command_queues=2,
                                      all_transfers_on_separate_command_queue=False),
                model=runner,
                device=device,
                dram_input_memory_config=dram_cfg,
                l1_input_memory_config=l1_cfg,
            )
            p.compile(self.seed_host)
            return p

        if self.host_decode:
            try:
                runner = (YoloV4ReducedRunner(self.infra, self.reduce_ctx) if self.card_reduce
                          else YoloV4HeadsRunner(self.infra, concat=getattr(self, "concat_heads", False)))
                self.pipeline = build(runner)
            except Exception as e:
                if CUSTOM_RES:
                    raise RuntimeError(f"RESOLUTION {RESOLUTION}: heads-only trace failed ({type(e).__name__}: {e}); "
                                       f"use (320, 320) or (640, 640)")
                print(f"  [submission] heads-only trace failed ({type(e).__name__}: {e}); using the supplied path")
                self.host_decode = False
                try:
                    ttnn.synchronize_device(device)
                except Exception:
                    pass
        if not self.host_decode:
            self.pipeline = build(YoloV4PipelineRunner(self.infra))

        self.streaming = STREAMING and hasattr(ttnn, "event_synchronize")
        if self.streaming:
            assert RING > IN_FLIGHT, "RING must exceed IN_FLIGHT or a slot is overwritten before it is read"
            self.executor = self.pipeline.executor
            self.read_cq = self.executor.get_read_cq()
            self.pipeline.preallocate_output_tensors_on_host(RING)
            self.slots = self.pipeline.output_tensors
        self.lut = self._build_input_lut() if FAST_INPUT else None

        self.pool = ThreadPoolExecutor(max_workers=DECODERS)
        self.post_pool = ThreadPoolExecutor(max_workers=POST_WORKERS)

    def _enable_custom_resolution(self):
        """The supplied model code only knows 320 and 640. Its layers branch on
        'resolution == 320' and otherwise compute shard specs from tensor shapes
        (the 640 path), so for other sizes: build the conv settings as for 640
        (shapes still come from the real input), and stub the on-card box decoder,
        which refuses other sizes and is not used with the host decode."""
        R = RESOLUTION[0]
        if RESOLUTION[0] != RESOLUTION[1] or R % 32 or (R * R // 64) % 32:
            raise RuntimeError(f"RESOLUTION {RESOLUTION} is not usable: use 320, 384, 448, 512, 576 or 640")
        from models.demos.yolov4.tt import model_preprocessing as mp
        from models.demos.yolov4.tt import yolov4 as y4
        if getattr(mp, "_semicon_custom_res", False):
            return
        for name in ("_create_ds1_model_parameters", "_create_ds2_model_parameters", "_create_ds3_model_parameters",
                     "_create_ds4_model_parameters", "_create_ds5_model_parameters", "_create_neck_model_parameters",
                     "_create_head_model_parameters"):
            orig = getattr(mp, name)

            def wrapped(conv_args, resolution=None, *rest, _orig=orig):
                if resolution not in ((320, 320), (640, 640)):
                    resolution = (640, 640)
                return _orig(conv_args, resolution, *rest)
            setattr(mp, name, wrapped)

        class _NoCardBoxDecode:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, *args, **kwargs):
                raise RuntimeError("on-card box decode is not available at this resolution")
        y4.TtGenBoxes = _NoCardBoxDecode
        mp._semicon_custom_res = True
        print(f"  [submission] resolution {RESOLUTION}: 640 conv settings, host box decode only")

    def _check_card_reduce(self, outs, raws):
        """Run the on-card class reduction untraced on the real head tensors,
        verify it against torch (max exact, class index tie-tolerant) and verify
        the detections against the host-only path. Picks the API variants that
        work on this ttnn. Any failure -> the host-only path stays."""
        ctx = _build_reduce_ctx(self.device)
        last = None
        for pad in ("val_padding", "pad_then_tilize"):
            for eq in ("eq", "sub_eqz", "bcast"):
                ctx["pad"], ctx["eq"] = pad, eq
                red = None
                try:
                    red = _reduce_heads(outs, ctx)
                    ttnn.synchronize_device(self.device)
                    scales = _reduced_to_torch(red)
                    ok = True
                    for (bo, m, cid), raw, S in zip(scales, raws, _GRIDS):
                        hw = S * S
                        t = raw[:hw, :255].reshape(hw, 3, 85).transpose(0, 1)      # [3,hw,85]
                        if not torch.equal(bo, t[:, :, :5].transpose(1, 2)):
                            ok = False; last = "box/obj rows differ"; break
                        cls = t[:, :, 5:85]
                        mt = cls.max(dim=-1).values
                        if not torch.equal(m, mt):
                            ok = False; last = f"max differs (max abs {float((m - mt).abs().max()):.3g})"; break
                        ci = (81 - cid.long()).clamp(0, 79)
                        picked = cls.gather(2, ci.unsqueeze(-1)).squeeze(-1)
                        if not torch.equal(picked, mt) or bool((cid < 1).any()):
                            ok = False; last = "class index does not point at the max"; break
                    if ok:
                        host_dets = detections_from_heads(raws, 640, 480)
                        red_dets = detections_from_reduced(red, 640, 480)
                        same = len(host_dets) == len(red_dets) and all(
                            a["category_id"] == b["category_id"] and abs(a["score"] - b["score"]) < 1e-5
                            and max(abs(x - y) for x, y in zip(a["bbox"], b["bbox"])) < 1e-2
                            for a, b in zip(host_dets, red_dets))
                        print(f"  [submission] card reduction ({pad}, {eq}): exact; detections "
                              f"{len(red_dets)} vs {len(host_dets)} host-only, {'identical' if same else 'DIFFERENT'}")
                        if same:
                            self.reduce_ctx = ctx
                            self._try_reduce_concat(outs, ctx, scales)
                            return True
                        last = "detections differ"
                except Exception as e:
                    last = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
                finally:
                    _free(red)
        print(f"  [submission] card reduction unavailable ({last}); host-only decode")
        return False

    def _try_reduce_concat(self, outs, ctx, ref_scales):
        """Enable the single-tensor output if the on-card concat works and
        returns exactly the same values as the three-tensor form."""
        global _RED_WIDTHS
        red = None
        try:
            widths = [int(m.shape[-1]) for (_, m, _) in _reduce_heads(outs, dict(ctx, concat=False))]
            ctx["concat"] = True
            red = _reduce_heads(outs, ctx)
            ttnn.synchronize_device(self.device)
            _RED_WIDTHS = widths
            got = _reduced_to_torch(red)
            ok = all(torch.equal(a, b) for g, r in zip(got, ref_scales) for a, b in zip(g, r))
            if ok:
                print(f"  [submission] card reduction output: one tensor {list(red.shape)} per frame (widths {widths})")
                return
            raise ValueError("values differ from the three-tensor form")
        except Exception as e:
            ctx["concat"] = False
            print(f"  [submission] single-tensor reduction unavailable ({type(e).__name__}: {str(e).splitlines()[0][:120]}); 9 reads per frame")
        finally:
            _free(red)

    def _check_host_decode(self, dram_cfg, l1_cfg):
        """Run the heads-only model once, untraced, on the infra's reference frame
        (giraffe.jpg), decode on the host, and compare with the float32 PyTorch
        reference the infra already computed (printed for information), and with
        the card's own decode of the same frame. Not timed."""
        dram = ttnn.allocate_tensor_on_device(self.seed_host.shape, self.seed_host.dtype,
                                              self.seed_host.layout, self.device, dram_cfg)
        ttnn.copy_host_to_device_tensor(self.seed_host, dram)
        x = ttnn.reshard(dram, l1_cfg)
        outs = YoloV4HeadsRunner(self.infra)(x)
        raws = [ttnn.to_torch(o).reshape(-1, o.shape[-1]).float() for o in outs]
        self._raw_specs = [tuple(int(v) for v in r.shape) for r in raws]
        self._check_raws = [r.to(torch.bfloat16) for r in raws]
        self.card_reduce = False
        if CARD_REDUCE:
            try:
                self.card_reduce = self._check_card_reduce(outs, raws)
            except Exception as e:
                print(f"  [submission] card reduction check raised {type(e).__name__}: {e}")
                self.card_reduce = False
        self.concat_heads = False
        if not self.card_reduce:
            try:
                c_out = ttnn.concat(outs, dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.synchronize_device(self.device)
                c_torch = ttnn.to_torch(c_out).reshape(-1, c_out.shape[-1]).float()
                ref_torch = torch.cat([r[:S * S, :] for r, S in zip(raws, _GRIDS)], dim=0)
                if torch.equal(c_torch[:ref_torch.shape[0], :ref_torch.shape[1]], ref_torch):
                    self.concat_heads = True
                    print(f"  [submission] card heads concatenation verified: 1 tensor {list(c_out.shape)} per frame")
                _free(c_out)
            except Exception as e:
                print(f"  [submission] heads concat on card unavailable ({type(e).__name__}: {e}); using 3 head tensors")
                self.concat_heads = False
        _free(outs)
        card_b = card_c = None
        if not CUSTOM_RES:
            ttnn.copy_host_to_device_tensor(self.seed_host, dram)
            card_out = self.infra.ttnn_yolov4_model(ttnn.reshard(dram, l1_cfg))       # supplied path, untraced
            card_b, card_c = get_model_result(card_out, self.resolution, mesh_composer=self.out_composer)
            _free(card_out)
        _free(dram)
        ttnn.synchronize_device(self.device)
        shapes = [tuple(r.shape) for r in raws]
        rows_ok = all(r.shape[0] >= S * S and r.shape[1] >= 255 for r, S in zip(raws, _GRIDS))
        if not rows_ok:
            print(f"  [submission] unexpected head shapes {shapes}")
            return False
        boxes, confs = decode_heads(raws)
        ref_b = self.infra.ref_boxes.reshape(1, -1, 4)
        ref_c = self.infra.ref_confs.reshape(1, -1, 80)
        pb, pc = _pcc(boxes, ref_b), _pcc(confs, ref_c)
        print(f"  [submission] host decode vs PyTorch reference: boxes PCC {pb:.5f}, confs PCC {pc:.5f} "
              f"{'(gate: boxes >= 0.98; no card decode at this size)' if CUSTOM_RES else '(information)'}")
        if CUSTOM_RES:
            cb, cc = pb, 1.0
        else:
            cb = _pcc(boxes, card_b.float().reshape(1, -1, 4))
            cc = _pcc(confs, card_c.float().reshape(1, -1, 80))
            print(f"  [submission] host decode vs card decode:       boxes PCC {cb:.5f}, confs PCC {cc:.5f} "
                  f"(gate {DECODE_MIN_PCC[0]} / {DECODE_MIN_PCC[1]})")
        self.decode_check = {"head_shapes": shapes, "ref_pcc": (pb, pc), "card_pcc": (cb, cc)}
        full = detections_full(raws, 640, 480)
        fast = detections_from_heads(raws, 640, 480)
        same = len(full) == len(fast) and all(
            a["category_id"] == b["category_id"] and abs(a["score"] - b["score"]) < 1e-6
            and max(abs(x - y) for x, y in zip(a["bbox"], b["bbox"])) < 1e-3 for a, b in zip(full, fast))
        print(f"  [submission] cheap decode vs full decode on giraffe: {len(fast)} vs {len(full)} detections, "
              f"{'identical' if same else 'DIFFERENT'} (torchvision: {HAS_TORCHVISION})")
        if CUSTOM_RES:
            return same and pb >= 0.98
        return same and cb >= DECODE_MIN_PCC[0] and cc >= DECODE_MIN_PCC[1]

    # ---- input conversion -------------------------------------------------
    def _to_host_reference(self, img_bgr):
        """Milestone-7 input path: float32 /255, then ttnn converts to bf16."""
        t = image_to_tensor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                               mesh_mapper=self.in_mapper)

    def _to_host_fast(self, img_bgr):
        """uint8 BGR HWC -> bf16 RGB NCHW in one LUT gather, no float32 pass."""
        chw_rgb = img_bgr.transpose(2, 0, 1)[::-1]
        t = torch.from_numpy(self.lut[chw_rgb]).view(torch.bfloat16).unsqueeze(0)
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                               mesh_mapper=self.in_mapper)

    def _build_input_lut(self):
        """Ask ttnn how it rounds each of the 256 possible pixel values, then
        verify on a random frame that the fast path is bit-identical."""
        try:
            vals = torch.arange(256, dtype=torch.float32).div(255.0).reshape(1, 1, 1, 256)
            bf = ttnn.to_torch(ttnn.from_torch(vals, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT))
            self.lut = bf.to(torch.bfloat16).contiguous().view(torch.int16).reshape(256).numpy().copy()
            rng = np.random.default_rng(0)
            probe = rng.integers(0, 256, size=(self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
            a = ttnn.to_torch(self._to_host_reference(probe)).view(torch.int16)
            b = ttnn.to_torch(self._to_host_fast(probe)).view(torch.int16)
            if a.shape == b.shape and torch.equal(a, b):
                return self.lut
            print("  [submission] fast input path not bit-identical, using reference path")
        except Exception as e:  # never let an optimisation break setup
            print(f"  [submission] fast input path disabled: {type(e).__name__}: {e}")
        return None

    def _decode(self, path):
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            return None, 1, 1, True
        H, W = frame.shape[:2]
        img = cv2.resize(frame, self.resolution)
        if HOST_SALIENCY_THRESH > 0.0 and float(img.std()) < HOST_SALIENCY_THRESH:
            return None, W, H, True
        host_t = self._to_host_fast(img) if self.lut is not None else self._to_host_reference(img)
        return host_t, W, H, False

    # ---- timed --------------------------------------------------------------
    def warmup(self, paths):
        self.run(paths)

    def run(self, paths):
        paths = list(paths)
        if not paths:
            return {}
        if not self.streaming:
            return self._run_chunked(paths)

        n = len(paths)
        out = {}
        decoded = {}
        if SWITCH_INTERVAL:
            sys.setswitchinterval(SWITCH_INTERVAL)
        submitted = 0

        def top_up(limit):
            nonlocal submitted
            limit = min(n, limit)
            while submitted < limit:
                decoded[submitted] = self.pool.submit(self._decode, paths[submitted])
                submitted += 1

        inflight = deque()   # (path, W, H, host_slot, event, host_input)
        post = []            # (path, future)
        slot = 0
        self._slot_busy = [None] * RING   # future of the worker still reading each slot
        self._st = {"card_wait": 0.0, "decode_wait": 0.0, "slot_copy": 0.0, "post": []}
        _READ_TIMES.clear()
        _READ_CALLS[0] = 0
        t_run = time.perf_counter()
        top_up(LOOKAHEAD)

        for k in range(n):
            top_up(k + LOOKAHEAD)
            t0 = time.perf_counter()
            host_t, W, H, empty = decoded.pop(k).result()
            self._st["decode_wait"] += time.perf_counter() - t0
            if empty:
                out[paths[k]] = []
                continue

            busy = self._slot_busy[slot]
            if busy is not None:                  # a worker may still be reading this slot
                t0 = time.perf_counter()
                busy.result()
                self._st["slot_copy"] += time.perf_counter() - t0
                self._slot_busy[slot] = None

            # Feed the card immediately (H2D on CQ1, reshard + trace on CQ0)
            dev_out = next(iter(self.executor.execute([host_t])))
            host_slot = self.slots[slot]
            _copy_struct_to_host(dev_out, host_slot, self.read_cq)
            self.executor.signal_read_complete()
            ev = ttnn.record_event(self.device, self.read_cq)
            inflight.append((paths[k], W, H, host_slot, ev, host_t, slot))
            slot = (slot + 1) % RING

            # Retire the oldest frame while the card executes queued frames in the background
            if len(inflight) >= IN_FLIGHT:
                self._retire(inflight.popleft(), post)

        while inflight:
            self._retire(inflight.popleft(), post)
        t_tail = time.perf_counter()
        for p, fut in post:
            out[p] = fut.result()
        if STATS:
            st, now = self._st, time.perf_counter()
            pt = st["post"]
            ms = lambda v: 1000.0 * v / max(1, n)
            print(f"  [submission] {n} frames, {ms(now - t_run):.3f} ms/frame: card-wait {ms(st['card_wait']):.3f}, "
                  f"jpeg-wait {ms(st['decode_wait']):.3f}, slot {ms(st['slot_copy']):.3f}, "
                  f"final post-wait {ms(now - t_tail):.3f} ms/frame; post work "
                  f"{1000.0 * sum(pt) / max(1, len(pt)):.3f} ms per frame per worker "
                  f"(of which slot read {1000.0 * sum(_READ_TIMES) / max(1, len(_READ_TIMES)):.3f}, "
                  f"{_READ_CALLS[0] / max(1, n):.1f} reads/frame) "
                  f"({'host' if self.host_decode else 'card'} decode, {DECODERS} decoders, {POST_WORKERS} post workers)")
        return out

    def _retire(self, item, post):
        """Wait for one frame's output, hand it to worker thread for decode + NMS."""
        p, W, H, host_slot, ev, _host_t, slot_idx = item
        t0 = time.perf_counter()
        ttnn.event_synchronize(ev)
        self._st["card_wait"] += time.perf_counter() - t0
        if self.host_decode:
            fn = detections_from_reduced if self.card_reduce else _slot_detections
            fut = self.post_pool.submit(self._timed, fn, (host_slot, W, H))
            self._slot_busy[slot_idx] = fut
        else:
            boxes, confs = get_model_result(host_slot, self.resolution, mesh_composer=self.out_composer)
            fut = self.post_pool.submit(self._timed, fast_detections, (boxes, confs, W, H))
        post.append((p, fut))

    def _timed(self, fn, args):
        t0 = time.perf_counter()
        r = fn(*args)
        self._st["post"].append(time.perf_counter() - t0)
        return r

    # ---- milestone-7 path, kept for A/B -------------------------------------
    def _run_chunked(self, paths):
        out = {}
        chunks = [paths[i:i + CHUNK] for i in range(0, len(paths), CHUNK)]
        pending = [self.pool.submit(self._decode, p) for p in chunks[0]]
        prev = None
        for i, chunk in enumerate(chunks):
            frames = [f.result() for f in pending]
            pending = ([self.pool.submit(self._decode, p) for p in chunks[i + 1]]
                       if i + 1 < len(chunks) else None)
            act_p, act_wh, act_h = [], [], []
            for p, (host_t, W, H, empty) in zip(chunk, frames):
                if empty:
                    out[p] = []
                else:
                    act_p.append(p); act_wh.append((W, H)); act_h.append(host_t)
            if act_h:
                self.pipeline.enqueue(act_h)
            if prev is not None:
                for p, (W, H), raw in zip(*prev):
                    out[p] = self._post(raw, W, H)
            prev = (act_p, act_wh, self.pipeline.pop_all()) if act_h else None
        if prev is not None:
            for p, (W, H), raw in zip(*prev):
                out[p] = self._post(raw, W, H)
        return out

    def _post(self, raw, W, H):
        if self.host_decode:
            if isinstance(raw, (list, tuple)):
                return detections_from_heads(_raw_to_float(raw), W, H)
            return detections_from_heads(ttnn.to_torch(raw).reshape(-1, 256), W, H)
        boxes, confs = get_model_result(raw, self.resolution, mesh_composer=self.out_composer)
        return fast_detections(boxes, confs, W, H)

    def teardown(self):
        sys.setswitchinterval(0.005)
        for name in ("pool", "post_pool"):
            p = getattr(self, name, None)
            if p is not None:
                p.shutdown(wait=True)
        pipeline = getattr(self, "pipeline", None)
        if pipeline is not None:
            try:
                pipeline.cleanup()
            except Exception:
                pass
        gc.enable()
        gc.collect()
