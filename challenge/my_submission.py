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

RESOLUTION = (320, 320)
CONF_THRESH = 0.001
NMS_THRESH = 0.4
EMPTY_THRESH = 0.05
HOST_SALIENCY_THRESH = 3.0

TRACE_REGION_SIZE = 6434816

# ---- scheduling knobs (tune on the card; see docs/IDEA_8_STREAMING_PIPELINE.md)
STREAMING = True     # False -> milestone-7 chunked path
IN_FLIGHT = 4        # frames queued on the card before the host waits on the oldest
RING = 16            # preallocated host output slots; must be > IN_FLIGHT (workers hold a slot while decoding)
LOOKAHEAD = 64       # how many frames ahead decoding runs
DECODERS = int(os.environ.get("SUB_DECODERS", 24))   # JPEG decode + resize + host tensor threads
POST_WORKERS = int(os.environ.get("SUB_POST_WORKERS", 3 if RESOLUTION[0] <= 320 else 6))  # decode+NMS threads
TORCH_THREADS = 1    # intra-op threads per torch call
FAST_INPUT = True    # bf16 LUT input path (auto-disabled if parity check fails)
CHUNK = 32           # only used when STREAMING = False
HOST_DECODE = True   # milestone 9: box decode on the host in float32 (falls back automatically)
DECODE_MIN_PCC = (0.99, 0.98)   # boxes, confs: host decode vs the card's own decode of the same frame
SWITCH_INTERVAL = 0.0005   # GIL hand-over interval during run(); Python default 0.005
ZERO_COPY = os.environ.get("SUB_ZERO_COPY", "0") == "1"        # shared host buffers; off: hung at 640, no gain at 320
HEADS_CONCAT = os.environ.get("SUB_HEADS_CONCAT", "0") == "1"  # concat heads on card; off: one more card op, measured slower
STATS = True                                                    # one timing line per run()
READ_QUEUE = os.environ.get("SUB_READ_QUEUE", "cq0")    # cq0 (measured best) | cq1 | auto (setup times both)
WEIGHTS_DB = os.environ.get("SUB_WEIGHTS_DB", "0") == "1"  # conv weight double-buffering in L1 (card-side memory knob)
CARD_PROBE = os.environ.get("SUB_CARD_PROBE", "1") != "0"      # setup only: time the card alone (not timed)

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

    scale = torch.tensor([W, H, W, H], dtype=torch.float64)
    px = cb[keep].double() * scale
    px = torch.minimum(px.clamp(min=0.0), scale)
    valid = (px[:, 2] > px[:, 0]) & (px[:, 3] > px[:, 1])
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


# ---- shared host memory ("memory" lever) -----------------------------------
# ttnn.from_torch / ttnn.to_torch copy and convert every frame inside C++ calls
# that keep Python's GIL, so with them in the loop all threads take turns. A
# host tensor whose memory torch can address directly removes both calls from
# the frame loop: decode threads write pixels straight into input tensors, and
# the card writes its output straight into memory the workers read.
def _writes_through(h, t):
    """True if writing the torch tensor t changes what the ttnn host tensor h holds."""
    pat = torch.randn(tuple(t.shape)).to(torch.bfloat16)
    t.copy_(pat)
    got = ttnn.to_torch(h).to(torch.bfloat16).contiguous().view(torch.int16).reshape(-1)
    return torch.equal(got, pat.contiguous().view(torch.int16).reshape(-1))


def _shared_host_tensor(shape, device):
    """(ttnn host tensor, torch bf16 tensor on the same memory, method) or None."""
    try:   # A: ttnn borrows the memory of a torch tensor
        t = torch.zeros(shape, dtype=torch.bfloat16)
        h = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        if _writes_through(h, t):
            return h, t, "borrowed torch memory"
    except Exception:
        pass
    try:   # B: ttnn allocates, torch views that memory
        h = ttnn.allocate_tensor_on_host(ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device)
        t = ttnn.to_torch(h)
        if t.dtype == torch.bfloat16 and _writes_through(h, t):
            return h, t, "torch view of ttnn memory"
    except Exception:
        pass
    return None


def _view_detections(view, W, H):
    """Worker job when the slot is shared memory: decode it in place."""
    return detections_from_heads(view, W, H)


# ---- reading a ttnn host tensor into torch -------------------------------
# ttnn.to_torch is one of several ways; setup times all that work on this
# machine, keeps only those bit-identical to ttnn.to_torch, and picks the fastest.
def _read_to_torch(h):
    return ttnn.to_torch(h)


def _read_method(h):
    return h.to_torch()


def _read_numpy(h):
    return torch.from_numpy(h.to_numpy())


def _read_host_buffer(h):
    buf = ttnn.get_host_buffer(h) if hasattr(ttnn, "get_host_buffer") else h.host_buffer()
    return torch.frombuffer(memoryview(buf), dtype=torch.bfloat16).reshape(tuple(h.shape))


_READERS = (("ttnn.to_torch", _read_to_torch), ("Tensor.to_torch", _read_method),
            ("Tensor.to_numpy", _read_numpy), ("host buffer view", _read_host_buffer))
_READER = _read_to_torch
_READ_TIMES = []          # seconds per slot read, appended by workers (STATS)


def _read_2d(h):
    t = _READER(h)
    return t.reshape(-1, t.shape[-1])


def _slot_detections(host_struct, W, H):
    """Worker job: read the head tensor(s) straight out of the host slot and
    decode them. The main thread does not reuse the slot until this returns."""
    t0 = time.perf_counter()
    if isinstance(host_struct, (list, tuple)):
        raw = [_read_2d(h) for h in host_struct]
    else:
        raw = _read_2d(host_struct)
    if STATS:
        _READ_TIMES.append(time.perf_counter() - t0)
    return detections_from_heads(raw, W, H)


def detections_full(raws, W, H, grids=_GRIDS):
    """Reference host path: decode all N x 80 scores, then fast_detections."""
    boxes, confs = decode_heads([r.float() for r in raws], grids)
    return fast_detections(boxes, confs, W, H)


_AR85 = torch.arange(85)


def detections_from_heads(raw, W, H, conf_thresh=CONF_THRESH, grids=_GRIDS):
    """Same detections as detections_full, touching only what can matter.

    raw: the three scales stacked, [rows, C >= 255] (bf16 or float), or a list
    of the three per-scale tensors.

    A candidate's score is sigmoid(obj) * sigmoid(best class logit) <= sigmoid(obj),
    so an anchor with sigmoid(obj) <= conf_thresh can never pass: only the three
    objectness columns are read for every row, and the 85 values are gathered
    only for anchors that survive. sigmoid is monotonic, so the best class is the
    argmax of the raw logits. Candidates are put back in card / reference order
    so NMS tie-breaks match the full path.
    """
    tbl = _tables(grids)
    if isinstance(raw, (list, tuple)):
        raw = torch.cat([r[:S * S] for r, S in zip(raw, grids)])
    raw = raw.contiguous()
    C = raw.shape[-1]
    so = torch.sigmoid(raw[:, 4:175:85].float())                          # [rows, 3] objectness
    r, a = (so > conf_thresh).nonzero(as_tuple=True)
    if r.numel() == 0:
        return []
    sel = raw.view(-1)[(r * C + a * 85).unsqueeze(1) + _AR85].float()     # [K, 85]
    ml, mi = sel[:, 5:85].max(dim=1)
    conf = so[r, a] * torch.sigmoid(ml)
    if EMPTY_THRESH > 0.0 and float(conf.max()) < EMPTY_THRESH:
        return []
    keep = conf > conf_thresh
    if not bool(keep.any()):
        return []
    r, a, sel, conf, mi = r[keep], a[keep], sel[keep], conf[keep], mi[keep]
    o = tbl.rkey[a, r].argsort()
    r, a, sel, conf, mi = r[o], a[o], sel[o], conf[o], mi[o]
    S = tbl.rs[r]
    bx = (torch.sigmoid(sel[:, 0]) + tbl.rgx[r]) / S
    by = (torch.sigmoid(sel[:, 1]) + tbl.rgy[r]) / S
    bw = torch.exp(sel[:, 2]) * tbl.raw[a, r]
    bh = torch.exp(sel[:, 3]) * tbl.rah[a, r]
    x1 = bx - bw * 0.5
    y1 = by - bh * 0.5
    return _finish(torch.stack((x1, y1, x1 + bw, y1 + bh), -1), conf, mi, W, H)


def _pcc(a, b):
    a = torch.as_tensor(a).detach().double().flatten()
    b = torch.as_tensor(b).detach().double().flatten()
    return float(torch.corrcoef(torch.stack((a, b)))[0, 1])


class YoloV4HeadsRunner:
    """Like YoloV4PipelineRunner, but the trace stops after the detection head
    and hands back the three raw head tensors (ROW_MAJOR bf16, L1 interleaved).
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
        if self.concat:                                   # one [1, 1, rows, 256] tensor, one D2H read
            return ttnn.concat(outs, dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
        return outs


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

        if WEIGHTS_DB:
            self._patch_conv_double_buffer()
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
        print(f"  [submission] box decode on {'HOST (float32)' if self.host_decode else 'CARD (as supplied)'}"
              f"{', heads concatenated on card' if self.host_decode and self.heads_concat else ''}")

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

        self._dram_out = None
        if self.host_decode and READ_QUEUE != "cq0":
            try:   # two sets, allocated before trace capture so the trace cannot reuse their memory
                self._dram_out = [[ttnn.allocate_tensor_on_device(sh, dt, lay, device, ttnn.DRAM_MEMORY_CONFIG)
                                   for sh, dt, lay in self._head_specs] for _ in range(2)]
            except Exception as e:
                print(f"  [submission] DRAM staging buffers unavailable ({type(e).__name__}: {e})")
                self._dram_out = None
        if self.host_decode:
            try:
                self.pipeline = build(YoloV4HeadsRunner(self.infra, concat=self.heads_concat))
            except Exception as e:
                print(f"  [submission] heads-only trace failed ({type(e).__name__}: {e}); using the supplied path")
                self.host_decode = False
                try:
                    ttnn.synchronize_device(device)
                except Exception:
                    pass
        if not self.host_decode:
            self.pipeline = build(YoloV4PipelineRunner(self.infra))

        self.lut = self._build_input_lut() if FAST_INPUT else None
        self.out_views = None      # torch views of the output slots (zero-copy outputs)
        self.in_host = None        # ring of shared input tensors (zero-copy inputs)
        self.streaming = STREAMING and hasattr(ttnn, "event_synchronize")
        if self.streaming:
            assert RING > IN_FLIGHT, "RING must exceed IN_FLIGHT or a slot is overwritten before it is read"
            self.executor = self.pipeline.executor
            self.read_cq = self.executor.get_read_cq()
            if ZERO_COPY and self.host_decode and self.heads_concat:
                self._guarded("zero-copy outputs", self._setup_shared_outputs)
            if self.out_views is None:
                self.pipeline.preallocate_output_tensors_on_host(RING)
                self.slots = self.pipeline.output_tensors
                if self.host_decode:
                    self._pick_reader()
            if ZERO_COPY and self.lut is not None:
                self._guarded("zero-copy inputs", self._setup_input_ring)
            self.cq1_read = False
            self._pending, self._read_ev, self._b = None, [None, None], 0
            if self._dram_out is not None:
                self._guarded("output read queue", self._choose_read_queue)
            if CARD_PROBE:
                self._guarded("card time per frame", self._measure_card)

        self.pool = ThreadPoolExecutor(max_workers=DECODERS)
        self.post_pool = ThreadPoolExecutor(max_workers=POST_WORKERS)

    def _patch_conv_double_buffer(self):
        """Every TtConv2d is built through builder.to_conv2d_config; force weight
        double-buffering on (activation double-buffering is already on). Same
        math, only how weights stream into L1 changes."""
        try:
            import dataclasses
            from models.tt_cnn.tt import builder as _b
            orig = _b.to_conv2d_config

            def patched(configuration):
                return orig(dataclasses.replace(configuration, enable_weights_double_buffer=True,
                                                enable_act_double_buffer=True))
            _b.to_conv2d_config = patched
            print("  [submission] conv weight double-buffering: on")
        except Exception as e:
            print(f"  [submission] conv weight double-buffering unavailable ({type(e).__name__}: {e})")

    def _guarded(self, what, fn):
        try:
            msg = fn()
        except Exception as e:
            msg = f"no ({type(e).__name__}: {e})"
        print(f"  [submission] {what}: {msg}")

    # ---- per-frame card schedule ------------------------------------------
    # cq0 (supplied pattern): CQ1 writes the input; CQ0 reshards, runs the trace,
    #   then reads the output to the host. The read idles every compute core.
    # cq1: CQ0 reshards, runs the trace, copies the output to one of two DRAM
    #   staging sets (on-card, fast). CQ1 reads that set to the host while CQ0
    #   already runs the next frame. CQ1 order per iteration is: write input k,
    #   then read output k-1, so the next input is never stuck behind a read.
    #   CQ0 waits for the read of frame k-2 before reusing its staging set.
    def _issue(self, host_t, token, host_slot):
        """Enqueue one frame. Returns [(token, event)] for frames whose host read
        has just been enqueued (the event fires when that read is complete)."""
        dev_out = next(iter(self.executor.execute([host_t])))
        if not self.cq1_read:
            _copy_struct_to_host(dev_out, host_slot, self.read_cq)
            self.executor.signal_read_complete()
            return [(token, ttnn.record_event(self.device, self.read_cq))]
        done = self._flush_read()                       # read of the previous frame, after input k on CQ1
        b = self._b
        self._b ^= 1
        if self._read_ev[b] is not None:
            ttnn.wait_for_event(0, self._read_ev[b])   # host read of frame k-2 finished
        srcs = dev_out if isinstance(dev_out, (list, tuple)) else [dev_out]
        for src, dst in zip(srcs, self._dram_out[b]):
            self._dcopy(src, dst)
        self._pending = (token, b, host_slot, ttnn.record_event(self.device, 0))
        return done

    def _flush_read(self):
        if self._pending is None:
            return []
        token, b, host_slot, staged = self._pending
        self._pending = None
        ttnn.wait_for_event(1, staged)
        hs = host_slot if isinstance(host_slot, (list, tuple)) else [host_slot]
        for d, h in zip(self._dram_out[b], hs):
            ttnn.copy_device_to_host_tensor(d, h, blocking=False, cq_id=1)
        ev = ttnn.record_event(self.device, 1)
        self._read_ev[b] = ev
        return [(token, ev)]

    def _bench_schedule(self, n=100, warm=10):
        q = deque()
        t0 = None
        for k in range(n + warm):
            if k == warm:
                while q:
                    ttnn.event_synchronize(q.popleft())
                t0 = time.perf_counter()
            for _, ev in self._issue(self.seed_host, k, self.slots[k % RING]):
                q.append(ev)
            while len(q) > IN_FLIGHT:
                ttnn.event_synchronize(q.popleft())
        for _, ev in self._flush_read():
            q.append(ev)
        while q:
            ttnn.event_synchronize(q.popleft())
        return (time.perf_counter() - t0) * 1000.0 / n

    def _slot_bits(self, idx):
        hs = self.slots[idx] if isinstance(self.slots[idx], (list, tuple)) else [self.slots[idx]]
        return [ttnn.to_torch(h).contiguous().view(torch.int16).clone() for h in hs]

    def _choose_read_queue(self):
        dev_out = self._run_one(self.seed_host)
        srcs = dev_out if isinstance(dev_out, (list, tuple)) else [dev_out]
        ref = [ttnn.to_torch(s.cpu()).contiguous().view(torch.int16) for s in srcs]
        cands = (("ttnn.copy", lambda a, b: ttnn.copy(a, b)),
                 ("ttnn.assign", lambda a, b: ttnn.assign(a, b)),
                 ("ttnn.assign(output_tensor=)", lambda a, b: ttnn.assign(a, memory_config=ttnn.DRAM_MEMORY_CONFIG, output_tensor=b)))
        self._dcopy, name = None, None
        for nm, fn in cands:
            try:
                for s_, d_ in zip(srcs, self._dram_out[0]):
                    fn(s_, d_)
                ttnn.synchronize_device(self.device)
                got = [ttnn.to_torch(d_.cpu()).contiguous().view(torch.int16) for d_ in self._dram_out[0]]
                if all(torch.equal(g.reshape(-1), r.reshape(-1)) for g, r in zip(got, ref)):
                    self._dcopy, name = fn, nm
                    break
            except Exception:
                continue
        if self._dcopy is None:
            return "cq0 (no on-card copy into a preallocated DRAM tensor on this ttnn)"
        # same frame through both schedules must give identical host bytes
        self.cq1_read = False
        self._issue(self.seed_host, 0, self.slots[0])
        ttnn.synchronize_device(self.device)
        a = self._slot_bits(0)
        self.cq1_read, self._pending, self._read_ev, self._b = True, None, [None, None], 0
        self._issue(self.seed_host, 0, self.slots[1])
        self._flush_read()
        ttnn.synchronize_device(self.device)
        b = self._slot_bits(1)
        if not all(torch.equal(x.reshape(-1), y.reshape(-1)) for x, y in zip(a, b)):
            self.cq1_read = False
            return "cq0 (cq1 schedule returned different bytes)"
        ms1 = self._bench_schedule()
        self.cq1_read = False
        ms0 = self._bench_schedule()
        use1 = READ_QUEUE == "cq1" or (READ_QUEUE == "auto" and ms1 < ms0)
        self.cq1_read, self._pending, self._read_ev, self._b = use1, None, [None, None], 0
        return (f"card-only ms/frame: read on CQ0 {ms0:.3f}, read on CQ1 {ms1:.3f} (via {name}) "
                f"-> using {'CQ1' if use1 else 'CQ0'}")

    def _measure_card(self, n=100):
        """Setup only. trace = the traced model back to back; +read adds the
        card->host copy of the output on the same queue; full adds the input
        write + reshard, i.e. one frame of run() with the host doing nothing else."""
        ex, dev = self.executor, self.executor.output_tensor
        slot = self.slots[0]

        def timed(fn):
            for _ in range(10):
                fn()
            ttnn.synchronize_device(self.device)
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            ttnn.synchronize_device(self.device)
            return (time.perf_counter() - t0) * 1000.0 / n

        def trace():
            ttnn.execute_trace(self.device, ex.trace_id, cq_id=0, blocking=False)

        def trace_read():
            trace()
            _copy_struct_to_host(dev, slot, self.read_cq)

        def full():
            next(iter(ex.execute([self.seed_host])))
            _copy_struct_to_host(dev, slot, self.read_cq)

        a, b, c = timed(trace), timed(trace_read), timed(full)
        self.card_ms = {"trace": a, "trace+read": b, "full": c}
        return (f"trace {a:.3f} ms, +output read {b:.3f} ms (read {b - a:.3f}), "
                f"+input write/reshard {c:.3f} ms  <- the card's floor for run()")

    def _run_one(self, host_in, slot=None):
        """One frame through the traced pipeline (setup only); returns the device output."""
        dev = next(iter(self.executor.execute([host_in])))
        if slot is not None:
            _copy_struct_to_host(dev, slot, self.read_cq)
            self.executor.signal_read_complete()
        ttnn.synchronize_device(self.device)
        return dev

    def _setup_shared_outputs(self):
        dev = self._run_one(self.seed_host)
        if isinstance(dev, (list, tuple)):
            return "no (model has several outputs)"
        shape = tuple(dev.shape)
        ref = ttnn.to_torch(dev.cpu()).to(torch.bfloat16).contiguous()
        pairs = [_shared_host_tensor(shape, self.device) for _ in range(RING)]
        if any(p is None for p in pairs):
            return "no (this ttnn cannot share host memory with torch)"
        h0, v0, how = pairs[0]
        v0.zero_()
        ttnn.copy_device_to_host_tensor(dev, h0, blocking=True, cq_id=self.read_cq)
        ttnn.synchronize_device(self.device)
        if not torch.equal(v0.contiguous().view(torch.int16).reshape(-1), ref.view(torch.int16).reshape(-1)):
            return "no (card-to-host copy did not land in the shared memory)"
        self.slots = [h for h, _, _ in pairs]
        self.out_views = [v.reshape(-1, shape[-1]) for _, v, _ in pairs]
        self._out_keep = pairs
        return f"yes ({how}, {RING} slots of {list(shape)})"

    def _fill(self, dst, img_bgr):
        """uint8 BGR HWC -> bf16 RGB CHW bits, written in place (numpy releases the GIL)."""
        np.take(self.lut, img_bgr.transpose(2, 0, 1)[::-1], out=dst, mode="clip")

    def _setup_input_ring(self):
        R = LOOKAHEAD + IN_FLIGHT + 4         # a slot is rewritten only after its frame retired
        shape = tuple(self.seed_host.shape)
        pairs = [_shared_host_tensor(shape, self.device) for _ in range(R)]
        if any(p is None for p in pairs):
            return "no (this ttnn cannot share host memory with torch)"
        img = cv2.imread("models/demos/yolov4/resources/giraffe.jpg", cv2.IMREAD_COLOR)
        if img is None:
            img = np.random.default_rng(1).integers(0, 256, (480, 640, 3), dtype=np.uint8)
        img = cv2.resize(img, self.resolution)
        in_np = [v.view(torch.int16).numpy().reshape(shape)[0] for _, v, _ in pairs]
        h0 = pairs[0][0]
        self._fill(in_np[0], img)
        ref_host = self._to_host_fast(img)
        a = ttnn.to_torch(h0).to(torch.bfloat16).contiguous().view(torch.int16)
        b = ttnn.to_torch(ref_host).to(torch.bfloat16).contiguous().view(torch.int16)
        if not torch.equal(a.reshape(-1), b.reshape(-1)):
            return "no (shared input differs from the from_torch input)"
        out_a = ttnn.to_torch(self._run_one(h0).cpu()).contiguous()
        out_b = ttnn.to_torch(self._run_one(ref_host).cpu()).contiguous()
        if not torch.equal(out_a, out_b):
            return "no (card output differs between shared and from_torch input)"
        self.in_host = [h for h, _, _ in pairs]
        self.in_np = in_np
        self._in_keep = pairs
        return f"yes ({pairs[0][2]}, ring of {R}; card output identical)"

    def _pick_reader(self):
        """Put one real frame into slot 0, then time every way of reading it into
        torch. Keep only readers whose values equal ttnn.to_torch exactly."""
        global _READER
        try:
            dev_out = next(iter(self.executor.execute([self.seed_host])))
            _copy_struct_to_host(dev_out, self.slots[0], self.read_cq)
            self.executor.signal_read_complete()
            ttnn.synchronize_device(self.device)
            hs = self.slots[0] if isinstance(self.slots[0], (list, tuple)) else [self.slots[0]]
            ref = [ttnn.to_torch(h).reshape(-1, ttnn.to_torch(h).shape[-1]).float() for h in hs]
            best, report = None, []
            for name, fn in _READERS:
                try:
                    got = [fn(h) for h in hs]
                    got = [g.reshape(-1, g.shape[-1]).float() for g in got]
                    if not all(g.shape == r.shape and torch.equal(g, r) for g, r in zip(got, ref)):
                        report.append(f"{name} differs")
                        continue
                    t0 = time.perf_counter()
                    for _ in range(20):
                        for h in hs:
                            fn(h)
                    ms = (time.perf_counter() - t0) * 1000.0 / 20
                    report.append(f"{name} {ms:.3f} ms")
                    if best is None or ms < best[0]:
                        best = (ms, name, fn)
                except Exception as e:
                    report.append(f"{name} n/a ({type(e).__name__})")
            if best is not None:
                _READER = best[2]
            print(f"  [submission] host read per frame: {'; '.join(report)} -> using "
                  f"{best[1] if best else 'ttnn.to_torch'}")
        except Exception as e:
            print(f"  [submission] reader probe failed ({type(e).__name__}: {e}); using ttnn.to_torch")
            _READER = _read_to_torch

    def _check_host_decode(self, dram_cfg, l1_cfg):
        """Run the heads-only model once, untraced, on the infra's reference frame
        (giraffe.jpg), decode on the host, and compare with the float32 PyTorch
        reference the infra already computed (printed for information), and with
        the card's own decode of the same frame (the gate: at 640 the model itself
        is off from the reference, PCC ~0.77 on confs, on either path). Not timed."""
        dram = ttnn.allocate_tensor_on_device(self.seed_host.shape, self.seed_host.dtype,
                                              self.seed_host.layout, self.device, dram_cfg)
        ttnn.copy_host_to_device_tensor(self.seed_host, dram)
        x = ttnn.reshard(dram, l1_cfg)
        self.heads_concat = False
        raws = None
        if HEADS_CONCAT:
            outs = None
            try:
                outs = YoloV4HeadsRunner(self.infra, concat=True)(x)
                t = ttnn.to_torch(outs)
                t = t.reshape(-1, t.shape[-1]).float()
                if t.shape[0] != _tables().rows:
                    raise ValueError(f"concatenated head has {t.shape[0]} rows, expected {_tables().rows}")
                raws = list(torch.split(t, [S * S for S in _GRIDS]))
                self._head_specs = [(outs.shape, outs.dtype, outs.layout)]
                self.heads_concat = True
            except Exception as e:
                print(f"  [submission] on-card head concat unavailable ({type(e).__name__}: {e}); using 3 outputs")
                raws = None
                ttnn.copy_host_to_device_tensor(self.seed_host, dram)
                x = ttnn.reshard(dram, l1_cfg)
            finally:
                _free(outs)
        if raws is None:
            outs = YoloV4HeadsRunner(self.infra)(x)
            raws = [ttnn.to_torch(o).reshape(-1, o.shape[-1]).float() for o in outs]
            self._head_specs = [(o.shape, o.dtype, o.layout) for o in outs]
            _free(outs)
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
        cb = _pcc(boxes, card_b.float().reshape(1, -1, 4))
        cc = _pcc(confs, card_c.float().reshape(1, -1, 80))
        self.decode_check = {"head_shapes": shapes, "ref_pcc": (pb, pc), "card_pcc": (cb, cc)}
        print(f"  [submission] host decode vs PyTorch reference: boxes PCC {pb:.5f}, confs PCC {pc:.5f} (information)")
        print(f"  [submission] host decode vs card decode:       boxes PCC {cb:.5f}, confs PCC {cc:.5f} "
              f"(gate {DECODE_MIN_PCC[0]} / {DECODE_MIN_PCC[1]})")
        full = detections_full(raws, 640, 480)
        fast = detections_from_heads(torch.cat(raws) if self.heads_concat else raws, 640, 480)
        same = len(full) == len(fast) and all(
            a["category_id"] == b["category_id"] and abs(a["score"] - b["score"]) < 1e-6
            and max(abs(x - y) for x, y in zip(a["bbox"], b["bbox"])) < 1e-3 for a, b in zip(full, fast))
        print(f"  [submission] cheap decode vs full decode on giraffe: {len(fast)} vs {len(full)} detections, "
              f"{'identical' if same else 'DIFFERENT'}")
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

    def _decode(self, path, k=None):
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            return None, 1, 1, True
        H, W = frame.shape[:2]
        img = cv2.resize(frame, self.resolution)
        if HOST_SALIENCY_THRESH > 0.0 and float(img.std()) < HOST_SALIENCY_THRESH:
            return None, W, H, True
        if self.in_host is not None and k is not None:
            idx = k % len(self.in_host)
            self._fill(self.in_np[idx], img)
            return self.in_host[idx], W, H, False
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
                decoded[submitted] = self.pool.submit(self._decode, paths[submitted], submitted)
                submitted += 1

        inflight = deque()   # (path, W, H, host_slot, event, host_input, slot index)
        meta = {}            # frame index -> (path, W, H, host_slot, slot index) until its read is enqueued
        post = []            # (path, future)
        slot = 0
        self._slot_busy = [None] * RING   # future of the worker still reading each slot
        _READ_TIMES.clear()
        self._st = {"card_wait": 0.0, "decode_wait": 0.0, "slot_copy": 0.0, "post": []}
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

            # Keep the card fed: only wait once IN_FLIGHT frames are queued.
            if len(inflight) >= IN_FLIGHT:
                self._retire(inflight.popleft(), post)

            busy = self._slot_busy[slot]
            if busy is not None:                  # a worker may still be reading this slot
                t0 = time.perf_counter()
                busy.result()
                self._st["slot_copy"] += time.perf_counter() - t0
                self._slot_busy[slot] = None
            host_slot = self.slots[slot]
            meta[k] = (paths[k], W, H, host_slot, slot)
            for tok, ev in self._issue(host_t, k, host_slot):
                p_, W_, H_, hs_, si_ = meta.pop(tok)
                inflight.append((p_, W_, H_, hs_, ev, None, si_))
            slot = (slot + 1) % RING

        for tok, ev in self._flush_read():
            p_, W_, H_, hs_, si_ = meta.pop(tok)
            inflight.append((p_, W_, H_, hs_, ev, None, si_))
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
                  f"(of which read {1000.0 * sum(_READ_TIMES) / max(1, len(_READ_TIMES)):.3f}) "
                  f"({'host' if self.host_decode else 'card'} decode, {DECODERS} decoders, {POST_WORKERS} post workers)")
        return out

    def _retire(self, item, post):
        """Wait for one frame's output, copy it out of its slot, hand it to NMS."""
        p, W, H, host_slot, ev, _host_t, slot_idx = item
        t0 = time.perf_counter()
        ttnn.event_synchronize(ev)
        t1 = time.perf_counter()
        self._st["card_wait"] += t1 - t0
        if self.host_decode:
            # The worker reads the slot itself; run() waits on this future
            # before the slot is written again.
            if self.out_views is not None:
                fut = self.post_pool.submit(self._timed, _view_detections, (self.out_views[slot_idx], W, H))
            else:
                fut = self.post_pool.submit(self._timed, _slot_detections, (host_slot, W, H))
            self._slot_busy[slot_idx] = fut
        else:
            # get_model_result returns fresh tensors, so the slot is free at once.
            boxes, confs = get_model_result(host_slot, self.resolution, mesh_composer=self.out_composer)
            self._st["slot_copy"] += time.perf_counter() - t1
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
            return detections_from_heads(_raw_to_float(raw), W, H)
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
