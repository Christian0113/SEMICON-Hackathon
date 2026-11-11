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

RESOLUTION = (512, 512)   # measured: 4.797 ms, mAP 0.6007 (320: 3.88 ms, 0.4656; 448: 4.739, 0.5026; 640: 7.75, 0.5179)
# Square, a multiple of 32, and (side*side/64) a multiple of 32 for the input sharding:
# 320, 384, 448, 512, 576 or 640. Sizes other than 320/640 use the 640 conv settings,
# need the host box decode, and are checked against the PyTorch reference at setup.
CUSTOM_RES = RESOLUTION not in ((320, 320), (640, 640))
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
DECODERS = int(os.environ.get("SUB_DECODERS", 4))   # JPEG decode threads; sweep 4..48: 4 best (fewer threads, less GIL contention)
POST_WORKERS = int(os.environ.get("SUB_POST_WORKERS", 2 if RESOLUTION[0] <= 320 else 3))  # sweep at 320: 2 best (3.960 ms), 4+ much slower; 640 not swept yet
TORCH_THREADS = 1    # intra-op threads per torch call
FAST_INPUT = True    # bf16 LUT input path (auto-disabled if parity check fails)
CHUNK = 32           # only used when STREAMING = False
HOST_DECODE = True   # milestone 9: box decode on the host in float32 (falls back automatically)
DECODE_MIN_PCC = (0.99, 0.98)   # boxes, confs: host decode vs the card's own decode of the same frame
SWITCH_INTERVAL = 0.0005   # GIL hand-over interval during run(); Python default 0.005
STATS = True         # one line per run(): where the main thread waited (diagnostics, ~free)   # boxes, confs vs the PyTorch reference; the model's own validate() floors

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


def _slot_detections(host_struct, W, H):
    """Worker job: read the three head tensors straight out of the host slot and
    decode them. The main thread does not reuse the slot until this returns."""
    raws = []
    for h in host_struct:
        t = ttnn.to_torch(h)
        raws.append(t.reshape(-1, t.shape[-1]))
    return detections_from_heads(raws, W, H)


def detections_full(raws, W, H, grids=_GRIDS):
    """Reference host path: decode all N x 80 scores, then fast_detections."""
    boxes, confs = decode_heads([r.float() for r in raws], grids)
    return fast_detections(boxes, confs, W, H)


def detections_from_heads(raws, W, H, conf_thresh=CONF_THRESH, grids=_GRIDS):
    """Same detections as detections_full in ~20 torch calls per frame.

    sigmoid is monotonic, so the best class is the argmax of the raw class
    logits and its score is sigmoid(obj) * sigmoid(max logit): no 80-wide
    sigmoid. All three scales are processed as one tensor, boxes are decoded
    only for candidates above conf_thresh, and candidates are put back in the
    card / reference order so NMS tie-breaks match the full path.
    """
    tbl = _tables(grids)
    t = torch.cat([r[:S * S, :255] for r, S in zip(raws, grids)]).float()
    t = t.view(tbl.rows, 3, 85).transpose(0, 1)                            # [3, rows, 85]
    ml, mi = t[..., 5:85].max(dim=-1)
    conf = torch.sigmoid(t[..., 4]) * torch.sigmoid(ml)                    # [3, 2100]
    if EMPTY_THRESH > 0.0 and float(conf.max()) < EMPTY_THRESH:
        return []
    a, r = (conf > conf_thresh).nonzero(as_tuple=True)
    if a.numel() == 0:
        return []
    o = tbl.rkey[a, r].argsort()
    a, r = a[o], r[o]
    tb = t[a, r, :4]
    S = tbl.rs[r]
    bx = (torch.sigmoid(tb[:, 0]) + tbl.rgx[r]) / S
    by = (torch.sigmoid(tb[:, 1]) + tbl.rgy[r]) / S
    bw = torch.exp(tb[:, 2]) * tbl.raw[a, r]
    bh = torch.exp(tb[:, 3]) * tbl.rah[a, r]
    x1 = bx - bw * 0.5
    y1 = by - bh * 0.5
    return _finish(torch.stack((x1, y1, x1 + bw, y1 + bh), -1), conf[a, r], mi[a, r], W, H)


def _pcc(a, b):
    a = torch.as_tensor(a).detach().double().flatten()
    b = torch.as_tensor(b).detach().double().flatten()
    return float(torch.corrcoef(torch.stack((a, b)))[0, 1])


class YoloV4HeadsRunner:
    """Like YoloV4PipelineRunner, but the trace stops after the detection head
    and hands back the three raw head tensors (ROW_MAJOR bf16, L1 interleaved).
    The first block is yolov4.py TtYOLOv4.__call__ verbatim (320x320)."""

    def __init__(self, infra):
        self.infra = infra

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
        print(f"  [submission] box decode on {'HOST (float32)' if self.host_decode else 'CARD (as supplied)'}")
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
                self.pipeline = build(YoloV4HeadsRunner(self.infra))
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
        outs = YoloV4HeadsRunner(self.infra)(x)
        raws = [ttnn.to_torch(o).reshape(-1, o.shape[-1]).float() for o in outs]
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
              f"{'identical' if same else 'DIFFERENT'}")
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

            # H2D on CQ1, reshard + trace on CQ0 (the executor's own event logic).
            dev_out = next(iter(self.executor.execute([host_t])))
            busy = self._slot_busy[slot]
            if busy is not None:                  # a worker may still be reading this slot
                t0 = time.perf_counter()
                busy.result()
                self._st["slot_copy"] += time.perf_counter() - t0
                self._slot_busy[slot] = None
            host_slot = self.slots[slot]
            _copy_struct_to_host(dev_out, host_slot, self.read_cq)
            self.executor.signal_read_complete()
            ev = ttnn.record_event(self.device, self.read_cq)
            inflight.append((paths[k], W, H, host_slot, ev, host_t, slot))
            slot = (slot + 1) % RING

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
