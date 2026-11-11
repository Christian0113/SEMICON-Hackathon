# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""High-Throughput YOLOv4 Object Detection Pipeline on Tenstorrent Wormhole n150.

Architecture & Acceleration Highlights:
1. Pipelined 2-Command-Queue Hardware Execution:
   - Command Queue 0 (CQ0): Traced execution of the YOLOv4 backbone, neck, and
     detection heads, resharding inputs into L1 interleaved memory.
   - Command Queue 1 (CQ1): Asynchronous host-to-device (H2D) PCIe streaming
     operating concurrently with device computation.
2. Truncated On-Device Graph (Heads-Only Execution):
   - The device trace executes downsample1 through downsample5, the PANet neck,
     and the three detection heads.
   - Outputs are concatenated on-device into a single [1, 1, 2100, 256] bfloat16
     tensor, reducing PCIe D2H transfers from 3 calls to 1 non-blocking transfer.
3. Overlapped Per-Frame Streaming Pipeline:
   - A bounded in-flight sliding window (IN_FLIGHT = 4) keeps the Wormhole Tensix
     cores 100% saturated without device-wide pipeline stalls.
   - A preallocated 48-slot pinned host ring buffer eliminates per-frame host
     memory allocation during inference.
   - Lightweight hardware event synchronization (ttnn.event_synchronize) ensures
     smooth asynchronous hand-off between card completion and CPU post-processing.
4. Asynchronous Parallel Post-Processing & Decode:
   - 4 background decoder threads perform concurrent JPEG decoding and uint8-to-bf16
     normalization using a precomputed 256-entry lookup table (LUT).
   - 2 dedicated post-processing worker threads perform empty-frame gating, direct
     bfloat16 class-max candidate selection, float32 box decoding, and vectorized
     torchvision batched Non-Maximum Suppression (NMS).
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

# ----------------------------------------------------------------------------
# Pipeline & Detection Configuration
# ----------------------------------------------------------------------------
RESOLUTION = (320, 320)
CUSTOM_RES = RESOLUTION not in ((320, 320), (640, 640))
CONF_THRESH = float(os.environ.get("SUB_CONF", 0.001))
NMS_THRESH = 0.4
EMPTY_THRESH = 0.05
HOST_SALIENCY_THRESH = 3.0

_OBJ_GATE = math.log(CONF_THRESH / (1.0 - CONF_THRESH)) - 0.05
_EMPTY_GATE = math.log(EMPTY_THRESH / (1.0 - EMPTY_THRESH)) if EMPTY_THRESH > 0.0 else -999.0

TRACE_REGION_SIZE = 6434816

# Concurrency & scheduling knobs (tuned for 96-core host + Wormhole n150)
IN_FLIGHT = int(os.environ.get("SUB_IN_FLIGHT", 4))
RING = int(os.environ.get("SUB_RING", 48))
LOOKAHEAD = 64
DECODERS = int(os.environ.get("SUB_DECODERS", 4))
POST_WORKERS = int(os.environ.get("SUB_POST_WORKERS", 2))
TORCH_THREADS = 1
FAST_INPUT = True
HOST_DECODE = True
DECODE_MIN_PCC = (0.99, 0.98)
SWITCH_INTERVAL = float(os.environ.get("SUB_SWITCH_INTERVAL", 0.005))
STATS = True

_COCO_IDS = [api.to_coco_id(i) for i in range(80)]

# Precomputed anchor dimensions per scale (stride 8, 16, 32)
_ANCHORS = (
    torch.tensor([[1.5, 2.0], [2.375, 4.5], [5.0, 3.5]]),                     # 40x40 grid (stride 8)
    torch.tensor([[2.25, 4.6875], [4.75, 3.4375], [4.5, 9.125]]),             # 20x20 grid (stride 16)
    torch.tensor([[4.4375, 3.4375], [6.0, 7.59375], [14.34375, 12.53125]]),   # 10x10 grid (stride 32)
)
_GRIDS = tuple(RESOLUTION[0] // st for st in (8, 16, 32))  # (40, 20, 10) at 320x320


# ----------------------------------------------------------------------------
# Post-Processing & Non-Maximum Suppression (NMS)
# ----------------------------------------------------------------------------
def _pure_torch_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
    """Pure PyTorch fallback for Non-Maximum Suppression when torchvision is absent."""
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


def _class_nms(boxes: torch.Tensor, scores: torch.Tensor, classes: torch.Tensor, iou_thresh: float) -> torch.Tensor:
    """Class-aware batched Non-Maximum Suppression."""
    if HAS_TORCHVISION:
        return torchvision.ops.batched_nms(boxes, scores, classes, iou_thresh)
    keep_list = []
    for cls in torch.unique(classes):
        idx = (classes == cls).nonzero(as_tuple=True)[0]
        keep_list.append(idx[_pure_torch_nms(boxes[idx], scores[idx], iou_thresh)])
    return torch.cat(keep_list) if keep_list else torch.empty(0, dtype=torch.long)


def fast_detections(boxes, confs, W: int, H: int, conf_thresh=CONF_THRESH, nms_thresh=NMS_THRESH):
    """Post-processes raw decoded candidate boxes and confidence tensors into COCO detections."""
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


def _finish(cb: torch.Tensor, cs: torch.Tensor, ci: torch.Tensor, W: int, H: int, nms_thresh=NMS_THRESH):
    """Filters candidates through NMS and maps coordinates to the original image dimensions."""
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
# Host-Side Float32 Detection Head Decoding
# ----------------------------------------------------------------------------
class _GridTables:
    """Precomputed spatial grid offsets and anchor ratios across scales."""

    def __init__(self, grids):
        self.grids = grids
        self.rows = sum(S * S for S in grids)
        self.grid_xy = []
        rs, rgx, rgy = [], [], []
        for S in grids:
            i = torch.arange(S * S)
            gx, gy = (i % S).float(), torch.div(i, S, rounding_mode="floor").float()
            self.grid_xy.append((gx.reshape(-1, 1), gy.reshape(-1, 1)))
            rs.append(torch.full((S * S,), float(S)))
            rgx.append(gx)
            rgy.append(gy)
        self.rs, self.rgx, self.rgy = torch.cat(rs), torch.cat(rgx), torch.cat(rgy)
        self.raw = torch.cat([(_ANCHORS[g][:, 0:1] / S).expand(3, S * S) for g, S in enumerate(grids)], 1)
        self.rah = torch.cat([(_ANCHORS[g][:, 1:2] / S).expand(3, S * S) for g, S in enumerate(grids)], 1)
        self.rkey = torch.cat([torch.arange(3).view(3, 1) * (S * S) + torch.arange(S * S).view(1, -1)
                               + 3 * sum(T * T for T in grids[:g]) for g, S in enumerate(grids)], 1)


_CACHED_TABLES: dict[tuple[int, ...], _GridTables] = {}


def _get_grid_tables(grids=_GRIDS) -> _GridTables:
    tbl = _CACHED_TABLES.get(grids)
    if tbl is None:
        tbl = _CACHED_TABLES[grids] = _GridTables(grids)
    return tbl


def decode_heads(raws, grids=_GRIDS):
    """Reference float32 decode: expands all bounding boxes and 80 class confidences."""
    tbl = _get_grid_tables(grids)
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
        b = torch.stack((x1, y1, x1 + bw, y1 + bh), -1)
        c = torch.sigmoid(t[..., 4:5]) * torch.sigmoid(t[..., 5:85])
        boxes.append(b.transpose(0, 1).reshape(-1, 4))
        confs.append(c.transpose(0, 1).reshape(-1, 80))
    return torch.cat(boxes).unsqueeze(0), torch.cat(confs).unsqueeze(0)


def _slot_detections(host_struct, W: int, H: int):
    """Worker job: extracts concatenated head tensor from host slot and decodes detections."""
    if isinstance(host_struct, (list, tuple)):
        raws = [ttnn.to_torch(h).reshape(-1, h.shape[-1]) for h in host_struct]
        return detections_from_heads(raws, W, H)
    t = ttnn.to_torch(host_struct)
    return detections_from_heads(t.reshape(-1, t.shape[-1]), W, H)


def detections_full(raws, W: int, H: int, grids=_GRIDS):
    """Full-grid reference host path for validation against PyTorch and on-card decoder."""
    boxes, confs = decode_heads([r.float() for r in raws], grids)
    return fast_detections(boxes, confs, W, H)


def detections_from_heads(raws, W: int, H: int, conf_thresh=CONF_THRESH, grids=_GRIDS):
    """Fast host decode of detection heads:
    1. Early empty-frame shortcut via objectness logit threshold.
    2. Direct zero-allocation view [3, rows, 85].
    3. Direct bfloat16 max over 80 classes to avoid intermediate float32 tensor allocation.
    4. Candidate selection: conf = sigmoid(obj) * sigmoid(cls_max) > conf_thresh.
    5. Float32 coordinate decoding for candidate anchors only.
    """
    tbl = _get_grid_tables(grids)
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
    hw_half = bw * 0.5
    hh_half = bh * 0.5
    return _finish(torch.stack((bx - hw_half, by - hh_half, bx + hw_half, by + hh_half), -1), conf, mi, W, H)


def _pcc(a, b) -> float:
    """Computes Pearson Correlation Coefficient between two tensors."""
    a = torch.as_tensor(a).detach().double().flatten()
    b = torch.as_tensor(b).detach().double().flatten()
    return float(torch.corrcoef(torch.stack((a, b)))[0, 1])


def _free(struct):
    """Deallocates on-device tensors safely."""
    if isinstance(struct, (list, tuple)):
        for s in struct:
            _free(s)
    elif struct is not None and hasattr(struct, "is_allocated") and struct.is_allocated():
        ttnn.deallocate(struct, force=True)


def _copy_struct_to_host(dev, host, cq_id: int):
    """Issues a non-blocking device-to-host transfer on the specified command queue."""
    if isinstance(dev, (list, tuple)):
        for d, h in zip(dev, host):
            _copy_struct_to_host(d, h, cq_id)
    else:
        ttnn.copy_device_to_host_tensor(dev, host, blocking=False, cq_id=cq_id)


# ----------------------------------------------------------------------------
# Model Runner: Truncated Graph (Heads Only)
# ----------------------------------------------------------------------------
class YoloV4HeadsRunner:
    """Executes YOLOv4 up to the detection heads and concatenates the output tensors on-device."""

    def __init__(self, infra, concat: bool = False):
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
        for t in heads:
            t = ttnn.to_memory_config(t, ttnn.L1_MEMORY_CONFIG)
            outs.append(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT))

        if self.concat:
            return ttnn.concat(outs, dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
        return outs


# ----------------------------------------------------------------------------
# Competition Submission Implementation
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
            except Exception as e:
                print(f"  [submission] host decode check raised {type(e).__name__}: {e}")
                self.host_decode = False

        print(f"  [submission] box decode on {'HOST (float32)' if self.host_decode else 'CARD (as supplied)'}")

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
                runner = YoloV4HeadsRunner(self.infra, concat=getattr(self, "concat_heads", False))
                self.pipeline = build(runner)
            except Exception as e:
                print(f"  [submission] heads-only trace failed ({type(e).__name__}: {e}); falling back to supplied path")
                self.host_decode = False
                try:
                    ttnn.synchronize_device(device)
                except Exception:
                    pass

        if not self.host_decode:
            self.pipeline = build(YoloV4PipelineRunner(self.infra))

        assert RING > IN_FLIGHT, "RING must exceed IN_FLIGHT to prevent slot overwrite before read"
        self.executor = self.pipeline.executor
        self.read_cq = self.executor.get_read_cq()
        self.pipeline.preallocate_output_tensors_on_host(RING)
        self.slots = self.pipeline.output_tensors

        self.lut = self._build_input_lut() if FAST_INPUT else None
        self.pool = ThreadPoolExecutor(max_workers=DECODERS)
        self.post_pool = ThreadPoolExecutor(max_workers=POST_WORKERS)

    def _enable_custom_resolution(self):
        """Allows non-standard resolutions by overriding conv parameters and stubbing on-card box decode."""
        R = RESOLUTION[0]
        if RESOLUTION[0] != RESOLUTION[1] or R % 32 or (R * R // 64) % 32:
            raise RuntimeError(f"RESOLUTION {RESOLUTION} is not usable: must be divisible by 32")
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
                raise RuntimeError("on-card box decode is not available at custom resolutions")
        y4.TtGenBoxes = _NoCardBoxDecode
        mp._semicon_custom_res = True

    def _check_host_decode(self, dram_cfg, l1_cfg) -> bool:
        """Runs the model untraced on the reference frame, verifying PCC and detection parity."""
        dram = ttnn.allocate_tensor_on_device(self.seed_host.shape, self.seed_host.dtype,
                                              self.seed_host.layout, self.device, dram_cfg)
        ttnn.copy_host_to_device_tensor(self.seed_host, dram)
        x = ttnn.reshard(dram, l1_cfg)
        outs = YoloV4HeadsRunner(self.infra)(x)
        raws = [ttnn.to_torch(o).reshape(-1, o.shape[-1]).float() for o in outs]

        self.concat_heads = False
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
            card_out = self.infra.ttnn_yolov4_model(ttnn.reshard(dram, l1_cfg))
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
        print(f"  [submission] host decode vs PyTorch reference: boxes PCC {pb:.5f}, confs PCC {pc:.5f}")

        if CUSTOM_RES:
            cb, cc = pb, 1.0
        else:
            cb = _pcc(boxes, card_b.float().reshape(1, -1, 4))
            cc = _pcc(confs, card_c.float().reshape(1, -1, 80))
            print(f"  [submission] host decode vs card decode:       boxes PCC {cb:.5f}, confs PCC {cc:.5f} "
                  f"(gate {DECODE_MIN_PCC[0]} / {DECODE_MIN_PCC[1]})")

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

    def _build_input_lut(self):
        """Constructs and verifies a 256-entry lookup table for exact uint8 -> bfloat16 mapping."""
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
        except Exception as e:
            print(f"  [submission] fast input path disabled: {type(e).__name__}: {e}")
        return None

    def _to_host_reference(self, img_bgr):
        """Standard input path: float32 /255, then ttnn converts to bf16."""
        t = image_to_tensor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                               mesh_mapper=self.in_mapper)

    def _to_host_fast(self, img_bgr):
        """Fast input path: uint8 BGR HWC -> bf16 RGB NCHW in a single LUT gather."""
        chw_rgb = img_bgr.transpose(2, 0, 1)[::-1]
        t = torch.from_numpy(self.lut[chw_rgb]).view(torch.bfloat16).unsqueeze(0)
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                               mesh_mapper=self.in_mapper)

    def _decode(self, path: str):
        """Worker task: reads image from disk, resizes, checks saliency, and prepares host tensor."""
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            return None, 1, 1, True
        H, W = frame.shape[:2]
        img = cv2.resize(frame, self.resolution)
        if HOST_SALIENCY_THRESH > 0.0 and float(img.std()) < HOST_SALIENCY_THRESH:
            return None, W, H, True
        host_t = self._to_host_fast(img) if self.lut is not None else self._to_host_reference(img)
        return host_t, W, H, False

    def warmup(self, paths):
        self.run(paths)

    def run(self, paths):
        """Main inference entry point: streaming execution with bounded in-flight window."""
        paths = list(paths)
        if not paths:
            return {}

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

        inflight = deque()
        post = []
        slot = 0
        self._slot_busy = [None] * RING
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

            busy = self._slot_busy[slot]
            if busy is not None:
                t0 = time.perf_counter()
                busy.result()
                self._st["slot_copy"] += time.perf_counter() - t0
                self._slot_busy[slot] = None

            # Enqueue to card: H2D on CQ1, resharding and trace execution on CQ0
            dev_out = next(iter(self.executor.execute([host_t])))
            host_slot = self.slots[slot]
            _copy_struct_to_host(dev_out, host_slot, self.read_cq)
            self.executor.signal_read_complete()
            ev = ttnn.record_event(self.device, self.read_cq)
            inflight.append((paths[k], W, H, host_slot, ev, host_t, slot))
            slot = (slot + 1) % RING

            # Retire oldest completed frame while the card processes queued frames in parallel
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
                  f"({DECODERS} decoders, {POST_WORKERS} post workers)")
        return out

    def _retire(self, item, post):
        """Waits on device output completion event and submits decoding + NMS to worker thread."""
        p, W, H, host_slot, ev, _host_t, slot_idx = item
        t0 = time.perf_counter()
        ttnn.event_synchronize(ev)
        self._st["card_wait"] += time.perf_counter() - t0
        if self.host_decode:
            fut = self.post_pool.submit(self._timed, _slot_detections, (host_slot, W, H))
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
