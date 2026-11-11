# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Team Submission, lever L04: batch-2 trace on the milestone-8 streaming pipeline.

Milestone 8 (BATCH = 1 here) runs 5.30 ms/frame — at the single-frame card
floor. L03 established WHERE the time is: the per-frame D2H readback (~1 MB)
is nearly free, while every small traced op costs real time (adding ~6 tile
ops cost +0.4 ms). The frame is dominated by the fixed per-op cost of ~140
small post-head ops (3x TtGenBoxes + the output tail) plus ~110 convs.

This lever runs TWO images through one trace (BATCH = 2). The conv stack is
batch-agnostic (batch folds into the flattened NHW dim), so its op count per
batch is unchanged = halved per frame. The post-head tail hardcodes batch 1
upstream, so we monkey-patch it batch-aware with the SAME op count — each op
just processes 2x the rows. Per frame, the entire traced-op count halves.
Nothing about thresholds, NMS or scoring changes; per-image outputs feed the
exact milestone-8 fast_detections.

Feasibility notes (why this should compile): the convs/maxpools see folded
NHW = 204,800, half of what the supported 640x640 batch-1 config (409,600)
already runs on this card; genboxes head-1 extent becomes 3200 rows vs 6400
in the 640 config. All patched-op shapes are therefore <= shapes the pinned
tree already executes.

Batch plumbing:
  - infra built with batch_size=2; its torch seed input is repeated to batch
    2 before trace capture (upstream builds the giraffe seed per-device, not
    per-batch).
  - TtGenBoxes.__call__ patched: group/H/W derived from HW//BATCH, grid
    tensors pre-repeated per image at setup. All slice bounds already use the
    folded extent.
  - TtYOLOv4.__call__ tail patched: output reshapes scaled by BATCH (head-3
    rows pad 100B -> 128B, keeping the milestone-8 padding scheme).
  - Device output is [1,1,12768,4] boxes + [1,12600,80] confs, (head, anchor,
    image, pos)-ordered. Host splits per frame with precomputed row indices
    that reproduce milestone 8's 6300-row order exactly (for BATCH=1 the same
    formula reproduces get_model_result's unpad indices 0:6100 / 6128:6228 /
    6256:6356, verified by construction).
  - Streaming loop pairs consecutive non-empty frames; an odd tail duplicates
    the last frame and discards the extra result. IN_FLIGHT/RING now count
    batches.

BATCH = 1 is the exact milestone-8 path (no patches applied) for same-session
A/B. Measurement mode: a failed batch-2 compile raises loudly rather than
falling back, so a completed run PROVES which mode ran (teardown prints it).
"""
from __future__ import annotations

import gc
import math
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
    from models.demos.yolov4.tt.genboxes import TtGenBoxes
    from models.demos.yolov4.tt.yolov4 import TtYOLOv4
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
BATCH = 2            # images per trace execution; 1 = exact milestone 8
STREAMING = True     # False -> milestone-7 chunked path (BATCH must be 1)
IN_FLIGHT = 4        # batches queued on the card before the host waits on the oldest
RING = 8             # preallocated host output slots (batches); must be > IN_FLIGHT
LOOKAHEAD = 64       # how many frames ahead decoding runs
DECODERS = 24        # JPEG decode + resize + host tensor threads
POST_WORKERS = 8     # NMS threads
TORCH_THREADS = 1    # intra-op threads per torch call
FAST_INPUT = True    # bf16 LUT input path (auto-disabled if parity check fails)
CHUNK = 32           # only used when STREAMING = False

_COCO_IDS = [api.to_coco_id(i) for i in range(80)]


# ----------------------------------------------------------------------------
# Post-processing: identical to milestones 7/8.
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
    cb, cs, ci = b[m], max_conf[m], max_id[m]

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


def _copy_struct_to_host(dev, host, cq_id):
    """Non-blocking card->host copy of a (possibly nested) output structure."""
    if isinstance(dev, (list, tuple)):
        for d, h in zip(dev, host):
            _copy_struct_to_host(d, h, cq_id)
    else:
        ttnn.copy_device_to_host_tensor(dev, host, blocking=False, cq_id=cq_id)


# ----------------------------------------------------------------------------
# L04 batch-aware patches. Bodies are verbatim copies of the pinned tree
# (models/demos/yolov4/tt/genboxes.py __call__ and tt/yolov4.py __call__)
# except where marked [L04]. Applied only when BATCH > 1.
# ----------------------------------------------------------------------------
def _l04_genboxes_call(self, device, input_tensor):
    B, __, HW, dim = input_tensor.shape        # HW is the folded BATCH*H*W extent
    HW_img = HW // BATCH                       # [L04] per-image spatial extent
    H = W = int(math.sqrt(HW_img))             # [L04] was sqrt(HW)
    AHW = self.num_anchors * HW
    A = self.num_anchors

    # [L04] group detection on the per-image extent (was on HW)
    if self.resolution[0] == 320:
        group = {1600: 0, 400: 1, 100: 2}[HW_img]
    else:
        group = {6400: 0, 1600: 1, 400: 2}[HW_img]

    # Pre-derived from the torch function
    if group == 0:
        anchor_w_a = 1.5
        anchor_w_b = 2.375
        anchor_w_c = 5.0
        anchor_h_a = 2.0
        anchor_h_b = 4.5
        anchor_h_c = 3.5
    elif group == 1:
        anchor_w_a = 2.25
        anchor_w_b = 4.75
        anchor_w_c = 4.5
        anchor_h_a = 4.6875
        anchor_h_b = 3.4375
        anchor_h_c = 9.125
    elif group == 2:
        anchor_w_a = 4.4375
        anchor_w_b = 6.0
        anchor_w_c = 14.34375
        anchor_h_a = 3.4375
        anchor_h_b = 7.59375
        anchor_h_c = 12.53125

    input_tensor_i = ttnn.to_memory_config(input_tensor, ttnn.L1_MEMORY_CONFIG)
    input_tensor_i = ttnn.to_layout(input_tensor_i, ttnn.ROW_MAJOR_LAYOUT)
    input_tensor_i = ttnn.permute(input_tensor_i, (0, 1, 3, 2))

    # first anchor
    bx_a = ttnn.slice(input_tensor_i, [0, 0, 0, 0], [1, 1, 1, HW])
    by_a = ttnn.slice(input_tensor_i, [0, 0, 1, 0], [1, 1, 2, HW])
    bw_a = ttnn.slice(input_tensor_i, [0, 0, 2, 0], [1, 1, 3, HW])
    bh_a = ttnn.slice(input_tensor_i, [0, 0, 3, 0], [1, 1, 4, HW])
    det_confs_a = ttnn.slice(input_tensor_i, [0, 0, 4, 0], [1, 1, 5, HW])
    cls_confs_a = ttnn.slice(input_tensor_i, [0, 0, 5, 0], [1, 1, 85, HW])
    # second anchor
    bx_b = ttnn.slice(input_tensor_i, [0, 0, 85, 0], [1, 1, 86, HW])
    by_b = ttnn.slice(input_tensor_i, [0, 0, 86, 0], [1, 1, 87, HW])
    bw_b = ttnn.slice(input_tensor_i, [0, 0, 87, 0], [1, 1, 88, HW])
    bh_b = ttnn.slice(input_tensor_i, [0, 0, 88, 0], [1, 1, 89, HW])
    det_confs_b = ttnn.slice(input_tensor_i, [0, 0, 89, 0], [1, 1, 90, HW])
    cls_confs_b = ttnn.slice(input_tensor_i, [0, 0, 90, 0], [1, 1, 170, HW])
    # third anchor
    bx_c = ttnn.slice(input_tensor_i, [0, 0, 170, 0], [1, 1, 171, HW])
    by_c = ttnn.slice(input_tensor_i, [0, 0, 171, 0], [1, 1, 172, HW])
    bw_c = ttnn.slice(input_tensor_i, [0, 0, 172, 0], [1, 1, 173, HW])
    bh_c = ttnn.slice(input_tensor_i, [0, 0, 173, 0], [1, 1, 174, HW])
    det_confs_c = ttnn.slice(input_tensor_i, [0, 0, 174, 0], [1, 1, 175, HW])
    cls_confs_c = ttnn.slice(input_tensor_i, [0, 0, 175, 0], [1, 1, 255, HW])

    #############
    # Confs
    #############

    det_confs = ttnn.concat([det_confs_a, det_confs_b, det_confs_c], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
    cls_confs = ttnn.concat([cls_confs_a, cls_confs_b, cls_confs_c], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)

    det_confs = ttnn.to_layout(det_confs, ttnn.TILE_LAYOUT)
    cls_confs = ttnn.to_layout(cls_confs, ttnn.TILE_LAYOUT)

    det_confs = ttnn.sigmoid(det_confs, memory_config=ttnn.L1_MEMORY_CONFIG)
    cls_confs = ttnn.sigmoid(cls_confs, memory_config=ttnn.L1_MEMORY_CONFIG)

    confs = ttnn.multiply(det_confs, cls_confs, memory_config=ttnn.L1_MEMORY_CONFIG)
    confs = ttnn.permute(confs, (0, 1, 3, 2))
    confs = ttnn.reshape(confs, (B, AHW, self.num_classes))

    #################
    ## Boxes
    #################

    bx = ttnn.concat([bx_a, bx_b, bx_c], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
    by = ttnn.concat([by_a, by_b, by_c], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
    bw = ttnn.concat([bw_a, bw_b, bw_c], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
    bh = ttnn.concat([bh_a, bh_b, bh_c], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)

    # expensive TilizeWithValPadding
    bx = ttnn.to_layout(bx, ttnn.TILE_LAYOUT)
    by = ttnn.to_layout(by, ttnn.TILE_LAYOUT)
    bw = ttnn.to_layout(bw, ttnn.TILE_LAYOUT)
    bh = ttnn.to_layout(bh, ttnn.TILE_LAYOUT)

    bx = ttnn.sigmoid(bx, memory_config=ttnn.L1_MEMORY_CONFIG)
    by = ttnn.sigmoid(by, memory_config=ttnn.L1_MEMORY_CONFIG)
    bw = ttnn.exp(bw, memory_config=ttnn.L1_MEMORY_CONFIG)
    bh = ttnn.exp(bh, memory_config=ttnn.L1_MEMORY_CONFIG)

    ####
    ## Grid tensor derivation
    ####

    # [L04] self.grid_x/self.grid_y were re-tiled per image at setup
    grid_x = self.grid_x[group]
    grid_y = self.grid_y[group]

    bx = ttnn.add(bx, grid_x, memory_config=ttnn.L1_MEMORY_CONFIG)
    by = ttnn.add(by, grid_y, memory_config=ttnn.L1_MEMORY_CONFIG)

    bx = ttnn.multiply(bx, 1 / W, memory_config=ttnn.L1_MEMORY_CONFIG)
    by = ttnn.multiply(by, 1 / H, memory_config=ttnn.L1_MEMORY_CONFIG)

    ######

    bw_a = ttnn.slice(bw, [0, 0, 0, 0], [1, 1, 1, HW])
    bw_b = ttnn.slice(bw, [0, 1, 0, 0], [1, 2, 1, HW])
    bw_c = ttnn.slice(bw, [0, 2, 0, 0], [1, 3, 1, HW])

    bh_a = ttnn.slice(bh, [0, 0, 0, 0], [1, 1, 1, HW])
    bh_b = ttnn.slice(bh, [0, 1, 0, 0], [1, 2, 1, HW])
    bh_c = ttnn.slice(bh, [0, 2, 0, 0], [1, 3, 1, HW])

    bw_a = bw_a * (anchor_w_a / W)
    bw_b = bw_b * (anchor_w_b / W)
    bw_c = bw_c * (anchor_w_c / W)

    bh_a = bh_a * (anchor_h_a / H)
    bh_b = bh_b * (anchor_h_b / H)
    bh_c = bh_c * (anchor_h_c / H)

    bw = ttnn.concat([bw_a, bw_b, bw_c], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
    bh = ttnn.concat([bh_a, bh_b, bh_c], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)

    bw_half = bw * (0.5)
    bh_half = bh * (0.5)

    ####

    bx1 = bx - bw_half
    by1 = by - bh_half
    bx2 = bx1 + bw
    by2 = by1 + bh

    bx1 = ttnn.to_layout(bx1, ttnn.ROW_MAJOR_LAYOUT)
    bx2 = ttnn.to_layout(bx2, ttnn.ROW_MAJOR_LAYOUT)
    by1 = ttnn.to_layout(by1, ttnn.ROW_MAJOR_LAYOUT)
    by2 = ttnn.to_layout(by2, ttnn.ROW_MAJOR_LAYOUT)

    boxes = ttnn.concat((bx1, by1, bx2, by2), dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
    boxes = ttnn.permute(boxes, (0, 2, 1, 3))

    return boxes, confs


def _l04_yolov4_call(self, x):
    N, C, H, W = x.shape
    min_channels = 16
    if C < min_channels:
        channel_padding_needed = min_channels - C
        nchw = ttnn.pad(x, ((0, 0), (0, channel_padding_needed), (0, 0), (0, 0)), value=0.0)
    else:
        nchw = x
    nhwc = ttnn.permute(nchw, (0, 2, 3, 1))  # NCHW -> NHWC
    ttnn.deallocate(nchw)
    ttnn.deallocate(x)
    nhwc = ttnn.reallocate(nhwc)
    x = ttnn.reshape(nhwc, [1, 1, nhwc.shape[0] * nhwc.shape[1] * nhwc.shape[2], nhwc.shape[-1]])
    d1 = self.down1(x)
    d2 = self.down2(d1)
    ttnn.deallocate(d1)
    d3 = self.down3(d2)
    ttnn.deallocate(d2)
    d4 = self.down4(d3)
    d5 = self.down5(d4)
    x20, x13, x6 = self.neck([d5, d4, d3])
    x4, x5, x6 = self.head([x20, x13, x6])

    x4_boxes_confs = self.boxes_confs_0(self.device, x4)
    x5_boxes_confs = self.boxes_confs_1(self.device, x5)
    x6_boxes_confs = self.boxes_confs_2(self.device, x6)

    confs_1 = ttnn.to_layout(x4_boxes_confs[1], ttnn.ROW_MAJOR_LAYOUT)
    confs_2 = ttnn.to_layout(x5_boxes_confs[1], ttnn.ROW_MAJOR_LAYOUT)
    confs_3 = ttnn.to_layout(x6_boxes_confs[1], ttnn.ROW_MAJOR_LAYOUT)
    confs = ttnn.concat([confs_1, confs_2, confs_3], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)

    boxes_1 = ttnn.to_layout(x4_boxes_confs[0], ttnn.ROW_MAJOR_LAYOUT)
    boxes_2 = ttnn.to_layout(x5_boxes_confs[0], ttnn.ROW_MAJOR_LAYOUT)
    boxes_3 = ttnn.to_layout(x6_boxes_confs[0], ttnn.ROW_MAJOR_LAYOUT)
    # [L04] output reshapes scaled by BATCH; head-3 per-anchor rows pad
    # 100*BATCH -> 128*BATCH (same scheme as upstream's 100 -> 128).
    if self.resolution[0] == 320:
        boxes_1 = ttnn.reshape(boxes_1, (1, 4, 1, 4800 * BATCH))
        boxes_2 = ttnn.reshape(boxes_2, (1, 4, 1, 1200 * BATCH))
        boxes_3 = ttnn.pad(boxes_3, ((0, 0), (0, 0), (0, 0), (0, 28 * BATCH)), 0)
        boxes_3 = ttnn.reshape(boxes_3, (1, 4, 1, 384 * BATCH))
    else:
        boxes_1 = ttnn.reshape(boxes_1, (1, 4, 1, 19200 * BATCH))
        boxes_2 = ttnn.reshape(boxes_2, (1, 4, 1, 4800 * BATCH))
        boxes_3 = ttnn.pad(boxes_3, ((0, 0), (0, 0), (0, 0), (0, 28 * BATCH)), 0)
        boxes_3 = ttnn.reshape(boxes_3, (1, 4, 1, 1284 * BATCH))

    boxes_1 = ttnn.permute(boxes_1, (0, 2, 3, 1))
    boxes_2 = ttnn.permute(boxes_2, (0, 2, 3, 1))
    boxes_3 = ttnn.permute(boxes_3, (0, 2, 3, 1))
    boxes = ttnn.concat([boxes_1, boxes_2, boxes_3], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)

    return boxes, confs


def _frame_rows(i, padded_head3):
    """Row indices of image i inside the batched device output.

    Layout per head h: 3 anchor blocks of stride BATCH*hw_h (boxes head 3:
    128*BATCH, tile-padded); inside a block, image i's hw_h rows start at
    i*hw_h. Concatenating heads/anchors in order reproduces, for BATCH=1,
    exactly get_model_result's unpad (0:6100, 6128:6228, 6256:6356 for boxes,
    0:6300 contiguous for confs) -> milestone-8 row order per image.
    """
    rows = []
    base = 0
    for h, hw in enumerate((1600, 400, 100)):
        stride = 128 * BATCH if (padded_head3 and h == 2) else BATCH * hw
        for a in range(3):
            start = base + a * stride + i * hw
            rows.append(torch.arange(start, start + hw, dtype=torch.long))
        base += 3 * stride
    return torch.cat(rows)


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
        self.batch = BATCH

        if get_mesh_mappers is not None:
            self.in_mapper, _, self.out_composer = get_mesh_mappers(device)
        else:
            self.in_mapper = self.out_composer = None

        if BATCH > 1:
            # Patch BEFORE trace capture; measurement mode = no fallback, a
            # compile failure must fail the run so a completed run proves
            # batch-2 actually executed.
            TtGenBoxes._l04_original_call = TtGenBoxes.__call__
            TtYOLOv4._l04_original_call = TtYOLOv4.__call__
            TtGenBoxes.__call__ = _l04_genboxes_call
            TtYOLOv4.__call__ = _l04_yolov4_call

        self.infra = YOLOv4PerformanceRunnerInfra(
            device, BATCH, ttnn.bfloat8_b, ttnn.bfloat8_b,
            model_location_generator=None,
            resolution=self.resolution,
            mesh_mapper=self.in_mapper,
            mesh_composer=self.out_composer,
        )

        if BATCH > 1:
            # Upstream builds the giraffe seed per-device, not per-batch:
            # repeat it to the batch so the trace is captured at batch size.
            ti = self.infra.torch_input_tensor
            if ti.shape[0] != BATCH:
                self.infra.torch_input_tensor = ti.repeat(BATCH // ti.shape[0], 1, 1, 1)
            self._repeat_grids(self.infra.ttnn_yolov4_model)
            self._box_rows = [_frame_rows(i, True) for i in range(BATCH)]
            self._conf_rows = [_frame_rows(i, False) for i in range(BATCH)]

        self.seed_host, dram_cfg, l1_cfg = self.infra.setup_dram_sharded_input(device)
        self.pipeline = create_pipeline_from_config(
            config=PipelineConfig(use_trace=True, num_command_queues=2,
                                  all_transfers_on_separate_command_queue=False),
            model=YoloV4PipelineRunner(self.infra),
            device=device,
            dram_input_memory_config=dram_cfg,
            l1_input_memory_config=l1_cfg,
        )
        self.pipeline.compile(self.seed_host)

        self.streaming = STREAMING and hasattr(ttnn, "event_synchronize")
        if BATCH > 1:
            assert self.streaming, "BATCH > 1 requires the streaming path"
        if self.streaming:
            assert RING > IN_FLIGHT, "RING must exceed IN_FLIGHT or a slot is overwritten before it is read"
            self.executor = self.pipeline.executor
            self.read_cq = self.executor.get_read_cq()
            self.pipeline.preallocate_output_tensors_on_host(RING)
            self.slots = self.pipeline.output_tensors
        self.lut = self._build_input_lut() if FAST_INPUT else None

        self.pool = ThreadPoolExecutor(max_workers=DECODERS)
        self.post_pool = ThreadPoolExecutor(max_workers=POST_WORKERS)

    def _repeat_grids(self, model):
        """Re-tile each genboxes grid constant per image: [1,3,1,HH] ->
        [1,3,1,BATCH*HH], matching images laid out contiguously along the
        folded row dim. Done once at setup, outside trace capture."""
        for inst in (model.boxes_confs_0, model.boxes_confs_1, model.boxes_confs_2):
            for lst in (inst.grid_x, inst.grid_y):
                for g in range(len(lst)):
                    t = ttnn.to_torch(lst[g])
                    lst[g] = ttnn.from_torch(
                        t.repeat(1, 1, 1, BATCH), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=self.device,
                    )

    # ---- input conversion -------------------------------------------------
    def _to_host_reference(self, img_bgr):
        """Milestone-7 input path: float32 /255, then ttnn converts to bf16."""
        t = image_to_tensor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                               mesh_mapper=self.in_mapper)

    def _to_host_fast(self, img_bgr):
        """uint8 BGR HWC -> bf16 RGB NCHW in one LUT gather, no float32 pass."""
        return ttnn.from_torch(self._to_torch_fast(img_bgr), dtype=ttnn.bfloat16,
                               layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.in_mapper)

    def _to_torch_fast(self, img_bgr):
        chw_rgb = img_bgr.transpose(2, 0, 1)[::-1]
        return torch.from_numpy(self.lut[chw_rgb]).view(torch.bfloat16).unsqueeze(0)

    def _to_torch_reference(self, img_bgr):
        t = image_to_tensor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        return t.to(torch.bfloat16)

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
        if BATCH == 1:
            host_t = self._to_host_fast(img) if self.lut is not None else self._to_host_reference(img)
            return host_t, W, H, False
        # Batched path: return a torch tensor; the main loop concatenates a
        # batch and does one from_torch per BATCH frames.
        t = self._to_torch_fast(img) if self.lut is not None else self._to_torch_reference(img)
        return t, W, H, False

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
        submitted = 0

        def top_up(limit):
            nonlocal submitted
            limit = min(n, limit)
            while submitted < limit:
                decoded[submitted] = self.pool.submit(self._decode, paths[submitted])
                submitted += 1

        inflight = deque()   # (frames, host_slot, event, host_input)
        post = []            # (path, future)
        slot = 0
        pending = []         # [( (path, W, H), tensor ), ...] not yet a full batch

        def submit(items):
            nonlocal slot
            if BATCH == 1:
                host_t = items[0][1]
            else:
                ts = [t for _, t in items]
                if len(ts) < BATCH:          # odd tail: pad with the last frame
                    ts = ts + [ts[-1]] * (BATCH - len(ts))
                host_t = ttnn.from_torch(torch.cat(ts, 0), dtype=ttnn.bfloat16,
                                         layout=ttnn.ROW_MAJOR_LAYOUT,
                                         mesh_mapper=self.in_mapper)
            if len(inflight) >= IN_FLIGHT:
                self._retire(inflight.popleft(), post)
            # H2D on CQ1, reshard + trace on CQ0 (the executor's own event logic).
            dev_out = next(iter(self.executor.execute([host_t])))
            host_slot = self.slots[slot]
            slot = (slot + 1) % RING
            _copy_struct_to_host(dev_out, host_slot, self.read_cq)
            self.executor.signal_read_complete()
            ev = ttnn.record_event(self.device, self.read_cq)
            inflight.append(([f for f, _ in items], host_slot, ev, host_t))

        top_up(LOOKAHEAD)
        for k in range(n):
            top_up(k + LOOKAHEAD)
            t, W, H, empty = decoded.pop(k).result()
            if empty:
                out[paths[k]] = []
                continue
            pending.append(((paths[k], W, H), t))
            if len(pending) == BATCH:
                submit(pending)
                pending = []
        if pending:
            submit(pending)

        while inflight:
            self._retire(inflight.popleft(), post)
        for p, fut in post:
            out[p] = fut.result()
        return out

    def _retire(self, item, post):
        """Wait for one batch's output, split it per frame, hand each to NMS."""
        frames, host_slot, ev, _host_t = item
        ttnn.event_synchronize(ev)
        if BATCH == 1:
            # get_model_result returns fresh tensors (cat / dtype change), so
            # the slot can be reused as soon as this returns.
            boxes, confs = get_model_result(host_slot, self.resolution, mesh_composer=self.out_composer)
            p, W, H = frames[0]
            post.append((p, self.post_pool.submit(fast_detections, boxes, confs, W, H)))
            return
        bt = ttnn.to_torch(host_slot[0], mesh_composer=self.out_composer)
        ct = ttnn.to_torch(host_slot[1], mesh_composer=self.out_composer)
        bt = bt.reshape(-1, bt.shape[-1])     # [BATCH*6384, 4]
        ct = ct.reshape(-1, ct.shape[-1])     # [BATCH*6300, 80]
        for i, (p, W, H) in enumerate(frames):
            # index_select copies, so the slot is reusable after this loop;
            # .to(float16) matches get_model_result's output dtype exactly.
            b = bt.index_select(0, self._box_rows[i]).to(torch.float16).unsqueeze(0)
            c = ct.index_select(0, self._conf_rows[i]).to(torch.float16).unsqueeze(0)
            post.append((p, self.post_pool.submit(fast_detections, b, c, W, H)))

    # ---- milestone-7 path, kept for A/B (BATCH = 1 only) ---------------------
    def _run_chunked(self, paths):
        assert BATCH == 1, "the chunked path only supports BATCH = 1"
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
        boxes, confs = get_model_result(raw, self.resolution, mesh_composer=self.out_composer)
        return fast_detections(boxes, confs, W, H)

    def teardown(self):
        print(f"  [submission] batch this run: {getattr(self, 'batch', BATCH)}")
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
        if HAS_TTNN and BATCH > 1 and hasattr(TtGenBoxes, "_l04_original_call"):
            TtGenBoxes.__call__ = TtGenBoxes._l04_original_call
            TtYOLOv4.__call__ = TtYOLOv4._l04_original_call
        gc.enable()
        gc.collect()
