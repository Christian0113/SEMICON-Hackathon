# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Team Submission, milestone 8: per-frame streaming pipeline.

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

import os
import submission_api as api

# Resolution selection:
# Candidate resolutions: (320, 320), (384, 384), (416, 416), (480, 480), (512, 512), (640, 640)
# Can be overridden via environment variable YOLO_RESOLUTION="416,416"
_env_res = os.environ.get("YOLO_RESOLUTION", "").strip()
if _env_res:
    try:
        _parts = [int(p.strip()) for p in _env_res.split(",")]
        RESOLUTION = (_parts[0], _parts[1]) if len(_parts) == 2 else (_parts[0], _parts[0])
    except Exception:
        RESOLUTION = (416, 416)
else:
    RESOLUTION = (416, 416)

CONF_THRESH = 0.001
NMS_THRESH = 0.4
EMPTY_THRESH = 0.05
HOST_SALIENCY_THRESH = 3.0

TRACE_REGION_SIZE = 6434816 if RESOLUTION == (320, 320) else 16777216

# ---- scheduling knobs (tune on the card; see docs/IDEA_8_STREAMING_PIPELINE.md)
STREAMING = True     # False -> milestone-7 chunked path
IN_FLIGHT = 4        # frames queued on the card before the host waits on the oldest
RING = 8             # preallocated host output slots; must be > IN_FLIGHT
LOOKAHEAD = 64       # how many frames ahead decoding runs
DECODERS = 24        # JPEG decode + resize + host tensor threads
POST_WORKERS = 8     # NMS threads
TORCH_THREADS = 1    # intra-op threads per torch call
FAST_INPUT = True    # bf16 LUT input path (auto-disabled if parity check fails)
CHUNK = 32           # only used when STREAMING = False

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


def fast_detections(boxes, confs, W, H, conf_thresh=CONF_THRESH, nms_thresh=NMS_THRESH, max_ids=None):
    boxes = torch.as_tensor(boxes)
    b = boxes.reshape(boxes.shape[0], boxes.shape[1], 4)[0].float()

    if max_ids is not None:
        max_conf = torch.as_tensor(confs).reshape(-1).float()
        max_id = torch.as_tensor(max_ids).reshape(-1).long()
    else:
        confs = torch.as_tensor(confs)
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


class TruncatedYoloV4Runner:
    """Wraps YOLOv4 to compute max and argmax over class dimension on Wormhole Tensix cores.
    Shrinks confs output from [1, N, 80] to [1, N, 1] + [1, N, 1], achieving a 40x reduction
    in PCIe DMA data and eliminating host-side max(dim=1).
    """
    def __init__(self, test_infra):
        self._test_infra = test_infra

    def __call__(self, l1_input_tensor):
        self._test_infra.input_tensor = l1_input_tensor
        self._test_infra.run()
        boxes, confs = self._test_infra.output_tensor
        max_confs = ttnn.max(confs, dim=-1, keepdim=True)
        max_ids = ttnn.argmax(confs, dim=-1, keepdim=True)
        ttnn.deallocate(confs, force=True)
        self._test_infra.output_tensor = None
        return boxes, max_confs, max_ids


def patch_yolov4_dynamic_resolution(resolution):
    """Unlocks arbitrary resolutions (H, W multiples of 32) and repairs 640x640 bfloat8_b precision."""
    try:
        import models.demos.yolov4.common as _common
        import models.demos.yolov4.tt.yolov4 as _yolov4_mod
        import models.demos.yolov4.tt.genboxes as _genboxes_mod
        import models.demos.yolov4.tt.model_preprocessing as _mp
        import models.demos.yolov4.tt.downsample1 as _ds1_mod
        import models.demos.yolov4.runner.performant_runner_infra as _infra_mod
        from models.demos.utils.common_demo_utils import get_mesh_mappers
        from ttnn.model_preprocessing import infer_ttnn_module_args, preprocess_model_parameters

        # 1. Dynamic get_model_result
        def _patched_get_model_result(ttnn_output_tensor, res, mesh_composer=None):
            result_boxes_padded = ttnn.to_torch(ttnn_output_tensor[0], mesh_composer=mesh_composer)
            result_confs = ttnn.to_torch(ttnn_output_tensor[1], mesh_composer=mesh_composer)
            result_boxes_padded = result_boxes_padded.permute(0, 2, 1, 3)

            H, W = res
            hw1 = (H // 8) * (W // 8)
            hw2 = (H // 16) * (W // 16)
            hw3 = (H // 32) * (W // 32)
            pad_len = (32 - (hw3 % 32)) % 32

            n1 = 3 * hw1
            n2 = 3 * hw2
            b0_end = n1 + n2 + hw3
            b1_start = b0_end + pad_len
            b1_end = b1_start + hw3
            b2_start = b1_end + pad_len
            b2_end = b2_start + hw3

            result_boxes_list = [
                result_boxes_padded[:, :b0_end],
                result_boxes_padded[:, b1_start:b1_end],
                result_boxes_padded[:, b2_start:b2_end],
            ]
            result_boxes = torch.cat(result_boxes_list, dim=1)
            return [result_boxes.to(torch.float16), result_confs.to(torch.float16)]

        _common.get_model_result = _patched_get_model_result

        # 2. Dynamic TtGenBoxes
        def _patched_genboxes_init(self, device, res):
            self.thresh = 0.6
            self.num_classes = 80
            self.num_anchors = 3
            self.resolution = res
            H, W = res
            h1, h2, h3 = H // 8, H // 16, H // 32
            w1, w2, w3 = W // 8, W // 16, W // 32

            self.grid_x = []
            self.grid_y = []
            for H_g, W_g in ((h1, w1), (h2, w2), (h3, w3)):
                gx = torch.linspace(0, W_g - 1, W_g, dtype=torch.float32).repeat(H_g, 1).view(1, 1, 1, H_g * W_g).repeat(1, 3, 1, 1)
                gy = torch.linspace(0, H_g - 1, H_g, dtype=torch.float32).unsqueeze(1).repeat(1, W_g).view(1, 1, 1, H_g * W_g).repeat(1, 3, 1, 1)
                self.grid_x.append(ttnn.from_torch(gx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device))
                self.grid_y.append(ttnn.from_torch(gy, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device))

        def _patched_genboxes_call(self, device, input_tensor):
            B, __, HW, dim = input_tensor.shape
            H, W = self.resolution
            hw1 = (H // 8) * (W // 8)
            hw2 = (H // 16) * (W // 16)
            hw3 = (H // 32) * (W // 32)
            if HW == hw1:
                group = 0
                cur_h, cur_w = H // 8, W // 8
            elif HW == hw2:
                group = 1
                cur_h, cur_w = H // 16, W // 16
            elif HW == hw3:
                group = 2
                cur_h, cur_w = H // 32, W // 32
            else:
                raise ValueError(f"Unknown HW={HW} for resolution={self.resolution}")

            if group == 0:
                anchor_w_a, anchor_w_b, anchor_w_c = 1.5, 2.375, 5.0
                anchor_h_a, anchor_h_b, anchor_h_c = 2.0, 4.5, 3.5
            elif group == 1:
                anchor_w_a, anchor_w_b, anchor_w_c = 2.25, 4.75, 4.5
                anchor_h_a, anchor_h_b, anchor_h_c = 4.6875, 3.4375, 9.125
            elif group == 2:
                anchor_w_a, anchor_w_b, anchor_w_c = 4.4375, 6.0, 14.34375
                anchor_h_a, anchor_h_b, anchor_h_c = 3.4375, 7.59375, 12.53125

            input_tensor_i = ttnn.to_memory_config(input_tensor, ttnn.L1_MEMORY_CONFIG)
            input_tensor_i = ttnn.to_layout(input_tensor_i, ttnn.ROW_MAJOR_LAYOUT)
            input_tensor_i = ttnn.permute(input_tensor_i, (0, 1, 3, 2))

            bx_a = ttnn.slice(input_tensor_i, [0, 0, 0, 0], [1, 1, 1, HW])
            by_a = ttnn.slice(input_tensor_i, [0, 0, 1, 0], [1, 1, 2, HW])
            bw_a = ttnn.slice(input_tensor_i, [0, 0, 2, 0], [1, 1, 3, HW])
            bh_a = ttnn.slice(input_tensor_i, [0, 0, 3, 0], [1, 1, 4, HW])
            det_confs_a = ttnn.slice(input_tensor_i, [0, 0, 4, 0], [1, 1, 5, HW])
            cls_confs_a = ttnn.slice(input_tensor_i, [0, 0, 5, 0], [1, 1, 85, HW])

            bx_b = ttnn.slice(input_tensor_i, [0, 0, 85, 0], [1, 1, 86, HW])
            by_b = ttnn.slice(input_tensor_i, [0, 0, 86, 0], [1, 1, 87, HW])
            bw_b = ttnn.slice(input_tensor_i, [0, 0, 87, 0], [1, 1, 88, HW])
            bh_b = ttnn.slice(input_tensor_i, [0, 0, 88, 0], [1, 1, 89, HW])
            det_confs_b = ttnn.slice(input_tensor_i, [0, 0, 89, 0], [1, 1, 90, HW])
            cls_confs_b = ttnn.slice(input_tensor_i, [0, 0, 90, 0], [1, 1, 170, HW])

            bx_c = ttnn.slice(input_tensor_i, [0, 0, 170, 0], [1, 1, 171, HW])
            by_c = ttnn.slice(input_tensor_i, [0, 0, 171, 0], [1, 1, 172, HW])
            bw_c = ttnn.slice(input_tensor_i, [0, 0, 172, 0], [1, 1, 173, HW])
            bh_c = ttnn.slice(input_tensor_i, [0, 0, 173, 0], [1, 1, 174, HW])
            det_confs_c = ttnn.slice(input_tensor_i, [0, 0, 174, 0], [1, 1, 175, HW])
            cls_confs_c = ttnn.slice(input_tensor_i, [0, 0, 175, 0], [1, 1, 255, HW])

            det_confs = ttnn.concat([det_confs_a, det_confs_b, det_confs_c], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
            cls_confs = ttnn.concat([cls_confs_a, cls_confs_b, cls_confs_c], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)

            det_confs = ttnn.to_layout(det_confs, ttnn.TILE_LAYOUT)
            cls_confs = ttnn.to_layout(cls_confs, ttnn.TILE_LAYOUT)

            det_confs = ttnn.sigmoid(det_confs, memory_config=ttnn.L1_MEMORY_CONFIG)
            cls_confs = ttnn.sigmoid(cls_confs, memory_config=ttnn.L1_MEMORY_CONFIG)

            confs = ttnn.multiply(det_confs, cls_confs, memory_config=ttnn.L1_MEMORY_CONFIG)
            confs = ttnn.permute(confs, (0, 1, 3, 2))
            confs = ttnn.reshape(confs, (B, self.num_anchors * HW, self.num_classes))

            bx = ttnn.concat([bx_a, bx_b, bx_c], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
            by = ttnn.concat([by_a, by_b, by_c], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
            bw = ttnn.concat([bw_a, bw_b, bw_c], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
            bh = ttnn.concat([bh_a, bh_b, bh_c], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)

            bx = ttnn.to_layout(bx, ttnn.TILE_LAYOUT)
            by = ttnn.to_layout(by, ttnn.TILE_LAYOUT)
            bw = ttnn.to_layout(bw, ttnn.TILE_LAYOUT)
            bh = ttnn.to_layout(bh, ttnn.TILE_LAYOUT)

            bx = ttnn.sigmoid(bx, memory_config=ttnn.L1_MEMORY_CONFIG)
            by = ttnn.sigmoid(by, memory_config=ttnn.L1_MEMORY_CONFIG)
            bw = ttnn.exp(bw, memory_config=ttnn.L1_MEMORY_CONFIG)
            bh = ttnn.exp(bh, memory_config=ttnn.L1_MEMORY_CONFIG)

            grid_x = self.grid_x[group]
            grid_y = self.grid_y[group]

            bx = ttnn.add(bx, grid_x, memory_config=ttnn.L1_MEMORY_CONFIG)
            by = ttnn.add(by, grid_y, memory_config=ttnn.L1_MEMORY_CONFIG)

            bx = ttnn.multiply(bx, 1.0 / cur_w, memory_config=ttnn.L1_MEMORY_CONFIG)
            by = ttnn.multiply(by, 1.0 / cur_h, memory_config=ttnn.L1_MEMORY_CONFIG)

            bw_a = ttnn.slice(bw, [0, 0, 0, 0], [1, 1, 1, HW])
            bw_b = ttnn.slice(bw, [0, 1, 0, 0], [1, 2, 1, HW])
            bw_c = ttnn.slice(bw, [0, 2, 0, 0], [1, 3, 1, HW])

            bh_a = ttnn.slice(bh, [0, 0, 0, 0], [1, 1, 1, HW])
            bh_b = ttnn.slice(bh, [0, 1, 0, 0], [1, 2, 1, HW])
            bh_c = ttnn.slice(bh, [0, 2, 0, 0], [1, 3, 1, HW])

            bw_a = bw_a * (anchor_w_a / cur_w)
            bw_b = bw_b * (anchor_w_b / cur_w)
            bw_c = bw_c * (anchor_w_c / cur_w)

            bh_a = bh_a * (anchor_h_a / cur_h)
            bh_b = bh_b * (anchor_h_b / cur_h)
            bh_c = bh_c * (anchor_h_c / cur_h)

            bw = ttnn.concat([bw_a, bw_b, bw_c], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
            bh = ttnn.concat([bh_a, bh_b, bh_c], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)

            bw_half = bw * 0.5
            bh_half = bh * 0.5

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

        _genboxes_mod.TtGenBoxes.__init__ = _patched_genboxes_init
        _genboxes_mod.TtGenBoxes.__call__ = _patched_genboxes_call

        # 3. Dynamic TtYOLOv4.__call__
        def _patched_yolov4_call(self, x):
            if x.layout != ttnn.ROW_MAJOR_LAYOUT:
                nchw = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            else:
                nchw = x
            nhwc = ttnn.permute(nchw, (0, 2, 3, 1))
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

            H, W = self.resolution
            hw1 = (H // 8) * (W // 8)
            hw2 = (H // 16) * (W // 16)
            hw3 = (H // 32) * (W // 32)
            pad_len = (32 - (hw3 % 32)) % 32

            boxes_1 = ttnn.reshape(boxes_1, (1, 4, 1, 3 * hw1))
            boxes_2 = ttnn.reshape(boxes_2, (1, 4, 1, 3 * hw2))
            if pad_len > 0:
                boxes_3 = ttnn.pad(boxes_3, ((0, 0), (0, 0), (0, 0), (0, pad_len)), 0)
            boxes_3 = ttnn.reshape(boxes_3, (1, 4, 1, 3 * (hw3 + pad_len)))

            boxes_1 = ttnn.permute(boxes_1, (0, 2, 3, 1))
            boxes_2 = ttnn.permute(boxes_2, (0, 2, 3, 1))
            boxes_3 = ttnn.permute(boxes_3, (0, 2, 3, 1))
            boxes = ttnn.concat([boxes_1, boxes_2, boxes_3], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
            return boxes, confs

        _yolov4_mod.TtYOLOv4.__call__ = _patched_yolov4_call

        # 4. Universal bfloat8_b model parameters across all resolutions
        def _patched_create_ds1_params(conv_args, res):
            is_640 = (res == (640, 640))
            conv_args.c1["act_block_h"] = 1024 if is_640 else 128
            conv_args.c1["deallocate_activation"] = True
            conv_args.c1["reshard_if_not_optimal"] = False
            conv_args.c1["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            conv_args.c1["dtype"] = ttnn.bfloat8_b
            conv_args.c1["num_cores_nhw"] = None

            conv_args.c2["act_block_h"] = 320 if is_640 else None
            conv_args.c2["deallocate_activation"] = True
            conv_args.c2["reshard_if_not_optimal"] = False
            conv_args.c2["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            conv_args.c2["dtype"] = ttnn.bfloat8_b
            conv_args.c2["num_cores_nhw"] = None

            for name in ("c3", "c4", "c5", "c6", "c7", "c8"):
                c = getattr(conv_args, name)
                c["act_block_h"] = 1024 if is_640 else None
                c["deallocate_activation"] = (name not in ("c3", "c5"))
                c["reshard_if_not_optimal"] = False
                c["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                c["dtype"] = ttnn.bfloat8_b
                c["num_cores_nhw"] = None

        def _patched_create_yolov4_params(model, input_tensor, res, device):
            _, weights_mesh_mapper, _ = get_mesh_mappers(device)
            parameters = preprocess_model_parameters(
                initialize_model=lambda: model,
                custom_preprocessor=_mp.create_custom_mesh_preprocessor(weights_mesh_mapper),
            )
            parameters["resolution"] = res
            for sub in ("downsample1", "downsample2", "downsample3", "downsample4", "downsample5", "neck", "head"):
                getattr(parameters, sub)["resolution"] = res

            parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda m: m(input_tensor), device=None)

            _patched_create_ds1_params(parameters.conv_args.downsample1, res)
            # ds2..ds5, neck, head use (320, 320) to enforce bfloat8_b across all layers
            _mp._create_ds2_model_parameters(parameters.conv_args.downsample2, (320, 320))
            _mp._create_ds3_model_parameters(parameters.conv_args.downsample3, (320, 320))
            _mp._create_ds4_model_parameters(parameters.conv_args.downsample4, (320, 320))
            _mp._create_ds5_model_parameters(parameters.conv_args.downsample5, (320, 320))
            _mp._create_neck_model_parameters(parameters.conv_args.neck, (320, 320))
            _mp._create_head_model_parameters(parameters.conv_args.head, (320, 320))
            return parameters

        _mp.create_yolov4_model_parameters = _patched_create_yolov4_params
        _infra_mod.create_yolov4_model_parameters = _patched_create_yolov4_params

        # 5. Universal downsample1 height-sharded concat
        def _patched_sharded_concat(input_tensors, num_cores=64, dim=3):
            t0, t1 = input_tensors[0], input_tensors[1]
            mem_cfg = ttnn.create_sharded_memory_config(
                [
                    t0.memory_config().shard_spec.shape[0],
                    2 * t0.memory_config().shard_spec.shape[1],
                ],
                core_grid=t1.memory_config().shard_spec.grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                use_height_and_width_as_shard_shape=True,
            )
            return ttnn.concat(input_tensors, dim=dim, memory_config=mem_cfg)

        _ds1_mod.sharded_concat = _patched_sharded_concat

        # 6. Safe concat typecast for neck and head
        _orig_concat = ttnn.concat
        def _safe_concat(tensors, dim=0, memory_config=None, **kwargs):
            if len(tensors) > 1 and any(getattr(t, "dtype", None) == ttnn.bfloat8_b for t in tensors):
                casted = []
                for t in tensors:
                    if getattr(t, "dtype", None) != ttnn.bfloat8_b:
                        try:
                            t = ttnn.add(t, 0.0, dtype=ttnn.bfloat8_b)
                        except Exception:
                            pass
                    casted.append(t)
                tensors = casted
            return _orig_concat(tensors, dim=dim, memory_config=memory_config, **kwargs)

        ttnn.concat = _safe_concat
        print(f"  [submission] Dynamic resolution patches applied for {resolution}")
    except Exception as e:
        print(f"  [submission] Dynamic resolution patch skipped ({type(e).__name__}: {e})")


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

        # Apply dynamic resolution and precision patches
        patch_yolov4_dynamic_resolution(self.resolution)

        self.infra = YOLOv4PerformanceRunnerInfra(
            device, 1, ttnn.bfloat8_b, ttnn.bfloat8_b,
            model_location_generator=None,
            resolution=self.resolution,
            mesh_mapper=self.in_mapper,
            mesh_composer=self.out_composer,
        )
        self.seed_host, dram_cfg, l1_cfg = self.infra.setup_dram_sharded_input(device)

        self.is_truncated = False
        try:
            trunc_runner = TruncatedYoloV4Runner(self.infra)
            self.pipeline = create_pipeline_from_config(
                config=PipelineConfig(
                    use_trace=True,
                    num_command_queues=2,
                    all_transfers_on_separate_command_queue=False,
                ),
                model=trunc_runner,
                device=device,
                dram_input_memory_config=dram_cfg,
                l1_input_memory_config=l1_cfg,
            )
            self.pipeline.compile(self.seed_host)
            self.is_truncated = True
        except Exception as e:
            print(f"  [submission] On-device truncation fallback: {type(e).__name__}: {e}")
            self.is_truncated = False
            for cq in (0, 1):
                for tid in (0, 1, 2):
                    try:
                        ttnn.end_trace_capture(device, tid, cq_id=cq)
                    except Exception:
                        pass
            for tid in (0, 1, 2):
                try:
                    ttnn.release_trace(device, tid)
                except Exception:
                    pass
            try:
                ttnn.synchronize_device(device)
            except Exception:
                pass
            if hasattr(self, "pipeline") and self.pipeline is not None:
                try:
                    self.pipeline.cleanup()
                except Exception:
                    pass

            self.pipeline = create_pipeline_from_config(
                config=PipelineConfig(
                    use_trace=True,
                    num_command_queues=2,
                    all_transfers_on_separate_command_queue=False,
                ),
                model=YoloV4PipelineRunner(self.infra),
                device=device,
                dram_input_memory_config=dram_cfg,
                l1_input_memory_config=l1_cfg,
            )
            self.pipeline.compile(self.seed_host)
            self.is_truncated = False

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
        interp = cv2.INTER_AREA if (W > self.resolution[0] or H > self.resolution[1]) else cv2.INTER_LINEAR
        img = cv2.resize(frame, self.resolution, interpolation=interp)
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
        top_up(LOOKAHEAD)

        for k in range(n):
            top_up(k + LOOKAHEAD)
            host_t, W, H, empty = decoded.pop(k).result()
            if empty:
                out[paths[k]] = []
                continue

            # Keep the card fed: only wait once IN_FLIGHT frames are queued.
            if len(inflight) >= IN_FLIGHT:
                self._retire(inflight.popleft(), post)

            # H2D on CQ1, reshard + trace on CQ0 (the executor's own event logic).
            dev_out = next(iter(self.executor.execute([host_t])))
            host_slot = self.slots[slot]
            slot = (slot + 1) % RING
            _copy_struct_to_host(dev_out, host_slot, self.read_cq)
            self.executor.signal_read_complete()
            ev = ttnn.record_event(self.device, self.read_cq)
            inflight.append((paths[k], W, H, host_slot, ev, host_t))

        while inflight:
            self._retire(inflight.popleft(), post)
        for p, fut in post:
            out[p] = fut.result()
        return out

    def _retire(self, item, post):
        """Wait for one frame's output, copy it out of its slot, hand it to NMS."""
        p, W, H, host_slot, ev, _host_t = item
        ttnn.event_synchronize(ev)
        # get_model_result returns fresh tensors (cat / dtype change), so the
        # slot can be reused as soon as this returns.
        if getattr(self, "is_truncated", False):
            raw_boxes, raw_max_confs, raw_max_ids = host_slot
            boxes, max_confs = get_model_result([raw_boxes, raw_max_confs], self.resolution, mesh_composer=self.out_composer)
            max_ids = ttnn.to_torch(raw_max_ids, mesh_composer=self.out_composer)
            post.append((p, self.post_pool.submit(fast_detections, boxes, max_confs, W, H, max_ids=max_ids)))
        else:
            boxes, confs = get_model_result(host_slot, self.resolution, mesh_composer=self.out_composer)
            post.append((p, self.post_pool.submit(fast_detections, boxes, confs, W, H)))

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
        if getattr(self, "is_truncated", False):
            raw_boxes, raw_max_confs, raw_max_ids = raw
            boxes, max_confs = get_model_result([raw_boxes, raw_max_confs], self.resolution, mesh_composer=self.out_composer)
            max_ids = ttnn.to_torch(raw_max_ids, mesh_composer=self.out_composer)
            return fast_detections(boxes, max_confs, W, H, max_ids=max_ids)
        else:
            boxes, confs = get_model_result(raw, self.resolution, mesh_composer=self.out_composer)
            return fast_detections(boxes, confs, W, H)

    def teardown(self):
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
