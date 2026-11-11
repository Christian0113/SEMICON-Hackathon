# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""YOLOv4 double-buffer experiment based on my_submission.py.

Select YOLO_DOUBLE_BUFFER=original|A|B|C|D before launching evaluate.py.
Each mode must run in a fresh process. Only Conv2dConfiguration double-buffer
flags are overridden; tracing, queues, decoding and postprocessing are inherited.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import json
import os

import models.tt_cnn.tt.builder as cnn_builder

import cv2
import torch
import ttnn

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

from models.demos.yolov4.common import get_model_result, image_to_tensor
from models.demos.yolov4.runner.performant_runner_infra import YOLOv4PerformanceRunnerInfra
from models.demos.yolov4.runner.pipeline_runner import YoloV4PipelineRunner
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config

try:
    from models.demos.utils.common_demo_utils import get_mesh_mappers
except Exception:
    get_mesh_mappers = None

import submission_api as api

RESOLUTION = (320, 320)
CONF_THRESH = 0.001        # Official floor for competitive mAP
NMS_THRESH = 0.4           # Standard IoU suppression threshold

# Hardware trace memory size allocated at device open
TRACE_REGION_SIZE = 6434816
CHUNK = 16
DECODERS = 4


DOUBLE_BUFFER_FLAGS = {
    "A": (False, False),
    "B": (True, False),
    "C": (False, True),
    "D": (True, True),
}


def _double_buffer_mode():
    value = os.environ.get("YOLO_DOUBLE_BUFFER", "original").strip()
    mode = "original" if value.lower() == "original" else value.upper()
    if mode not in {"original", *DOUBLE_BUFFER_FLAGS}:
        raise ValueError(
            f"Invalid YOLO_DOUBLE_BUFFER={value!r}; expected original, A, B, C or D"
        )
    return mode


@contextmanager
def _double_buffer_configuration(mode):
    """Patch only during model construction/compilation, restoring on any exit.

    Model setup must be serial: the builder module is shared within a process.
    The observed counts describe configured layers, not proof that device kernels
    use double buffering. The original configuration is also counted unchanged.
    """
    original = cnn_builder.to_conv2d_config
    stats = {"mode": mode, "configured_convs": 0, "effective_flag_counts": {}}

    def configured(configuration):
        if mode != "original":
            act, weights = DOUBLE_BUFFER_FLAGS[mode]
            configuration = replace(
                configuration,
                enable_act_double_buffer=act,
                enable_weights_double_buffer=weights,
            )
        config = original(configuration)
        observed = tuple(bool(getattr(config, name)) for name in (
            "enable_act_double_buffer", "enable_weights_double_buffer"
        ))
        if mode != "original" and observed != DOUBLE_BUFFER_FLAGS[mode]:
            raise RuntimeError(f"Double-buffer config was not applied: {observed}")
        stats["configured_convs"] += 1
        key = f"act={observed[0]},weights={observed[1]}"
        counts = stats["effective_flag_counts"]
        counts[key] = counts.get(key, 0) + 1
        return config

    cnn_builder.to_conv2d_config = configured
    try:
        yield stats
    finally:
        cnn_builder.to_conv2d_config = original


def _pure_torch_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
    """Pure-PyTorch NMS fallback if torchvision C++ extension is absent."""
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


class Submission(api.Submission):
    # Request hardware trace region and 2 command queues at device open
    DEVICE_OPTIONS = {
        "trace_region_size": TRACE_REGION_SIZE,
        "num_command_queues": 2,
    }

    def setup(self, device):
        """Build model, compile traced pipeline, and initialize thread pool."""
        if device is None:
            raise RuntimeError("This submission requires a Tenstorrent device.")
        self.device = device
        self.resolution = RESOLUTION

        if get_mesh_mappers is not None:
            self.in_mapper, _, self.out_composer = get_mesh_mappers(device)
        else:
            self.in_mapper = self.out_composer = None

        self.double_buffer_mode = _double_buffer_mode()
        with _double_buffer_configuration(self.double_buffer_mode) as stats:
            self.infra = YOLOv4PerformanceRunnerInfra(
                device,
                1,
                ttnn.bfloat8_b,
                ttnn.bfloat8_b,
                model_location_generator=None,
                resolution=self.resolution,
                mesh_mapper=self.in_mapper,
                mesh_composer=self.out_composer,
            )

            self.seed_host, dram_cfg, l1_cfg = self.infra.setup_dram_sharded_input(device)
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
        self.double_buffer_stats = stats
        if stats["configured_convs"] == 0:
            raise RuntimeError(
                "No Conv2dConfig creation was observed; verify the model builder path."
            )
        print("[double_buffer] " + json.dumps(stats, sort_keys=True), flush=True)
        self.pool = ThreadPoolExecutor(max_workers=DECODERS)

    def warmup(self, paths):
        """Warm up kernel caches outside the timed region."""
        self.run(paths)

    def run(self, paths):
        """TIMED. Overlapped chunk decoding + hardware trace + vectorized NMS."""
        paths = list(paths)
        out = {}
        chunks = [paths[i:i + CHUNK] for i in range(0, len(paths), CHUNK)]
        if not chunks:
            return out

        pending = self._submit(chunks[0])
        for i, chunk in enumerate(chunks):
            # Asynchronously decode next chunk on CPU threads while card works on this one
            nxt = self._submit(chunks[i + 1]) if i + 1 < len(chunks) else None
            frames = [f.result() for f in pending]
            pending = nxt

            # Convert to host bfloat16 tensors
            hosts = [
                ttnn.from_torch(
                    t,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=self.in_mapper,
                )
                for t, _, _ in frames
            ]

            # Replay trace on Wormhole accelerator
            results = self.pipeline.enqueue(hosts).pop_all()

            # Fast vectorized post-processing and bounding box conversion
            for path, (_, W, H), raw in zip(chunk, frames, results):
                out[path] = self._fast_detections(raw, W, H)

        return out

    def _submit(self, chunk):
        return [self.pool.submit(self._decode, p) for p in chunk]

    def _decode(self, path):
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            return torch.zeros(1, 3, self.resolution[1], self.resolution[0]), 1, 1
        H, W = frame.shape[:2]
        img = cv2.resize(frame, self.resolution)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return image_to_tensor(img), W, H

    def _fast_detections(self, raw, W: int, H: int):
        """Postprocessing inherited from my_submission.py, unchanged."""
        boxes, confs = get_model_result(
            raw, self.resolution, mesh_composer=self.out_composer
        )

        # boxes: [1, 6300, 1, 4] normalized to [0, 1]
        # confs: [1, 6300, 80] class scores in [0, 1]
        if not isinstance(boxes, torch.Tensor):
            boxes = torch.from_numpy(boxes)
        if not isinstance(confs, torch.Tensor):
            confs = torch.from_numpy(confs)

        boxes = boxes.float()
        confs = confs.float()

        if boxes.dim() == 4:
            boxes = boxes[:, :, 0]  # [1, 6300, 4]

        # 1. Vectorized single-label max across all 80 classes
        max_scores, class_ids = torch.max(confs[0], dim=-1)  # [6300]

        # 2. Filter candidates above CONF_THRESH
        mask = max_scores > CONF_THRESH
        if not mask.any():
            return []

        cand_boxes = boxes[0, mask]      # [M, 4]
        cand_scores = max_scores[mask]   # [M]
        cand_classes = class_ids[mask]   # [M]

        # 3. Compiled C++ batched NMS (multi-class simultaneous suppression)
        if HAS_TORCHVISION:
            keep = torchvision.ops.batched_nms(
                cand_boxes, cand_scores, cand_classes, NMS_THRESH
            )
        else:
            unique_classes = torch.unique(cand_classes)
            keep_list = []
            for cls in unique_classes:
                cls_idx = (cand_classes == cls).nonzero(as_tuple=True)[0]
                cls_keep = _pure_torch_nms(cand_boxes[cls_idx], cand_scores[cls_idx], NMS_THRESH)
                keep_list.append(cls_idx[cls_keep])
            keep = torch.cat(keep_list) if keep_list else torch.empty(0, dtype=torch.long)

        if keep.numel() == 0:
            return []

        final_boxes = cand_boxes[keep]
        final_scores = cand_scores[keep]
        final_classes = cand_classes[keep]

        # 4. Convert directly to competition output format
        dets = []
        for k in range(final_boxes.shape[0]):
            x1 = float(final_boxes[k, 0])
            y1 = float(final_boxes[k, 1])
            x2 = float(final_boxes[k, 2])
            y2 = float(final_boxes[k, 3])
            s = float(final_scores[k])
            score = s * s  # Exact parity with baseline det[4] * det[5]
            cls_id = int(final_classes[k])

            dets.append({
                "category_id": api.to_coco_id(cls_id),
                "bbox": api.xyxy_to_xywh(x1 * W, y1 * H, x2 * W, y2 * H),
                "score": score,
            })

        return dets

    def teardown(self):
        pool = getattr(self, "pool", None)
        if pool is not None:
            pool.shutdown(wait=True)
        pipeline = getattr(self, "pipeline", None)
        if pipeline is not None:
            try:
                pipeline.cleanup()
            except Exception:
                pass
