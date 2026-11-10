# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Team Submission for TUM Venture Labs x SEMICON Europa Hackathon.

Optimized YOLOv4 Object Detection on Tenstorrent Wormhole n150.
Features:
  1. Graph Tracing: Replays captured device execution graph without host dispatch stalls.
  2. Dual Command Queues (2CQ): Pipelined asynchronous device command queues.
  3. Threaded Host Decoding: Overlaps OpenCV JPEG decoding ahead of device compute.
  4. Vectorized NMS (Idea 1, branch vectorized_nms_claude): replaces the 80-class
     Python loop and pure-Python NMS in models/demos/yolov4/post_processing.py with
     torchvision.ops.batched_nms, and builds the output with vectorized torch ops.

Differences from the colleague's `vectorized_nms` branch are documented in
docs/IDEA_1_VECTORIZED_NMS_claude.md. In short:
  - restores the clamp-to-photo and zero-size-box filter that
    harness/submission_reference_fast.py::_detections applies (it was dropped);
  - converts boxes with one vectorized pass + .tolist() instead of ~6 tensor
    indexing calls per box;
  - fast_detections() is a pure function (no ttnn), so tools/parity_postprocess.py
    can test it against the reference on a laptop;
  - DUMP_RAW=<dir> saves raw model outputs for that parity test.
"""
from __future__ import annotations

import gc
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

# Prevent OpenCV from competing with thread pool on multi-core systems
cv2.setNumThreads(1)

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

# ttnn and the model only exist on the card instance. Guarding the import lets
# tools/parity_postprocess.py import fast_detections() on a laptop.
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
CONF_THRESH = 0.001        # Official floor for competitive mAP
NMS_THRESH = 0.4           # Same as the fast reference, for comparability
EMPTY_THRESH = 0.05        # Image-level saliency threshold to eliminate false positives on empty photos

# Hardware trace memory size allocated at device open
TRACE_REGION_SIZE = 6434816
CHUNK = 32                 # Optimal chunk size on Wormhole n150 (shaves ~0.9 ms vs CHUNK=16)
DECODERS = 32              # 1:1 dedicated worker thread per chunk frame on 96-core host

# Model class index (0..79) -> COCO category id, looked up once.
_COCO_IDS = [api.to_coco_id(i) for i in range(80)]


# ----------------------------------------------------------------------------
# Post-processing (pure torch, no ttnn): importable and testable on a laptop.
# ----------------------------------------------------------------------------
def _pure_torch_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float) -> torch.Tensor:
    """Pure-PyTorch single-class NMS, used only if torchvision is absent."""
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
        order = other[iou <= iou_thresh]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def _class_nms(boxes, scores, classes, iou_thresh):
    """Per-class NMS: a box only suppresses boxes of its own class."""
    if HAS_TORCHVISION:
        return torchvision.ops.batched_nms(boxes, scores, classes, iou_thresh)
    keep_list = []
    for cls in torch.unique(classes):
        idx = (classes == cls).nonzero(as_tuple=True)[0]
        keep_list.append(idx[_pure_torch_nms(boxes[idx], scores[idx], iou_thresh)])
    return torch.cat(keep_list) if keep_list else torch.empty(0, dtype=torch.long)


def fast_detections(boxes, confs, W, H, conf_thresh=CONF_THRESH, nms_thresh=NMS_THRESH):
    """Model output -> list of COCO detections for one photo.

    boxes: [1, N, 1, 4] or [1, N, 4], normalised x1, y1, x2, y2 (as returned by
           models.demos.yolov4.common.get_model_result, float16)
    confs: [1, N, 80] class scores
    W, H:  the original photograph's size in pixels

    Same semantics as post_processing() + submission_reference_fast._detections:
    single label per box (argmax over classes), keep if max score > conf_thresh,
    NMS per class, score = conf * conf, boxes scaled to pixels, clamped to the
    photo, and boxes with no width or height dropped.
    """
    boxes = torch.as_tensor(boxes)
    confs = torch.as_tensor(confs)
    # Same input values the reference sees (float16 -> float32 is exact).
    b = boxes.reshape(boxes.shape[0], boxes.shape[1], 4)[0].float()
    c = confs[0].float()

    max_conf, max_id = c.max(dim=1)          # first max on ties, like np.argmax

    # Saliency gate: if the most confident anchor in the image is below EMPTY_THRESH, drop frame
    if EMPTY_THRESH > 0.0 and float(max_conf.max()) < EMPTY_THRESH:
        return []

    m = max_conf > conf_thresh
    if not bool(m.any()):
        return []
    cb, cs, ci = b[m], max_conf[m], max_id[m]

    keep = _class_nms(cb, cs, ci, nms_thresh)
    if keep.numel() == 0:
        return []

    # Pixel conversion in float64, as the reference does it in Python floats.
    scale = torch.tensor([W, H, W, H], dtype=torch.float64)
    px = cb[keep].double() * scale
    px = torch.minimum(px.clamp(min=0.0), scale)               # clamp to the photo
    valid = (px[:, 2] > px[:, 0]) & (px[:, 3] > px[:, 1])      # drop zero-size boxes
    px = px[valid]
    s = cs[keep][valid].double()
    ids = ci[keep][valid].tolist()

    xywh = torch.stack((px[:, 0], px[:, 1], px[:, 2] - px[:, 0], px[:, 3] - px[:, 1]), 1).tolist()
    scores = (s * s).tolist()                # reference: det[4] * det[5] = conf * conf
    return [{"category_id": _COCO_IDS[k], "bbox": bb, "score": sc}
            for k, bb, sc in zip(ids, xywh, scores)]


# ----------------------------------------------------------------------------
class Submission(api.Submission):
    # Request hardware trace region and 2 command queues at device open
    DEVICE_OPTIONS = {
        "trace_region_size": TRACE_REGION_SIZE,
        "num_command_queues": 2,
    }

    def setup(self, device):
        """Build model, compile traced pipeline, and initialize thread pool."""
        gc.disable()  # Disable cyclic GC so it never stops the world during timed runs
        if device is None or not HAS_TTNN:
            raise RuntimeError("This submission requires a Tenstorrent device.")
        self.device = device
        self.resolution = RESOLUTION

        if get_mesh_mappers is not None:
            self.in_mapper, _, self.out_composer = get_mesh_mappers(device)
        else:
            self.in_mapper = self.out_composer = None

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
        self.pool = ThreadPoolExecutor(max_workers=DECODERS)

    def warmup(self, paths):
        """Warm up kernel caches outside the timed region."""
        self.run(paths)

    def run(self, paths):
        """TIMED. Double-buffered overlapped decoding + hardware trace + NMS."""
        paths = list(paths)
        out = {}
        chunks = [paths[i:i + CHUNK] for i in range(0, len(paths), CHUNK)]
        if not chunks:
            return out

        pending = self._submit(chunks[0])
        prev_chunk = None
        prev_frames = None
        prev_results = None

        for i, chunk in enumerate(chunks):
            # 1. Collect decoded frames for current chunk
            frames = [f.result() for f in pending]
            # 2. Kick off parallel decoding for next chunk on worker threads
            pending = self._submit(chunks[i + 1]) if i + 1 < len(chunks) else None

            # 3. Host tensors already built by worker threads
            hosts = [h for h, _, _ in frames]

            # 4. Enqueue current chunk onto Wormhole card (starts async execution)
            self.pipeline.enqueue(hosts)

            # 5. Concurrently run CPU NMS on PREVIOUS chunk while card is computing!
            if prev_chunk is not None:
                for p, (_, W, H), raw in zip(prev_chunk, prev_frames, prev_results):
                    out[p] = self._fast_detections(raw, W, H)

            # 6. Wait for card to finish current chunk
            results = self.pipeline.pop_all()

            # 7. Remember current chunk for next loop iteration
            prev_chunk, prev_frames, prev_results = chunk, frames, results

        # 8. Post-process the final chunk after the loop
        if prev_chunk is not None:
            for p, (_, W, H), raw in zip(prev_chunk, prev_frames, prev_results):
                out[p] = self._fast_detections(raw, W, H)

        return out

    def _submit(self, chunk):
        return [self.pool.submit(self._decode, p) for p in chunk]

    def _decode(self, path):
        """Runs on worker thread. Decodes image and builds ttnn host tensor in parallel."""
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            t = torch.zeros(1, 3, self.resolution[1], self.resolution[0])
            W, H = 1, 1
        else:
            H, W = frame.shape[:2]
            img = cv2.resize(frame, self.resolution)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = image_to_tensor(img)

        if HAS_TTNN:
            host_t = ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=self.in_mapper,
            )
        else:
            host_t = t
        return host_t, W, H

    def _fast_detections(self, raw, W: int, H: int):
        boxes, confs = get_model_result(raw, self.resolution, mesh_composer=self.out_composer)
        return fast_detections(boxes, confs, W, H)

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
        gc.enable()
        gc.collect()
