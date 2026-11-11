# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Team Submission, L03: streaming pipeline + on-device confs truncation.

Builds on milestone 8 (per-frame streaming pipeline, 5.301 ms/frame, mAP
0.4640, 25,119 detections) and merges in the trunc_confs lever (on-device
max/argmax over the 80 classes). Lever F1 of fable_suggestion.md.

Why: milestone 8 sits at the measured card floor (~5.45 ms). That floor still
contains the D2H read of confs [1, 6300, 80] bf16 = 1,008 KB per frame, issued
on CQ0 — the compute queue — so the Tensix cores idle during the read. Reducing
confs on the card to max [1, 6300, 1] + argmax [1, 6300, 1] shrinks the read
~12x (1,059 KB -> ~89 KB including boxes) and also removes the 80-class max
from the host NMS workers.

What changes vs milestone 8 (scheduling is untouched):

  1. TruncatedYoloV4Runner wraps the infra: after the model runs (inside the
     trace), ttnn.max + ttnn.argmax reduce confs over dim=-1 and only the
     reduced tensors are read back. From the trunc_confs branch.
  2. fast_detections accepts precomputed max_ids; thresholds, gates, NMS,
     scaling and scoring are unchanged.
  3. If the truncated trace fails to compile (argmax may not be traceable on
     this tree — never verified on the card), setup recovers the device and
     falls back to the exact milestone-8 pipeline. TRUNC_CONFS = False forces
     that path for A/B.

Parity watch-item: torch's c.max(dim=1) returns the FIRST index on ties;
device argmax may tie-break differently, and bf16 scores do collide. The
acceptance gate decides: mAP 0.4640 AND detections_returned == 25,119 on the
practice set. If detections drift by a handful, diff per-image against
milestone 8 before accepting or rejecting.

Set TRUNC_CONFS = False for milestone-8 behaviour; STREAMING = False for the
milestone-7 chunked path (both A/B switches work in any combination).
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

import submission_api as api

RESOLUTION = (320, 320)
CONF_THRESH = 0.001
NMS_THRESH = 0.4
EMPTY_THRESH = 0.05
HOST_SALIENCY_THRESH = 3.0

# +1 MiB headroom over the perf test's 6434816: the truncated trace holds two
# extra ops (max, argmax) and a compile failure here would silently cost the
# whole lever via the fallback.
TRACE_REGION_SIZE = 6434816 + (1 << 20)

# ---- scheduling knobs (tune on the card; see docs/IDEA_8_STREAMING_PIPELINE.md)
STREAMING = True     # False -> milestone-7 chunked path
TRUNC_CONFS = True   # False -> milestone-8 full-confs readback, for A/B
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
# Post-processing: identical semantics to milestones 7/8; max_ids added for
# the truncated output path (from the trunc_confs branch).
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
    """Model output -> list of COCO detections for one photo.

    boxes:   [1, N, 1, 4] or [1, N, 4], normalised x1, y1, x2, y2
    confs:   [1, N, 80] class scores, or [1, N, 1] max scores when max_ids given
    max_ids: optional [N] class ids from on-device argmax truncation
    """
    boxes = torch.as_tensor(boxes)
    b = boxes.reshape(boxes.shape[0], boxes.shape[1], 4)[0].float()

    if max_ids is not None:
        max_conf = torch.as_tensor(confs).reshape(-1).float()
        max_id = torch.as_tensor(max_ids).reshape(-1).long()
    else:
        c = torch.as_tensor(confs)[0].float()
        max_conf, max_id = c.max(dim=1)   # first max on ties, like np.argmax

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


class TruncatedYoloV4Runner:
    """YOLOv4 + on-device max/argmax over the class dim, inside the trace.

    Shrinks the confs readback from [1, 6300, 80] bf16 (1,008 KB) to
    [1, 6300, 1] max + [1, 6300, 1] argmax (~50 KB). From the trunc_confs
    branch; the confs deallocate keeps L1 addresses stable for trace capture.
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

        self.infra = YOLOv4PerformanceRunnerInfra(
            device, 1, ttnn.bfloat8_b, ttnn.bfloat8_b,
            model_location_generator=None,
            resolution=self.resolution,
            mesh_mapper=self.in_mapper,
            mesh_composer=self.out_composer,
        )
        self.seed_host, dram_cfg, l1_cfg = self.infra.setup_dram_sharded_input(device)
        self.pipeline, self.is_truncated = self._build_pipeline(device, dram_cfg, l1_cfg)
        print(f"  [submission] confs truncation {'ON' if self.is_truncated else 'OFF'}")

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

    # ---- pipeline construction ----------------------------------------------
    def _make_pipeline(self, model, device, dram_cfg, l1_cfg):
        p = create_pipeline_from_config(
            config=PipelineConfig(use_trace=True, num_command_queues=2,
                                  all_transfers_on_separate_command_queue=False),
            model=model,
            device=device,
            dram_input_memory_config=dram_cfg,
            l1_input_memory_config=l1_cfg,
        )
        p.compile(self.seed_host)
        return p

    def _build_pipeline(self, device, dram_cfg, l1_cfg):
        """Truncated pipeline if possible, else the exact milestone-8 one."""
        if TRUNC_CONFS:
            try:
                return self._make_pipeline(TruncatedYoloV4Runner(self.infra), device, dram_cfg, l1_cfg), True
            except Exception as e:
                print(f"  [submission] truncated trace failed, falling back: {type(e).__name__}: {e}")
                self._recover_device(device)
        return self._make_pipeline(YoloV4PipelineRunner(self.infra), device, dram_cfg, l1_cfg), False

    @staticmethod
    def _recover_device(device):
        """If compile died mid-trace-capture the device is left in capture mode;
        close any open capture and release traces so the fallback can compile."""
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

    # ---- output conversion --------------------------------------------------
    def _convert(self, raw):
        """Host-tensor output struct -> torch tensors, as (boxes, confs, max_ids).

        Runs on the retiring thread: every returned tensor is a fresh copy, so
        the host slot can be reused the moment this returns.
        """
        if self.is_truncated:
            raw_boxes, raw_max_confs, raw_max_ids = raw
            boxes, max_confs = get_model_result([raw_boxes, raw_max_confs], self.resolution,
                                                mesh_composer=self.out_composer)
            # copy=True: to_torch may alias the slot buffer; the NMS worker
            # reads this after the slot is reused.
            max_ids = ttnn.to_torch(raw_max_ids, mesh_composer=self.out_composer).to(torch.int64, copy=True)
            return boxes, max_confs, max_ids
        boxes, confs = get_model_result(raw, self.resolution, mesh_composer=self.out_composer)
        return boxes, confs, None

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
        boxes, confs, max_ids = self._convert(host_slot)
        post.append((p, self.post_pool.submit(fast_detections, boxes, confs, W, H, max_ids=max_ids)))

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
        boxes, confs, max_ids = self._convert(raw)
        return fast_detections(boxes, confs, W, H, max_ids=max_ids)

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
