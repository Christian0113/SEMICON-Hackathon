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
    try:
        import models.demos.yolov4.common as _common_mod
        _common_mod.YOLOV4_L1_SMALL_SIZE = 16384
    except Exception:
        pass
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


# ----------------------------------------------------------------------------
# WP-3: Host-side float32 box and confidence decode
# ----------------------------------------------------------------------------
_ANCHORS = (
    torch.tensor([[1.5, 2.0], [2.375, 4.5], [5.0, 3.5]]),            # 40x40, stride 8
    torch.tensor([[2.25, 4.6875], [4.75, 3.4375], [4.5, 9.125]]),    # 20x20, stride 16
    torch.tensor([[4.4375, 3.4375], [6.0, 7.59375], [14.34375, 12.53125]]), # 10x10, stride 32
)
_SCALES = ((1600, 40), (400, 20), (100, 10))


def decode_raw(raw):
    """Host-side float32 box and confidence decode (WP-3).
    Replaces ~230 device-side ops in TtGenBoxes with pure vector PyTorch math.
    Coordinates reach true float32 precision, eliminating bf16 grid quantization.
    """
    t_all = raw.reshape(-1, 256).float()
    off = 0
    boxes = []
    scores = []
    for g, (hw, S) in enumerate(_SCALES):
        t = t_all[off:off + hw, :255].view(hw, 3, 85)
        off += hw
        gy, gx = torch.meshgrid(
            torch.arange(S, device=t.device),
            torch.arange(S, device=t.device),
            indexing="ij",
        )
        gx = gx.reshape(-1, 1).float()
        gy = gy.reshape(-1, 1).float()
        anchors = _ANCHORS[g].to(device=t.device, dtype=torch.float32)

        bx = (torch.sigmoid(t[..., 0]) + gx) / S
        by = (torch.sigmoid(t[..., 1]) + gy) / S
        bw = torch.exp(t[..., 2]) * anchors[:, 0] / S
        bh = torch.exp(t[..., 3]) * anchors[:, 1] / S

        b = torch.stack((bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2), -1)
        s = torch.sigmoid(t[..., 4:5]) * torch.sigmoid(t[..., 5:85])

        boxes.append(b.transpose(0, 1).reshape(-1, 4))
        scores.append(s.transpose(0, 1).reshape(-1, 80))

    return torch.cat(boxes, dim=0), torch.cat(scores, dim=0)


def _decode_and_detect(raw_torch, W, H):
    boxes, confs = decode_raw(raw_torch)
    return fast_detections(boxes, confs, W, H)


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
        "l1_small_size": 16384,
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

        # WP-8a: Upgrade Scale 1 detection head (c2: 40x40 stride-8, 4800 anchors / 76.2% of all boxes)
        # from bfloat8_b to bfloat16 to match head.c10 and head.c18.
        try:
            import models.demos.yolov4.tt.model_preprocessing as _mp

            _orig_head_params = _mp._create_head_model_parameters

            def _patched_head_params(conv_args, resolution):
                _orig_head_params(conv_args, resolution)
                if resolution == (320, 320):
                    conv_args.c2["dtype"] = ttnn.bfloat16

            _mp._create_head_model_parameters = _patched_head_params
        except Exception as e:
            print(f"  [submission] Head precision patch skipped ({type(e).__name__}: {e})")

        # WP-3: Device-side monkey-patch for TtYOLOv4.__call__
        # Ends device execution at head convolutions and returns [1, 1, 2100, 256] raw logits.
        # Strips ~230 tiny genboxes/epilogue ops from the trace, saving ~1.0-1.5 ms.
        from models.demos.yolov4.tt import yolov4 as _y4

        def _heads_only(model_self, x):
            N, C, H, W = x.shape
            min_channels = 16
            if C < min_channels:
                channel_padding_needed = min_channels - C
                nchw = ttnn.pad(x, ((0, 0), (0, channel_padding_needed), (0, 0), (0, 0)), value=0.0)
            else:
                nchw = x
            nhwc = ttnn.permute(nchw, (0, 2, 3, 1))
            ttnn.deallocate(nchw)
            ttnn.deallocate(x)
            nhwc = ttnn.reallocate(nhwc)
            x = ttnn.reshape(nhwc, [1, 1, nhwc.shape[0] * nhwc.shape[1] * nhwc.shape[2], nhwc.shape[-1]])
            d1 = model_self.down1(x)
            d2 = model_self.down2(d1)
            ttnn.deallocate(d1)
            d3 = model_self.down3(d2)
            ttnn.deallocate(d2)
            d4 = model_self.down4(d3)
            d5 = model_self.down5(d4)
            x20, x13, x6 = model_self.neck([d5, d4, d3])
            x4, x5, x6 = model_self.head([x20, x13, x6])

            outs = []
            for t in (x4, x5, x6):
                t = ttnn.to_memory_config(t, ttnn.L1_MEMORY_CONFIG)
                t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
                outs.append(t)
            return ttnn.concat(outs, dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)

        _y4.TtYOLOv4.__call__ = _heads_only

        self.infra = YOLOv4PerformanceRunnerInfra(
            device, 1, ttnn.bfloat8_b, ttnn.bfloat8_b,
            model_location_generator=None,
            resolution=self.resolution,
            mesh_mapper=self.in_mapper,
            mesh_composer=self.out_composer,
        )
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
        # Convert host slot to torch tensor; clone so the ring slot is free immediately
        raw_torch = ttnn.to_torch(host_slot, mesh_composer=self.out_composer).clone()
        post.append((p, self.post_pool.submit(_decode_and_detect, raw_torch, W, H)))

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
        raw_torch = ttnn.to_torch(raw, mesh_composer=self.out_composer)
        return _decode_and_detect(raw_torch, W, H)

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
