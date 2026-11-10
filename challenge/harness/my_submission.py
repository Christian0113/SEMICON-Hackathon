# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Team submission: YOLOv4 on a Wormhole n150, built up one lever at a time.

L00 (this file's starting point) is harness/submission_reference_fast.py
unchanged in behaviour (trace + 2 command queues + threaded decode), plus:

  * LEVERS: one on/off switch per optimisation, overridable without editing
    code, e.g.  LEVERS="L01=0,L02=1" python evaluate.py ...
    Each lever branch (L01-..., L02-...) adds its code behind its own switch,
    so after merging the team can still measure each lever on and off.
  * STAGE_TIMES=1 prints where the time went inside run() (host decode wait,
    host->card conversion, card + readback, box step). Off by default: never
    set it for a scored run.

The jury runs a single file, so all levers live in this file. Documentation of
each change: docs/changes/<branch>.md.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import torch

try:
    import ttnn
    from models.demos.yolov4.common import get_model_result, image_to_tensor
    from models.demos.yolov4.post_processing import post_processing
    from models.demos.yolov4.runner.performant_runner_infra import YOLOv4PerformanceRunnerInfra
    from models.demos.yolov4.runner.pipeline_runner import YoloV4PipelineRunner
    from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config
    HAS_TTNN = True
except ImportError:          # laptop: lets tools import this file without a card
    HAS_TTNN = False

try:
    from models.demos.utils.common_demo_utils import get_mesh_mappers
except Exception:                                     # older trees
    get_mesh_mappers = None

import submission_api as api

RESOLUTION = (320, 320)
CONF_THRESH = 0.001        # the same floor as the plain reference, for comparability
NMS_THRESH = 0.4

# Enough trace room for this model at 320x320; the value bench_levers.py uses.
TRACE_REGION_SIZE = 6434816

# Frames handed to the card at a time, and threads decoding ahead of it.
# DECODERS: an env value always wins; otherwise 4 (the reference), or, with
# L01 on, a count derived from the cores this process may use (see setup()).
CHUNK = int(os.environ.get("CHUNK", "16"))
DECODERS_ENV = os.environ.get("DECODERS")
DECODERS = int(DECODERS_ENV or "4")


# ----------------------------------------------------------------------------
# Lever switches. Defaults are what this branch claims; LEVERS env overrides.
# ----------------------------------------------------------------------------
LEVERS = {
    "L01": True,     # L01-thread-limits: one OpenCV/torch thread per worker, decoders from core count
    # "L02": True,   # added by branch L02-overlap-postprocess
}


def _apply_lever_env(levers, spec):
    for item in filter(None, (s.strip() for s in spec.split(","))):
        name, _, val = item.partition("=")
        levers[name.strip().upper()] = val.strip().lower() not in ("0", "false", "off", "no")
    return levers


_apply_lever_env(LEVERS, os.environ.get("LEVERS", ""))


def lever(name):
    return LEVERS.get(name, False)


STAGE_TIMES = os.environ.get("STAGE_TIMES", "") not in ("", "0")


def _usable_cores():
    """Cores this process may run on (affinity/cpuset aware), not the host's total."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:                       # not Linux
        return os.cpu_count() or 4


def _apply_thread_limits():
    """L01. OpenCV and torch each start a thread pool per call by default. With
    DECODERS worker threads already decoding in parallel, those pools put many
    more runnable threads than cores on the machine, and the extra switching
    slows every stage. One thread each inside our own workers avoids that.
    Returns the decoder count to use."""
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    cores = _usable_cores()
    # Leave cores for the main thread (card feeding, boxes) and the runtime's
    # own threads; never fewer than 2 decoders, never more than 8.
    decoders = int(DECODERS_ENV) if DECODERS_ENV else max(2, min(8, cores - 2))
    return decoders, cores


class _Stages:
    """Accumulates wall time per stage of run(). No-op unless STAGE_TIMES=1."""

    def __init__(self):
        self.t = {}

    def add(self, name, t0):
        if STAGE_TIMES:
            self.t[name] = self.t.get(name, 0.0) + (time.perf_counter() - t0)

    def report(self, n_frames, total):
        if not STAGE_TIMES or not n_frames:
            return
        parts = "  ".join(f"{k} {1e3 * v / n_frames:.3f}" for k, v in self.t.items())
        print(f"    [STAGE_TIMES ms/frame] total {1e3 * total / n_frames:.3f}  {parts}  "
              f"levers {sorted(k for k, v in LEVERS.items() if v) or 'none'}")


# ----------------------------------------------------------------------------
class Submission(api.Submission):
    # Without this the harness opens the card with no trace region and one
    # command queue, and setup() fails. Both are fixed when the card is opened.
    DEVICE_OPTIONS = {"trace_region_size": TRACE_REGION_SIZE,
                      "num_command_queues": 2}

    def setup(self, device):
        """Build the model and capture the graph. Not timed."""
        if device is None or not HAS_TTNN:
            raise RuntimeError("this needs a card; the harness could not open one")
        self.device = device
        self.resolution = RESOLUTION
        if get_mesh_mappers is not None:
            self.in_mapper, _, self.out_composer = get_mesh_mappers(device)
        else:
            self.in_mapper = self.out_composer = None

        self.infra = YOLOv4PerformanceRunnerInfra(
            device, 1, ttnn.bfloat8_b, ttnn.bfloat8_b,
            model_location_generator=None, resolution=self.resolution,
            mesh_mapper=self.in_mapper, mesh_composer=self.out_composer)

        self.seed_host, dram_cfg, l1_cfg = self.infra.setup_dram_sharded_input(device)
        self.pipeline = create_pipeline_from_config(
            config=PipelineConfig(use_trace=True, num_command_queues=2,
                                  all_transfers_on_separate_command_queue=False),
            model=YoloV4PipelineRunner(self.infra), device=device,
            dram_input_memory_config=dram_cfg, l1_input_memory_config=l1_cfg)
        self.pipeline.compile(self.seed_host)

        decoders, cores = DECODERS, _usable_cores()
        if lever("L01"):
            decoders, cores = _apply_thread_limits()
        self.pool = ThreadPoolExecutor(max_workers=decoders)
        print(f"  my_submission: CHUNK={CHUNK} DECODERS={decoders} usable cores={cores} "
              f"cv2 threads={cv2.getNumThreads()} torch threads={torch.get_num_threads()} "
              f"levers on: {sorted(k for k, v in LEVERS.items() if v) or 'none'}")

    def warmup(self, paths):
        """Settle the card and let the kernel cache fill. Not timed."""
        self.run(paths)

    def run(self, paths):
        """TIMED. Decode ahead on the host, replay the graph on the card."""
        t_start = time.perf_counter()
        st = _Stages()
        paths = list(paths)
        out = {}
        chunks = [paths[i:i + CHUNK] for i in range(0, len(paths), CHUNK)]
        if not chunks:
            return out

        pending = self._submit(chunks[0])
        for i, chunk in enumerate(chunks):
            # Start decoding the next chunk before asking the card for this one.
            nxt = self._submit(chunks[i + 1]) if i + 1 < len(chunks) else None
            t0 = time.perf_counter()
            frames = [f.result() for f in pending]
            st.add("decode_wait", t0)
            pending = nxt

            t0 = time.perf_counter()
            hosts = self._to_host(frames)
            st.add("from_torch", t0)

            t0 = time.perf_counter()
            results = self.pipeline.enqueue(hosts).pop_all()
            st.add("card+readback", t0)

            t0 = time.perf_counter()
            for path, (_, W, H), raw in zip(chunk, frames, results):
                out[path] = self._detections(raw, W, H)
            st.add("boxes", t0)

        st.report(len(paths), time.perf_counter() - t_start)
        return out

    # ------------------------------------------------------------------ host
    def _submit(self, chunk):
        return [self.pool.submit(self._decode, p) for p in chunk]

    def _to_host(self, frames):
        return [ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                mesh_mapper=self.in_mapper)
                for t, _, _ in frames]

    def _decode(self, path):
        """Runs on a worker thread."""
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            return torch.zeros(1, 3, self.resolution[1], self.resolution[0]), 1, 1
        H, W = frame.shape[:2]
        img = cv2.resize(frame, self.resolution)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return image_to_tensor(img), W, H

    def _detections(self, raw, W, H):
        """Identical conversions to submission_reference_fast.py."""
        boxes, confs = get_model_result(raw, self.resolution,
                                        mesh_composer=self.out_composer)
        result = post_processing(None, CONF_THRESH, NMS_THRESH, [boxes, confs])
        dets = []
        for det in (result[0] if result else []):
            x1, y1, x2, y2 = (float(det[0]), float(det[1]), float(det[2]), float(det[3]))
            score = float(det[4]) * float(det[5])
            cls_id = int(det[6])
            px1 = min(max(x1 * W, 0.0), W)
            py1 = min(max(y1 * H, 0.0), H)
            px2 = min(max(x2 * W, 0.0), W)
            py2 = min(max(y2 * H, 0.0), H)
            if px2 <= px1 or py2 <= py1:
                continue
            dets.append({"category_id": api.to_coco_id(cls_id),
                         "bbox": api.xyxy_to_xywh(px1, py1, px2, py2),
                         "score": score})
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
