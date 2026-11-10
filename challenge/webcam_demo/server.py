#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Webcam object detection served from a Tenstorrent Console instance.

    Date      2026-09-23
    Version   v0.1
    Measured  2026-09-24, output in ../data/roundtrip_20260924_094258.jsonl

WHAT IT IS
    A small HTTP server that keeps YOLOv4 resident on the card and answers
    /detect with bounding boxes as JSON. The browser on your laptop captures the
    webcam, posts frames here, and draws the boxes it gets back. In the spirit of
    the TT-Studio object detection front end, but with no dependencies
    beyond what the Console image already has.

WHY IT IS BUILT THIS WAY
    The camera is on your laptop and the card is in the instance, so frames
    have to travel. Measured on an n150 at 320x320, the round trip is roughly
    five times the whole of the work done on the machine, so the network is
    the bottleneck here in a way it would not be in a product. The panel
    keeps the two apart on purpose.

    Standard library only: http.server, not FastAPI or Flask, so there is nothing
    extra to install. ThreadingHTTPServer, because HTTP/1.1 keep-alive on a
    single-threaded server deadlocks the moment the browser opens a second
    connection: the idle first connection holds the only thread and the proxy
    reports "socket hang up". Threads mean the device needs DEVICE_LOCK.

    The model is built once at startup. Rebuilding per request would add seconds
    a frame.

REQUIRES
    The setup in ../setup/setup_yolov4_on_ttmetal.sh, i.e. the TTNN yolov4 in
    ~/tt-metal with both patches applied. Run from the tt-metal root:

        cd ~/tt-metal
        python /path/to/webcam_demo/server.py

    Then accept the port-forwarding prompt in VS Code, or open the PORTS tab and
    click the forwarded address. Chrome will only grant camera access over
    https or on localhost, and the VS Code forward is https, so this works.

STATUS
    The model calls below are taken from code paths that ran successfully on
    a Console instance on 2026-09-23: the input preparation from
    tests/pcc/test_ttnn_yolov4.py, the runner from runner/runner.py, and
    post_processing from post_processing.py, whose `img` argument is unused.
    This has been run on a Console instance with a Wormhole n150 and produced
    the round-trip records in ../data/. Timings it shows on screen are
    dominated by the network, so use bench/ for anything you quote.
"""

import contextlib
import io
import json
import os
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import torch

import ttnn
from models.demos.utils.common_demo_utils import load_coco_class_names
from models.demos.yolov4.common import YOLOV4_L1_SMALL_SIZE, image_to_tensor, load_torch_model
from models.demos.yolov4.post_processing import post_processing
from models.demos.yolov4.runner.runner import YOLOv4Runner
from models.demos.yolov4.tt.model_preprocessing import create_yolov4_model_parameters

# 320x320 rather than 640x640 deliberately. Measured on this card, the
# confidences agree with the PyTorch reference at 0.997 at 320 against 0.768 at
# 640, where the branch's tiling overrides are rejected much more heavily. So
# 320 is both quicker and closer to the reference. Change it and measure.
RESOLUTION = (320, 320)
CONF_THRESH = 0.3
NMS_THRESH = 0.4
PORT = int(os.environ.get("PORT", "8000"))

HERE = os.path.dirname(os.path.abspath(__file__))

# Where per-frame round-trip records land. One JSON object per line, appended
# as the browser reports them, so a run that is interrupted still leaves usable
# data behind. Read it with summarise_roundtrip.py.
TELEMETRY_PATH = os.environ.get(
    "TELEMETRY",
    os.path.join(HERE, "roundtrip_%s.jsonl" % time.strftime("%Y%m%d_%H%M%S")),
)
TELEMETRY_LOCK = threading.Lock()

class Detector:
    """Holds the device and the compiled model for the process lifetime."""

    def __init__(self, resolution):
        self.resolution = resolution
        print("opening device ...", flush=True)
        self.device = ttnn.open_device(device_id=0, l1_small_size=YOLOV4_L1_SMALL_SIZE)

        print("loading torch weights ...", flush=True)
        torch_model = load_torch_model(None)

        # The tensor is used for shape inference when extracting conv arguments,
        # so zeros of the right shape are enough.
        dummy = torch.zeros(1, 3, resolution[1], resolution[0])
        print("building ttnn parameters ...", flush=True)
        parameters = create_yolov4_model_parameters(torch_model, dummy, resolution, self.device)
        self.runner = YOLOv4Runner(self.device, parameters, resolution)
        self.class_names = load_coco_class_names()

        print("warming up (first run compiles kernels, this is the slow one) ...", flush=True)
        t0 = time.time()
        self.detect(np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8))
        print(f"warm-up took {time.time() - t0:.1f} s", flush=True)

    def _to_device(self, tensor):
        # Input preparation copied from tests/pcc/test_ttnn_yolov4.py. The channel
        # count is padded from 3 to 16 for the sharded memory config.
        n, c, h, w = tensor.shape
        if c == 3:
            c = 16
        mem = ttnn.create_sharded_memory_config(
            [n, c, h, w],
            ttnn.CoreGrid(x=8, y=8),
            ttnn.ShardStrategy.HEIGHT,
        )
        tt_in = ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        return tt_in.to(self.device, mem)

    def detect(self, bgr_frame):
        """bgr_frame: HWC uint8 as OpenCV gives it. Returns a list of dicts."""
        # Matches common.load_image: resize to the model's square input, then
        # BGR to RGB. The aspect ratio is deliberately not preserved, which is
        # why normalised box coordinates map straight back onto the full frame.
        img = cv2.resize(bgr_frame, self.resolution)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = image_to_tensor(img)

        tt_in = self._to_device(tensor)
        boxes, confs = self.runner.run(tt_in)

        # post_processing prints timing on every call, which floods the terminal
        # at webcam rates. Its first argument is unused.
        with contextlib.redirect_stdout(io.StringIO()):
            batch = post_processing(None, CONF_THRESH, NMS_THRESH, [boxes, confs])

        out = []
        for b in batch[0]:
            cls_id = int(b[6])
            name = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
            out.append(
                {
                    "x1": float(b[0]), "y1": float(b[1]),
                    "x2": float(b[2]), "y2": float(b[3]),
                    "conf": float(b[5]),
                    "cls": name,
                }
            )
        return out

DETECTOR = None
# The device is not safe for concurrent access. The server is threaded so that
# keep-alive connections cannot block each other, so inference needs a lock.
DEVICE_LOCK = threading.Lock()

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # the per-frame access log is noise at webcam rates

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            page = os.path.join(HERE, "index.html")
            if not os.path.isfile(page):
                # index.html must sit beside server.py. Say so plainly rather
                # than raising, because this happens on every fresh instance.
                msg = (
                    "<h1>index.html is missing</h1>"
                    f"<p>Expected it next to server.py, at:<br><code>{page}</code></p>"
                    "<p>Copy index.html into that directory and reload. "
                    "The model is loaded and the card is ready.</p>"
                ).encode()
                self._send(500, msg, "text/html; charset=utf-8")
                return
            with open(page, "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif self.path == "/health":
            body = json.dumps({"ok": True, "resolution": list(RESOLUTION)}).encode()
            self._send(200, body, "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def _telemetry(self):
        """Append the browser's per-frame records to a JSONL file.

        The browser measures its own round trip and sends a batch every so
        often rather than one request per frame, because a telemetry POST
        interleaved with the detection POSTs would show up in the very
        measurement it is reporting.
        """
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        try:
            records = json.loads(raw or b"[]")
            if isinstance(records, dict):
                records = [records]
        except Exception as exc:  # noqa: BLE001
            self._send(400, json.dumps({"error": repr(exc)}).encode(),
                       "application/json")
            return
        with TELEMETRY_LOCK:
            with open(TELEMETRY_PATH, "a") as fh:
                for rec in records:
                    fh.write(json.dumps(rec) + "\n")
        self._send(200, json.dumps({"ok": True, "written": len(records),
                                    "path": TELEMETRY_PATH}).encode(),
                   "application/json")

    def do_POST(self):
        if self.path == "/telemetry":
            self._telemetry()
            return
        if self.path != "/detect":
            self._send(404, b"not found", "text/plain")
            return

        t_recv = time.perf_counter()
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        t_body = time.perf_counter()
        if not raw:
            self._send(400, b'{"error":"empty body"}', "application/json")
            return

        # Timing note. An earlier version started the clock here, after the
        # decode and before the lock, and reported the result as the frame's
        # cost on the machine. It was neither: it left out the JPEG decode and
        # it included waiting for up to MAX_IN_FLIGHT-1 other frames to finish
        # on the card. The three parts are now reported separately, because
        # only `work_ms` is a property of the hardware, and a queue wait is a
        # property of how the browser is driving it.
        t_start = time.perf_counter()
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self._send(400, b'{"error":"could not decode image"}', "application/json")
            return
        t_decoded = time.perf_counter()

        try:
            with DEVICE_LOCK:
                t_got_lock = time.perf_counter()
                boxes = DETECTOR.detect(frame)
                t_done = time.perf_counter()
        except Exception as exc:  # a bad frame must not kill the server
            body = json.dumps({"error": repr(exc)}).encode()
            self._send(500, body, "application/json")
            return

        decode_ms = (t_decoded - t_start) * 1000.0
        wait_ms = (t_got_lock - t_decoded) * 1000.0
        work_ms = (t_done - t_got_lock) * 1000.0
        read_ms = (t_body - t_recv) * 1000.0
        # Everything this process spent, from the first byte of the request
        # arriving to the answer being ready to write. The browser subtracts
        # this from its own round trip to get the cost of the rig. Both are
        # durations on their own clock and are never compared as timestamps,
        # so the two machines' clocks do not need to agree.
        server_total_ms = (time.perf_counter() - t_recv) * 1000.0

        body = json.dumps(
            {
                "boxes": boxes,
                # what the machine actually spent on this frame
                "decode_ms": round(decode_ms, 2),
                "work_ms": round(work_ms, 2),
                "onmachine_ms": round(decode_ms + work_ms, 2),
                # how long it sat behind other frames. Not a hardware cost.
                "wait_ms": round(wait_ms, 2),
                # reading the JPEG off the socket. Part of the server, but not
                # part of anything a camera on the same board would do.
                "read_ms": round(read_ms, 2),
                # the subtrahend: everything above, plus the odds and ends
                "server_total_ms": round(server_total_ms, 2),
                # kept so the page's existing panel still works
                "server_ms": round(decode_ms + wait_ms + work_ms, 2),
            }
        ).encode()
        self._send(200, body, "application/json")

def main():
    global DETECTOR
    DETECTOR = Detector(RESOLUTION)
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    srv.daemon_threads = True
    print(f"\nready on port {PORT}", flush=True)
    print(f"round-trip telemetry -> {TELEMETRY_PATH}", flush=True)
    print("summarise it afterwards with:", flush=True)
    print(f"    python summarise_roundtrip.py {TELEMETRY_PATH}", flush=True)
    print("Accept the VS Code port-forwarding prompt, or open the PORTS tab", flush=True)
    print("and click the forwarded address. Ctrl+C to stop.\n", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping", flush=True)
    finally:
        ttnn.close_device(DETECTOR.device)

if __name__ == "__main__":
    sys.exit(main())
