# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Copy this file, fill in the four marked places, and you have a submission.

    cp harness/submission_template.py my_submission.py

Then run it against the two-photograph sample, which takes seconds:

    python harness/evaluate.py \
      --submission my_submission.py \
      --images ~/tt-metal/models/demos/yolov4/resources \
      --annotations sample/instances_sample.json \
      --out my_result.json

Everything you need for the four TODOs already exists in
~/challenge/bench/bench_stages.py, which loads the detector and drives one
frame through it. Each TODO names the lines to find. Read that file once
before you start; it is about 280 lines and it is commented.

Only run() is timed. Put everything expensive in setup() and warmup().
"""
import os
import cv2
import numpy as np

import submission_api as api


class Submission(api.Submission):

    # ---------------------------------------------------------------- setup
    # Thresholds you choose. Keep the confidence low for scoring: the score
    # walks every detection in score order, so a box you cut here can never be
    # credited. Measured on the supplied detector over 200 practice
    # photographs: 0.3 gives mAP 0.3762, 0.001 gives 0.4639, and the extra cost
    # is about 4 ms of overlap removal. Raise it for the camera demonstration,
    # where nothing is scored and a page full of faint boxes looks bad.
    CONF_THRESH = 0.001
    NMS_THRESH = 0.4

    # What the harness passes to ttnn.open_device() for you. Leave it empty for
    # the plain path. The moment you want to capture a trace, or overlap the
    # copy with the compute, uncomment this: both settings are fixed when the
    # card is opened and the harness owns that call, so asking here is the only
    # way to get them. Without them, tracing fails inside setup().
    #
    # DEVICE_OPTIONS = {"trace_region_size": 6434816, "num_command_queues": 2}
    DEVICE_OPTIONS = {}

    def setup(self, device):
        """Build the model. NOT timed, so do everything slow here.

        TODO 1. Build the detector.

        bench_stages.py does this inside main(), around line 160. Find the line
        beginning `torch_model = load_torch_model`:

            from models.demos.yolov4.common import load_torch_model
            from models.demos.yolov4.tt.model_preprocessing import (
                create_yolov4_model_parameters)
            from models.demos.yolov4.runner.runner import YOLOv4Runner

            torch_model = load_torch_model(None)
            dummy = torch.zeros(1, 3, resolution[1], resolution[0])
            parameters = create_yolov4_model_parameters(
                torch_model, dummy, resolution, device)
            self.runner = YOLOv4Runner(device, parameters, resolution)

        Note the harness has already opened the card and passes it in as
        `device`, so you do not call ttnn.open_device yourself. It has also set
        the working directory to ~/tt-metal before calling this, because the
        model's own load_torch_model() looks for its weights at a relative path
        and would not find them otherwise. If you load weights yourself, use an
        absolute path rather than relying on that.

        If you want the fast path, this is also where you capture a trace and
        set up a second command queue. bench_loop.py shows how. Set
        DEVICE_OPTIONS above first, or the card will have been opened without
        room for a trace and this will fail with an allocation error.
        """
        self.device = device
        self.resolution = (320, 320)          # change this if you want to
        raise NotImplementedError("TODO 1: build the detector")

    # --------------------------------------------------------------- warmup
    def warmup(self, paths):
        """A few frames to settle. NOT timed.

        The first run compiles roughly 600 small programs for the card and
        takes a couple of minutes. Doing it here keeps it out of your result.
        Simply calling self.run(paths) is usually enough.
        """
        self.run(paths)

    # ------------------------------------------------------------------ run
    def run(self, paths):
        """TIMED. Return {path: [detection, ...]} for every path given.

        The loop below is the shape of the answer. Two TODOs inside it.
        """
        out = {}
        for path in paths:
            # decode and prepare, as bench_stages.py lines 180-186
            frame = cv2.imread(path, cv2.IMREAD_COLOR)
            H, W = frame.shape[:2]
            img = cv2.resize(frame, self.resolution)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # TODO 2. Hand it to the card and run the detector.
            #
            #   bench_stages.py does exactly this, around line 188. Find the
            #   line beginning `tt_in = to_device`:
            #     tensor = image_to_tensor(img)
            #     tt_in  = to_device(device, tensor)
            #     boxes, confs = self.runner.run(tt_in)
            #
            #   Both helpers are defined in that file. Copy them or import them.
            raise NotImplementedError("TODO 2: run the detector")

            # TODO 3. Turn the raw output into boxes and classes.
            #
            #   bench_stages.py calls it at line 205. Find `post_processing(`:
            #     from models.demos.yolov4.post_processing import post_processing
            #     result = post_processing(None, conf_thresh, nms_thresh,
            #                              [boxes, confs])
            #
            #   post_processing returns, per image, a list of
            #     [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]
            #   with the coordinates as FRACTIONS of the image, 0 to 1,
            #   not pixels.

            # TODO 4. Convert to what the harness expects, and append.
            #
            #   Three conversions, and getting any of them wrong scores you
            #   near zero with perfectly good detections:
            #
            #     a) fractions   -> pixels of the ORIGINAL photograph
            #                       multiply x by W and y by H, not by 320
            #     b) corners     -> left, top, width, height
            #                       api.xyxy_to_xywh(x1, y1, x2, y2)
            #     c) class index -> COCO category number
            #                       api.to_coco_id(cls_id)
            #
            #   dets = []
            #   for x1, y1, x2, y2, obj_conf, cls_conf, cls_id in result[0]:
            #       box = api.xyxy_to_xywh(x1 * W, y1 * H, x2 * W, y2 * H)
            #       dets.append({"category_id": api.to_coco_id(int(cls_id)),
            #                    "bbox": box,
            #                    "score": float(obj_conf * cls_conf)})
            #   out[path] = dets

        return out

    # ------------------------------------------------------------- teardown
    def teardown(self):
        """Release anything you allocated. NOT timed. The harness closes the
        card itself, so you usually need nothing here."""
        pass
