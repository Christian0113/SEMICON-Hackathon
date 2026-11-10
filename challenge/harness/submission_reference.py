# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The detector exactly as supplied, driven through the submission interface.

Two uses. For the organisers it is the baseline: run it over the hold-out
photographs and every team's point is read against it. For a team it is the
worked answer to the template, so if you are stuck on a TODO, read this.

It is deliberately the plain path: one frame at a time, no trace, one command
queue. That is the thing you are being asked to beat.

    python harness/evaluate.py --submission harness/submission_reference.py \
      --images practice/images --annotations practice/instances.json \
      --team as-supplied --out as-supplied.json
"""
import cv2
import torch
import ttnn

from models.demos.yolov4.common import YOLOV4_L1_SMALL_SIZE, image_to_tensor, load_torch_model
from models.demos.yolov4.post_processing import post_processing
from models.demos.yolov4.runner.runner import YOLOv4Runner
from models.demos.yolov4.tt.model_preprocessing import create_yolov4_model_parameters

import submission_api as api

RESOLUTION = (320, 320)

# Keep the confidence threshold low for scoring. Measured on 200 practice
# photographs, 26 October 2026, on a Wormhole n150:
#
#     confidence 0.3     mAP 0.3762    78.863 ms a frame
#     confidence 0.001   mAP 0.4639    82.708 ms a frame
#
# The score walks every detection in score order and credits the correct ones,
# so a box cut here can never be credited. At 0.3 roughly two thirds of the
# real objects had no box submitted for them at all. The extra work is entirely
# in overlap removal, about 4 ms, and it is worth it.
#
# For the live camera demonstration, raise it. A page drawing every faint box
# looks terrible, and nothing there is being scored.
CONF_THRESH = 0.001
NMS_THRESH = 0.4


class Submission(api.Submission):

    def setup(self, device):
        """TODO 1 of the template, answered. Not timed."""
        if device is None:
            raise RuntimeError("this needs a card; the harness could not open one")
        self.device = device
        self.resolution = RESOLUTION
        torch_model = load_torch_model(None)
        dummy = torch.zeros(1, 3, self.resolution[1], self.resolution[0])
        parameters = create_yolov4_model_parameters(torch_model, dummy, self.resolution, device)
        self.runner = YOLOv4Runner(device, parameters, self.resolution)

    def warmup(self, paths):
        """Compiles the kernels and settles the card. Not timed."""
        self.run(paths)

    def run(self, paths):
        """TIMED. One frame at a time, which is the point of a baseline."""
        out = {}
        for path in paths:
            frame = cv2.imread(path, cv2.IMREAD_COLOR)
            if frame is None:
                out[path] = []
                continue
            H, W = frame.shape[:2]

            img = cv2.resize(frame, self.resolution)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = image_to_tensor(img)

            tt_in = self._to_device(tensor)
            boxes, confs = self.runner.run(tt_in)

            result = post_processing(None, CONF_THRESH, NMS_THRESH, [boxes, confs])

            dets = []
            for det in (result[0] if result else []):
                # post_processing returns [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]
                # with the coordinates as fractions of the image, 0 to 1.
                x1, y1, x2, y2 = float(det[0]), float(det[1]), float(det[2]), float(det[3])
                score = float(det[4]) * float(det[5])
                cls_id = int(det[6])
                # The detector routinely predicts a little past the edge of the
                # photograph. Every real pipeline clips; without it the boxes
                # are still roughly right but they describe something that is
                # not in the picture.
                px1 = min(max(x1 * W, 0.0), W)
                py1 = min(max(y1 * H, 0.0), H)
                px2 = min(max(x2 * W, 0.0), W)
                py2 = min(max(y2 * H, 0.0), H)
                if px2 <= px1 or py2 <= py1:
                    continue
                dets.append({
                    "category_id": api.to_coco_id(cls_id),
                    "bbox": api.xyxy_to_xywh(px1, py1, px2, py2),
                    "score": score,
                })
            out[path] = dets
        return out

    def teardown(self):
        pass

    # ---- copied from bench/bench_stages.py, which measured the baseline -----
    def _to_device(self, tensor):
        """The channel count is padded from 3 to 16 for the sharded memory config."""
        n, c, h, w = tensor.shape
        if c == 3:
            c = 16
        mem = ttnn.create_sharded_memory_config(
            [n, c, h, w], ttnn.CoreGrid(x=8, y=8), ttnn.ShardStrategy.HEIGHT
        )
        tt_in = ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        return tt_in.to(self.device, mem)
