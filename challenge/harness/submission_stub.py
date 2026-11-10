# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A submission that needs no card, used to test the harness end to end.

It decodes each photograph and returns boxes derived from the image, so the
timing path and the JPEG decode are real even though no detection happens.

Set STUB_MODE to exercise the checks:
    good        plausible output with correct COCO category numbers
    bad_ids     class numbers left as the model's own 0..79  (the common trap)
    bad_scale   boxes left in 320x320 space (the other common trap)
    silent      returns nothing
    oracle      returns the ground truth, which must score mAP 1.0
"""
import os
from PIL import Image
import json
import submission_api as api

_GT = os.environ.get("STUB_GT")   # annotations file, for oracle mode

MODE = os.environ.get("STUB_MODE", "good")


class Submission(api.Submission):
    def setup(self, device):
        self.device = device
        self.truth = {}
        if MODE == "oracle":
            gt = json.load(open(_GT))
            name = {i["id"]: i["file_name"] for i in gt["images"]}
            for a in gt["annotations"]:
                if not a.get("iscrowd"):
                    self.truth.setdefault(name[a["image_id"]], []).append(
                        {"category_id": a["category_id"],
                         "bbox": [float(v) for v in a["bbox"]], "score": 1.0})

    def warmup(self, paths):
        for p in paths[:2]:
            Image.open(p).convert("RGB").resize((320, 320))

    def run(self, paths):
        out = {}
        for i, p in enumerate(paths):
            im = Image.open(p).convert("RGB")       # real decode, real cost
            W, H = im.size
            im.resize((320, 320))                   # real resize, real cost
            if MODE == "silent":
                out[p] = []
                continue
            if MODE == "oracle":
                out[p] = self.truth.get(os.path.basename(p), [])
                continue
            dets = []
            for k in range(3):
                idx = (i + k) % 80
                cid = idx if MODE == "bad_ids" else api.to_coco_id(idx)
                if MODE == "bad_scale":
                    box = [10 + 20 * k, 10 + 20 * k, 60, 60]          # 320-space
                elif k == 0:
                    box = [W * 0.03, H * 0.03, W * 0.92, H * 0.92]   # fills the frame
                else:
                    box = [W * 0.1 * (k + 1), H * 0.1 * (k + 1), W * 0.2, H * 0.2]
                dets.append({"category_id": cid, "bbox": box, "score": 0.9 - 0.2 * k})
            out[p] = dets
        return out

    def teardown(self):
        pass
