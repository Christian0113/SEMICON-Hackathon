# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The interface your submission implements, and the helpers that keep you out
of the two traps that cost teams their whole score.

You write a class called Submission with four methods. Only run() is timed.

    class Submission:
        def setup(self, device):   ...   # build the model. Not timed.
        def warmup(self, paths):   ...   # a few frames to settle. Not timed.
        def run(self, paths):      ...   # TIMED. Returns the detections.
        def teardown(self):        ...   # free the card. Not timed.

run() is given a list of file paths and must return a dict mapping each path to
a list of detections. Because it receives paths rather than decoded images,
unpacking the JPEG and resizing are inside the timed region, which is where a
meaningful part of the remaining time sits.

A detection is a dict:

    {"category_id": 1, "bbox": [x, y, w, h], "score": 0.93}

  category_id   COCO's number for the class. NOT 0-79. See COCO_IDS below.
  bbox          left edge, top edge, width, height, in pixels, in the ORIGINAL
                photograph's coordinates. If you shrank the image to run it,
                scale the boxes back up before returning them.
  score         confidence between 0 and 1, used to rank detections.
"""
from __future__ import annotations
import json, os, statistics as st

# COCO category numbers in the order every YOLO-family model emits them.
# They are NOT 0-79: the range is 1-90 with ten numbers missing
# (12, 26, 29, 30, 45, 66, 68, 69, 71, 83). Index into this list with the
# model's own class index and you get the right number.
COCO_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]
assert len(COCO_IDS) == 80


def to_coco_id(model_class_index: int) -> int:
    """Convert a model's own 0-79 class index to COCO's category number."""
    if not 0 <= model_class_index < 80:
        raise ValueError(f"class index {model_class_index} is outside 0..79")
    return COCO_IDS[model_class_index]


def xyxy_to_xywh(x1, y1, x2, y2):
    """Corner form to COCO's left-top-width-height."""
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def cxcywh_to_xywh(cx, cy, w, h):
    """Centre form to COCO's left-top-width-height."""
    return [float(cx - w / 2), float(cy - h / 2), float(w), float(h)]


def scale_boxes(boxes, from_size, to_size):
    """Rescale boxes from the size you ran at back to the original photograph.

    boxes      list of [x, y, w, h]
    from_size  (width, height) you ran the model at
    to_size    (width, height) of the original photograph
    """
    fw, fh = from_size
    tw, th = to_size
    sx, sy = tw / fw, th / fh
    return [[x * sx, y * sy, w * sx, h * sy] for x, y, w, h in boxes]


class Submission:
    """Subclass this. The harness calls the methods in this order."""

    # What the harness should pass to ttnn.open_device() when it opens the card
    # for you. It reads this before opening, so anything here is in force by the
    # time setup() is called.
    #
    # This matters more than it looks. trace_region_size and num_command_queues
    # can ONLY be set when the device is opened, and the harness owns that call,
    # so a submission that wants to capture a trace or overlap transfers with
    # compute has to ask for them here. Without them, tracing fails inside
    # setup() with an allocation error.
    #
    #     DEVICE_OPTIONS = {"trace_region_size": 6434816,
    #                       "num_command_queues": 2}
    #
    # 6434816 is the value bench/bench_levers.py uses and is enough for this
    # model at 320x320. l1_small_size is set for you; you do not need it here.
    DEVICE_OPTIONS = {}

    def setup(self, device):
        """Build the model and put it on the card. Not timed, so do everything
        expensive here: loading weights, compiling, allocating buffers."""
        raise NotImplementedError

    def warmup(self, paths):
        """Run a few frames so the kernel cache is warm and any trace is
        captured. Not timed. The harness passes a short list."""
        pass

    def run(self, paths):
        """TIMED. Return {path: [detection, ...]} for every path given."""
        raise NotImplementedError

    def teardown(self):
        """Release the card. Not timed."""
        pass


# --------------------------------------------------------------------------
# Checks the harness runs on your output before scoring it. You can run them
# yourself: python submission_api.py --check my_output.json
# --------------------------------------------------------------------------

def check_detections(by_path, sizes=None, max_report=8):
    """Return a list of problems, worst first. Empty list means it looks sane.

    sizes, if given, maps path -> (width, height) of the original photograph,
    which lets the check catch boxes left in the wrong coordinate space.
    """
    problems, valid_ids = [], set(COCO_IDS)
    n_det = sum(len(v) for v in by_path.values())
    if n_det == 0:
        return ["no detections at all were returned"]

    looks_contiguous = 0
    out_of_range = 0
    oversize = 0
    bad_shape = 0
    bad_score = 0
    ratios = {}

    for path, dets in by_path.items():
        W, H = (sizes or {}).get(path, (None, None))
        for d in dets:
            cid = d.get("category_id")
            if cid not in valid_ids:
                out_of_range += 1
                if isinstance(cid, int) and 0 <= cid <= 79:
                    looks_contiguous += 1
            b = d.get("bbox")
            if not (isinstance(b, (list, tuple)) and len(b) == 4):
                bad_shape += 1
                continue
            x, y, w, h = b
            if w <= 0 or h <= 0:
                bad_shape += 1
            s = d.get("score")
            if not isinstance(s, (int, float)) or not 0.0 <= float(s) <= 1.0:
                bad_score += 1
            if W:
                if x + w > W * 1.02 or y + h > H * 1.02:
                    oversize += 1
                ratios.setdefault(max(W, H), []).append(max(x + w, y + h) / max(W, H))

    if looks_contiguous:
        problems.append(
            f"{looks_contiguous} detections use a class number in 0..79, which is the "
            f"model's own index and not COCO's category number. Convert with "
            f"to_coco_id(), or you will score close to zero. See COCO_IDS.")
    elif out_of_range:
        problems.append(f"{out_of_range} detections carry a category_id that is not a "
                        f"COCO category number")
    # Across hundreds of photographs a real detector always finds at least one
    # object filling most of a frame. Boxes left in the size the model ran at
    # are capped there, so their largest relative extent stops well short. This
    # works even when every photograph is the same size, which is the case for
    # most of COCO, where a correlation between box size and image size cannot
    # be measured.
    all_ratios = [r for rs in ratios.values() for r in rs]
    if len(all_ratios) >= 100:
        biggest = max(all_ratios)
        if biggest < 0.6:
            problems.append(
                f"the largest box anywhere covers only {biggest:.0%} of its photograph. "
                f"Across this many photographs something should fill most of a frame, so "
                f"the boxes are probably still in the size you ran at rather than the "
                f"original photograph's. Use scale_boxes().")
    # A detector predicting slightly past the edge is normal and every real
    # pipeline clips it away. Only say something when it is the rule rather
    # than the exception, which is what a coordinate error looks like.
    if oversize > max(10, n_det * 0.05):
        problems.append(f"{oversize} of {n_det} boxes fall outside the photograph, "
                        f"which is more than stray edge predictions. Clip your boxes "
                        f"to the image, or check the scaling.")
    if bad_shape:
        problems.append(f"{bad_shape} boxes are not four positive numbers")
    if bad_score:
        problems.append(f"{bad_score} scores are missing or outside 0..1")
    return problems[:max_report]


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--check":
        d = json.load(open(sys.argv[2]))
        probs = check_detections(d)
        print("looks sane" if not probs else "\n".join("PROBLEM: " + p for p in probs))
    else:
        print(__doc__)
        print(f"COCO_IDS has {len(COCO_IDS)} entries, index 0 -> {COCO_IDS[0]}, "
              f"index 79 -> {COCO_IDS[79]}")
