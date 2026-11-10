"""Two-photo check: does the card really see the photograph we hand it?

The challenge README warns that one supplied function takes the photograph you
pass it, then ignores it and uses the one it already had. This runs two clearly
different photographs through a submission's run() and fails unless the two
sets of boxes differ. Run it after every change to how frames are fed.

Instance only (needs the card). Opens the card the same way harness/evaluate.py does.

    python tools/liveness_check.py --submission challenge/harness/my_submission.py
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default=os.path.join(REPO, "challenge", "harness", "my_submission.py"))
    ap.add_argument("--harness", default=os.path.expanduser("~/challenge/harness"),
                    help="folder with submission_api.py (the supplied harness)")
    ap.add_argument("--images", nargs=2,
                    default=[os.path.expanduser("~/tt-metal/models/demos/yolov4/resources/giraffe.jpg"),
                             os.path.expanduser("~/tt-metal/models/demos/yolov4/resources/dinning.png")])
    a = ap.parse_args()
    sub_path = os.path.abspath(os.path.expanduser(a.submission))
    imgs = [os.path.abspath(os.path.expanduser(p)) for p in a.images]

    sys.path.insert(0, os.path.abspath(os.path.expanduser(a.harness)))
    spec = importlib.util.spec_from_file_location("team_submission", sub_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["team_submission"] = mod
    spec.loader.exec_module(mod)
    sub = mod.Submission()

    import ttnn
    from models.demos.yolov4.common import YOLOV4_L1_SMALL_SIZE as L1
    opts = dict(getattr(sub, "DEVICE_OPTIONS", None) or {})
    opts.pop("device_id", None)
    opts.pop("l1_small_size", None)
    device = ttnn.open_device(device_id=0, l1_small_size=L1, **opts)

    cwd = os.getcwd()
    os.chdir(os.environ.get("TT_METAL_HOME") or os.path.expanduser("~/tt-metal"))  # model weights path
    try:
        sub.setup(device)
        sub.warmup(imgs)
        out = sub.run(imgs)
        # Also run them one at a time and in the other order, to catch stale inputs
        # that only show up when a frame is alone or comes second.
        alone = {p: sub.run([p])[p] for p in reversed(imgs)}
    finally:
        try:
            sub.teardown()
        finally:
            ttnn.close_device(device)
            os.chdir(cwd)

    def sig(dets):
        return sorted((d["category_id"], tuple(round(v, 1) for v in d["bbox"])) for d in dets)

    a_, b_ = (sig(out[p]) for p in imgs)
    ok = a_ != b_ and bool(a_) and bool(b_)
    consistent = all(sig(alone[p]) == sig(out[p]) for p in imgs)
    for p in imgs:
        print(f"  {os.path.basename(p):12s} {len(out[p]):4d} detections")
    print(f"  together vs one at a time: {'same' if consistent else 'DIFFERENT (stale input?)'}")
    if ok and consistent:
        print("DIFFERENT: the two photographs give different boxes. Pass.")
        sys.exit(0)
    print("FAIL: the photographs gave the same (or empty) boxes, or results depend on "
          "batch order. Do not trust any timing from this submission.")
    sys.exit(1)


if __name__ == "__main__":
    main()
