#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Draw a submission's own detections on the photographs and save them.

Runs the submission the way harness/evaluate.py does (same device options, same
setup/run calls, same working directory for the weights), then draws every
detection at or above --conf on the ORIGINAL photograph. Nothing here is timed
or scored; it is only for looking at the results.

    source ~/tt-metal/python_env/bin/activate
    python ~/SEMICON-Hackathon/tools/draw_detections.py \
        --submission ~/challenge/claude.py --images ~/challenge/practice/images \
        --out ~/challenge/vis --limit 30

Scores are what the submission returns (for the YOLOv4 submissions: the class
confidence squared, as in the supplied reference), the same scale evaluate.py's
plain readout uses with its 0.25 threshold.
"""
import argparse
import glob
import importlib.util
import json
import os
import sys

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default=os.path.expanduser("~/challenge/vis"))
    ap.add_argument("--limit", type=int, default=30, help="first N photographs (sorted by name); 0 = all")
    ap.add_argument("--conf", type=float, default=0.25, help="draw detections with score >= this")
    a = ap.parse_args()

    sub_path = os.path.abspath(os.path.expanduser(a.submission))
    img_dir = os.path.abspath(os.path.expanduser(a.images))
    out_dir = os.path.abspath(os.path.expanduser(a.out))
    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(p for p in glob.glob(os.path.join(img_dir, "*"))
                   if p.lower().endswith((".jpg", ".jpeg", ".png")))
    if a.limit:
        paths = paths[:a.limit]

    harness = os.path.expanduser("~/challenge/harness")
    sys.path.insert(0, harness)
    import submission_api as api

    spec = importlib.util.spec_from_file_location("team_submission", sub_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["team_submission"] = mod
    spec.loader.exec_module(mod)
    sub = mod.Submission()

    import ttnn
    from models.demos.yolov4.common import YOLOV4_L1_SMALL_SIZE
    opts = dict(getattr(sub, "DEVICE_OPTIONS", None) or {})
    opts.pop("device_id", None)
    opts.pop("l1_small_size", None)
    device = ttnn.open_device(device_id=0, l1_small_size=YOLOV4_L1_SMALL_SIZE, **opts)

    tt_home = os.environ.get("TT_METAL_HOME") or os.path.expanduser("~/tt-metal")
    names_file = os.path.join(tt_home, "models/demos/yolov4/resources/coco.names")
    names = [l.strip() for l in open(names_file)] if os.path.exists(names_file) else [str(i) for i in range(80)]
    cwd = os.getcwd()
    os.chdir(tt_home)
    try:
        sub.setup(device)
        sub.warmup(paths[:8])
        out = sub.run(list(paths))
    finally:
        try:
            sub.teardown()
        finally:
            os.chdir(cwd)
            ttnn.close_device(device)

    all_dets = {}
    for p in paths:
        dets = out.get(p, [])
        all_dets[os.path.basename(p)] = dets
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        shown = 0
        for d in sorted(dets, key=lambda d: d["score"]):
            if d["score"] < a.conf:
                continue
            x, y, w, h = d["bbox"]
            idx = api.COCO_IDS.index(d["category_id"]) if d["category_id"] in api.COCO_IDS else -1
            label = f"{names[idx] if idx >= 0 else d['category_id']} {d['score']:.2f}"
            color = tuple(int(c) for c in ((idx * 67) % 256, (idx * 151) % 256, (idx * 211 + 80) % 256))
            p1, p2 = (int(x), int(y)), (int(x + w), int(y + h))
            cv2.rectangle(img, p1, p2, color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (p1[0], max(0, p1[1] - th - 4)), (p1[0] + tw + 2, p1[1]), color, -1)
            cv2.putText(img, label, (p1[0] + 1, max(th, p1[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            shown += 1
        name = os.path.splitext(os.path.basename(p))[0] + ".jpg"
        cv2.imwrite(os.path.join(out_dir, name), img)
        print(f"  {name}: {shown} boxes drawn (of {len(dets)} returned)")
    with open(os.path.join(out_dir, "detections.json"), "w") as f:
        json.dump(all_dets, f)
    print(f"\n  {len(paths)} images written to {out_dir} (plus detections.json with every detection)")


if __name__ == "__main__":
    main()
