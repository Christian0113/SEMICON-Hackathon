# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Cross-check cocomap.evaluate against pycocotools on the real hold-out set.

Builds synthetic detections from the ground truth under several distortions,
so the comparison covers perfect output, missed objects, invented objects,
displaced boxes and score noise.
"""
import json, sys, numpy as np, contextlib, io
sys.path.insert(0, __import__("os").path.dirname(__file__))
import cocomap

if len(sys.argv) < 2:
    sys.exit("usage: validate_cocomap.py <annotations.json>\n"
             "Needs pycocotools, which is not installed on a challenge instance.\n"
             "This is the organisers' check that the scorer agrees with the reference\n"
             "implementation. You are welcome to run it if you install pycocotools.")
GT = json.load(open(sys.argv[1]))
rng = np.random.default_rng(20261110)


def make(kind):
    dets = []
    for a in GT["annotations"]:
        if a.get("iscrowd"):
            continue
        x, y, w, h = a["bbox"]
        s = float(rng.uniform(0.3, 1.0))
        if kind == "perfect":
            s = 1.0
        elif kind == "miss_half" and rng.random() < 0.5:
            continue
        elif kind == "jitter":
            j = 0.10
            x += rng.normal(0, j * w); y += rng.normal(0, j * h)
            w *= float(rng.uniform(1 - j, 1 + j)); h *= float(rng.uniform(1 - j, 1 + j))
        dets.append({"image_id": a["image_id"], "category_id": a["category_id"],
                     "bbox": [float(x), float(y), float(w), float(h)], "score": s})
    if kind in ("false_pos", "mixed"):
        imgs = [i["id"] for i in GT["images"]]
        cats = [c["id"] for c in GT["categories"]]
        for _ in range(3000):
            dets.append({"image_id": int(rng.choice(imgs)),
                         "category_id": int(rng.choice(cats)),
                         "bbox": [float(rng.uniform(0, 400)), float(rng.uniform(0, 300)),
                                  float(rng.uniform(10, 150)), float(rng.uniform(10, 150))],
                         "score": float(rng.uniform(0.05, 0.9))})
    return dets


def ref_map(dets):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        c = COCO(); c.dataset = GT; c.createIndex()
        dt = c.loadRes([dict(d) for d in dets])
        E = COCOeval(c, dt, "bbox")
        E.params.iouThrs = np.array([0.5])
        E.evaluate(); E.accumulate(); E.summarize()
    return float(E.stats[0])


print(f"  {'case':<12} {'ours':>9} {'pycocotools':>13} {'delta':>11}")
worst = 0.0
for kind in ("perfect", "miss_half", "jitter", "false_pos", "mixed"):
    d = make(kind)
    ours, _ = cocomap.evaluate(GT, d)
    ref = ref_map(d)
    delta = abs(ours - ref); worst = max(worst, delta)
    print(f"  {kind:<12} {ours:9.6f} {ref:13.6f} {delta:11.2e}   ({len(d)} detections)")
print(f"\n  worst disagreement: {worst:.2e}")
sys.exit(0 if worst < 1e-6 else 1)
