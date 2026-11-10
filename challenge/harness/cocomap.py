# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""mAP@0.5 for COCO-format detections, numpy only.

Written so the hour-30 score does not depend on pycocotools being present in
whatever image an instance happens to carry. Follows COCOeval for the bbox
task at a single IoU threshold: 101-point interpolated AP, maxDets 100 per
image, and iscrowd regions treated as ignore rather than as false positives.

Validated against pycocotools on the hold-out set; see validate_cocomap.py.
"""
import numpy as np
from collections import defaultdict

IOU_THR = 0.5
MAX_DETS = 100
REC_THRS = np.linspace(0.0, 1.0, 101)


def _iou_matrix(dets, gts, iscrowd):
    """dets, gts as [N,4] xywh. For crowd gt, COCO uses intersection/det_area."""
    if len(dets) == 0 or len(gts) == 0:
        return np.zeros((len(dets), len(gts)))
    d = np.asarray(dets, dtype=float)
    g = np.asarray(gts, dtype=float)
    dx1, dy1 = d[:, 0], d[:, 1]
    dx2, dy2 = d[:, 0] + d[:, 2], d[:, 1] + d[:, 3]
    gx1, gy1 = g[:, 0], g[:, 1]
    gx2, gy2 = g[:, 0] + g[:, 2], g[:, 1] + g[:, 3]
    iw = np.minimum(dx2[:, None], gx2[None, :]) - np.maximum(dx1[:, None], gx1[None, :])
    ih = np.minimum(dy2[:, None], gy2[None, :]) - np.maximum(dy1[:, None], gy1[None, :])
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    da = (d[:, 2] * d[:, 3])[:, None]
    ga = (g[:, 2] * g[:, 3])[None, :]
    union = da + ga - inter
    crowd = np.asarray(iscrowd, dtype=bool)[None, :]
    denom = np.where(crowd, da, union)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, inter / denom, 0.0)


def evaluate(gt, dets, iou_thr=IOU_THR, max_dets=MAX_DETS):
    """gt: COCO instances dict. dets: list of {image_id, category_id, bbox, score}.

    Returns (mAP, per_class_ap).
    """
    gt_by = defaultdict(list)
    for a in gt["annotations"]:
        gt_by[(a["image_id"], a["category_id"])].append(a)
    img_ids = {i["id"] for i in gt["images"]}
    cat_ids = sorted({c["id"] for c in gt["categories"]})

    det_by = defaultdict(list)
    for d in dets:
        if d["image_id"] in img_ids:
            det_by[(d["image_id"], d["category_id"])].append(d)

    per_class = {}
    for cid in cat_ids:
        n_gt = sum(1 for (i, c), A in gt_by.items() if c == cid for a in A if not a.get("iscrowd"))
        if n_gt == 0:
            per_class[cid] = float("nan")      # COCO skips classes with no gt
            continue
        scores, tps, fps = [], [], []
        for iid in img_ids:
            D = sorted(det_by.get((iid, cid), []), key=lambda x: -x["score"])[:max_dets]
            G = gt_by.get((iid, cid), [])
            # COCO sorts gt so that non-crowd come first and crowd last
            order = sorted(range(len(G)), key=lambda k: G[k].get("iscrowd", 0))
            G = [G[k] for k in order]
            crowd = [bool(g.get("iscrowd")) for g in G]
            ious = _iou_matrix([d["bbox"] for d in D], [g["bbox"] for g in G], crowd)
            used = [False] * len(G)
            for di, d in enumerate(D):
                best, bj = iou_thr, -1
                for gj in range(len(G)):
                    if used[gj] and not crowd[gj]:
                        continue
                    if bj > -1 and not crowd[bj] and crowd[gj]:
                        break                  # a real match beats a crowd match
                    if ious[di, gj] >= best:
                        best, bj = ious[di, gj], gj
                scores.append(d["score"])
                if bj == -1:
                    tps.append(0); fps.append(1)
                elif crowd[bj]:
                    tps.append(0); fps.append(0)   # ignored, neither tp nor fp
                else:
                    used[bj] = True; tps.append(1); fps.append(0)
        if not scores:
            per_class[cid] = 0.0
            continue
        o = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
        tp = np.cumsum(np.asarray(tps)[o])
        fp = np.cumsum(np.asarray(fps)[o])
        rc = tp / n_gt
        pr = tp / np.maximum(tp + fp, np.finfo(float).eps)
        for i in range(len(pr) - 1, 0, -1):            # make precision monotone
            pr[i - 1] = max(pr[i - 1], pr[i])
        idx = np.searchsorted(rc, REC_THRS, side="left")
        q = np.where(idx < len(pr), pr[np.clip(idx, 0, len(pr) - 1)], 0.0)
        per_class[cid] = float(q.mean())

    vals = [v for v in per_class.values() if not np.isnan(v)]
    return (float(np.mean(vals)) if vals else 0.0), per_class


def plain_readout(gt, dets, conf=0.25, iou_thr=IOU_THR):
    """The readable line beside the score: how many real objects were found,
    and how many detections corresponded to nothing."""
    gt_by = defaultdict(list)
    for a in gt["annotations"]:
        if not a.get("iscrowd"):
            gt_by[(a["image_id"], a["category_id"])].append(a)
    det_by = defaultdict(list)
    for d in dets:
        if d["score"] >= conf:
            det_by[(d["image_id"], d["category_id"])].append(d)

    n_gt = sum(len(v) for v in gt_by.values())
    found = fp = 0
    for key in set(det_by) | set(gt_by):
        D = sorted(det_by.get(key, []), key=lambda x: -x["score"])
        G = gt_by.get(key, [])
        ious = _iou_matrix([d["bbox"] for d in D], [g["bbox"] for g in G], [False] * len(G))
        used = [False] * len(G)
        for di in range(len(D)):
            best, bj = iou_thr, -1
            for gj in range(len(G)):
                if not used[gj] and ious[di, gj] >= best:
                    best, bj = ious[di, gj], gj
            if bj == -1:
                fp += 1
            else:
                used[bj] = True; found += 1
    empty_imgs = {i["id"] for i in gt["images"]} - {k[0] for k in gt_by}
    fp_empty = sum(1 for d in dets if d["score"] >= conf and d["image_id"] in empty_imgs)
    return {"objects_present": n_gt, "objects_found": found,
            "recall": found / n_gt if n_gt else 0.0,
            "false_detections": fp,
            "false_detections_on_empty_images": fp_empty,
            "conf": conf, "iou": iou_thr}
