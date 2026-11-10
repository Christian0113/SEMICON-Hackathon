"""Check a submission's fast post-processing against the supplied reference.

Runs, on the same raw model outputs:
  reference: models/demos/yolov4/post_processing.post_processing()
             + the conversion in harness/submission_reference_fast._detections
  candidate: fast_detections() from the submission file
and reports, per frame, whether the two detection lists are identical, plus the
time each takes. No card needed: it runs on a laptop with torch + torchvision.

Inputs are either raw outputs dumped on the instance (DUMP_RAW=<dir> with
vectorized_nms_claude's my_submission.py) or synthetic tensors.

    # on the instance: dump 20 frames' raw outputs
    DUMP_RAW=~/raw python ~/challenge/harness/evaluate.py --submission <file> \
        --images ~/challenge/practice/images --annotations ~/challenge/practice/instances.json \
        --limit 20 --repeats 1 --out /tmp/dump.json
    # anywhere: compare
    python tools/parity_postprocess.py --dumps ~/raw --tt-metal ~/tt-metal
    python tools/parity_postprocess.py --synthetic 50 --tt-metal SEMICON/yolov4_pinned_3a0e398
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import importlib.util
import io
import os
import statistics as st
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def load_reference(tt_metal_root):
    sys.path.insert(0, os.path.abspath(os.path.expanduser(tt_metal_root)))
    from models.demos.yolov4.post_processing import post_processing  # noqa: E402
    return post_processing


def load_submission(path):
    sys.path.insert(0, os.path.join(REPO, "challenge", "harness"))   # for submission_api
    spec = importlib.util.spec_from_file_location("candidate_submission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_detections(post_processing, api, boxes, confs, W, H, conf, nms):
    """Exactly submission_reference_fast._detections, with its prints silenced."""
    with contextlib.redirect_stdout(io.StringIO()):
        result = post_processing(None, conf, nms, [boxes, confs])
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


def _key(d, nd):
    return (d["category_id"], *(round(v, nd) for v in d["bbox"]), round(d["score"], 9))


def compare(ref, cand, nd=4):
    """Multiset comparison, independent of output order (the scorer sorts anyway)."""
    a = sorted(_key(d, nd) for d in ref)
    b = sorted(_key(d, nd) for d in cand)
    if a == b:
        return 0, 0
    sa, sb = set(a), set(b)
    return len(sa - sb), len(sb - sa)


def synthetic_frames(n, seed=0):
    """Plausible YOLOv4 320x320 outputs, float16 like get_model_result's:
    6300 anchors, ~1000 above the 0.001 floor (scores spread over 1e-3..1),
    clustered around ~30 objects so NMS has real overlaps to resolve."""
    g = torch.Generator().manual_seed(seed)
    frames = []
    for i in range(n):
        N, K = 6300, 1000
        cxcy = torch.rand(N, 2, generator=g)
        wh = torch.rand(N, 2, generator=g) * 0.4 + 0.01
        obj = torch.rand(30, 2, generator=g)                               # object centres
        hot = torch.randperm(N, generator=g)[:K]
        cxcy[hot] = obj[torch.randint(0, 30, (K,), generator=g)] + torch.randn(K, 2, generator=g) * 0.02
        boxes = torch.cat([cxcy - wh / 2, cxcy + wh / 2], 1)              # may exceed [0,1]
        confs = torch.rand(N, 80, generator=g) * 0.0009                    # below the floor
        confs[hot, torch.randint(0, 80, (K,), generator=g)] = 10 ** (-3 * torch.rand(K, generator=g))
        frames.append((boxes.reshape(1, N, 1, 4).half(), confs.reshape(1, N, 80).half(),
                       [640, 500, 427][i % 3], [480, 375, 640][i % 3], f"synthetic_{i:03d}"))
    return frames


def dumped_frames(folder):
    out = []
    for f in sorted(glob.glob(os.path.join(os.path.expanduser(folder), "*.npz"))):
        z = np.load(f)
        out.append((torch.from_numpy(z["boxes"]), torch.from_numpy(z["confs"]),
                    int(z["W"]), int(z["H"]), os.path.basename(f)))
    if not out:
        raise SystemExit(f"no .npz files in {folder}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default=os.path.join(REPO, "challenge", "harness", "my_submission.py"))
    ap.add_argument("--tt-metal", default=os.environ.get("TT_METAL_HOME", "~/tt-metal"),
                    help="folder that contains models/demos/yolov4 at the pinned commit")
    ap.add_argument("--dumps", help="folder of .npz raw outputs from DUMP_RAW")
    ap.add_argument("--synthetic", type=int, default=0, help="use N synthetic frames instead")
    ap.add_argument("--conf", type=float, default=None, help="default: the submission's CONF_THRESH")
    ap.add_argument("--nms", type=float, default=None, help="default: the submission's NMS_THRESH")
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()

    post_processing = load_reference(a.tt_metal)
    sub = load_submission(a.submission)
    api = sub.api
    conf = sub.CONF_THRESH if a.conf is None else a.conf
    nms = sub.NMS_THRESH if a.nms is None else a.nms
    frames = dumped_frames(a.dumps) if a.dumps else synthetic_frames(a.synthetic or 20)
    print(f"{len(frames)} frames, conf {conf}, nms {nms}, torchvision {sub.HAS_TORCHVISION}")

    t_ref, t_new, same, n_ref, n_new, miss, extra = [], [], 0, 0, 0, 0, 0
    for boxes, confs, W, H, name in frames:
        ref = new = None
        for _ in range(a.repeats):
            t0 = time.perf_counter(); ref = reference_detections(post_processing, api, boxes, confs, W, H, conf, nms)
            t_ref.append((time.perf_counter() - t0) * 1e3)
            t0 = time.perf_counter(); new = sub.fast_detections(boxes, confs, W, H, conf, nms)
            t_new.append((time.perf_counter() - t0) * 1e3)
        m, e = compare(ref, new)
        same += (m == 0 and e == 0)
        n_ref += len(ref); n_new += len(new); miss += m; extra += e
        if m or e:
            print(f"  DIFFERENT {name}: {len(ref)} ref vs {len(new)} new detections, "
                  f"{m} only in reference, {e} only in new")

    print(f"\nidentical frames      {same} of {len(frames)}")
    print(f"detections            reference {n_ref}, new {n_new}  "
          f"({miss} only in reference, {extra} only in new)")
    print(f"time per frame (median) reference {st.median(t_ref):.3f} ms, new {st.median(t_new):.3f} ms "
          f"-> {st.median(t_ref) / max(st.median(t_new), 1e-9):.1f}x")
    print("Laptop timings are indicative only; quote evaluate.py numbers from the instance.")
    sys.exit(0 if same == len(frames) else 1)


if __name__ == "__main__":
    main()
