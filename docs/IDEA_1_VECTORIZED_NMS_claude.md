# Idea 1 (Claude's version): Vectorized NMS with matching output

**Branch:** `vectorized_nms_claude`, started from `vectorized_nms` (commit `691ccba`)
**Compare with:** [`IDEA_1_VECTORIZED_NMS.md`](IDEA_1_VECTORIZED_NMS.md) on `vectorized_nms`
**Status:** code done and tested locally on synthetic data; waiting for instance numbers
**Goal:** same detections as the fast reference (`harness/submission_reference_fast.py`), with a much cheaper box step.

---

## 1. What changed compared with `vectorized_nms`

All changes are in `challenge/harness/my_submission.py`. The run loop, the card setup and the decode are unchanged.

| # | `vectorized_nms` (colleague) | `vectorized_nms_claude` (this branch) | Why |
|---|---|---|---|
| 1 | Boxes scaled to pixels with **no clamp** and **no zero-size filter** | Clamp each corner to the photo (`0..W`, `0..H`) and drop boxes with `x2 <= x1` or `y2 <= y1` | That is what `submission_reference_fast._detections` does. Without it, boxes stick out of the photo and some degenerate boxes are submitted, so the output differs from the reference even though the doc says "100% parity". |
| 2 | Per-box Python loop: 6 `float(tensor[k, i])` calls per box, ~120 boxes per photo | One vectorized pass in torch, then a single `.tolist()`, then a list comprehension | Indexing a torch tensor one element at a time costs microseconds per call, which adds up to milliseconds per photo. |
| 3 | `api.to_coco_id()` called per box | Lookup table `_COCO_IDS` built once | Same result, no per-box function call and range check. |
| 4 | Post-processing lives inside a method that needs `ttnn` | `fast_detections(boxes, confs, W, H, conf, nms)` is a plain function; `ttnn` imports are guarded | It can be tested against the reference on a laptop (`tools/parity_postprocess.py`). |
| 5 | n/a | `DUMP_RAW=<dir>` (and `DUMP_N`, default 20) saves raw model outputs as `.npz` | Real card outputs for the laptop parity test. Off unless set. Never enable it in a scored run. |
| 6 | Pixel conversion in float32 | Pixel conversion in float64 | The reference computes `x1 * W` in Python floats (float64). float32 would shift some box edges by about 1e-5 px. |

Kept from the colleague's version: `torchvision.ops.batched_nms` (C++), the pure-torch fallback when torchvision is missing (the instance has it, see `setup/setup_yolov4_on_ttmetal.sh` line 65), single-label semantics (argmax over classes), `score = conf * conf` (the reference multiplies `det[4] * det[5]`, both equal to the max class score).

### Considered and not done here
- **Removing the float16 round trip in `get_model_result`.** The reference removes overlaps on float16 boxes, so using the float32 values from the card would change which boxes survive. This belongs to a separate readback lever (see `PLAN.md` §7), measured on its own.
- **Multi-label decoding and a top-K cut.** These change the output on purpose, so they need their own branches and an mAP measurement.

## 2. Known small difference from the reference

The reference runs NMS in **numpy float16** (its boxes arrive as float16). torchvision computes IoU in **float32**. In addition, float16 scores tie often, and the two implementations order tied scores differently. When two tied, overlapping boxes compete, each version can keep a different one. Expect identical output on almost all photos, and at most a handful of borderline boxes different over a run. `tools/parity_postprocess.py` counts them; `evaluate.py`'s mAP is the final check.

## 3. How to test

### On a laptop (no card)
```bash
# venv with torch + torchvision + opencv, e.g. C:\Users\20225396\venvs\semicon
python tools/parity_postprocess.py --synthetic 50 --tt-metal SEMICON/yolov4_pinned_3a0e398
```

### On the instance
```bash
cd ~/SEMICON-Hackathon && git switch vectorized_nms_claude && git pull
SUB=~/SEMICON-Hackathon/challenge/harness/my_submission.py

# 1. Real-output parity: dump 20 frames, then compare (runs on the instance too)
DUMP_RAW=~/raw_vnms python ~/challenge/harness/evaluate.py --submission $SUB \
  --images ~/challenge/practice/images --annotations ~/challenge/practice/instances.json \
  --limit 20 --repeats 1 --out /tmp/dump.json
python tools/parity_postprocess.py --dumps ~/raw_vnms --tt-metal ~/tt-metal

# 2. Scored practice run (no DUMP_RAW!)
mkdir -p results
python ~/challenge/harness/evaluate.py --submission $SUB \
  --images ~/challenge/practice/images --annotations ~/challenge/practice/instances.json \
  --team vectorized_nms_claude --out results/vectorized_nms_claude.json
```

**Pass:** no `PROBLEM:` lines, mAP **0.464**, `detections_returned` within a few of **25,176** (the fast reference), ms/frame below **17.8**.

## 4. Measured

### Laptop, synthetic outputs (not card data)
`tools/parity_postprocess.py --synthetic 50 --repeats 5`, Windows laptop, torch 2.15 CPU, torchvision 0.29, 10 Nov 2026. Synthetic frames: 6300 anchors, ~1000 above the 0.001 floor, clustered around 30 objects, float16 like the card's output.

| | result |
|---|---|
| identical frames | **50 of 50** (46,480 detections in both) |
| box step, median per frame | reference 66.0 ms, new 8.8 ms (**7.5×**). Laptop numbers; the instance will be faster for both |
| float16 check | also 50/50 identical when the reference is given float32 boxes, so the float16-vs-float32 NMS difference (section 2) did not show up on this data |

A first synthetic set with unrealistically many candidates (~5,400 surviving boxes per frame) gave 4/30 identical frames, with 113 of 161,576 boxes differing (0.07%). That is the tie and float16 effect from section 2. It only appears when thousands of near-identical boxes compete, which real frames (~126 detections) do not have. The real-output check on the instance settles it.

### Instance (to be filled in by whoever runs it)
| Run | ms/frame | mAP | detections_returned | identical frames (parity tool) | date / who |
|---|---|---|---|---|---|
| fast reference (baseline) | 17.803 | 0.464 | 25,176 | n/a | 10 Nov, team |
| `vectorized_nms` (colleague) | | | | | |
| `vectorized_nms_claude` | | | | | |

## 5. Risks
- If mAP moves by more than 0.002, the float16/float32 NMS difference matters more than expected: check the parity tool's `DIFFERENT` lines.
- The box step is now cheap, but the card still waits for it: the run loop works out boxes only after `pop_all()`. Overlapping the two is lever L02, on its own branch.
