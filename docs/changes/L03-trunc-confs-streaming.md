# L03 — trunc_confs × streaming pipeline (lever F1)

**Branch:** `L03-trunc-confs-streaming` (off `main` @ `82a2e43`)
**Files:** `challenge/my_submission.py`, `challenge/harness/my_submission.py` (identical copies)
**Status:** written on the laptop, syntax-checked only — NOT yet run on the card.

## What

Merges the `trunc_confs` lever (on-device `ttnn.max` + `ttnn.argmax` over the
80 classes, commits `9c11d10`/`827c339`) into the milestone-8 per-frame
streaming pipeline (`main` @ `d6f913e`, 5.301 ms/frame). The two had never been
combined: trunc_confs was written against the pre-streaming submission.

Per-frame D2H read shrinks from `confs [1,6300,80]` bf16 (1,008 KB) + boxes
(51 KB) to `max [1,6300,1]` + `argmax [1,6300,1]` + boxes ≈ 89 KB. The read
rides CQ0 (the compute queue), so the saving is card-idle time:
**expected 5.30 → ~4.8–5.0 ms** if device argmax is cheap (see risks).

## How (delta vs milestone 8)

- `TruncatedYoloV4Runner` wraps the infra inside the trace; appends max/argmax,
  deallocates the full confs tensor (keeps L1 addresses stable for capture).
- `fast_detections(..., max_ids=)` consumes precomputed ids; thresholds, empty
  gate, NMS, scaling, score squaring unchanged.
- `_convert()` centralises output conversion for both streaming `_retire` and
  the chunked A/B path; `max_ids` is copied (`.to(torch.int64, copy=True)`)
  before the host slot is reused.
- Setup tries the truncated trace first; on any compile failure it recovers the
  device (end/release stray trace captures) and falls back to the exact
  milestone-8 pipeline. `TRUNC_CONFS = False` forces milestone-8 behaviour.
- `TRACE_REGION_SIZE` +1 MiB headroom for the two extra traced ops.

## Risks / open questions

1. **argmax device cost is unmeasured.** If `ttnn.argmax` on `[1,6300,80]`
   costs more on CQ0 than the ~0.3–0.5 ms read it saves, the lever inverts.
   Time it standalone first (F0 in `fable_suggestion.md`).
2. **argmax may not be traceable / may not support ROW_MAJOR input** on this
   tree — never verified on the card. The fallback catches it (look for
   `[submission] confs truncation OFF` in the log: that means the lever is
   silently inactive and the run is just milestone 8).
3. **Tie-breaking parity:** torch `max(dim=1)` returns the first index on
   ties; device argmax may differ, and bf16 scores do collide. Acceptance gate
   decides; if detections drift slightly, diff per-image vs milestone 8.

## How to test on the instance

```bash
cd ~/challenge
# 0. plumbing + check the log says "confs truncation ON"
python harness/evaluate.py --submission my_submission.py \
  --images ~/tt-metal/models/demos/yolov4/resources \
  --annotations sample/instances_sample.json --out /tmp/check_l03.json
# 1. practice set, fastest of 3
python harness/evaluate.py --submission my_submission.py \
  --images practice/images --annotations practice/instances.json \
  --out runs/11_trunc_streaming.json
```

**Acceptance:** no `PROBLEM:` lines, mAP@50 = 0.4640, `detections_returned`
= 25,119, and ms/frame < 5.301. If accuracy fails: set `TRUNC_CONFS = False`
→ must reproduce milestone 8 exactly (that isolates the lever).

## Result

_(fill in after the card run)_

| Config | ms/frame | FPS | mAP@50 | detections |
|---|---|---|---|---|
| milestone 8 (main) | 5.301 | 188.6 | 0.4640 | 25,119 |
| L03 TRUNC_CONFS=True | — | — | — | — |
