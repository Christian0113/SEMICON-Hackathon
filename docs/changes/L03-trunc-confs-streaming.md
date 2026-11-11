# L03 — trunc_confs × streaming pipeline (lever F1)

**Branch:** `L03-trunc-confs-streaming` (off `main` @ `82a2e43`)
**Files:** `challenge/my_submission.py`, `challenge/harness/my_submission.py` (identical copies)
**Status:** v1 (ttnn.argmax) measured on the card 2026-11-11 and REJECTED
(19.232 ms/frame, mAP 0.4639 — see Result). v2 ("maxmin") replaces argmax
with multicore tile ops and exact torch tie-breaking; awaiting card run.

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

## v2: "maxmin" (replaces argmax, 2026-11-11)

The v1 card run confirmed risks 1 and 3 simultaneously: `ttnn.argmax` runs
single-core (~14 ms for 6300×80 — 2.7× the entire rest of the frame) and its
tie-breaking is not torch's (mAP 0.4639, 315 false detections vs 314). Power
dropped 34 → 27 W, consistent with the card idling in a scalar loop.

v2 computes the class id with multicore tile ops only (the softmax pattern):

```
c = tilize(confs)                     # [1,6300,80]
m = max(c, dim=-1)                    # row max  -> read back (scores)
d = c - m                             # bcast sub, exact in bf16
z = eqz(d)                            # 1.0 at every tied max
r = z * rev_iota                      # rev_iota[j] = 80 - j, constant in DRAM
f = max(r, dim=-1)                    # = 80 - FIRST max index -> read back
```

Host undoes the reversal: `class_id = 80 - f`. Torch's `c.max(dim=1)` returns
the first index of the max; the reversed iota makes the row max select exactly
that, so v2 should reproduce milestone-8 outputs bit-identically (gate: mAP
0.4640, detections 25,119). Tile padding is benign: scores ≥ 0 so zero-padding
never wins a row max, and rev_iota's padding is zero.

Knob is now `TRUNC_MODE`: `"maxmin"` (default) | `"argmax"` (v1, reference
only) | `"off"` (exact milestone 8). Open risk: `ttnn.subtract` must broadcast
`[1,6300,80] - [1,6300,1]` on this tree (the standard softmax pattern); if the
truncated compile fails, the log prints `confs truncation: off` and the run is
milestone 8 — report that back rather than treating the timing as v2's.

## Result

| Config | ms/frame | FPS | mAP@50 | detections | Verdict |
|---|---|---|---|---|---|
| milestone 8 (main) | 5.301 | 188.6 | 0.4640 | 25,119 | reference |
| L03 v1 argmax (2026-11-11) | 19.232 | 52.0 | 0.4639 | — (315 false @0.25) | REJECTED: single-core argmax |
| L03 v2 maxmin (13:40 UTC) | 5.626 | 177.8 | 0.4640 | 25,119 | mode unproven (pre-b60f633) |
| L03 v2 maxmin (13:48 UTC, proven) | 5.915 | 169.1 | 0.4640 | 25,119 | parity EXACT; speed verdict needs same-session off reference — card drifted slower all day (m8 was 5.301 at 08:57) |
| L03 TRUNC_MODE=off (14:05 UTC, same session) | 5.880 | 170.1 | 0.4640 | 25,119 | reference for v2 |
| L03 v3 headmax | — | — | — | — | awaiting card run (A/B vs off, same session) |

**v2 verdict (2026-11-11, same-session A/B): BREAK-EVEN.** maxmin 5.915 vs
off 5.880 (+0.035 ms, inside the 2% spread): the six reduction ops cost about
what the 1 MB read saves. Not merged as-is. Also learned: the card drifted
+11% over the day (identical milestone-8 code: 5.301 at 08:57, 5.880 at
14:05) — **every speed verdict must come from a same-session A/B.**

## v3: "headmax" — reduce per head inside the model (2026-11-11)

The model already pays for 3 confs untilizes (~1 MB total) and a 1 MB
row-major concat in `TtYOLOv4.__call__` purely to ship the full score matrix.
v3 monkey-patches `__call__` (from inside the submission — single-file
constraint respected) to run the proven v2 reduction chain per head on the
`[1,4800/1200/300,80]` genboxes outputs while still tiled, concatenating only
the tiny `[1,N,1]` results. The big untilizes, the 1 MB concat, and v2's own
tilize all drop out of the trace. Boxes tail is verbatim; parity argument is
unchanged from v2 (which measured bit-exact: 25,119 / 314 false twice).

Caveat: if genboxes' trailing permute already returns confs ROW_MAJOR, the
per-head `to_layout(TILE)` adds back what the original untilizes saved and v3
lands near v2. The A/B decides; if v3 is also break-even, L03 is closed as
rejected and the next lever is F2/F4.
