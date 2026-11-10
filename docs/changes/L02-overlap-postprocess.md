# L02: overlap the box step with the card

**Branch:** `L02-overlap-postprocess`, started from `L00-base-framework` (`6758fc3`)
**Status:** coded, compiles; not yet run on the card
**Switch:** `LEVERS["L02"]`, on by default on this branch. `LEVERS="L02=0"` gives back the L00 loop.

## What changed and why
In the fast reference, each chunk of 16 photos goes: send to the card → **wait** (`pop_all()`) → work out the boxes for all 16. While the boxes are worked out, the card has nothing to do. At `CONF_THRESH=0.001` the supplied box step is the most expensive host stage, so the card sits idle for a large share of every chunk.

L02 reorders the same work:

```
for chunk i:
    decode (already running on worker threads) -> host tensors
    pipeline.enqueue(chunk i)          # card starts, the call returns without waiting
    boxes for chunk i-1                # host works while the card computes
    pop_all()                          # wait for chunk i, keep its outputs
boxes for the last chunk
```

Ideal result: per frame ≈ max(card time, box time) instead of their sum.

| File | Change |
|---|---|
| `challenge/harness/my_submission.py` | `_run_overlapped()`, used by `run()` when L02 is on; `_boxes_into()` helper. The original loop stays for `L02=0`. |
| same | Stage timer splits `enqueue`, `boxes(overlapped)`, `card_wait`, `boxes(last)` |

### Why this is safe (read from the pinned tt-metal code)
- `Pipeline.enqueue()` (`models/tt_cnn/tt/pipeline.py`) without preallocated outputs creates **new** host tensors on each call (`device_struct.cpu(blocking=False)`), and `pop_all()` just synchronises the card and returns that list. So chunk i−1's outputs survive chunk i's readback. **`preallocate_output_tensors_on_host()` must not be used with this loop**: those buffers are reused on every `enqueue()`. This contradicts PLAN.md §2.A.2 / Phase 3, which suggest adding preallocation.
- `MultiCQTracedModelOverlappedInputExecutor._execute_single()` records events so the input copy for frame k+1 waits for frame k's ops. Ordering on the card is unchanged.
- Box results are identical to L00: same functions, same inputs, only the order on the host changes.

## How to run (instance)
```bash
cd ~/SEMICON-Hackathon && git fetch && git switch L02-overlap-postprocess
python tools/liveness_check.py                         # stale-input check: order of work changed
bash tools/run_eval.sh                                 # L02 on
LEVERS="L02=0" bash tools/run_eval.sh L02-off          # same file, lever off (= L00)
STAGE_TIMES=1 bash tools/run_eval.sh L02-stages --repeats 1
git add results/ && git commit -m "[L02] practice runs" && git push
```

## Expected effect
With the supplied (slow) box step: the card's ~5.6 ms/frame is hidden behind the box step, saving up to ~5 ms/frame. Combined later with the fast NMS branch, the box step gets cheap and this lever matters less. The two should be measured together after merging.

**What the stage timer tells you:**
- `enqueue` large (several ms per frame) → `enqueue()` blocks when the card's command queue is full, so the overlap is partial. Next step: smaller CHUNK (`CHUNK=4 bash tools/run_eval.sh L02-chunk4`).
- `card_wait` near 0 → the card finishes before the host does; the host is now the limit (box step, from_torch).

## Pass criteria
mAP 0.464 and 25,176 detections (identical to L00), liveness check passes, ms/frame below L00's on the same day.

## Measured
| Run | ms/frame | mAP | detections | notes |
|---|---|---|---|---|
| L02-off (= L00) | | | | |
| L02 | | | | |
| L02-stages | | | | enqueue / boxes(overlapped) / card_wait: |
| L02-chunk4 / chunk8 | | | | |

## Risks
- If `enqueue()` blocks until the card has run most of the chunk, the gain shrinks. The stage timer shows it.
- Extra host memory: one chunk of outputs is held while the next runs (16 × ~1 MB). Negligible.
