# L01: thread limits

**Branch:** `L01-thread-limits`, started from `L00-base-framework` (`6758fc3`)
**Status:** coded, compiles; not yet run on the card
**Switch:** `LEVERS["L01"]`, on by default on this branch. `LEVERS="L01=0"` gives back L00 behaviour.

## What changed and why
OpenCV (`cv2.resize`, `cv2.cvtColor`, decode) and torch (`image_to_tensor`, `get_model_result`) each start their own thread pool, sized to the machine, for every call. The submission already decodes on `DECODERS` worker threads, so each worker's call can start another pool of threads. The result is many more runnable threads than cores, constant switching between them, and slower decode and box steps. With the work already parallel across photos, one library thread per worker is enough.

| File | Change |
|---|---|
| `challenge/harness/my_submission.py` | `_apply_thread_limits()`, called in `setup()` (not timed) when L01 is on: `cv2.setNumThreads(1)`, `torch.set_num_threads(1)`, and decoder count `max(2, min(8, usable_cores - 2))` |
| same | `_usable_cores()` uses `os.sched_getaffinity(0)` (the cores this container may use), not the host's total |
| same | `DECODERS` from the environment always wins, for sweeps |
| same | `setup()` prints CHUNK, decoders, usable cores and the cv2/torch thread counts actually in force |

## How to run (instance)
```bash
cd ~/SEMICON-Hackathon && git fetch && git switch L01-thread-limits
nproc                                              # record it in the Measured table
python tools/liveness_check.py
bash tools/run_eval.sh                             # L01 with automatic decoder count
LEVERS="L01=0" bash tools/run_eval.sh L01-off      # same file, lever off (= L00)
for d in 2 4 8; do DECODERS=$d bash tools/run_eval.sh L01-decoders$d; done
STAGE_TIMES=1 bash tools/run_eval.sh L01-stages --repeats 1
git add results/ && git commit -m "[L01] practice runs" && git push
```

## Expected effect
0–2 ms/frame, more if the stage timer shows `decode_wait` or `boxes` shrinking. Output identical to L00: same mAP and detection count, because only thread counts change.

## Pass criteria
mAP 0.464, detections 25,176 (identical to L00), ms/frame below L00's on the same day.

## Measured
`nproc` = 

| Run | ms/frame | mAP | detections | decoders | notes |
|---|---|---|---|---|---|
| L01-off (= L00) | | | | 4 | |
| L01 (auto) | | | | | |
| L01-decoders2 | | | | 2 | |
| L01-decoders4 | | | | 4 | |
| L01-decoders8 | | | | 8 | |

## Risks
- `torch.set_num_threads(1)` also applies to torch work on the main thread (`get_model_result`, and the NMS branch's torch ops). If `boxes` in the stage timer gets slower, try `torch.set_num_threads(2)`.
- If the instance has very few cores, `cores - 2` hits the floor of 2 decoders; the sweep shows whether 4 is better anyway.
