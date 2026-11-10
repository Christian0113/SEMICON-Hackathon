# L00: base framework

**Branch:** `L00-base-framework`, started from `main` at `5b132a4`
**Status:** coded, compiles, harness selftest passes on the laptop; not yet run on the card
**Expected effect on speed:** none. This is the starting point every lever is measured against (fast reference, 17.8 ms/frame, mAP 0.464, 25,176 detections).

## What changed and why

| File | Change | Why |
|---|---|---|
| `challenge/harness/my_submission.py` | Replaced the empty template with `submission_reference_fast.py`, with identical behaviour | Every lever starts from the fastest working version we have. |
| same | `LEVERS` dict + `LEVERS="L01=1,L02=0"` env override + `lever(name)` helper | The jury runs one file. With switches, each lever can be measured on and off after merging, for the change log. |
| same | `CHUNK` and `DECODERS` read from env (defaults 16 and 4, as in the reference) | Sweeps without editing code. |
| same | `STAGE_TIMES=1` prints ms/frame per stage of `run()`: `decode_wait`, `from_torch`, `card+readback`, `boxes` | Shows where the host time goes. Off by default: a few `perf_counter` calls when on, none when off. |
| same | Building the host tensors moved into `_to_host()` (same code) | Gives L03 one place to change. |
| same | `ttnn`/model imports guarded (`HAS_TTNN`) | Tools can import the file on a laptop. `setup()` still fails clearly without a card. |
| `tools/liveness_check.py` | Runs giraffe and dining room through `run()` together and one at a time; fails unless the boxes differ and do not depend on batch order | The README's stale-input trap. Run after every change to how frames reach the card. |
| `tools/run_eval.sh` | Practice-set `evaluate.py` run → `results/<tag>.json` + one line in `results/LOG.md` (branch, commit, LEVERS env, ms, each run, mAP, detections, disturbed, problems). Warns about uncommitted changes and other card jobs | Every number has a file, a commit and a branch attached. `results/` is committed (`runs/` is git-ignored). |
| `docs/LEVERS.md` | Lever registry: ID, branch, change, expected gain, status | Branch names and numbering in one place. Kept out of `PLAN.md` so it doesn't conflict with `docs/plan-review`. |

## How to run (instance)
```bash
cd ~/SEMICON-Hackathon && git fetch && git switch L00-base-framework
python tools/liveness_check.py                 # expect: DIFFERENT ... Pass.
STAGE_TIMES=1 bash tools/run_eval.sh L00-stages --repeats 1   # where the time goes
bash tools/run_eval.sh                          # scored baseline for this framework
git add results/ && git commit -m "[L00] practice run" && git push
```

## Pass criteria
Same as the fast reference within noise: ~17.8 ms/frame, mAP 0.464, 25,176 detections, no `PROBLEM:`.

## Measured
| Run | ms/frame | mAP | detections | notes |
|---|---|---|---|---|
| L00 (STAGE_TIMES) | | | | per-stage split: |
| L00 | | | | |

## Risks
- `STAGE_TIMES` measures wall time on the main thread. `decode_wait` is only the part of decoding not hidden by the worker threads.
