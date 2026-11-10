# Lever registry

Every optimisation gets an ID, its own branch and its own change doc. Branch name = `L<nn>-<what-changed>`. Commit messages start with `[Lnn]`. Each lever's code sits behind its switch in `LEVERS` in `challenge/harness/my_submission.py`, so it can be turned off without editing code (`LEVERS="L02=0"`).

Status: **planned** → **coded** (on its branch, not measured) → **measured** (numbers in `results/LOG.md`) → **merged** (in `main`) or **dropped** (with the reason in its doc).

| ID | Branch | What it changes | Expected gain | Status | Doc |
|---|---|---|---|---|---|
| L00 | `L00-base-framework` | Fast reference + lever switches, stage timer, two-photo check, `run_eval.sh` + results log | none (baseline 17.8 ms) | coded | [L00](changes/L00-base-framework.md) |
| NMS | `vectorized_nms_claude` | Box step: torchvision batched NMS + vectorized conversion, output matches reference (colleague's version: `vectorized_nms`) | several ms (box step) | coded | [IDEA_1 claude](IDEA_1_VECTORIZED_NMS_claude.md) |
| L01 | `L01-thread-limits` | One OpenCV/torch thread per worker; decoder count from the core count | 0–2 ms, more if threads were fighting for cores | coded | [L01](changes/L01-thread-limits.md) |
| L02 | `L02-overlap-postprocess` | Card runs chunk i+1 while the host works out boxes for chunk i | up to the box step's share per frame | coded | [L02](changes/L02-overlap-postprocess.md) |
| L03 | `L03-convert-in-workers` | `ttnn.from_torch` and `get_model_result` moved to worker threads | ~2 ms on the main thread | planned | |
| L04 | `L04-decode-reduced` | Half-size JPEG decode for photos with short side ≥ 400 (lossy) | ~22% of decode (laptop) | planned | |
| L05 | `L05-device-argmax` | max/argmax over classes on the card; read back 6300×2 instead of 6300×80 | part of ~1.9 ms readback | planned | |
| L06 | `L06-uint8-upload` | Copy uint8 to the card, convert there | part of ~2 ms conversion + copy | planned | |
| L07 | `L07-batch2` | Two frames per card call | unknown | planned | |

Accuracy levers (change the output on purpose; need mAP checks) will be numbered `A01`, `A02`, ... (multi-label decoding, overlap threshold sweep, resize interpolation).
