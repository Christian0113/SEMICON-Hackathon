# Lever branch log

One line per lever branch: status and official evaluate.py numbers (practice
set, fastest of N). Details in `docs/changes/<branch>.md`.

| Branch | Lever | Status | ms/frame | mAP@50 | Detections |
|---|---|---|---|---|---|
| (main @ d6f913e) | milestone 8 streaming | merged | 5.301 | 0.4640 | 25,119 |
| L03-trunc-confs-streaming v1 | F1: on-device max/argmax × streaming | rejected (single-core ttnn.argmax) | 19.232 | 0.4639 | — |
| L03-trunc-confs-streaming v2 | F1: multicore "maxmin" id reduction | REJECTED: 5.701 vs 5.301 off (10-rep same-session), +0.4 ms | 5.701 | 0.4640 | 25,119 |
| L03-trunc-confs-streaming v3 | F1: per-head "headmax" inside patched model | REJECTED: 5.735 vs 5.284 off (10-rep same-session, file verified), +0.45 ms. **L03 closed** | 5.735 | 0.4640 | 25,119 |
| L04-batch2-genboxes (BATCH=1) | F4 control: same file, batching off | reproduces milestone 8 exactly — the L04 scaffolding is free | 5.291 | 0.4640 | 25,119 |
| L04-batch2-genboxes (BATCH=2) | F4: two images per trace execution | awaiting first card run (`bench/probe_batch2.py` for correctness, then the 10-rep A/B) | — | — | — |
