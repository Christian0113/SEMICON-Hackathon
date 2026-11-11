# Lever branch log

One line per lever branch: status and official evaluate.py numbers (practice
set, fastest of 3). Details in `docs/changes/<branch>.md`.

| Branch | Lever | Status | ms/frame | mAP@50 | Detections |
|---|---|---|---|---|---|
| (main @ d6f913e) | milestone 8 streaming | merged | 5.301 | 0.4640 | 25,119 |
| L03-trunc-confs-streaming v1 | F1: on-device max/argmax × streaming | rejected (single-core ttnn.argmax) | 19.232 | 0.4639 | — |
| L03-trunc-confs-streaming v2 | F1: multicore "maxmin" id reduction | break-even (5.915 vs 5.880 off, same session) — not merged | 5.915 | 0.4640 | 25,119 |
| L03-trunc-confs-streaming v2 (10-rep) | F1: maxmin, clean A/B — run was mislabelled "headmax" (cloud file not pulled) | REJECTED: 5.701 vs 5.301 off, distributions disjoint (+0.4 ms) | 5.701 | 0.4640 | 25,119 |
| L03-trunc-confs-streaming v3 | F1: per-head "headmax" inside patched model | REJECTED: 5.735 vs 5.284 off (10-rep same-session, file verified), distributions disjoint (+0.45 ms). **L03 closed** | 5.735 | 0.4640 | 25,119 |
