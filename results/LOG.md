# Lever branch log

One line per lever branch: status and official evaluate.py numbers (practice
set, fastest of 3). Details in `docs/changes/<branch>.md`.

| Branch | Lever | Status | ms/frame | mAP@50 | Detections |
|---|---|---|---|---|---|
| (main @ d6f913e) | milestone 8 streaming | merged | 5.301 | 0.4640 | 25,119 |
| L03-trunc-confs-streaming v1 | F1: on-device max/argmax × streaming | rejected (single-core ttnn.argmax) | 19.232 | 0.4639 | — |
| L03-trunc-confs-streaming v2 | F1: multicore "maxmin" id reduction | awaiting card run | — | — | — |
