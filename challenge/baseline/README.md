# Official Baseline Measurements

Recorded directly on the team's allocated Tenstorrent Wormhole n150 accelerator.

- **Host:** `app-c6af5617-deployment-f5f89f7cf-w2ghb`
- **Board Type:** Wormhole `n150 L` (`board_id: 010001861190402c`)
- **PCIe:** Gen4 x16, 12 GB GDDR6, AICLK 1000 MHz# Official Baseline Measurements

Recorded directly on the team's allocated Tenstorrent Wormhole n150 accelerator.

- **Host:** `app-c6af5617-deployment-f5f89f7cf-w2ghb`
- **Board Type:** Wormhole `n150 L` (`board_id: 010001861190402c`)
- **PCIe:** Gen4 x16, 12 GB GDDR6, AICLK 1000 MHz
- **Environment:** Ubuntu 22.04, Python 3.10.19, PyTorch 2.11.0+cpu, OpenCV 4.8.1, TT-Metal `9f9cd4fd590`
- **Timestamp:** 2026-11-10 13:20 – 13:24 UTC

---

## 1. Plain Untraced Baseline (`run_bench.sh`)
Folder: [`bench_results/`](bench_results/) (`bench_320.json`, 300 measured frames, 320x320)

| Stage | Mean (ms) | Median (ms) | p95 (ms) | p99 (ms) | Share |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **decode** | 2.362 | 2.357 | 3.007 | 3.117 | 3.3% |
| **prepare** | 0.829 | 0.819 | 0.899 | 0.966 | 1.2% |
| **upload** | 2.390 | 2.123 | 4.124 | 4.904 | 3.3% |
| **model** | 63.695 | 63.217 | 66.705 | 69.120 | 89.3% |
| **boxes** | 2.084 | 1.993 | 2.688 | 2.817 | 2.9% |
| **TOTAL** | **71.360** | **70.901** | **75.234** | **77.400** | **100%** |

- **Mean FPS:** `14.0 FPS` (worst frame `11.4 FPS`)

---

## 2. Traced Pipeline Baseline (`run_loop.sh`)
Folder: [`loop_results/`](loop_results/) (`loop_320_2cq.json`, 300 measured frames, 320x320, 2 Command Queues)

| Stage | Mean (ms) | Median (ms) | p95 (ms) | p99 (ms) | Share |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **decode** | 2.443 | 2.533 | 3.045 | 3.490 | 18.2% |
| **prepare** | 0.765 | 0.669 | 0.767 | 2.140 | 5.7% |
| **to_host** | 1.954 | 1.908 | 2.217 | 2.417 | 14.6% |
| **model (device)** | **5.598** | **5.578** | **5.728** | **5.868** | **41.8%** |
| **boxes** | 2.646 | 2.619 | 3.105 | 3.668 | 19.7% |
| **WHOLE LOOP** | **13.406** | **13.287** | **14.354** | **15.426** | **100%** |

- **Numerical Check:** PASSED (`validated: "passed"`, `trace_is_live: "passed"`)
- **Mean FPS:** `74.6 FPS` (worst frame `29.4 FPS`)

---


## 3. Official End-to-End Evaluation Baseline (`evaluate.py`)
File: [`01_reference.json`](01_reference.json) (`submission_reference.py`, 200 practice photographs, 3 repeats)

- **Latency:** **`85.815 ms / frame`** (fastest of 3 repeats: `85.815`, `86.511`, `85.971 ms`)
- **Throughput:** **`11.65 FPS`**
- **Spread:** `0.8%` (`disturbed: false`)
- **Accuracy (mAP@50):** **`0.464`**
- **Detections Returned:** 25,176
- **False Detections on Empty Images:** `1` (out of empty practice photographs)
- **Power Consumption:** Mean `24.24 W` (max `26.0 W`)
- **Problems:** `[]` (clean run)


## Dazhi's Measurements

Each configuration used 300 measured frames after 20 warmup frames. The V2 untraced runs were recorded at 13:21–13:22 UTC and the V2 traced runs at 12:26–12:27 UTC on 2026-11-10.

| Configuration | Mean (ms) | Median (ms) | p95 (ms) | p99 (ms) | Mean FPS | Worst-frame FPS |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Untraced 320×320 ([JSON](bench_results_V2/bench_320.json)) | 73.441 | 73.080 | 77.097 | 81.933 | 13.6 | 11.8 |
| Untraced 640×640 ([JSON](bench_results_V2/bench_640.json)) | 85.501 | 84.527 | 91.726 | 108.631 | 11.7 | 8.5 |
| Traced 320×320, 1 CQ ([JSON](loop_results_V2/loop_320_1cq.json)) | **12.855** | 12.653 | 14.047 | 16.942 | **77.8** | 35.7 |
| Traced 320×320, 2 CQ ([JSON](loop_results_V2/loop_320_2cq.json)) | 13.748 | 13.070 | 18.269 | 25.448 | 72.7 | 36.1 |
| Traced 640×640, 1 CQ ([JSON](loop_results_V2/loop_640_1cq.json)) | 32.621 | 31.186 | 40.775 | 45.493 | 30.7 | 21.0 |

Latency is the per-frame total for untraced runs and the whole-loop time for traced runs. CQ stands for command queue.

| Configuration | Decode (ms) | Prepare (ms) | Upload (ms) | To host (ms) | Model (ms) | Boxes (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Untraced 320×320 | 2.652 | 0.868 | 2.704 | — | 65.098 | 2.119 |
| Untraced 640×640 | 2.157 | 1.739 | 7.205 | — | 69.352 | 5.049 |
| Traced 320×320, 1 CQ | 1.978 | 0.695 | — | 1.929 | 5.565 | 2.686 |
| Traced 320×320, 2 CQ | 2.276 | 0.961 | — | 1.979 | 5.699 | 2.833 |
| Traced 640×640, 1 CQ | 2.211 | 4.938 | — | 7.055 | 11.874 | 6.542 |

Stage times are means. A dash indicates that the run did not report the stage separately.

**Numerical check:** Both traced 320×320 runs passed. The traced 640×640 run with 1 CQ reported `FAILED: 0.7678787216744429`. The trace liveness check passed in all three traced runs.

- **Environment:** Ubuntu 22.04, Python 3.10.19, PyTorch 2.11.0+cpu, OpenCV 4.8.1, TT-Metal `9f9cd4fd590`
- **Timestamp:** 2026-11-10 13:20 – 13:24 UTC

---

## 1. Plain Untraced Baseline (`run_bench.sh`)
Folder: [`bench_results/`](bench_results/) (`bench_320.json`, 300 measured frames, 320x320)

| Stage | Mean (ms) | Median (ms) | p95 (ms) | p99 (ms) | Share |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **decode** | 2.362 | 2.357 | 3.007 | 3.117 | 3.3% |
| **prepare** | 0.829 | 0.819 | 0.899 | 0.966 | 1.2% |
| **upload** | 2.390 | 2.123 | 4.124 | 4.904 | 3.3% |
| **model** | 63.695 | 63.217 | 66.705 | 69.120 | 89.3% |
| **boxes** | 2.084 | 1.993 | 2.688 | 2.817 | 2.9% |
| **TOTAL** | **71.360** | **70.901** | **75.234** | **77.400** | **100%** |

- **Mean FPS:** `14.0 FPS` (worst frame `11.4 FPS`)

---

## 2. Traced Pipeline Baseline (`run_loop.sh`)
Folder: [`loop_results/`](loop_results/) (`loop_320_2cq.json`, 300 measured frames, 320x320, 2 Command Queues)

| Stage | Mean (ms) | Median (ms) | p95 (ms) | p99 (ms) | Share |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **decode** | 2.443 | 2.533 | 3.045 | 3.490 | 18.2% |
| **prepare** | 0.765 | 0.669 | 0.767 | 2.140 | 5.7% |
| **to_host** | 1.954 | 1.908 | 2.217 | 2.417 | 14.6% |
| **model (device)** | **5.598** | **5.578** | **5.728** | **5.868** | **41.8%** |
| **boxes** | 2.646 | 2.619 | 3.105 | 3.668 | 19.7% |
| **WHOLE LOOP** | **13.406** | **13.287** | **14.354** | **15.426** | **100%** |

- **Numerical Check:** PASSED (`validated: "passed"`, `trace_is_live: "passed"`)
- **Mean FPS:** `74.6 FPS` (worst frame `29.4 FPS`)

---

## 3. V2 Measurements

Each configuration used 300 measured frames after 20 warmup frames. The V2 untraced runs were recorded at 13:21–13:22 UTC and the V2 traced runs at 12:26–12:27 UTC on 2026-11-10.

| Configuration | Mean (ms) | Median (ms) | p95 (ms) | p99 (ms) | Mean FPS | Worst-frame FPS |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Untraced 320×320 ([JSON](bench_results_V2/bench_320.json)) | 73.441 | 73.080 | 77.097 | 81.933 | 13.6 | 11.8 |
| Untraced 640×640 ([JSON](bench_results_V2/bench_640.json)) | 85.501 | 84.527 | 91.726 | 108.631 | 11.7 | 8.5 |
| Traced 320×320, 1 CQ ([JSON](loop_results_V2/loop_320_1cq.json)) | **12.855** | 12.653 | 14.047 | 16.942 | **77.8** | 35.7 |
| Traced 320×320, 2 CQ ([JSON](loop_results_V2/loop_320_2cq.json)) | 13.748 | 13.070 | 18.269 | 25.448 | 72.7 | 36.1 |
| Traced 640×640, 1 CQ ([JSON](loop_results_V2/loop_640_1cq.json)) | 32.621 | 31.186 | 40.775 | 45.493 | 30.7 | 21.0 |

Latency is the per-frame total for untraced runs and the whole-loop time for traced runs. CQ stands for command queue.

| Configuration | Decode (ms) | Prepare (ms) | Upload (ms) | To host (ms) | Model (ms) | Boxes (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Untraced 320×320 | 2.652 | 0.868 | 2.704 | — | 65.098 | 2.119 |
| Untraced 640×640 | 2.157 | 1.739 | 7.205 | — | 69.352 | 5.049 |
| Traced 320×320, 1 CQ | 1.978 | 0.695 | — | 1.929 | 5.565 | 2.686 |
| Traced 320×320, 2 CQ | 2.276 | 0.961 | — | 1.979 | 5.699 | 2.833 |
| Traced 640×640, 1 CQ | 2.211 | 4.938 | — | 7.055 | 11.874 | 6.542 |

Stage times are means. A dash indicates that the run did not report the stage separately.

**Numerical check:** Both traced 320×320 runs passed. The traced 640×640 run with 1 CQ reported `FAILED: 0.7678787216744429`. The trace liveness check passed in all three traced runs.

