# Official Baseline Measurements

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

