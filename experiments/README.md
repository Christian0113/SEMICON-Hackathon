# Optimization Experiments & Milestone Archive
**TUM Venture Labs × SEMICON Europa Hackathon — Tenstorrent Track**  
**Device:** Tenstorrent Wormhole n150 (`board_id: 010001861190402c`, PCIe Gen4 x16, 1 GHz AICLK)  
**Host:** 96-core server (Ubuntu 22.04, Python 3.10.19, PyTorch 2.11.0+cpu, TT-Metal `9f9cd4fd590`)

---

## 1. Executive Engineering Progress Table

Every measurement is taken with the official competition harness (`challenge/harness/evaluate.py`) across the 200 practice images with 3 repeated runs (fastest counts).

| Step | Milestone / Lever | Latency (ms) | Speedup vs Prev | Throughput (FPS) | mAP@50 | Total Detections | Mean Power | Submission Source | Result File |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **0** | **Plain Baseline**<br>Un-traced sequential dispatch | **`85.815 ms`** | — | `11.65` | **`0.4640`** | 25,176 | 24.2 W | [`submissions/01_submission_baseline.py`](submissions/01_submission_baseline.py) | [`runs/01_reference_plain.json`](runs/01_reference_plain.json) |
| **1** | **Traced 2CQ Baseline**<br>Hardware graph trace + 2 command queues + 4 decoder threads | **`17.803 ms`** | **-68.01 ms** (4.82×) | `56.17` | **`0.4640`** | 25,176 | 25.9 W | [`submissions/02_submission_fast_reference.py`](submissions/02_submission_fast_reference.py) | [`runs/02_reference_fast.json`](runs/02_reference_fast.json) |
| **2** | **Vectorized C++ NMS v1**<br>Replaced Python 80-class loop with `torchvision.ops.batched_nms` | **`15.801 ms`** | **-2.00 ms** (1.13×) | `63.29` | **`0.4639`** | 25,133 | 25.3 W | [`submissions/03_submission_vectorized_nms_v1.py`](submissions/03_submission_vectorized_nms_v1.py) | [`runs/03_vectorized_nms.json`](runs/03_vectorized_nms.json) |
| **3** | **Host Concurrency Tuning**<br>`CHUNK=32`, `DECODERS=16`, `cv2.setNumThreads(1)` on 96 cores | **`14.938 ms`** | **-0.86 ms** (1.06×) | `66.95` | **`0.4639`** | 25,133 | 23.0 W | [`submissions/03_submission_vectorized_nms_v1.py`](submissions/03_submission_vectorized_nms_v1.py) | [`runs/04_vectorized_nms_chunk32.json`](runs/04_vectorized_nms_chunk32.json) |
| **4** | **Vectorized `.tolist()` & Clamping**<br>Batch C++ tensor dump + image border clamp + `_COCO_IDS` table | **`12.073 ms`** | **-2.86 ms** (1.24×) | `82.83` | **`0.4640`** | 25,133 | 31.1 W | [`submissions/04_submission_vectorized_nms_claude.py`](submissions/04_submission_vectorized_nms_claude.py) | [`runs/05_vectorized_nms_claude.json`](runs/05_vectorized_nms_claude.json) |
| **5** | **Empty Photo Saliency Gate**<br>Image-level gate at `EMPTY_THRESH=0.05` (prunes 14 noise detections) | **`11.537 ms`** | **-0.54 ms** (1.05×) | **`86.68`** | **`0.4640`** | 25,119 | 27.5 W | [`submissions/05_submission_empty_gate.py`](submissions/05_submission_empty_gate.py) | [`runs/06_vectorized_nms_claude_v2.json`](runs/06_vectorized_nms_claude_v2.json) |
| **6** | **Parallel from_torch & DECODERS=32**<br>Pre-build TTNN host tensors in 32 background threads | **`9.828 ms`** | **-1.71 ms** (1.17×) | **`101.75`** | **`0.4640`** | 25,119 | 32.2 W | [`submissions/06_submission_parallel_from_torch.py`](submissions/06_submission_parallel_from_torch.py) | [`runs/07_parallel_from_torch.json`](runs/07_parallel_from_torch.json) |
| **7** | **Double-Buffered Overlap & Clean GC**<br>Overlap card execution of chunk $i$ with CPU NMS of chunk $i-1$ | **`7.360 ms`** | **-2.47 ms** (1.34×) | **`135.88`** | **`0.4640`** | 25,119 | 28.6 W | [`submissions/07_submission_double_buffered_overlap.py`](submissions/07_submission_double_buffered_overlap.py) | [`runs/09_double_buffered_overlap.json`](runs/09_double_buffered_overlap.json) |
| **8** | **Per-Frame Streaming Pipeline**<br>Bounded `IN_FLIGHT=4` window + preallocated `RING=8` slots + event sync + fast bf16 LUT | **`5.301 ms`** | **-2.06 ms** (1.39×) | **`188.63`** | **`0.4640`** | 25,119 | 34.2 W | [`submissions/08_submission_streaming_pipeline.py`](submissions/08_submission_streaming_pipeline.py) | [`runs/10_streaming_pipeline.json`](runs/10_streaming_pipeline.json) |
| **9** | **Gemini Fix: On-Device Concat + bfloat16 Class-Max + Pipelining**<br>1 D2H tensor transfer + direct bfloat16 class-max + zero-copy view + feed-card-first + `RING=48` | **`3.344 ms`** 🚀 | **-1.957 ms** (1.59×) | **`299.09`** | **`0.4656`** | 24,829 | 41.5 W | [`submissions/09_submission_gemini_fix.py`](submissions/09_submission_gemini_fix.py) | [`runs/34_gemini.json`](runs/34_gemini.json) |

**Total Cumulative Speedup:** **25.66× faster** (from 85.815 ms down to 3.344 ms) with **higher accuracy (`0.4656` mAP)**! Throughput reaches **299.09 FPS**!

---

## 2. Pareto Frontier Curve

The Pareto curve across all milestones is plotted in [`plots/pareto_curve_current.svg`](plots/pareto_curve_current.svg):

```text
     label                     fps   ms/frame      mAP    watts   
  *  champion_300fps         299.1       3.34   0.4656     41.5   <- Frontier Champion (Dominates all on speed & accuracy)
     streaming_pipeline      188.6       5.30   0.4640     34.2   
     parallel_torch_building 135.9       7.36   0.4640     28.6   
     parallel_torch_building 101.8       9.83   0.4640     32.2   
     vectorized_nms_claude    86.7      11.54   0.4640     27.5   
     parallel_torch_building  85.4      11.70   0.4640     24.4   
     vectorized_nms_claude    82.8      12.07   0.4640     31.1   
     team-vectorized-nms      67.0      14.94   0.4639     23.0   
     team-vectorized-nms      63.3      15.80   0.4639     25.3   
     ref-fast                 56.2      17.80   0.4640     25.9   
     ref-plain                11.7      85.81   0.4640     24.2   <- as supplied

  * on the frontier: nobody beat them on speed and accuracy at once
```

---

## 3. Directory Structure

```text
experiments/
├── README.md                  # This master engineering record and table
├── submissions/               # Frozen Python code snapshots for each milestone
│   ├── 01_submission_baseline.py
│   ├── 02_submission_fast_reference.py
│   ├── 03_submission_vectorized_nms_v1.py
│   └── 04_submission_vectorized_nms_claude.py
├── runs/                      # Official raw evaluate.py JSON outputs
│   ├── 01_reference_plain.json
│   ├── 02_reference_fast.json
│   ├── 03_vectorized_nms.json
│   ├── 04_vectorized_nms_chunk32.json
│   └── 05_vectorized_nms_claude.json
└── plots/                     # Output SVG plots generated by plot_curve.py
    └── pareto_curve_current.svg
```

---

## 4. How to Reproduce or Re-plot

To regenerate the Pareto trade-off plot:
```bash
python challenge/harness/plot_curve.py \
  experiments/runs/01_reference_plain.json \
  experiments/runs/02_reference_fast.json \
  experiments/runs/03_vectorized_nms.json \
  experiments/runs/04_vectorized_nms_chunk32.json \
  experiments/runs/05_vectorized_nms_claude.json \
  --out experiments/plots/pareto_curve_current.svg
```
