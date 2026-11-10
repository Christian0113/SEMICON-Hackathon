# Idea 1: Vectorized Post-Processing & Batched C++ NMS

**Branch:** `vectorized_nms`  
**Status:** In Progress / Ready for Testing  
**Goal:** Reduce CPU post-processing time from **`~6.5 ms`** to **`< 0.25 ms`** with **100% arithmetic parity** against baseline.

---

## 1. Problem Statement

In the baseline implementation ([`models/demos/yolov4/post_processing.py`](../tt-metal/models/demos/yolov4/post_processing.py)), the function `post_processing()` runs on CPU after device execution.

### The 4 Bottlenecks
1. **Host-to-Numpy Conversion & Memory Copies:**
   Takes PyTorch tensors from device/host, copies them to NumPy, creating memory copies on the main Python thread.
2. **80-Iteration Python Loop (`for j in range(num_classes):`):**
   Loops in Python 80 times for every single image.
3. **Pure-Python $O(N^2)$ NMS (`nms_cpu`):**
   Runs a Python `while order.size > 0:` loop. When `CONF_THRESH = 0.001`, thousands of low-confidence anchors enter this loop, requiring thousands of interpreted Python slicing and arithmetic operations.
4. **Terminal Stdout Flushes:**
   Calls `print()` 5 times on every frame, generating 12,000 terminal flushes across an 800-image evaluation run.

---

## 2. The Solution (Idea 1)

We implement `fast_post_process_v1()` directly in Python using PyTorch tensors and compiled C++ routines (`torchvision.ops.batched_nms` with fallback to vectorized PyTorch).

### Key Architectural Choices
1. **Exact 1:1 Parity:**
   In Idea 1, we preserve the exact same single-label semantics (`max` and `argmax` per anchor) as baseline `post_processing.py`. This guarantees zero divergence in bounding boxes, confidence scores, or category IDs.
2. **Vectorized Boolean Masking:**
   Instead of NumPy, we mask candidate boxes in a single vector operation:
   $$\text{mask} = \max(\text{confs}, \text{dim}=-1) > \text{conf\_thresh}$$
3. **Compiled C++ Batched NMS (`torchvision.ops.batched_nms`):**
   Runs NMS per class simultaneously in compiled C++ SIMD instructions.
4. **Clean Execution:**
   All stdout `print()` calls are eliminated.

---

## 3. Dedicated Benchmark: `challenge/bench/bench_postprocess.py`

To verify Idea 1 in complete isolation from the Tenstorrent hardware, we built [`challenge/bench/bench_postprocess.py`](../challenge/bench/bench_postprocess.py):

### What the Benchmark Does:
1. Generates / loads realistic raw YOLOv4 output tensors:
   - `boxes`: `[1, 6300, 1, 4]`
   - `confs`: `[1, 6300, 80]`
2. Runs baseline `post_processing()` across 300 iterations.
3. Runs `fast_post_process_v1()` across 300 iterations.
4. Tests at both:
   - `CONF_THRESH = 0.3` (fast path)
   - `CONF_THRESH = 0.001` (competition evaluation path)
5. **Numerical Parity Assertion:**
   Compares output bounding boxes, categories, and scores. Fails if there is any coordinate difference greater than $10^{-5}$.
6. **Reports Latency & Speedup.**

---

## 4. How to Run the Benchmark

On your remote instance (or any Python environment with PyTorch):

```bash
cd ~/challenge
python bench/bench_postprocess.py
```
