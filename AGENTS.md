# AI Agents Instructions & Hackathon Playbook

> **Challenge:** Make A Camera See Fast Enough To Act On  
> **Event:** TUM Venture Labs × SEMICON Europa Hackathon (Munich, Nov 10–12, 2026)  
> **Track Sponsor:** Tenstorrent (Hardware: Wormhole n150 Cloud Instance)

---

## 1. Challenge & Technical Objective

A moving vehicle at 50 km/h travels 13.9 mm per millisecond of delay. At the starting baseline latency of **69.122 ms** (~14.4 FPS), it moves ~96 cm blind before detecting an object. 
Our primary objective is to cut total pipeline latency to **< 15 ms (> 66 FPS)** on the Tenstorrent Wormhole n150 card with **zero or minimal accuracy loss** on 800 evaluation images.

### Baseline Latency Breakdown (Wormhole n150, 320x320)
| Stage | Baseline Time | Description | Optimization Opportunities |
| :--- | :--- | :--- | :--- |
| **1. Unpack photo** | 2.135 ms | Compressed JPEG to pixels | Fast decoders (TurboJPEG, OpenCV) |
| **2. Resize & convert** | 0.827 ms | Shrink to 320x320, color ordering | Fused preprocessing, SIMD / GPU/NPU |
| **3. Hand to card** | 2.903 ms | Transfer host RAM to card memory | Pinned memory, asynchronous DMA, batching |
| **4. Run detector** | **61.062 ms** | YOLOv4 execution on card (TTNN) | Layer fusion, INT8/FP8 quantization, pruning, kernel tuning in TT-Metalium |
| **5. Work out boxes** | 2.195 ms | NMS & bounding box decode | Vectorized NMS, score threshold filtering |
| **Total Measured** | **69.122 ms** | End-to-end frame latency | Pipelining / double-buffering stages |

---

## 2. Hardware & Environment Reference

All development and benchmarking runs on the Tenstorrent cloud instance:
* `~/challenge/README.md` — Authoritative challenge guide, traps, and guidelines.
* `~/challenge/harness/` — Scoring harness and `submission_template.py`.
* `~/challenge/bench/` — `run_bench.sh` (stage timing), `run_levers.sh`, `run_loop.sh`.
* `~/challenge/practice/` — 200 labeled practice images to measure speed vs accuracy tradeoffs.
* `~/challenge/sample/` — 2 sample images for instant verification of submission format.
* `~/challenge/data/` — Official reference logs, board specs, and firmware versions.
* `~/challenge/webcam_demo/` — Pitch demo using laptop camera on the card.
* `~/tt-metal/` — Tenstorrent TT-Metalium and TT-NN source and pre-built libraries.

### Key Commands on Remote Machine
```bash
# 1. Check initial baseline measurement:
cd ~/challenge/bench && bash run_bench.sh

# 2. Setup your submission file:
cp ~/challenge/harness/submission_template.py ~/challenge/my_submission.py

# 3. Test on practice set & plot tradeoff curve:
python3 ~/challenge/harness/score.py --submission ~/challenge/my_submission.py
python3 ~/challenge/harness/plot_curve.py
```

---

## 3. Multi-AI Collaboration Protocol (Codex + Claude Code + Antigravity)

Our 3-person team uses 3 different AI assistants. Every assistant must obey these rules:

### A. Modular Ownership & Branching
* **Never commit directly to `main`**. Work on isolated feature branches:
  * `feat/claude/<feature-name>`
  * `feat/codex/<feature-name>`
  * `feat/antigravity/<feature-name>`
* **Division of Focus**:
  * **Track A (Model & TTNN Kernels):** Quantization, TTNN layer optimization, model replacement/pruning in TT-Metalium (`run detector`).
  * **Track B (Host Pipeline & I/O):** Image decoding, fast preprocessing, asynchronous memory transfer to Wormhole n150 card (`unpack`, `resize`, `hand to card`).
  * **Track C (Post-processing, Scoring & Benchmarks):** Vectorized NMS, threshold tuning, submission packaging (`my_submission.py`), logging the trade-off curve, and the pitch/business deck.

### B. Code Formatting (Zero Merge Conflicts)
* Use standard Python 3.10+ conventions.
* Before committing, format code using:
  ```bash
  black . --line-length 100
  # or ruff format
  ```
* Do not reformat existing files unnecessarily if they belong to another module.

### C. Logging Required Deliverables
Keep an ongoing log of every optimization attempt. At hour 30, the judges require:
1. `my_submission.py` (filled configuration file).
2. **One-page log** detailing each change in order, measured speedup (via `run_bench.sh`), and accuracy impact (via practice set).
3. Stated final settings (input resolution, model variant, confidence and overlap thresholds).

---

## 4. Key References & Documents
* Supervisor Challenge Specification: [`NextSemi Hackathon_Tenstorrent_Challenge.pdf`](./NextSemi%20Hackathon_Tenstorrent_Challenge.pdf)
* Tenstorrent Hardware & Resource Pack: [`Tenstorrent_Technical Details_Fact Pack.pdf`](./Tenstorrent_Technical%20Details_Fact%20Pack.pdf)
* Official TT-Metalium Docs: `https://docs.tenstorrent.com`
