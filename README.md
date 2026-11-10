# SEMICON-Hackathon: Make A Camera See Fast Enough To Act On

**TUM Venture Labs × SEMICON Europa Hackathon (Munich, 10–12 November 2026)**  
**Track:** Chip Architecture & Integration — Sponsored by **Tenstorrent**

---

## 📄 Documents & Challenge Resources

The challenge supervisor provided two official specification documents located in the repository root:

1. **[`NextSemi Hackathon_Tenstorrent_Challenge.pdf`](./NextSemi%20Hackathon_Tenstorrent_Challenge.pdf)** — *The Official Challenge Briefing*
   - **Scenario:** A self-driving vehicle moving at 50 km/h moves 13.9 mm per millisecond of delay. At 69.1 ms baseline, the car travels ~96 cm blind before detecting objects.
   - **Target:** Reduce latency from **69.122 ms to < 15 ms** on Tenstorrent Wormhole n150 hardware with minimal accuracy degradation.
   - **Evaluation Criteria (30 points total):**
     - Technical Feasibility (1x, 5 pts)
     - Innovation (1x, 5 pts)
     - Market Potential (1x, 5 pts)
     - Business Viability (1x, 5 pts)
     - **Challenge-specific: Speed & Accuracy Result (2x, 10 pts)** (Tested on 800 unseen photographs at hour 30)
   - **Submission Requirements:** Single submission file (`my_submission.py`), 1-page trade-off curve log, declared settings (resolution, thresholds, model).

2. **[`Tenstorrent_Technical Details_Fact Pack.pdf`](./Tenstorrent_Technical%20Details_Fact%20Pack.pdf)** — *Hardware & Resource Guide*
   - **Cloud Environment:** Dedicated browser-accessible cloud instances with Tenstorrent Wormhole n150 cards.
   - **Filesystem Layout on Card Instance:**
     - `~/challenge/README.md` — Authoritative instructions and known traps.
     - `~/challenge/harness/` — Scoring harness and `submission_template.py`.
     - `~/challenge/bench/` — Timing scripts (`run_bench.sh`, `run_levers.sh`, `run_loop.sh`).
     - `~/challenge/practice/` — 200 labeled practice photographs.
     - `~/challenge/sample/` — 2 test photographs for quick sanity checks.
     - `~/challenge/webcam_demo/` — Pitch camera demo.
     - `~/tt-metal/` — TT-Metalium and TT-NN source tree.
   - **Official Docs:** [docs.tenstorrent.com](https://docs.tenstorrent.com) & Discord community.

---

## ⚡ Baseline Measurements (Wormhole n150, 320x320)

| Stage | Baseline Time | Task |
|:---|:---|:---|
| 1. Unpack photo | 2.135 ms | JPEG decode to raw pixels |
| 2. Resize & convert | 0.827 ms | Resize & reorder color channels |
| 3. Hand to card | 2.903 ms | Transfer host RAM to card memory |
| 4. Run detector | **61.062 ms** | YOLOv4 execution on card (TTNN) |
| 5. Work out boxes | 2.195 ms | NMS & bounding box decode |
| **Total Pipeline** | **69.122 ms** | End-to-end frame processing (Target: **< 15.0 ms**) |

---

## 🚀 First Steps on Instance

```bash
# 1. Read the on-device guide
cat ~/challenge/README.md

# 2. Record your baseline speed
cd ~/challenge/bench && bash run_bench.sh

# 3. Create your submission file
cp ~/challenge/harness/submission_template.py ~/challenge/my_submission.py
```
