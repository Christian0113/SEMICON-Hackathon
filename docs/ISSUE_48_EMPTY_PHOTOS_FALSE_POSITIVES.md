# Issue: 48 Negative Trap (Empty) Photographs & False Positive Suppression

**Status:** Open  
**Severity:** Critical for Jury Evaluation  
**Relevant Deliverables:** Hour 30 Scorecard (`jury_hour30.sh`), mAP@50, Precision/Recall balance  

---

## 1. Problem Statement

In the official 800-photograph evaluation testset (`challenge/testset/images`):
- **752 photographs** contain annotated COCO objects.
- **48 photographs contain NO objects at all** (negative trap photographs).

When the jury runs [`jury_hour30.sh`](file:///Users/pazzclanker/Documents/SEMI/SEMICON-Hackathon/challenge/jury_hour30.sh), Step 5 explicitly extracts and highlights:
```python
print("false       ", p["false_detections"], 
      "  on empty photographs", p["false_detections_on_empty_images"], 
      " (as supplied: 14)")
```
In the untouched baseline detector:
- Total false detections across testset: `314` (on 200 practice images).
- False detections on empty trap photos: **14** (on 48 empty testset photos).

---

## 2. The Core Dilemma (mAP vs False Positives)

1. **The Low Confidence Trap (`CONF_THRESH = 0.001`):**
   - In standard COCO evaluation, `cocomap.py` walks every detection sorted descending by score to calculate 101-point interpolated precision.
   - Setting `CONF_THRESH = 0.001` recovers faint ground truth objects, keeping mAP at **`0.464`**.
   - However, at `CONF_THRESH = 0.001`, thousands of weak background noise anchors pass through. On empty photographs (where *any* prediction is an immediate false positive), this can cause dozens of spurious boxes to appear.
2. **The High Confidence Penalty (`CONF_THRESH = 0.25` or `0.30`):**
   - If we raise `CONF_THRESH` to `0.25` or `0.30` to eliminate false positives on empty images, mAP plunges from **`0.4639` down to `0.3762`** (-0.088 mAP drop!).

---

## 3. Hypotheses & Mitigation Strategies to Investigate

### Strategy A: Image-Level Background / Saliency Score Filter
- Compute the top detection score across the entire image:
  $$S_{\max} = \max_{k, c} (\text{confs}[k, c])$$
- If an image contains genuine objects, $S_{\max}$ is typically $> 0.50$ (or at least $> 0.30$).
- If an image is completely empty, all anchors have low scores ($S_{\max} < 0.15$ or $0.20$).
- **Action:** Analyze the score distribution of the 48 empty photos vs positive photos. If a clear separation threshold exists, return `[]` (empty list) when $S_{\max} < \tau_{\text{empty}}$, wiping out all 14+ false positives on empty photos at zero cost to positive recall!

### Strategy B: Class-Specific Dynamic Confidence Floors
- Certain classes (e.g., person, car, chair) generate frequent background false positives on textures.
- Calibrate per-class thresholds rather than a flat `0.001` across all 80 classes.

### Strategy C: Total Anchor Energy / Entropy Threshold
- On empty images, candidate scores are diffuse and uniformly low.
- Measure sum or mean of top-5 scores to gate whether an image is active or empty.

---

## 4. Next Steps & Experiments
1. [ ] Inspect false detection patterns on the practice set empty photos (`practice/instances.json` has empty image IDs).
2. [ ] Profile $S_{\max}$ distributions on positive vs empty practice images.
3. [ ] Implement a prototype empty-frame gating function in `my_submission.py` and benchmark mAP vs false detections on empty images.
