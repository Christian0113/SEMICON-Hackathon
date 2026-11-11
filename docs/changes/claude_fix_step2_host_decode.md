# claude_fix, step 2: host-side float32 box decode (milestone 9)

Everything is in `challenge/my_submission.py` (copied to `challenge/harness/my_submission.py`).
Switch: `HOST_DECODE = True`. Set it to `False` to get milestone 8 exactly.

What changed:
- The card trace stops after the three detection-head convolutions and returns the raw
  head tensors (`YoloV4HeadsRunner`). About 230 small genboxes/epilogue ops leave the trace.
- `decode_heads()` does sigmoid/exp/grid/anchor math on the host in float32, in the
  post-processing threads. Checked on a laptop against the PyTorch reference decode
  (`post_processing.YoloLayer`) on random logits: confs max diff 1e-7, boxes 3e-5.
- Setup runs the heads-only model once on giraffe.jpg and compares the host decode with the
  float32 PyTorch reference. It prints `host decode vs PyTorch reference: boxes PCC ..., confs PCC ...`
  and falls back to the supplied path if PCC is below 0.99 / 0.90 or anything raises.

Expected: faster per frame (the size of the gain is what this run measures), and
detections differ slightly from milestone 8 because coordinates are no longer rounded
to bf16 and float16. Accept if mAP >= 0.4640, no PROBLEM line, false detections on empty
photos not worse than 1.

## Run

```bash
cd ~/SEMICON-Hackathon && git fetch origin && git checkout claude_fix && git pull
cp challenge/my_submission.py ~/challenge/my_submission.py
cd ~/challenge
python harness/evaluate.py --submission my_submission.py \
  --images ~/tt-metal/models/demos/yolov4/resources --annotations sample/instances_sample.json --out /tmp/check9.json
python harness/evaluate.py --submission my_submission.py \
  --images practice/images --annotations practice/instances.json --team m9_host_decode --out run_m9.json
```

Send back: the `[submission]` lines printed during setup, and the final summary
(ms a frame, mAP, objects found, false detections, on empty photographs).
