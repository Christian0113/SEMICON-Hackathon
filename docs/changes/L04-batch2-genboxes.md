# L04 — batch-2 trace × streaming pipeline (lever F4)

**Branch:** `L04-batch2-genboxes` (off `main` @ `82a2e43`)
**Files:** `challenge/my_submission.py`, `challenge/harness/my_submission.py` (identical copies)
**Status:** implemented, awaiting first card run.

## Why this lever, after L03

L03 (closed REJECTED, see `docs/changes/L03-trunc-confs-streaming.md`)
established the frame's cost structure: the ~1 MB D2H readback is nearly
free, and every small traced op costs real time (+6 tile ops = +0.4 ms).
Milestone 8's 5.30 ms frame is dominated by the **fixed per-op cost** of
~140 small post-head ops (3× TtGenBoxes + output tail) and ~110 convs.
The only lever that attacks per-op fixed cost without touching accuracy is
running more work per op: **two images per trace execution**.

- Convs: batch folds into the flattened NHW dim (`yolov4.py` line 56), so
  the conv op count per batch is unchanged — halved per frame.
- Post-head tail: patched batch-aware with the **same op count**, each op
  processing 2× the rows — halved per frame.
- If per-op fixed cost dominates (L03's finding), per-frame latency should
  drop meaningfully below 5.3 ms; if compute dominates, batch-2 is
  break-even and the lever closes.

## Feasibility (why it should compile)

Upstream never tests `batch_size > 1` per device (throughput comes from
multi-device mesh), but:

- `batch_size` plumbs through `YOLOv4PerformanceRunnerInfra` and
  `create_yolov4_model_parameters` (conv configs are inferred from a
  batch-sized torch input).
- The supported 640×640 batch-1 config runs folded NHW = 409,600 and
  genboxes extents up to 6400 rows on this very card. Batch-2 at 320×320 is
  NHW = 204,800 and genboxes extents up to 3200 — **every patched op shape
  is ≤ a shape the pinned tree already executes**.
- `YoloV4PipelineRunner` and the tt_cnn pipeline/executor are shape-agnostic.

## What was batch-1-hardcoded, and the fix (all inside the single file)

1. **Infra seed input**: upstream repeats the giraffe seed per *device*,
   not per batch (`performant_runner_infra.py` line 58), so the trace would
   capture batch 1. Fixed by repeating `infra.torch_input_tensor` to BATCH
   before `setup_dram_sharded_input`.
2. **`TtGenBoxes.__call__`**: derived `H = W = sqrt(HW)` and the anchor
   group from the folded extent; grid constants sized per image. Patched
   copy (`_l04_genboxes_call`): group/H/W from `HW // BATCH`, grid tensors
   re-tiled per image once at setup (`[1,3,1,HH] -> [1,3,1,BATCH·HH]`,
   images contiguous along the folded row dim). All slice bounds already
   use the folded extent, so the op sequence is otherwise verbatim.
3. **`TtYOLOv4.__call__` output tail**: reshape extents scaled by BATCH;
   head-3 per-anchor rows pad `100·B -> 128·B` (same scheme as upstream's
   100 → 128). Everything upstream of genboxes is verbatim.
4. **Host-side per-frame split** replaces `get_model_result` for BATCH>1:
   device output is `[1,1,12768,4]` boxes + `[1,12600,80]` confs in
   (head, anchor, image, pos) order. Precomputed row indices
   (`_frame_rows`) extract each image's 6300 rows in exactly milestone 8's
   order. **Verified off-card** by a torch emulation of the full device op
   sequence (slices/concats/permutes/reshapes/pad): per-image extraction
   from the B=2 layout is bit-identical to the B=1 output, and the B=1
   formula reproduces `get_model_result`'s unpad indices
   (0:6100 / 6128:6228 / 6256:6356) exactly.
5. **Streaming loop**: pairs consecutive non-empty frames into one host
   tensor (`[2,3,320,320]`, one `from_torch` per batch); an odd tail
   duplicates the last frame and discards the extra result. IN_FLIGHT and
   RING now count batches (so 4 in-flight batches = 8 frames). Empty/
   unreadable frames still skip the card entirely.

`BATCH = 1` applies no patches and is the exact milestone-8 path — the
same-session A/B reference. `BATCH = 2` is measurement mode: a compile
failure raises loudly (no fallback), so a completed run proves batch-2 ran;
teardown prints `[submission] batch this run: N`.

## Risks / open questions

1. **Conv config inference at NHW=204,800** may pick a different
   act_block/core split than batch-1 for some layer and fail to compile, or
   (worse) change per-pixel accumulation order → small numeric drift. The
   acceptance gate decides; a compile failure is loud.
2. **`ttnn.reshape` on TILE confs** `[1,3,200,80] -> [1,600,80]` (head 3)
   crosses tile padding exactly as the batch-1 100→300 case does — proven
   pattern, but at a new extent.
3. **Host assembly cost**: one 1.2 MB `torch.cat` + `from_torch` per batch
   on the main thread (batch-1 does from_torch on decoder threads). If the
   main thread becomes the bottleneck, move batch assembly to the decoders.
4. **Expected size of the win is unknown.** Hypothesis says 5.3 →
   ~3.8–4.5 ms/frame if fixed per-op cost dominates; break-even if compute
   dominates. The A/B decides.

## How to test on the instance

```bash
cd ~/SEMICON-Hackathon && git fetch && git checkout L04-batch2-genboxes && git pull
cp challenge/my_submission.py ~/challenge/guillem.py
grep -c "_l04_genboxes_call" ~/challenge/guillem.py   # MUST print >= 1 (0 = wrong file)
cd ~/challenge
python harness/evaluate.py --submission guillem.py --team L04-batch2 --repeats 10 \
  --images practice/images --annotations practice/instances.json --out /tmp/l04_b2.json
sed -i 's/^BATCH = .*/BATCH = 1            # images per trace execution; 1 = exact milestone 8/' guillem.py
python harness/evaluate.py --submission guillem.py --team L04-b1 --repeats 10 \
  --images practice/images --annotations practice/instances.json --out /tmp/l04_b1.json
```

**Acceptance:** no `PROBLEM:` lines, mAP@50 = 0.4640, `detections_returned`
= 25,119 (bit-parity with milestone 8), and batch-2 faster than the
same-session batch-1 reference by ≥ 0.15 ms on the fastest-of-10. Protocol
per L03 lessons: same-session `--repeats 10` A/B only; verify which file ran
with the grep above.

## Result

| Config | ms/frame | FPS | mAP@50 | detections | Verdict |
|---|---|---|---|---|---|
| milestone 8 (main, 2026-11-11 refs) | 5.301 / 5.284 | ~189 | 0.4640 | 25,119 | reference |
| L04 BATCH=2 | — | — | — | — | awaiting first card run |
