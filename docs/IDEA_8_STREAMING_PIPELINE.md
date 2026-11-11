# Idea 8: Per-frame streaming pipeline (host/card overlap)

**File:** [`experiments/submissions/08_submission_streaming_pipeline.py`](../experiments/submissions/08_submission_streaming_pipeline.py)
**Starts from:** milestone 7 (`challenge/my_submission.py`, 7.360 ms/frame, mAP 0.4640, 25,119 detections)
**Status:** written on a laptop, NOT yet run on the card.

## 1. Correction to ARCHITECTURE.md §1.3

ARCHITECTURE.md says the 5.598 ms figure is "pure trace execution" and that the
remaining 1.76 ms (7.36 − 5.60) is the card-to-host PCIe transfer of `confs`.
The source code doesn't support that:

- `bench/bench_loop.py` times `model` as `pipeline.enqueue([host]).pop_all()`.
  So the 5.58 ms already includes the input upload, reshard, trace, **and** the
  output readback.
- `bench_levers.py` `trace_2cq_pipelined` measured **5.448 ms/frame** with the
  same 1 MB output read every frame.
- The output is `ROW_MAJOR` bf16 (`yolov4.py` `to_layout(..., ROW_MAJOR_LAYOUT)`),
  so there is no host-side untilize hiding in `to_torch` either.

So the card's floor *including* the D2H read is about 5.45–5.58 ms. The 1.8–1.9 ms
above that floor is **host/card serialisation in `run()`**, not PCIe.

## 2. Where the serialisation comes from (tt-metal `models/tt_cnn/tt/pipeline.py`, `executor.py`, identical at `3a0e398` and `9f9cd4fd590`)

`MultiCQTracedModelOverlappedInputExecutor._execute_single`, per frame:

```
CQ1: wait(op_event) -> copy_host_to_device_tensor(input) -> record write_event
CQ0: wait(write_event) -> reshard DRAM->L1 -> record op_event -> execute_trace
CQ0: .cpu(blocking=False)          # D2H read of boxes + confs (in Pipeline.enqueue)
```

`Pipeline.pop_all()` is `ttnn.synchronize_device()`, which drains **everything**.

Milestone 7 does `enqueue(32 frames)` → NMS of previous chunk → `pop_all()`.
- `enqueue()` writes all 32 inputs from the main thread. Each upload waits on the card
  for the previous frame's reshard, so the command queue backs up and `enqueue()`
  blocks while most of the chunk runs. The "overlapped" NMS mostly starts
  late.
- `pop_all()` then waits until the card is completely idle. Every chunk boundary
  is a full pipeline drain plus refill.

## 3. What milestone 8 changes

| # | Change | Targets | Accuracy impact |
|---|---|---|---|
| 1 | Per-frame streaming: drive `executor.execute()` one frame at a time and keep `IN_FLIGHT` frames queued on the card | card idle gaps at chunk boundaries | none (same frames, same order) |
| 2 | Per-frame `ttnn.record_event` after the output read + `ttnn.event_synchronize` instead of `synchronize_device` | full drains | none |
| 3 | `RING` preallocated host output slots (`copy_device_to_host_tensor`, non-blocking) | per-frame host allocation | none |
| 4 | NMS in `POST_WORKERS` threads (torch releases the GIL) | main thread stays free to feed the card | none |
| 5 | bf16 input via a 256-entry LUT built **by ttnn** at setup; BGR→RGB folded in | float32 pass + 2× memory traffic in decode threads | none: checked bit-identical at setup, auto-disabled otherwise |
| 6 | `torch.set_num_threads(1)` | 96-thread intra-op pools fighting 32 decode threads | none |

Ordering argument (why it's correct):
- CQ0 runs in order: trace k → read k → reshard k+1 → trace k+1, so read k
  finishes before the trace output buffer is overwritten.
- The host waits on the event recorded *after* read k before touching slot k.
- Slot k is reused only after `RING` more frames, and `RING > IN_FLIGHT`
  guarantees it was already converted (`get_model_result` returns new tensors).

## 4. How to test on the instance

```bash
cd ~/challenge
SUB=~/SEMICON-Hackathon/experiments/submissions/08_submission_streaming_pipeline.py
# 0. plumbing + PROBLEM lines
python harness/evaluate.py --submission $SUB --images ~/tt-metal/models/demos/yolov4/resources \
  --annotations sample/instances_sample.json --out /tmp/check8.json
# 1. practice: must match milestone 7 exactly on mAP (0.4640) and detections_returned (25,119)
python harness/evaluate.py --submission $SUB --images practice/images \
  --annotations practice/instances.json --out runs/10_streaming.json
```

Acceptance: no `PROBLEM:`, mAP 0.4640, `detections_returned` == milestone 7.
If detections differ, set `FAST_INPUT = False` first, then `STREAMING = False`,
to isolate which change did it.

Knobs to sweep (one at a time, practice set, `--repeats 3`):
`IN_FLIGHT` ∈ {2, 4, 8}, `POST_WORKERS` ∈ {4, 8, 16}, `DECODERS` ∈ {16, 24, 32}.

**Expected:** approaching the 5.45 ms card floor (about 180 FPS) if the host keeps up.
That's an estimate, not a measurement.

## 5. Next device-side levers (after this one is measured)

Once the host is fully hidden, the card's serial CQ0 path is the limit:
`reshard + trace (~5.0 ms?) + D2H 1 MB`. In order of value/risk:

1. **On-device max/argmax over the 80 classes** (ARCHITECTURE Lever 1). This is still
   worthwhile, but for a different reason than stated there. It shortens the D2H read
   that sits *in series* with the next trace on CQ0, and it removes the host-side
   `to_torch` + float16 + `max` work. Measure the D2H share first with a
   trace-only timing loop (no `.cpu()`) against `bench_levers` `trace_2cq_pipelined`.
2. **`all_transfers_on_separate_command_queue=True`** (`MultiCQTracedModelPipelinedIOExecutor`)
   moves the D2H read to CQ1 so it overlaps the next trace. It needs a single
   output tensor, so boxes and confs must be packed into one on the card first.
3. The 640×640 fix and model-level levers (bfloat4_b, pruning) change accuracy
   and need PCC/mAP gating as described in PLAN.md.
