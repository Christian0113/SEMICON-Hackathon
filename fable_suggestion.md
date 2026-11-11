# Fable suggestions: latency opportunities after milestone 8

**Author:** Claude Fable 5 analysis session, 2026-11-11 (for the SEMICON/Tenstorrent team)
**Starting point analysed:** `main` @ `d6f913e` — milestone 8 per-frame streaming pipeline
**Current official numbers** (`harness/evaluate.py`, practice set, fastest of 3):

| Metric | Value | Source |
|---|---|---|
| Latency | **5.301 ms/frame** (188.63 FPS) | `experiments/runs/10_streaming_pipeline.json` |
| mAP@50 | **0.4640** (baseline parity) | same |
| Detections | 25,119 | same |
| Card floor measured so far | **5.448 ms/frame** (`trace_2cq_pipelined`, bench_levers, 2026-10-26) | `challenge/bench/bench_levers.py` header |

Milestone 8 is **at the measured card-side floor**. The host (decode, NMS,
conversion) is fully overlapped and no longer the bottleneck. Every further
millisecond must come from making the card's per-frame work cheaper. The good
news: that floor is *not* pure convolution time — it still contains removable
work. This guide ranks the remaining levers by expected gain / risk / effort,
with implementation sketches and acceptance tests for each.

---

## 1. Where the 5.3 ms actually goes

With `MultiCQTracedModelOverlappedInputExecutor` (what
`PipelineConfig(use_trace=True, num_command_queues=2,
all_transfers_on_separate_command_queue=False)` selects), the per-frame
schedule is (tt-metal `models/tt_cnn/tt/executor.py`, identical at `3a0e398`):

```
CQ1 (writes only):  wait(op_event) → H2D input 614 KB → record write_event
CQ0 (everything else, SERIAL):
    wait(write_event)
    reshard DRAM→L1 (input)
    execute_trace            ← the conv model, incl. pad 3→16ch, permute, genboxes
    D2H read boxes+confs     ← 1.06 MB over PCIe, ON THE COMPUTE QUEUE
```

Per-frame CQ0 time = reshard + trace + **output read**. The output read is:

- `confs` `[1, 6300, 80]` bf16, ROW_MAJOR, L1 → **1,008 KB**
- `boxes` `[1, 6384, 1, 4]` bf16 → **51 KB**

≈ **1.06 MB/frame on CQ0**. At the 2–3.5 GB/s effective D2H rates typical of
this hugepage-staged host path, that is **≈ 0.3–0.5 ms of every frame during
which the Tensix cores are idle**. Nobody has measured the read in isolation
yet — do that first (see §2, F0).

Second removable item: the first three ops *inside* the trace are input
plumbing, not convolution — `ttnn.pad` (3→16 channels), `ttnn.permute`
(NCHW→NHWC), `ttnn.reshape` (`models/demos/yolov4/tt/yolov4.py:44-56`). All
three are pure data movement that the host can do bit-exactly for free in the
decode threads (which have huge headroom: 24 threads for a 5.3 ms cadence).

Third: the trace itself has never been profiled op-by-op. The run logs already
hint at suboptimal conv configs (`act_block_h_override 128 is not a valid
override ... 64 was selected`, repeated matmul memory-config mismatch warnings,
`tilize ... legacy sharded` fallbacks).

---

## 2. Ranked levers

| # | Lever | Expected gain | mAP risk | Effort | Status |
|---|---|---|---|---|---|
| F0 | Measure: D2H read cost + device op profile | enabler | none | ~1 h | not done |
| F1 | Merge `trunc_confs` (on-device max/argmax, 40× smaller read) into the milestone-8 streaming path | 0.3–0.5 ms | low (argmax tie-breaks) | ~2–4 h | both halves exist, never combined |
| F2 | Host-side pad+permute: feed NHWC 16-channel input, delete 3 ops from the trace | 0.1–0.3 ms | **none** (bit-exact) | ~3–5 h | new |
| F3 | Move D2H reads to CQ1 (pipelined-IO event protocol) | 0.3–0.5 ms, **overlaps with F1** | none | ~4–6 h | new |
| F4 | Batch-2 trace | potentially 0.5–1.5 ms/frame | low if PCC passes | 1 day, may dead-end | new, uncertain |
| F5 | Targeted conv tuning from the F0 profile | 0.2–0.8 ms (speculative) | none–low per knob | 0.5–1 day | blocked on F0 |
| F6 | bf8_b activations (wire the inert dtype lever) | unknown | **high** (confs PCC floor is 0.9) | ~0.5 day | last resort |
| F7 | Knob sweep IN_FLIGHT/RING/POST_WORKERS/DECODERS | ±0.1–0.2 ms | none | ~1 h card time | listed in IDEA_8, not swept |

Realistic combined outcome for F1+F2 (+F7): **≈ 4.6–4.9 ms/frame**. Stretch
with F4 or F5 landing: **≈ 3.8–4.5 ms**. Per team convention, each lever gets
its own `L<nn>-<slug>` branch off `main`, one lever per branch, documented in
`docs/changes/` and measured only via `harness/evaluate.py` on the practice set.

**Acceptance gate for every lever, no exceptions:** mAP@50 = 0.4640 and
`detections_returned` = 25,119 on the practice set, no `PROBLEM:` lines,
fastest-of-3 reported.

---

## 3. F0 — Measure before cutting (do this first, ~1 h of card time)

Two numbers decide how the next two days are spent:

**(a) The output-read cost in isolation.** After `pipeline.compile`, time the
read alone against trace-only replay:

```python
# trace only, no read: N × execute_trace, synchronize once
t0 = time.perf_counter()
for _ in range(300):
    ttnn.execute_trace(device, executor.trace_id, cq_id=0, blocking=False)
ttnn.synchronize_device(device)
trace_only = (time.perf_counter() - t0) / 300

# trace + read (what production does): add the two copy_device_to_host calls
```

`trace+read − trace_only` is the exact prize for F1/F3. If it is < 0.15 ms,
deprioritise both and go straight to F2/F4/F5.

**(b) An op-level profile of the trace.** Run one evaluated frame with the
device profiler (`TT_METAL_DEVICE_PROFILER=1`, or build with Tracy and use
`tt_metal/tools/profiler`) and produce a table: op name → device µs. This
tells you whether the ~4.8 ms of real compute is dominated by the big early
convs (then F4/F6 are the only routes) or has fat tails in data movement ops,
maxpools, upsample/concat, genboxes (then F5 is cheap meat). Attach the table
to `docs/changes/` so every later lever can cite it.

---

## 4. F1 — Merge `trunc_confs` into the streaming pipeline («milestone 9»)

**What exists:** `origin/trunc_confs` (commits `9c11d10`, `827c339`) appends
`ttnn.max(confs, dim=-1, keepdim=True)` + `ttnn.argmax(confs, dim=-1,
keepdim=True)` to the model inside the trace and transfers `[1,6300,1]` max +
`[1,6300,1]` argmax instead of the `[1,6300,80]` scores: **1,008 KB → ~38 KB**
(plus boxes 51 KB). It was written against the *pre-streaming* submission and
has never been measured on the card, nor combined with milestone 8.

**Why it composes cleanly with streaming:** the streaming path only needs the
output schema to change shape. `Pipeline.preallocate_output_tensors_on_host`
builds slots from the schema, and the submission's `_copy_struct_to_host`
already walks nested lists, so a 3-tensor output `(boxes, max_confs, max_ids)`
works without touching the scheduler. `_retire` then calls the trunc-aware
`fast_detections(..., max_ids=...)` from that branch. Host NMS gets cheaper
too (the 80-class max disappears from the POST workers).

**The one thing to check before trusting it (on-card, 10 min):** time
`ttnn.argmax` on `[1,6300,80]` standalone. argmax is a generic reduction and on
some versions runs embarrassingly slowly on a single core. If argmax costs
more on CQ0 than the ~0.3–0.5 ms read it saves, the lever inverts. Fallbacks
if so: (i) keep `ttnn.max` only (13 KB) and transfer the class *scores* for
surviving anchors — no: requires device gather, skip; (ii) do F3 instead, same
prize, no new device ops.

**Parity watch-item:** `torch.max` returns the *first* index on ties;
device argmax may return another. bf16 scores quantize coarsely, so exact ties
are not hypothetical. A changed class id changes a detection. The acceptance
gate (detections == 25,119, mAP 0.4640) decides; if it fails by a handful of
detections, compare per-image against milestone 8 and check they are tie
flips, then decide with the team whether jury parity rules allow it.

---

## 5. F2 — Pull pad/permute out of the trace (bit-exact, zero risk)

`TtYOLOv4.__call__` starts with: pad channels 3→16 with zeros → permute
NCHW→NHWC → reallocate → reshape to `[1, 1, H*W, 16]`
(`yolov4_pinned_3a0e398/models/demos/yolov4/tt/yolov4.py:44-56`). These run
on the card *every frame inside the trace*.

The host decode threads can produce exactly that final layout instead:

1. Extend the milestone-8 LUT path to write into a preallocated
   `(1, 320*320, 16)` int16 buffer: 3 RGB channels via the LUT gather, 13
   channels zero (memset once, reused). This is NHWC flattened — the same
   element order the device reshape produces. Zero-padding and transposition
   are bit-exact by construction, so the existing setup-time parity check
   (`_build_input_lut`) extends naturally: compare against
   `to_torch(pad+permute+reshape(reference input))`.
2. Open the pipeline with an input memory config for the `[1, 1, 102400, 16]`
   host tensor (the infra's `_setup_l1_sharded_input` already *computes* its
   shard config with `c=16` — reuse that logic for the new shape).
3. Subclass/copy `TtYOLOv4.__call__` minus the first block, so `down1` gets
   the resharded tensor directly. The tt-metal tree on the instance is
   editable; keep the edit in a fork branch of `~/tt-metal` or monkey-patch
   the method from the submission (smaller blast radius, survives harness
   copying).

**Cost:** H2D grows 614 KB → 3.3 MB/frame, but it rides CQ1, which currently
carries 614 KB per 5.3 ms and has ~10× headroom. Decode threads do a little
more memory traffic; 24 threads at a 5.3 ms cadence won't notice.

**Expected:** whatever pad+permute+reshape cost in the F0 profile — a
plausible 0.1–0.3 ms. Even at the low end it's free latency with *provably*
identical output, the only lever in this list with literally zero accuracy
risk.

---

## 6. F3 — Output reads on CQ1 (only if F1's argmax disappoints)

Same prize as F1 (idle-read time off CQ0), different route: keep the full 1 MB
read but issue it on CQ1. The stock `MultiCQTracedModelPipelinedIOExecutor`
(`all_transfers_on_separate_command_queue=True`) does exactly this but
**rejects multi-output models** (`_create_output_schema` raises on lists), so
YOLOv4's `(boxes, confs)` needs one of:

- **Patch the executor** (copy it into the submission, as the streaming path
  already half-does): allocate *two* persistent DRAM output tensors, after
  `execute_trace` reshard both L1 outputs → DRAM on CQ0 (cheap, on-device),
  then `copy_device_to_host_tensor` both from DRAM on CQ1. Follow the stock
  four-event contract: CQ0 must `wait(read_event)` before the next trace
  overwrites the outputs, and `read_event` is recorded on CQ1 *after* the D2H
  is enqueued (that is what `signal_read_complete` exists for — the streaming
  submission already calls it in the right place).
- Or concat boxes+confs into one flat tensor on device and un-concat on host
  — more trace surgery for the same effect; only worth it if the reshard
  variant fights you.

Note F1 and F3 are near-substitutes: after F1 the read is ~90 KB and moving it
off CQ0 is worth maybe 0.05 ms. Pick one based on F0(a) and the argmax timing;
do not spend card time landing both unless both come out free.

---

## 7. F4 — Batch-2 trace (highest variance, biggest possible win)

The head/neck layers run on 20×20 and 10×10 grids — at batch 1 they cannot
fill 64 cores and are dispatch/latency-bound, not compute-bound. Batch 2
doubles the useful work per trace replay and halves per-frame reshard +
trace-launch overhead. The scored metric is average ms/frame over 800 photos
(throughput), so pairing frames is legitimate; latency per *pair* just needs
`out[path]` filled for both.

Cheap feasibility probe (half a day, kill it fast if it fights back):

1. `YOLOv4PerformanceRunnerInfra(device, 2, ...)` — the infra takes
   `batch_size` and builds its torch input from it. See if
   `create_yolov4_model_parameters` + `TtYOLOv4` compile and `validate()`
   passes PCC at batch 2, resolution 320. Watch L1: 640×640 batch 1 (4× the
   rows) hits TT_FATAL; batch 2 at 320 is only 2× — it may fit, it may not.
2. Double `TRACE_REGION_SIZE`; adjust `get_model_result` unpadding (the anchor
   slicing is on dim 1/2, batch stays dim 0 — check shapes, don't assume).
3. In the streaming loop, feed pairs; pad the odd frame with a duplicate and
   drop its result.

If `validate()` fails or L1 overflows on the early convs, stop — don't sink
hackathon hours into per-layer resharding. If it passes, this is plausibly the
single biggest lever on the board (per-frame time of a batch-2 replay should
land well under 2× the batch-1 replay precisely because the small layers
amortize).

---

## 8. F5 / F6 — Trace-interior tuning (gated on the F0 profile)

**F5, targeted and safe:** the warnings in every run log are a to-do list:
- `act_block_h_override 128 ... 64 was selected` (`conv2d_utils.cpp:466`) —
  the override in `conv_args` for that conv is invalid for its shard shape;
  setting a valid one (or 0) changes blocking and can win real time on that
  conv.
- Repeated matmul `Mismatch between computed MemoryConfig ... Using computed
  config` — the provided shard grids are being discarded; align the configs in
  `tt/model_preprocessing.py` with what the op computes and the reshards those
  mismatches imply can disappear.
- `tilize ... legacy sharded` fallbacks — check if a supported layout avoids
  the legacy path.
Attack only the ops the F0 profile says are > 0.1 ms. Each change: PCC
validate + full evaluate.py gate.

**F6, last resort:** activations are hardcoded bf16 (`_setup_l1_sharded_input`)
and conv output dtype defaults to bf16; weights are already bf8_b with LoFi
math (`common.py:create_conv2d_config`), so compute is already cheap — bf8_b
activations mainly halve inter-op data movement. The confs PCC floor is
already only 0.9, i.e. accuracy headroom is thin. Only try if F0 shows the
trace is data-movement-bound, apply to the backbone only (keep heads/genboxes
bf16), and let the mAP gate decide. Expect to revert.

---

## 9. F7 — Scheduling knob sweep (cheap, do alongside anything)

One evaluate.py run each, one knob at a time, fastest-of-3:
`IN_FLIGHT` ∈ {2, 4, 6, 8} (RING = IN_FLIGHT+4), `POST_WORKERS` ∈ {4, 8, 16},
`DECODERS` ∈ {16, 24, 32}, and `LOOKAHEAD` 64 → 128 for the 800-photo run
(decode ramp at the start of each repeat is amortized over 4× more frames at
hour 30 — verify it's a non-issue, don't assume).

---

## 10. Things deliberately *not* on the list

- **Lower resolution / 640**: only 320 and 640 are wired (`get_model_result`
  hardcodes both); 640 fails validation at PCC 0.768 on this tree; anything
  else is a model-surgery project, and any resolution change moves mAP.
- **Decode shortcuts** (`IMREAD_REDUCED_COLOR_2`, PIL-SIMD, nvJPEG): decode is
  fully overlapped and not on the critical path; reduced decode also changes
  pixels → mAP gate fails.
- **Clock/power**: AICLK is at the n150 busy clock; nothing user-settable.
- **Harness games**: setup and warmup are already excluded; fastest-of-3 is
  the rule; the 3.3% spread is instance noise, not a lever.
- **Skipping more frames via saliency**: the host gate (`std < 3.0`) already
  catches the empty traps; raising it risks real photos on the unseen 800.

## 11. Protocol reminders (team conventions)

- One lever per `L<nn>-<slug>` branch off GitHub `main`; `[Lnn]` commit
  prefixes; a `docs/changes/<branch>.md` per branch; measure **only** with
  `harness/evaluate.py` fastest-of-3 on the practice set; merge only winners.
- Every merge must keep mAP@50 = 0.4640 and detections = 25,119 (or document
  exactly which images changed and why before the team accepts a delta).
- Keep `STREAMING`/`FAST_INPUT`-style kill switches in every lever so a
  failure at hour 30 degrades to the previous milestone instead of a zero.

*Numbers marked "expected" are estimates from transfer sizes and queue
structure, not measurements. F0 exists to replace them with facts before
anyone spends a day on F4–F6.*
