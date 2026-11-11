# Guide for implementing agents: speed AND accuracy, the "memory" track

**Written:** 2026-11-11 ~12:20 CET, against `main` @ `82a2e43` (milestone 8 + `fable_suggestion.md`).
**Audience:** the agents who implement the next changes to `challenge/my_submission.py`.
**Relation to `fable_suggestion.md` (repo root):** that file ranks latency levers F0-F7. This guide
keeps its numbering where it refers to them, adds the accuracy track it does not have, maps the
mentors' repeated "do something with memory" hint to concrete work, and changes the order: the
largest removable device-side item is the on-card box decode (`genboxes`), which F0-F7 only touch
indirectly via F5.

Everything marked *estimate* is derived from op counts, tensor sizes and queue structure, not
measured. Work package WP-1 replaces the estimates with numbers before anyone spends half a day.

---

## 0. Where we are (facts, with sources)

| Item | Value | Source |
|---|---|---|
| Scored speed, practice set, fastest of 3 | **5.301 ms/frame** (188.6 FPS) | `experiments/runs/10_streaming_pipeline.json` |
| Scored accuracy | **mAP@50 0.4640**, 25,119 detections, 1 false detection on empty photos at conf 0.25 | same |
| Detector as supplied | 85.8 ms, 0.4640 | `experiments/runs/01_reference_plain.json` |
| Card-side `model` stage (H2D wait + reshard + trace + 1.06 MB D2H, one frame at a time) | 5.598 ms mean, 5.484 ms min | `loop_results_20261110_132340/loop_320_2cq.json` |
| Same, whole batch enqueued | 5.448 ms | `bench_levers.py` header, 26 Oct |
| Host work per frame (decode 2.4, prepare 0.7, to_host 1.9, boxes 2.6 ms) | fully hidden behind the card since milestone 8 | same loop json + milestone 8 |
| Practice set | 200 photos, 10 empty, 2,458 objects: 1,061 small / 826 medium / 571 large (COCO areas) | `practice/instances.json` |
| Hour-30 set | 800 photos, 48 empty; jury reads mAP, fps, and "false on empty photographs" | `JURY_HOUR30.md` |

Milestone 8 is at the measured card floor. Every further millisecond has to come from making the
card's per-frame work cheaper. Every accuracy point has to come from the host side (free) or from
precision fixes on the card (cheap), because the model and resolution are fixed for the scored run.

**Branches in flight, not merged, not all measured** (check `git log --all --oneline` first, they move):

| Branch | What it does | State |
|---|---|---|
| `multi_label_anchors` (4 commits, today 11:28-12:10) | multi-label decoding via global top-K=800, NMS 0.50, MAX_DETS 150, `get_model_result` moved into workers, RING=16, POST_WORKERS=32 | no result json committed. Its settings are what WP-2 tunes offline. |
| `trunc_confs` (10 Nov 23:15-23:30) | on-device `ttnn.max` + `ttnn.argmax` over the 80 classes inside the trace, try/except fallback | never measured; 2nd commit fought an L1 address mismatch. Folded into WP-4 as the fallback option. |
| `upstream/double_buffer` (today 11:10-11:46) | conv `enable_act/weights_double_buffer` flags A-D on the old chunked base | B 16.20 / C 15.73 / D 15.42 ms, mAP 0.4639, spread 5-16%. A hint, not a result. Re-measured in WP-5. |

---

## 1. Diagnosis

### 1.1 Speed: the trace is launch- and data-movement-bound, not compute-bound

YOLOv4 at 320x320 is roughly 35 GFLOP per frame. In 5.3 ms that is about 7 TFLOP/s on a card whose
dense bf8 matmul peak is in the hundreds of TFLOP/s: low single-digit utilisation. The time is in
program launches and memory movement. Counting what the trace replays every frame
(`tt-metal/models/demos/yolov4/tt/yolov4.py`, `genboxes.py`, `head.py`, `neck.py`, `downsample*.py`):

| Block | Programs per frame | Kind |
|---|---|---|
| Input prologue: `pad` 3->16 ch, `permute` NCHW->NHWC, `reallocate`, `reshape` (`yolov4.py:44-56`) | 4 | data movement on a 3.3 MB tensor |
| Convolutions ds1-ds5, neck, head (each 3x3 sharded conv is a halo program + a matmul program) | ~115 convs, ~200-230 programs | compute; the 20x20 and 10x10 layers cannot fill 64 cores |
| Glue: `sharded_to_interleaved`, `concat`, `add(x, 0.0, dtype=bf8)` typecasts, upsample, maxpool | ~30 | data movement |
| **`genboxes` x 3 scales** (`genboxes.py:116-232`): per scale 1 `to_memory_config`, 11 `to_layout`, 2 `permute`, 24 `slice`, 10 `concat`, ~20 elementwise | **~220** | tiny ROW_MAJOR slice/concat/tilize ops on 1-8 cores |
| Output epilogue (`yolov4.py:71-93`): 6 `to_layout`, 1 `pad`, 3 `permute`, 2 `concat`, 2 `reshape` | ~14 | data movement |
| D2H of `confs` 1,008 KB + `boxes` 51 KB, on CQ0, serial with the next frame's trace | 1 | PCIe |

A tiny op inside a trace still costs its launch plus a kernel that runs on few cores; ROW_MAJOR
slices and `TilizeWithValPadding` (the comment in `genboxes.py:168` calls it expensive) are 5-30 us
each. 230 such ops plausibly cost **0.8-2.0 ms of the 5.3** (*estimate*). That is the single largest
removable item, and it is pure memory shuffling that the host can do in float32 in the post workers,
which have more than 10x headroom. WP-3 removes it.

The second and third items are known from `fable_suggestion.md`: the 1 MB output read sits on the
compute queue (F1/F3, here WP-4) and the input prologue is on the card (F2, here WP-6).

### 1.2 Accuracy: mAP is lost on the card before NMS ever runs

- The box decode runs in **bf16 on the card**. `bx = sigmoid(tx) + grid_x` with `grid_x` up to 39
  lands in [32, 64) where bf16 spacing is 0.25 grid cells: up to 1 px error at 320, 2-3 px on a
  640-wide photo, before the further bf16 rounding of `x 1/W` and `bx - bw/2`
  (`genboxes.py:186-221`). Boxes reach PCC 0.996 against the float32 torch reference, not 0.9999.
  43% of practice objects are COCO-small; for them 3 px moves IoU across the 0.5 threshold.
- `confs = sigmoid(obj) * sigmoid(cls)` in bf16 gives coarse scores with many exact ties (seen in
  `tools/parity_postprocess.py`); ties scramble the score ranking that mAP integrates over.
- The scale-1 output conv (`head.c2`, 40x40 grid = 4,800 of the 6,300 anchors) writes
  `bfloat8_b` (`model_preprocessing.py:704-706`); `c10` and `c18` write bf16.
- Decoding is single-label (argmax over classes), NMS IoU is 0.40, `EMPTY_THRESH` 0.05 and the
  candidate cap are unswept. For YOLO-family models COCO-style multi-label decoding with NMS 0.5
  is usually worth +0.5 to +1.5 mAP@50; the `multi_label_anchors` branch started this but has no
  measurement.

Decoding on the host in float32 (WP-3) fixes the first two points and removes the ops of 1.1 in
the same change. The rest is an offline sweep (WP-2) that costs no card time.

---

## 2. What the mentors' "memory" hint maps to

Tenstorrent engineers say "memory" for several distinct things. All of them apply here; each has a
work package. Nothing on this list is "cache the answers between the three timed repeats": the
harness times identical work three times on purpose, and the README asks for honest measurement.

| # | Meaning on this hardware | Symptom in our pipeline | Work package |
|---|---|---|---|
| M1 | **PCIe memory traffic on the compute queue**: a D2H read on CQ0 idles all 72 Tensix cores while it runs | 1.06 MB/frame read serial with the next trace | WP-4 (read on CQ1), fallback F1 shrink |
| M2 | **Data-movement ops instead of compute**: layout changes, memory-config changes, slices/concats are memory copies with launch cost | ~270 of ~400 programs per frame are copies (`genboxes`, epilogue, prologue, glue) | WP-3 (host decode), WP-6 (host prologue), WP-5c (sharded concat) |
| M3 | **L1 residency and double buffering** of conv weights and activations (`enable_weights_double_buffer`, `enable_act_double_buffer`, `act_block_h`, sharding layout) | builder defaults never tuned on this tree; `act_block_h_override 128 ... 64 was selected` warnings in every log | WP-5a/b |
| M4 | **Weight-bandwidth-bound small layers**: 10x10 and 20x20 convs with 512-1024 channels read 2-9 MB of weights from DRAM per frame on few cores | ds5 and the deep neck convs are tiny in FLOPs but long in the profile (*expected*, WP-1 verifies) | WP-5d |
| M5 | **Host memory path**: pinned, preallocated host buffers so DMA does not stage through pageable memory | done for outputs (RING slots via `allocate_tensor_on_host`); inputs still go through `from_torch` + staging | low priority, note in WP-6 |
| M6 | **Memory capacity as the limiter for batch 2**: L1 per core, trace region | batch 2 doubles shard heights; `TRACE_REGION_SIZE` must grow | WP-7 |
| M7 | **Numeric memory format**: bf16 for coordinates that need 12+ bits | 1.2 above | WP-3, WP-8a |

---

## 3. Rules of engagement (read before touching the card)

1. **One process owns the card.** Every card job goes through a lock so two agents never collide:
   ```bash
   alias card='flock -w 1800 /tmp/tt-card.lock'
   card python ~/challenge/harness/evaluate.py --submission ... --images ~/challenge/practice/images \
       --annotations ~/challenge/practice/instances.json --repeats 3 --team <lever> --out <file>.json
   ```
   Check `pgrep -af 'evaluate.py|pytest|server.py|bench_'` before starting anything; a leftover
   webcam demo inflates every number.
2. **One lever per branch**, branched from `main`, named `L<nn>-<slug>`, result json committed under
   `experiments/runs/`, one paragraph in `docs/changes/<branch>.md`: what, measured before/after,
   mAP, detections, spread.
3. **Gates.** A change is accepted only if, on the practice set, fastest of 3:
   - no `PROBLEM:` line, `disturbed: false` (else re-run), `photographs_with_no_entry: 0`;
   - speed-only change: mAP@50 and `detections_returned` unchanged from its base (0.4640 / 25,119
     today);
   - accuracy change: mAP@50 >= base + 0.002 (the README says a few thousandths is noise on 200
     photos), `false_detections_on_empty_images` at conf 0.25 not worse than 1, and the gain must
     not come from a knife-edge setting (neighbouring settings must also be >= base);
   - any change to how the model is fed or read: the **liveness check** passes (two clearly
     different photos give different outputs; `tools/liveness_check.py` exists on
     `upstream/L02-overlap-postprocess`, or do it by hand), and the per-stage PCC tests still pass
     where applicable (`models/demos/yolov4/tests/pcc/`).
4. **Kill switches.** Every lever keeps a module-level flag that falls back to the previous
   milestone's path (as `STREAMING`, `FAST_INPUT`, `MULTI_LABEL` do). The hour-30 run must degrade,
   never crash.
5. **Never**: edit `harness/evaluate.py`, `cocomap.py`, `submission_api.py` (md5-checked by the
   jury); leave `DUMP_RAW`-style file writes in a scored run; print per frame; cache detections or
   decoded frames across `run()` calls; measure with the webcam page.
6. **Freeze.** Agree the wall-clock time of hour 30 with the organisers now. At H-90 min the best
   fully gated candidate is copied to `~/challenge/my_submission.py`, re-run once on the sample
   (`~/tt-metal/models/demos/yolov4/resources` + `sample/instances_sample.json`) and once on the
   practice set, and the card is left idle from H-20 min. Everything not merged by H-90 continues
   only for the curve and the pitch.

---

## 4. Work packages

Each package: goal, why, steps, acceptance, expected gain, time. Card time per `evaluate.py` run on
the practice set is about 90 s, so card time is rarely the bottleneck; agent time is.

### WP-0  Freeze the baseline and the clock (30 min, no card)

1. Record today's reference run of the candidate that is on `main` (`experiments/runs/10_...json`).
   If the instance was rebooted since, re-run it: numbers drift a few percent between sessions and
   every later comparison must be same-session.
2. Write the hour-30 wall-clock time at the top of `experiments/README.md`.
3. Make sure `experiments/submissions/08_submission_streaming_pipeline.py` is byte-identical to the
   candidate that produced 5.301 ms (it is the fallback for everything below).

### WP-1  Measure before cutting (60-90 min of card time, the enabler for WP-3/4/5/6)

Produces the three numbers every later decision needs. Do it as one script,
`tools/probe_device_split.py`, run under the card lock, output committed to
`experiments/probes/<date>.json`.

**(a) Split the CQ0 chain.** After `pipeline.compile(...)` exactly as the submission does it:

```python
ex = pipeline.executor                      # MultiCQTracedModelOverlappedInputExecutor
slots = pipeline.preallocate_output_tensors_on_host(2).output_tensors
dev_out = ex.output_tensor                  # [boxes, confs] on device, addresses fixed by the trace

def timed(fn, n=300):
    ttnn.synchronize_device(device); t0 = time.perf_counter()
    for _ in range(n): fn()
    ttnn.synchronize_device(device); return (time.perf_counter() - t0) / n * 1e3

trace_only  = timed(lambda: ttnn.execute_trace(device, ex.trace_id, cq_id=0, blocking=False))

def trace_read():
    ttnn.execute_trace(device, ex.trace_id, cq_id=0, blocking=False)
    for d, h in zip(dev_out, slots[0]):
        ttnn.copy_device_to_host_tensor(d, h, blocking=False, cq_id=0)
trace_plus_read = timed(trace_read)

def production():                           # what run() does per frame
    next(iter(ex.execute([seed_host])))     # H2D on CQ1, reshard + trace on CQ0
    for d, h in zip(dev_out, slots[0]):
        ttnn.copy_device_to_host_tensor(d, h, blocking=False, cq_id=0)
full = timed(production)
print(dict(trace_only=trace_only, read=trace_plus_read - trace_only, reshard_and_sync=full - trace_plus_read))
```

Decision rules: `read` >= 0.15 ms justifies WP-4; `trace_only` is the budget WP-3/5/6/7 compete for.

**(b) Op-level profile of one frame.** The profiler is built into the default tt-metal build on the
instance (`build/tools/profiler/bin/`). Profile the *untraced* 2-CQ pipeline first (per-op kernel
durations are identical inside and outside a trace; untraced attribution is reliable on every
version), then the traced one if the version supports trace attribution:

```bash
cd ~/tt-metal && source python_env/bin/activate && export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
# exact test id differs between trees: list it first
python -m pytest --collect-only -q models/demos/yolov4/tests/perf/test_e2e_performant.py | head
TT_METAL_DEVICE_PROFILER=1 python -m tracy -r -p -v --no-web-server -m pytest \
    "models/demos/yolov4/tests/perf/test_e2e_performant.py::test_e2e_performant[<id for 320>]"
# report: generated/profiler/reports/<stamp>/ops_perf_results_<stamp>.csv
```

Reduce the CSV to a table `op name -> count, sum of DEVICE KERNEL DURATION, mean OP TO OP LATENCY,
core count`, grouped as in section 1.1 (prologue / convs by stage / glue / genboxes / epilogue).
Commit it as `docs/changes/profile_320_<date>.md`. If `python -m tracy` is unavailable on the
instance, fall back to timing sub-graphs with (a)'s `timed()` on hand-built traces (capture a trace
of `genboxes` alone on a saved head output; 20 lines).

**(c) Standalone cost of `ttnn.argmax` and `ttnn.max` on `[1, 6300, 80]` bf16 ROW_MAJOR** (5 min).
Decides whether the `trunc_confs` route (F1) is even admissible: on several versions `argmax` with
`dim=-1` is multi-core only when the row dimension is exactly one tile and otherwise runs on a single
core (see the warning in `tt-metal/models/demos/gemma4/tt/dflash_drafter.py:47-50`).

Acceptance: the three numbers plus the op table are committed. Expected gain: none directly; it
re-orders everything below.

### WP-2  Accuracy sweep offline (2-3 h, zero card time after one 3-minute dump)

Goal: choose the post-processing configuration (single vs multi-label, K, NMS IoU, conf, empty gate,
detection cap) by mAP on the practice set *without* a card run per setting.

1. **Dump raw outputs once.** Port the `DUMP_RAW` / `DUMP_N` mechanism from
   `experiments/submissions/04_submission_vectorized_nms_claude.py:73-78,266-275` into a throwaway
   copy of the milestone-8 submission, set `DUMP_N=200`, run `evaluate.py --repeats 1` on the
   practice set under the lock. 200 x (boxes `[1,6300,1,4]` + confs `[1,6300,80]`) float16 =
   ~210 MB in `~/raw_practice/`. Once WP-3 exists, dump its raw head tensor `[1,1,2100,256]` the
   same way so the float32 decode can be swept too.
2. **Write `tools/sweep_postprocess.py`.** Loads the dumps and `practice/instances.json`, imports
   `fast_detections` from a given submission file, evaluates a grid with `cocomap.evaluate`
   (numpy only, runs on a laptop), prints one row per setting: mAP, detections, CPU ms/frame
   (single thread, median), false detections on empty photos at conf 0.25 (`cocomap.plain_readout`).
   Grid to cover (about 150 settings, seconds each):
   - decoding: single-label; multi-label with top-K in {300, 800, 1500, 3000, all above conf}
   - NMS IoU in {0.40, 0.45, 0.50, 0.55, 0.60, 0.65}
   - conf in {0.001, 0.005}
   - `EMPTY_THRESH` in {0, 0.02, 0.05, 0.10}
   - post-NMS cap in {100 per class only (scorer default), 150 total, 300 total, none}
   - optional, if time: Soft-NMS (Gaussian, sigma 0.5) on the top-K candidates; class-agnostic NMS
     as a negative control (expect worse).
3. **Pick.** Highest mAP whose neighbours (adjacent IoU and K) are also above base, CPU cost <= 8
   ms/frame single-thread (8 post workers at a 5 ms cadence give ~40 ms, keep 5x margin for the
   shared host), and no worse on empty photos. Expect the winner near multi-label, K 800-1500,
   IoU 0.50-0.55, conf 0.001.
4. **Confirm with one card run** of the chosen setting (mAP from the dump equals the card's because
   the raw outputs are identical; only speed needs re-measuring).

Acceptance: mAP >= 0.466 with speed within 0.1 ms of base. Expected: +0.5 to +1.5 mAP@50
(*estimate from the literature on YOLO eval protocols; the sweep gives the real number*).

Note on the scorer: `cocomap.py` caps at 100 detections per image **per class**, sorted by score.
A total cap below ~300 (as the branch's `MAX_DETS = 150`) can only remove true positives; let the
sweep decide, do not assume.

### WP-3  Host-side float32 box decode: delete `genboxes` and the epilogue from the trace (3-5 h)

Goal: the trace ends at the three head convolutions; the card returns the raw logits; the host does
sigmoid/exp/grid/anchor math in float32 in the post workers.

Why: removes ~230 programs per frame (section 1.1, M2), fixes the bf16 coordinate loss (1.2, M7),
and keeps the D2H volume unchanged (`2100 x 256 x 2 B = 1.05 MB` vs 1.06 MB today), so it composes
with WP-4 and makes WP-7 easier (`genboxes` is the one block that assumes batch 1 via `sqrt(HW)`).

Steps:

1. **Device side: monkey-patch `TtYOLOv4.__call__` from the submission** (keeps the submission
   self-contained; `~/tt-metal` stays untouched). Copy the prologue lines `yolov4.py:45-56`
   verbatim, keep `down1..down5`, `neck`, `head`, then:
   ```python
   from models.demos.yolov4.tt import yolov4 as _y4

   def _heads_only(self, x):
       ...  # yolov4.py:45-66 verbatim: pad, permute, reallocate, reshape, down1..5, neck, head
       x4, x5, x6 = self.head([x20, x13, x6])          # [1,1,1600,256], [1,1,400,256], [1,1,100,256]
       outs = []
       for t in (x4, x5, x6):
           t = ttnn.to_memory_config(t, ttnn.L1_MEMORY_CONFIG)      # genboxes.py:116, proven on this tree
           t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)             # genboxes.py:117, bf8 -> bf16 untilize
           outs.append(t)
       return ttnn.concat(outs, dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)   # [1,1,2100,256] bf16
   _y4.TtYOLOv4.__call__ = _heads_only
   ```
   7 programs replace ~234. The output schema becomes a single tensor, which `Pipeline`'s
   preallocated slots and `_copy_struct_to_host` handle unchanged.
2. **Host side**, pure torch, importable without `ttnn` so it is testable on a laptop against the
   dumps from WP-2. Channel layout per anchor `a` in the 256 channels is `a*85 + [tx, ty, tw, th,
   obj, 80 classes]` (`genboxes.py:120-140`); channel 255 is the zero pad. Row `i` of an `SxS`
   grid is `(gy, gx) = (i // S, i % S)` (`genboxes.py:41-63`). Anchor sizes in grid cells, as in
   `genboxes.py:94-114`:
   ```python
   _ANCHORS = (torch.tensor([[1.5, 2.0], [2.375, 4.5], [5.0, 3.5]]),            # 40x40, stride 8
               torch.tensor([[2.25, 4.6875], [4.75, 3.4375], [4.5, 9.125]]),    # 20x20, stride 16
               torch.tensor([[4.4375, 3.4375], [6.0, 7.59375], [14.34375, 12.53125]]))  # 10x10, stride 32
   _SCALES = ((1600, 40), (400, 20), (100, 10))

   def decode_raw(raw):                 # raw: torch [1,1,2100,256] bf16 (a copy, not the slot)
       t_all = raw[0, 0].float(); off = 0; boxes = []; scores = []
       for g, (hw, S) in enumerate(_SCALES):
           t = t_all[off:off + hw, :255].view(hw, 3, 85); off += hw
           gy, gx = torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij")
           gx = gx.reshape(-1, 1).float(); gy = gy.reshape(-1, 1).float()
           bx = (torch.sigmoid(t[..., 0]) + gx) / S
           by = (torch.sigmoid(t[..., 1]) + gy) / S
           bw = torch.exp(t[..., 2]) * _ANCHORS[g][:, 0] / S
           bh = torch.exp(t[..., 3]) * _ANCHORS[g][:, 1] / S
           b = torch.stack((bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2), -1)   # [hw,3,4]
           s = torch.sigmoid(t[..., 4:5]) * torch.sigmoid(t[..., 5:85])              # [hw,3,80]
           boxes.append(b.transpose(0, 1).reshape(-1, 4))      # anchor-major, same order as the card path
           scores.append(s.transpose(0, 1).reshape(-1, 80))
       return torch.cat(boxes), torch.cat(scores)               # [6300,4], [6300,80] float32
   ```
   This is exactly `post_processing.py:_yolo_forward_dynamic` (the torch reference) in float32,
   so the reference PCC test is the oracle.
3. **Wire it in.** In `_retire`, replace `get_model_result(...)` by
   `raw = ttnn.to_torch(host_slot).clone()` (clone so the ring slot is free), then submit
   `decode_raw` + `fast_detections` to the post pool as one job.
4. **Validate** (in this order): (i) PCC of `decode_raw` output against
   `gen_yolov4_boxes_confs(torch_model(x))` on the giraffe frame, expect boxes >= 0.999 (today
   0.996) and confs >= the current value; (ii) liveness check; (iii) `evaluate.py` on the
   sample, then on practice: mAP must be >= 0.4640, detections will differ by design (float32 vs
   bf16); document the delta; (iv) speed.

Acceptance: gates of section 3 for an accuracy change; speed <= base. Expected: -0.8 to -2.0 ms
(*estimate; the WP-1 profile's genboxes + epilogue sum is the real prize*) and +0.0 to +0.5 mAP.

### WP-4  Read the output on CQ1 instead of the compute queue (2-4 h, do if WP-1 shows read >= 0.15 ms)

Goal: the D2H of frame k overlaps the trace of frame k+1 instead of sitting between them.

Why it needs a persistent DRAM copy: the trace writes its output to fixed L1 addresses that the next
replay's early convs reuse, so a read straight from the trace output cannot overlap the next trace.
The stock `MultiCQTracedModelPipelinedIOExecutor` (`tt-metal/models/tt_cnn/tt/executor.py:556-772`)
solves this with an out-of-trace copy to DRAM protected by four events, but rejects list outputs and
needs sharded in/out for its `ttnn.reshard`. After WP-3 the output is one tensor, so write the loop
in the submission:

```python
# setup, after compile: two persistent DRAM buffers for the packed output + the ring of host slots
spec = ex.output_tensor                                     # [1,1,2100,256] bf16 ROW_MAJOR in L1
self.dram_out = [ttnn.allocate_tensor_on_device(spec.shape, spec.dtype, spec.layout, device,
                                                 ttnn.DRAM_MEMORY_CONFIG) for _ in range(2)]
self.read_ev = [None, None]

# per frame k, b = k % 2
ttnn.wait_for_event(1, ex.op_event)                                     # input buffer free?
ttnn.copy_host_to_device_tensor(host_t, ex.dram_input_tensor, cq_id=1)
write_ev = ttnn.record_event(device, 1)
ttnn.wait_for_event(0, write_ev)
ex.l1_input_tensor = ttnn.reshard(ex.dram_input_tensor, ex.l1_input_memory_config, ex.l1_input_tensor)
ex.op_event = ttnn.record_event(device, 0)
ttnn.execute_trace(device, ex.trace_id, cq_id=0, blocking=False)
if self.read_ev[b] is not None:
    ttnn.wait_for_event(0, self.read_ev[b])                             # D2H of frame k-2 done?
_copy_into(ex.output_tensor, self.dram_out[b])                          # L1 -> DRAM, ~1 MB, on CQ0
done_ev = ttnn.record_event(device, 0)
ttnn.wait_for_event(1, done_ev)
ttnn.copy_device_to_host_tensor(self.dram_out[b], host_slot, blocking=False, cq_id=1)
self.read_ev[b] = ttnn.record_event(device, 1)
inflight.append((..., host_slot, self.read_ev[b]))                      # _retire: event_synchronize as today
```

`_copy_into` is the one API that differs between tt-metal versions; probe on the instance with
`help(ttnn.copy)`, `help(ttnn.assign)`, `help(ttnn.to_memory_config)` and use the first that takes a
preallocated output: `ttnn.copy(src, dst)` (in-place into `dst`), `ttnn.assign(src,
memory_config=..., output_tensor=dst)`, or `ttnn.to_memory_config(src, cfg, output_tensor=dst)` on
newer trees. If none exists, make the last op of `_heads_only` a `ttnn.concat(...,
memory_config=ttnn.DRAM_MEMORY_CONFIG)` and `ttnn.reshard`/`ttnn.copy` from there.

Fallback if WP-4 fights back for more than 2 h: F1 (`trunc_confs`), but only if WP-1(c) shows
`argmax` + `max` cost less than the read they save, and only if WP-2 concluded that multi-label is
not worth it (F1 throws the 80-class scores away, which kills multi-label).

Acceptance: speed-only gate (detections and mAP identical to WP-3's). Expected: -(read cost) from
WP-1(a), plausibly 0.3-0.5 ms (*estimate*).

### WP-5  Conv memory knobs: double buffering, blocking, sharded glue, weight bandwidth (2-6 h, profile-gated)

**(a) Weights double buffer, 30 min, do first.** The builder (`tt-metal/models/tt_cnn/tt/builder.py`)
exposes `enable_weights_double_buffer`; `common.py:create_conv2d_config` sets only
`enable_act_double_buffer=True`. Check the instance's default (`grep -n weights_double_buffer
~/tt-metal/models/tt_cnn/tt/builder.py`). If it is False there, port the `to_conv2d_config`
monkey-patch from `upstream/double_buffer:challenge/harness/double_buffer_submission.py` into the
streaming submission behind a `WEIGHTS_DOUBLE_BUFFER` flag and measure A/B same-session. The old-base
hint was -0.8 ms for "both on" vs "act only", at 5-16% spread, so treat it as unknown until measured.
Speed-only gate.

**(b) Invalid `act_block_h` override.** Every log prints `act_block_h_override 128 is not a valid
override ... 64 was selected` for ds1 `c1` (`model_preprocessing.py:106`). Try 0 (let the op choose)
and 256; measure. Speed-only gate.

**(c) Sharded concat instead of `sharded_to_interleaved` + `concat` + `add(x, 0.0, dtype=bf8)`** in
`head.py:172-181` and `:200-208`, and the same pattern in `downsample4.py:262-267`,
`downsample5.py:165-170`: the YOLOv4 tech report (`tech_reports/YoloV4-TTNN/yolov4.md` section 2.1)
shows the replacement. Also make the producing conv write bf8 directly instead of the
`add(..., dtype=bf8)` typecast. Only if the WP-1 profile shows these glue ops above ~0.1 ms in
total. PCC tests + speed-only gate.

**(d) Weight-bandwidth-bound tail.** From the profile, list convs whose device time tracks weight
bytes rather than MACs (expect ds5 and the 10x10/20x20 neck convs with 512-1024 channels). Options
in order of safety: more cores for those convs (shard layout / `override_sharding_config`),
`enable_weights_double_buffer` if (a) did not already apply, and as a last resort `bfloat4_b`
weights for only those layers (halves their DRAM traffic; accuracy-gated, expect to need the full
mAP gate). Do not apply bf4 to the backbone.

Expected: 0.1-0.6 ms total (*estimate*), each sub-item measured on its own.

### WP-6  Input prologue on the host (F2 in `fable_suggestion.md`, 2-3 h, profile-gated)

Unchanged from F2: produce the `[1, 1, 102400, 16]` bf16 NHWC tensor in the decode threads (LUT
gather into a preallocated int16 buffer with 13 zero channels), open the pipeline with the matching
sharded input config, and start `_heads_only` at `down1`. Bit-exact by construction; the H2D grows
to 3.3 MB/frame on CQ1, which has headroom. Do it only if the WP-1 profile prices the four prologue
ops above ~0.1 ms. Optional extra (M5): allocate the input ring with `ttnn.allocate_tensor_on_host`
and write the LUT output into it to skip the staging copy; measure, it may be invisible.

### WP-7  Batch-2 trace (F4, 1 day, highest variance; start only with >= 8 h before freeze)

What breaks today and must be handled, found by reading the tree:
- `YOLOv4PerformanceRunnerInfra` builds its own host input from giraffe.jpg at batch 1 regardless
  of `batch_size` (`performant_runner_infra.py:53-58`); build the `[2,3,320,320]` seed tensor and
  the two memory configs yourself (`_setup_l1_sharded_input` logic with `n=2`).
- `neck.py` hardcodes 320-only shard specs with batch-1 heights (`(20, 32)`, `(80, 32)`,
  `(64, 64)`, `(80, 16)`, `(320, 16)` at lines 331-350, 403-416, 442-461); scale the heights by N.
- `genboxes` (`sqrt(HW)`) and `get_model_result` unpadding assume batch 1: moot after WP-3, which
  must land first. `decode_raw` then takes `raw[n, 0]` per image.
- Double `TRACE_REGION_SIZE`; the harness passes `DEVICE_OPTIONS` through.
- Pair consecutive frames; duplicate the last odd frame and drop its result.
Go/no-go after 3 h: `validate()` PCC passes at batch 2 and the trace captures. Expected if it works:
per-frame time well under 2x the batch-1 replay because the 20x20/10x10 layers amortise
(*estimate* -1 to -1.5 ms/frame). If L1 overflows in ds1-ds3, stop.

### WP-8  Accuracy micro-experiments on the card (each: one 2-minute run, 30 min of code)

- **(a) `head.c2` output in bf16** instead of `bfloat8_b` (`model_preprocessing.py:704-706`,
  patchable from the submission by editing `conv_args` before `TtYOLOv4` is built): better logits
  for the 4,800 small-object anchors. Accuracy gate; expect a small plus and <= 0.05 ms.
- **(b) `cv2.INTER_AREA` for the downscale** instead of the default `INTER_LINEAR` in `_decode`:
  anti-aliased small objects. Hidden host cost. Accuracy gate; could go either way.
- **(c) Letterbox** (keep aspect ratio, pad to 320x320 with gray, un-letterbox the boxes): the
  supplied weights were trained and validated with stretch resize, so expect a loss; one run
  settles it. Accuracy gate.
Keep only measured winners.

### WP-9  640x640 as a second point on the curve (optional, time-boxed to 4 h, only after the freeze)

`PLAN.md` section 2C names the two suspected causes (resolution guards that skip the bf8 casts and
shard specs at 640; the 28-element padding gaps in the box slicing). After WP-3 the second cause
disappears (no `get_model_result` slicing). Gate: `test_ttnn_yolov4[1-pretrained_weight_true-0]`
PCC >= 0.99, then `evaluate.py`. The card ran 640 at 11.9 ms for the model stage alone
(`loop_640_1cq.json`), so a 640 point lands near 12-14 ms with a materially higher mAP if the
numerics are fixed: a strong second point for the jury's chart, never the scored run.

### WP-10  Deliverables (1 h, in parallel with everything, owner: one agent)

- One-page change log: table of levers in order, measured speed before/after (`evaluate.py`,
  practice), mAP, detections, which numbers were estimated. Source: `experiments/README.md` table.
- Curve: `python challenge/harness/plot_curve.py experiments/runs/*.json --out experiments/plots/curve.svg`.
- Settings stated: 320x320, YOLOv4 as supplied (plus the exact post-processing: conf, NMS IoU,
  multi-label K, empty gate), from the frozen file.
- Starting measurement: `bench_results_20261110_132001/bench_320.json` (71.4 ms mean on this card).
- Pitch demo: point `webcam_demo/server.py` at the frozen submission's pipeline; rehearse; record.

---

## 5. Order and parallel assignment

Agent roles (two need the card, two do not):

| Agent | Needs card | Packages, in order |
|---|---|---|
| A: device | yes | WP-0 -> WP-1 -> WP-3 -> WP-4 (if justified) -> WP-5a/b -> WP-6 -> WP-7 |
| B: accuracy | only for 1 dump + confirm runs | WP-2 (offline), then WP-8 runs queued through the lock, then `decode_raw` laptop parity for WP-3 |
| C: conv/memory | yes, short runs | WP-5a immediately (30 min), then WP-5c/d from the WP-1 profile |
| D: deliverables | no | WP-10, keeps `experiments/README.md` and the curve current, runs the freeze checklist |

T-minus plan relative to hour 30 (H). If H is more than 12 h away, pull WP-6/WP-7 before the freeze.

| Window | Card | Off-card |
|---|---|---|
| H-6h .. H-4h | WP-0, WP-1 (A); WP-5a (C) | WP-2 dump then sweep (B); WP-10 skeleton (D) |
| H-4h .. H-2h | WP-3 (A); WP-8a/b/c and WP-2 confirm run (B, through the lock) | B finishes sweep, hands settings to A |
| H-2h .. H-1.5h | WP-4 if WP-1 justified it, else WP-5b (A) | D drafts change log from committed jsons |
| H-1.5h | **Freeze**: copy best gated candidate to `~/challenge/my_submission.py`; sample run; practice run; liveness; leave the card idle from H-20min | curve, settings, demo rehearsal |
| after H | WP-6, WP-7, WP-9 for the pitch and the curve | |

Expected landing, practice set, if WP-2 + WP-3 + WP-5a succeed: **4.0-4.6 ms/frame at mAP 0.468-0.478**
(*estimate*); with WP-4 **3.7-4.3 ms**; with WP-7 on top **below 3.5 ms**. The jury metric is the
pair, plotted against 79 ms / 0.467 for the supplied detector.

---

## 6. Appendix

### 6.1 Commands

```bash
# environment (every shell)
source ~/tt-metal/python_env/bin/activate; export TT_METAL_HOME=~/tt-metal PYTHONPATH=~/tt-metal
alias card='flock -w 1800 /tmp/tt-card.lock'

# plumbing check (no PROBLEM lines is the only thing that matters here)
card python ~/challenge/harness/evaluate.py --submission $SUB \
  --images ~/tt-metal/models/demos/yolov4/resources --annotations ~/challenge/sample/instances_sample.json \
  --out /tmp/check.json

# scored-style practice run
card python ~/challenge/harness/evaluate.py --submission $SUB \
  --images ~/challenge/practice/images --annotations ~/challenge/practice/instances.json \
  --repeats 3 --team L<nn>-<slug> --out ~/SEMICON-Hackathon/experiments/runs/<nn>_<slug>.json

# per-stage PCC tests after any device-side change
cd ~/tt-metal && python -m pytest -q --tb=line models/demos/yolov4/tests/pcc/test_ttnn_head.py \
  models/demos/yolov4/tests/pcc/test_ttnn_neck.py "models/demos/yolov4/tests/pcc/test_ttnn_yolov4.py::test_yolov4[0-pretrained_weight_true-0]"
# (downsample4 and downsample5 tests fail on the supplied tree; the README says so)
```

### 6.2 Numbers to fill in (WP-1 output)

| Quantity | Expected (estimate) | Measured |
|---|---|---|
| trace_only (ms) | 4.5-5.0 | |
| read on CQ0 (ms) | 0.3-0.5 | |
| reshard + wait (ms) | <= 0.1 | |
| genboxes + epilogue kernel sum (ms) | 0.8-2.0 | |
| prologue kernel sum (ms) | 0.1-0.3 | |
| glue ops kernel sum (ms) | 0.1-0.4 | |
| convs kernel sum, ds1-ds3 / ds4-ds5 / neck / head (ms) | | |
| `ttnn.argmax` on [1,6300,80] (ms) | unknown, possibly single-core | |

### 6.3 File map

| Where | What |
|---|---|
| `challenge/my_submission.py` | the candidate the jury runs; keep it identical to the frozen experiment file |
| `experiments/submissions/`, `experiments/runs/`, `experiments/README.md` | one frozen copy + one result json per milestone; the table the change log is built from |
| `fable_suggestion.md` (root) | the earlier lever analysis F0-F7, still valid for F2/F4/F7 details |
| `tools/parity_postprocess.py` | laptop parity of `fast_detections` vs the reference on dumped raw outputs; extend for `decode_raw` |
| `tt-metal/models/demos/yolov4/tt/genboxes.py`, `yolov4.py` | the ops WP-3 removes from the trace |
| `tt-metal/models/tt_cnn/tt/executor.py` | the event protocol WP-4 copies (`MultiCQTracedModelPipelinedIOExecutor`) |
| `tt-metal/models/demos/yolov4/tt/model_preprocessing.py` | per-conv memory knobs for WP-5 and WP-8a |
| `tt-metal/tech_reports/YoloV4-TTNN/yolov4.md`, `tech_reports/CNNs/ttcnn.md` | Tenstorrent's own guidance on sharded concat, double buffering, `act_block_h`, shard layouts |
