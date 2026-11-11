# Merge handoff: differences from main

This document describes the workspace inspected on 2026-11-11. The project rules
in the root `AGENTS.md` still apply. Preserve existing work; do not reset, clean,
or restore files already deleted by the user.

## 1. Code locations and comparison baseline

- Current branch at inspection: `double_buffer`.
- HEAD: `c389868b498a50407a315f56a496ddb77c6a168d`.
- Local `main` and `origin/main`: `3f658101ae023cf7dff36e06e4ca44ddef512cfa`.
  No fetch was performed, so these refs may not represent the latest remote state.
- **Primary implementation to merge: `challenge/harness/my_submission.py`.**
- `challenge/my_submission.py` is a separate entry point with different contents.
  Do not overwrite it merely because the filename matches.
- `challenge/harness/initial_submission.py` is an untracked earlier double-buffer
  implementation, useful for provenance rather than as the current entry point.
- Historical filenames `submission_fused.py` and `submission_double_buffer.py`
  are not currently present. Do not treat them as existing entry points.

Before merging, inspect the current state again:

```bash
git status --short
git diff main -- challenge/harness/my_submission.py
```

## 2. Main implementation differences

### Conv2D double-buffer experiment controls: preserve these

Added `DOUBLE_BUFFER_FLAGS`, `_double_buffer_mode()`, and
`_double_buffer_configuration()`. The `YOLO_DOUBLE_BUFFER` environment variable
selects:

| Mode | Activation double buffer | Weight double buffer |
|---|---|---|
| original (default) | Preserve each layer's setting | Preserve each layer's setting |
| A | false | false |
| B | true | false |
| C | false | true |
| D | true | true |

During model construction, pipeline construction, and compilation in `setup()`,
a context manager temporarily wraps
`models.tt_cnn.tt.builder.to_conv2d_config`. It uses `dataclasses.replace` to
override the two fields in the immutable configuration, then calls the original
converter. A `finally` block restores the converter without changing external
source files.

The wrapper checks the resulting configuration fields and counts calls. Zero
calls cause an error; setup prints `[double_buffer]` statistics. These counts
confirm configuration propagation, not that kernels actually use double buffering.
The patched module is shared within the process: model setup must be serial,
and each experimental mode must run in a separate process. Original layer
settings may differ, so `original` is not automatically equivalent to A or C.

### Execution path: chunks replaced by per-frame streaming

In main, `run()` enqueues and drains chunks while processing the preceding
chunk's results. The current implementation adds `_copy_struct_to_host()`,
a per-frame `run()`, and `_retire()`, retaining `_run_chunked()` as a fallback.

- `YOLO_STREAMING=1` (default) requests streaming; `0` selects chunks.
- Missing `ttnn.event_synchronize` also selects the chunked path.
- Streaming directly uses `pipeline.executor`: per-frame execute, asynchronous
  device-to-host copy, `signal_read_complete`, and `record_event`.
- Per-frame event waits replace full-device synchronization at chunk boundaries,
  maintaining a bounded number of frames in flight.
- Host output slots are preallocated in a ring: `IN_FLIGHT=4`, `RING=8`;
  RING must exceed IN_FLIGHT.
- `LOOKAHEAD=64`; host input references remain alive until their frame completes.
- Added an NMS pool with `POST_WORKERS=8`; `DECODERS` changes from 32 in main to 24.
- `_retire()` explicitly clones boxes and confidences before submitting NMS,
  preventing reused host slots from overwriting data still used by another thread.

Do not merge `run()` alone: setup of the executor, command queues, host slots,
and event ordering must move together. These depend on internal TT-NN pipeline
interfaces. Availability of the event API alone does not establish compatibility.

### Input conversion and CPU threading

- Added `_to_host_reference()`, `_to_host_fast()`, and `_build_input_lut()`.
- A 256-entry BF16 lookup table generated through TT-NN combines pixel conversion
  with BGR-to-RGB and NCHW transformations.
- Setup compares the fast path bit-for-bit against the reference on a fixed
  random image; mismatch or failure selects the reference path.
- Added `TORCH_THREADS=1` to reduce oversubscription during parallel frame processing.
- `CHUNK=32` remains for the fallback path.
- Single-threaded OpenCV already exists in main; it is not a new contribution.

### Frame gates and postprocessing semantics

Main hardcodes `EMPTY_THRESH=0.05` and `HOST_SALIENCY_THRESH=3.0`. The current
implementation reads environment variables, with both defaulting to zero:

```bash
YOLO_EMPTY_THRESH=0 YOLO_HOST_SALIENCY_THRESH=0
```

To reproduce main's gate thresholds, explicitly select 0.05 and 3.0 respectively.
These gates change predictions and the amount of work performed. Hold them fixed
when comparing speed; do not attribute threshold differences to double buffering.

Vectorized NMS, cached category mapping, float64 pixel scaling, clipping to image
bounds, zero-area filtering, and score=confidence squared already exist in main
and are largely retained. Do not describe them as newly introduced algorithms.
Main's documentation mentions `DUMP_RAW`; verify the actual code rather than
assuming the feature exists based on a docstring.

### Resource cleanup

Setup records the original GC state and torch thread count. Teardown shuts down
both pools, cleans up the pipeline, and restores those settings. Review cleanup
when setup fails partway through and draining after asynchronous execution
errors; safety of every failure path has not been established.

## 3. Results and evidence limits

Four result files were added relative to main:

- `challenge/runs/double_buffer_B_old.json`
- `challenge/runs/double_buffer_C_old.json`
- `challenge/runs/double_buffer_D_old.json`
- `challenge/runs/double_buffer_D_optimized.json`

The last record reports 200 images, 5.187 ms/frame, and mAP_50=0.464. Timing is
the fastest of three repeats, not a median, isolated Conv device time, or
single-request latency.

Its recorded submission path is `/home/user/challenge/double_buffer_submission.py`.
No source hash or complete environment-variable record is provided, so it cannot
be conclusively tied to the current entry point and default gate settings.
The user reports that D is fastest, but complete profiler evidence explaining
why is still missing. Equal mAP does not establish bitwise-identical raw outputs.

The local environment has no runnable TT-NN/PyTorch/Wormhole stack. Earlier
checks covered syntax, configuration switching, exception restoration, and
simulated ownership of copied output-slot data. They do not replace device
correctness or performance validation.

## 4. Recommended merge and acceptance procedure

1. Preserve the latest mainline model, scoring interface, and postprocessing
   fixes. Port changes by feature rather than overwriting the entire file.
2. Fix gate thresholds, resolution, and dataset. Validate migrated outputs and
   both streaming/chunked paths using `original` first.
3. Run A/B/C/D separately. Keep the original model configuration as its own
   reference rather than assuming it matches one of those modes.
4. Start with a small execution check, then evaluate mAP, detection counts,
   missing-image entries, and repeated timings on the same practice dataset.
5. Test more images than RING and LOOKAHEAD to exercise slot reuse; include empty
   frames between normal frames and repeated `run()`/warmup calls.
6. Store source hashes, switches, software versions, and device details alongside
   results and logs. Separate performance timing from profiler diagnostics.
7. Check correctness and L1 allocation failures before fixing the winning mode.
   D working at 320 does not guarantee it works at other resolutions.

320x320 input, CONF_THRESH=0.001, NMS_THRESH=0.4, tracing, and two command queues
are retained. 640 input, flip augmentation, and tiled inference have only been
discussed, not implemented. Do not add them incidentally during this merge.

## 5. Existing workspace changes

`PLAN.md` and `challenge/harness/double_buffer_submission.py` were already marked
as deleted. There are also untracked Python files, `research/`, and a file with
an unusual name. Preserve these existing changes; do not clean, restore, or
automatically include them in a merge commit. `research/wormhole_l1/` contains
isolated environment probing and research notes.
