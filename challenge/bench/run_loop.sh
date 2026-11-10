#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

# =============================================================================
# run_loop.sh
#
# Measure one whole frame on the traced path, with one stopwatch, so that the
# 10.889 ms on Figure 1 of an earlier draft stops being a composition of two
# runs and becomes a measurement.
#
#   Date      2026-09-24
#   Version   v0.1
#
# WHAT YOU DO
#       cd ~/challenge/bench
#       bash run_loop.sh
#
# WHAT IT DOES, IN ORDER
#   1. Checks the machine and makes a timestamped results folder.
#   2. Records the board and the environment, so the numbers arrive traced.
#   3. PROBE. Prints the signatures bench_loop.py depends on, WITHOUT opening
#      the device. If the probe fails, stop and send its output; the harness
#      needs a one-line change and nothing else in this script will work.
#   4. Runs 320 with one command queue. This is the number that replaces the
#      composed 10.889 ms.
#   5. Runs 320 with two command queues, for the loop a webcam demo would use.
#   6. Runs 640, unless --320-only.
#
# HOW LONG
#   About 12 minutes at 320 on a prepared machine. Add --quick for 60 frames.
#
# WHAT YOU GET
#   loop_results_<date>_<time>/ containing board.txt, env.txt, probe.txt, and
#   a .json plus .txt for each run. Copy the whole folder back.
#
# OPTIONS
#   --quick        60 measured frames instead of 300
#   --320-only     skip the 640 run
#   --1cq-only     skip the two-command-queue run
#   --iters N      exact frame count
# =============================================================================

set -euo pipefail

# A VS Code terminal that has been through many nested shells can carry
# SHLVL up near bash's limit of 1000, and then every command substitution
# in this script prints "shell level too high". Harmless, but it buries the
# output. Reset it for this script and everything it spawns.
export SHLVL=0

DST="$HOME/tt-metal"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ITERS=300
DO640=1
DO2CQ=1

while [ $# -gt 0 ]; do
  case "$1" in
    --quick)     ITERS=60; shift ;;
    --320-only)  DO640=0; shift ;;
    --1cq-only)  DO2CQ=0; shift ;;
    --iters)     ITERS="$2"; shift 2 ;;
    -h|--help)   sed -n '2,46p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)           echo "unknown option: $1"; exit 2 ;;
  esac
done

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31mSTOP: %s\033[0m\n' "$*"; exit 1; }

# -----------------------------------------------------------------------------
step "1/6  Checking the machine"
# -----------------------------------------------------------------------------
[ -f "$HERE/bench_loop.py" ] || fail "bench_loop.py is not next to this script, in $HERE"
[ -d "$DST/build/lib" ] || fail "No built tt-metal at $DST/build/lib.
This expects the Tenstorrent Console development-environment image."
[ -d "$DST/models/demos/yolov4" ] || fail "yolov4 is not in $DST.
Run ../setup_yolov4_on_ttmetal.sh first."
[ -f "$DST/models/tt_cnn/tt/pipeline.py" ] || fail "models/tt_cnn/tt/pipeline.py is missing.
bench_loop.py needs create_pipeline_from_config from it."

OUTDIR="$HERE/loop_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"
echo "  results will go to $OUTDIR"

# -----------------------------------------------------------------------------
step "2/6  Recording what this was measured on"
# -----------------------------------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
if command -v tt-smi >/dev/null 2>&1; then
  tt-smi -s > "$OUTDIR/board.txt" 2>&1 || tt-smi > "$OUTDIR/board.txt" 2>&1 || true
  echo "  board snapshot saved"
else
  echo "tt-smi not on PATH" > "$OUTDIR/board.txt"
  echo "  note: tt-smi not found, board.txt will say so"
fi

# shellcheck disable=SC1091
source "$DST/python_env/bin/activate"
export TT_METAL_HOME="$DST"
export PYTHONPATH="$DST"
cd "$DST"

{
  echo "date        $(date -Is)"
  echo "host        $(hostname)"
  echo "python      $(python -c 'import sys; print(sys.version.split()[0])')"
  echo "torch       $(python -c 'import torch; print(torch.__version__)')"
  echo "opencv      $(python -c 'import cv2; print(cv2.__version__)')"
  echo "tt-metal    $(git -C "$DST" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "branch      $(git -C "$DST" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
} > "$OUTDIR/env.txt"
sed 's/^/  /' "$OUTDIR/env.txt"

# -----------------------------------------------------------------------------
step "3/6  Probe: can the infra take a fresh frame each time?"
# -----------------------------------------------------------------------------
# bench_loop.py has to build a new ttnn host tensor per frame from a torch
# tensor. The method that does it is named differently on different branches.
# This prints the real signatures without opening the device, so a mismatch
# costs seconds rather than a failed 12-minute run.
python - > "$OUTDIR/probe.txt" 2>&1 <<'PY' || true
import inspect
from models.demos.yolov4.runner.performant_runner_infra import YOLOv4PerformanceRunnerInfra as I
for name in ("__init__", "setup_l1_sharded_input", "setup_dram_sharded_input",
             "validate", "run"):
    m = getattr(I, name, None)
    print(f"{name}: {inspect.signature(m) if m else 'ABSENT'}")
print()
print("--- source of the two setup methods ---")
for name in ("setup_l1_sharded_input", "setup_dram_sharded_input"):
    m = getattr(I, name, None)
    if m:
        print(inspect.getsource(m))
PY
sed 's/^/  /' "$OUTDIR/probe.txt"

if ! grep -qE 'setup_(l1|dram)_sharded_input: \(self, device, [a-z_]+' "$OUTDIR/probe.txt"; then
  printf '\n\033[1;33m  NOTE: neither setup method obviously takes a per-frame tensor.\033[0m\n'
  printf '  bench_loop.py will probe again at runtime and stop with a clear\n'
  printf '  message if it cannot find one. If it does, send probe.txt.\n'
fi

# -----------------------------------------------------------------------------
step "4/6  320x320, one command queue ($ITERS frames)  <- the headline number"
# -----------------------------------------------------------------------------
echo "  the first frame captures the trace, so expect a quiet couple of minutes"
python "$HERE/bench_loop.py" \
  --resolution 320 --iters "$ITERS" --cqs 1 \
  --out "$OUTDIR/loop_320_1cq.json" 2>&1 | tee "$OUTDIR/loop_320_1cq.txt"

# -----------------------------------------------------------------------------
step "5/6  320x320, two command queues"
# -----------------------------------------------------------------------------
if [ "$DO2CQ" = "1" ]; then
  python "$HERE/bench_loop.py" \
    --resolution 320 --iters "$ITERS" --cqs 2 \
    --out "$OUTDIR/loop_320_2cq.json" 2>&1 | tee "$OUTDIR/loop_320_2cq.txt"
else
  echo "  skipped (--1cq-only)"
fi

# -----------------------------------------------------------------------------
step "6/6  640x640, one command queue"
# -----------------------------------------------------------------------------
if [ "$DO640" = "1" ]; then
  python "$HERE/bench_loop.py" \
    --resolution 640 --iters "$ITERS" --cqs 1 \
    --out "$OUTDIR/loop_640_1cq.json" 2>&1 | tee "$OUTDIR/loop_640_1cq.txt"
else
  echo "  skipped (--320-only)"
fi

cat <<NOTE

  Done. Everything is in:
      $OUTDIR

  Read the 320 one-command-queue run first, and read two lines before any
  timing: the trace check and the accuracy line. If the trace check says
  UNSOUND, the pipeline is not reading the frames it is handed and the
  timings mean nothing.

  To get the folder back to your laptop, from the instance:

      cd "$(dirname "$OUTDIR")" && tar czf - "$(basename "$OUTDIR")" | base64 -w0

  Copy the output, and on your Mac:

      pbpaste | base64 -d | tar xzf -

NOTE
