#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

# =============================================================================
# run_levers.sh
#
# Find out where the factor of ten between the plain model path (59.763 ms a
# frame, measured) and the traced path (5.917 ms, published) actually comes
# from. Sets the environment, runs the four-rung ladder, saves everything to a
# timestamped folder so the numbers are on disk rather than on a screen.
#
#   Date      2026-09-24
#   Version   v0.1
#
# WHAT YOU DO
#       cd ~/challenge/bench
#       bash run_levers.sh
#
# HOW LONG
#   Roughly 15 minutes at 320 on a prepared machine. Four configurations, each
#   reopening the device and compiling, because trace_region_size and
#   num_command_queues can only be set at device open. --quick for a sanity
#   check.
#
# WHAT YOU GET
#   A folder lever_results_<date>_<time>/ containing:
#       board.txt           tt-smi snapshot, so the numbers have a board and a
#                           firmware attached to them
#       env.txt             python, torch, git commit and branch of tt-metal
#       levers_320.json     per-configuration timings and the split
#       levers_320.txt      the same run's console output
#       levers_640.json     the same at 640  (only with --640)
#       levers_640.txt
#
# THE ONE THING TO LOOK AT FIRST
#   The trace_2cq row. It should land near 5.917 ms, because it is the same
#   configuration the branch's own perf test measures. If it does not, the
#   harness is wrong and the split it prints underneath means nothing.
#
# OPTIONS
#   --quick        60 measured frames instead of 300
#   --640          also run at 640x640
#   --iters N      exact frame count
#   --only NAMES   comma-separated subset, e.g. --only trace,trace_2cq
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
DO640=0
ONLY=""

while [ $# -gt 0 ]; do
  case "$1" in
    --quick)   ITERS=60; shift ;;
    --640)     DO640=1; shift ;;
    --iters)   ITERS="$2"; shift 2 ;;
    --only)    ONLY="$2"; shift 2 ;;
    -h|--help) sed -n '2,48p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)         echo "unknown option: $1"; exit 2 ;;
  esac
done

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31mSTOP: %s\033[0m\n' "$*"; exit 1; }

ONLY_ARG=()
[ -n "$ONLY" ] && ONLY_ARG=(--only "$ONLY")

# -----------------------------------------------------------------------------
step "1/4  Checking the machine"
# -----------------------------------------------------------------------------
[ -f "$HERE/bench_levers.py" ] || fail "bench_levers.py is not next to this script, in $HERE"
[ -d "$DST/build/lib" ] || fail "No built tt-metal at $DST/build/lib.
This expects the Tenstorrent Console development-environment image."
[ -d "$DST/models/demos/yolov4" ] || fail "yolov4 is not in $DST.
Run ../setup_yolov4_on_ttmetal.sh first."
[ -f "$DST/models/tt_cnn/tt/pipeline.py" ] || fail "models/tt_cnn/tt/pipeline.py is
missing from $DST. This bench drives the same pipeline the branch's perf test
uses, so without it there is nothing to measure."

OUTDIR="$HERE/lever_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"
echo "  results will go to $OUTDIR"

# -----------------------------------------------------------------------------
step "2/4  Recording what this was measured on"
# -----------------------------------------------------------------------------
# An untraced number is no better than a guess, so capture the board first.
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
  echo "tt-metal    $(git -C "$DST" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "branch      $(git -C "$DST" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
} > "$OUTDIR/env.txt"
sed 's/^/  /' "$OUTDIR/env.txt"

# -----------------------------------------------------------------------------
step "3/4  Running the ladder at 320x320 ($ITERS frames a rung)"
# -----------------------------------------------------------------------------
echo "  four configurations, each reopening the device and compiling first"
python "$HERE/bench_levers.py" \
  --resolution 320 --iters "$ITERS" "${ONLY_ARG[@]}" \
  --out "$OUTDIR/levers_320.json" 2>&1 | tee "$OUTDIR/levers_320.txt"

# -----------------------------------------------------------------------------
step "4/4  The same at 640x640"
# -----------------------------------------------------------------------------
if [ "$DO640" = "1" ]; then
  python "$HERE/bench_levers.py" \
    --resolution 640 --iters "$ITERS" "${ONLY_ARG[@]}" \
    --out "$OUTDIR/levers_640.json" 2>&1 | tee "$OUTDIR/levers_640.txt"
else
  echo "  skipped (pass --640 to include it)"
fi

cat <<NOTE

  Done. Everything is in:
      $OUTDIR

  Read the trace_2cq row against 5.917 ms before reading anything else.

  To get it back to your laptop, from the instance:

      cd "$(dirname "$OUTDIR")" && tar czf - "$(basename "$OUTDIR")" | base64 -w0

  Copy the output, and on your Mac:

      pbpaste | base64 -d | tar xzf -

NOTE
