#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

# =============================================================================
# run_bench.sh
#
# One command to find out where a frame's time really goes on the machine with
# the card. Sets the environment, runs the bench at both resolutions, and saves
# everything to a timestamped folder so the numbers are on disk rather than on
# a screen.
#
#   Date      2026-09-23
#   Version   v0.1
#
# WHAT YOU DO
#       cd ~/challenge/bench
#       bash run_bench.sh
#
# HOW LONG
#   About 10 minutes at 320 on a prepared machine, most of it the first frame
#   compiling kernels. Add --quick for 60 frames instead of 300 if you only
#   want a sanity check.
#
# WHAT YOU GET
#   A folder bench_results_<date>_<time>/ containing:
#       board.txt           tt-smi snapshot, so the numbers have a board and a
#                           firmware attached to them
#       env.txt             python, torch, opencv, git commit of tt-metal
#       bench_320.json      per-stage timings, 320x320
#       bench_320.txt       the same run's console output
#       bench_640.json      per-stage timings, 640x640  (skip with --320-only)
#       bench_640.txt
#
#   Copy that whole folder back and the figure can be rebuilt from it.
#
# OPTIONS
#   --quick        60 measured frames instead of 300
#   --320-only     skip the 640 run
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

while [ $# -gt 0 ]; do
  case "$1" in
    --quick)     ITERS=60; shift ;;
    --320-only)  DO640=0; shift ;;
    --iters)     ITERS="$2"; shift 2 ;;
    -h|--help)   sed -n '2,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)           echo "unknown option: $1"; exit 2 ;;
  esac
done

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31mSTOP: %s\033[0m\n' "$*"; exit 1; }

# -----------------------------------------------------------------------------
step "1/4  Checking the machine"
# -----------------------------------------------------------------------------
[ -f "$HERE/bench_stages.py" ] || fail "bench_stages.py is not next to this script, in $HERE"
[ -d "$DST/build/lib" ] || fail "No built tt-metal at $DST/build/lib.
This expects the Tenstorrent Console development-environment image."
[ -d "$DST/models/demos/yolov4" ] || fail "yolov4 is not in $DST.
Run ../setup_yolov4_on_ttmetal.sh first."

OUTDIR="$HERE/bench_results_$(date +%Y%m%d_%H%M%S)"
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
  echo "opencv      $(python -c 'import cv2; print(cv2.__version__)')"
  echo "tt-metal    $(git -C "$DST" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "branch      $(git -C "$DST" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
} > "$OUTDIR/env.txt"
cat "$OUTDIR/env.txt" | sed 's/^/  /'

# -----------------------------------------------------------------------------
step "3/4  Measuring at 320x320 ($ITERS frames)"
# -----------------------------------------------------------------------------
echo "  the first frame compiles kernels, so expect a few quiet minutes"
python "$HERE/bench_stages.py" \
  --resolution 320 --iters "$ITERS" \
  --out "$OUTDIR/bench_320.json" 2>&1 | tee "$OUTDIR/bench_320.txt"

# -----------------------------------------------------------------------------
step "4/4  Measuring at 640x640"
# -----------------------------------------------------------------------------
if [ "$DO640" = "1" ]; then
  python "$HERE/bench_stages.py" \
    --resolution 640 --iters "$ITERS" \
    --out "$OUTDIR/bench_640.json" 2>&1 | tee "$OUTDIR/bench_640.txt"
else
  echo "  skipped (--320-only)"
fi

cat <<NOTE

  Done. Everything is in:
      $OUTDIR

  To get it back to your laptop, from the instance:

      cd "$(dirname "$OUTDIR")" && tar czf - "$(basename "$OUTDIR")" | base64 -w0

  Copy the output, and on your Mac:

      pbpaste | base64 -d | tar xzf -

NOTE
