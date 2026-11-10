#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

# =============================================================================
# setup_yolov4_on_ttmetal.sh
#
# Brings up YOLOv4 object detection on a Tenstorrent Console VSCode Server
# instance, using the tt-metal build that ships in the image plus the
# hand-written TTNN YOLOv4 from tt-metal. For the Chip Architecture and
# Integration track at a student hackathon in Munich, 10-12 November 2026.
#
#   Date      2026-09-23
#   Version   v1.0
#
# Every stage below was executed successfully on a Console instance
# (Wormhole N150, board 010001851171e0c, firmware 19.11.0, KMD 2.9.0) on
# 2026-09-23. Results are in README.md.
#
# WHY THIS ROUTE AND NOT pip install ttnn
#   The Console image carries a fully built tt-metal at ~/tt-metal, matched to
#   the SFPI toolchain installed at /opt/tenstorrent/sfpi. No pip ttnn wheel
#   lines up with that toolchain: 0.70.1 fails to compile kernels with
#   "no matching function for call to setman", and a newer wheel fails with
#   "'lreg_pressure' is not a member of 'sfpi'". Use the image's build.
#
# WHY THE PATCHES
#   The pinned commit is dated 2026-05-16 and needs five mechanical changes to run
#   on a current tt-metal. None of them touch the model's arithmetic. See
#   README.md and the patches/ directory.
#
# Usage, from anywhere:
#     bash setup_yolov4_on_ttmetal.sh
# Then follow the printed verification commands.
#
# Roughly 10 minutes on a cold instance, most of it cloning tt-metal and
# compiling kernels on the first run.
# =============================================================================

set -euo pipefail

# Pinned so a machine built in October and one built in November are
# identical. This is the commit every figure in the brief was measured on.
COMMIT="3a0e3987db56711450ec04d388dadcd723d80c98"
SRC="$HOME/ttm"                 # scratch clone of the model source
DST="$HOME/tt-metal"            # the image's built tt-metal
WEIGHTS_REL="models/demos/yolov4/tests/pcc/yolov4.pth"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

say() { printf '\n========== %s ==========\n' "$*"; }

# -----------------------------------------------------------------------------
say "STAGE 0  sanity checks"
# -----------------------------------------------------------------------------
[ -d "$DST/build/lib" ] || { echo "No built tt-metal at $DST/build/lib. Wrong image?"; exit 1; }
ls "$DST/build/lib/_ttnncpp.so" >/dev/null
echo "found built tt-metal:"
ls -la "$DST/build/lib/_ttnncpp.so" | sed 's/^/  /'
export PATH="$HOME/.local/bin:$PATH"
tt-smi --help >/dev/null 2>&1 && echo "tt-smi present (run it interactively to see the board)"

# -----------------------------------------------------------------------------
say "STAGE 1  environment"
# -----------------------------------------------------------------------------
# The image ships its own virtual environment with torch, torchvision, cv2,
# datasets, loguru and pytest already present. Nothing needs pip installing for
# the model itself. Do NOT install ttnn, torch or torchvision into this
# environment: it is matched to the build and a stray install breaks the card
# with an undefined C++ symbol that gives no hint of the cause.
# shellcheck disable=SC1091
source "$DST/python_env/bin/activate"
export TT_METAL_HOME="$DST"
export PYTHONPATH="$DST"
cd "$DST"
python --version
python - <<'PYEOF'
import ttnn, torch, cv2
print("ttnn        :", getattr(ttnn, "__version__", "built from source"))
print("torch       :", torch.__version__)
print("opencv      :", cv2.__version__)
PYEOF

# -----------------------------------------------------------------------------
say "STAGE 2  model source"
# -----------------------------------------------------------------------------
# The repository is public, so this needs no permissions. Fetching the commit
# directly rather than a branch means the result cannot move under us.
if [ -d "$SRC/.git" ]; then
  echo "reusing existing clone at $SRC"
else
  git init -q "$SRC"
  git -C "$SRC" remote add origin https://github.com/tenstorrent/tt-metal.git
  # Materialise only the three paths this challenge uses. Without it the
  # checkout lands all 16,514 files of tt-metal at that commit, including the
  # other YOLO models, on this machine. We copy three of them out below.
  git -C "$SRC" config extensions.partialClone origin
  git -C "$SRC" sparse-checkout init --cone
  git -C "$SRC" sparse-checkout set models/demos/yolov4 models/demos/utils models/tt_cnn
  # Fetching one commit by its hash is by far the cheapest route and GitHub
  # allows it, but a proxy or an older git may refuse. Fall back to a full
  # fetch rather than fail, since the commit is what matters, not how it got
  # here. Either way the checkout below pins the result.
  if git -C "$SRC" fetch -q --depth 1 --filter=blob:none origin "$COMMIT" 2>/dev/null; then
    git -C "$SRC" checkout -q FETCH_HEAD
  else
    echo "  fetch by commit refused, falling back to a full fetch (slower)"
    git -C "$SRC" fetch -q --filter=blob:none origin
    git -C "$SRC" checkout -q "$COMMIT"
  fi
fi
git -C "$SRC" log -1 --format='model source: %h  %ad%n  %s' --date=short

cp -r "$SRC/models/demos/yolov4" models/demos/
[ -d models/demos/utils ] || cp -r "$SRC/models/demos/utils" models/demos/
[ -d models/tt_cnn ]      || cp -r "$SRC/models/tt_cnn"      models/
echo "copied: models/demos/yolov4, models/demos/utils, models/tt_cnn"

# -----------------------------------------------------------------------------
say "STAGE 3  pretrained weights"
# -----------------------------------------------------------------------------
# Public Google Drive download, file id 1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ, via
# the script that ships with the model. No internal cache and no IRD_LF_CACHE,
# which is the opposite of the tt-forge-models copy of yolov4 whose weights sit
# in models/tt-ci-models-private. External participants can be served from this.
if [ -f "$WEIGHTS_REL" ]; then
  echo "weights already present:"; ls -la "$WEIGHTS_REL" | sed 's/^/  /'
else
  # The Console image no longer guarantees pip inside python_env. Observed
  # 2026-09-24 on a fresh image: "No module named pip", which aborted this
  # script under set -e and silently skipped the patches in STAGE 4. Try three
  # routes before giving up, and put gdown on PATH however it gets installed.
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v gdown >/dev/null 2>&1; then
    python -m pip install --quiet gdown 2>/dev/null \
      || { python -m ensurepip --upgrade >/dev/null 2>&1 \
           && python -m pip install --quiet gdown; } \
      || python3 -m pip install --quiet --user gdown \
      || { echo "could not install gdown by any route."; \
           echo "install it by hand, then re-run this script."; exit 1; }
  fi
  command -v gdown | sed 's/^/  gdown: /'
  bash models/demos/yolov4/tests/pcc/yolov4_weights_download.sh
  ls -la "$WEIGHTS_REL" | sed 's/^/  /'
fi

# -----------------------------------------------------------------------------
say "STAGE 4  patches"
# -----------------------------------------------------------------------------
python3 "$HERE/patches/01_builder_activations.py"  "$DST"
python3 "$HERE/patches/02_demo_imports.py"         "$DST"
python3 "$HERE/patches/03_perblock_pcc_tests.py"   "$DST"
python3 "$HERE/patches/04_demo_all_images.py"     "$DST"

# -----------------------------------------------------------------------------
say "STAGE 5  warm the kernel cache"
# -----------------------------------------------------------------------------
# First run compiles roughly 600 kernels and takes a couple of minutes.
# Afterwards the cache reports close to 100% hits and the same run takes
# seconds. Do it during image preparation so a team's first run is quick.
#
# Warm the bench path rather than demo.py. Measured 14 Oct 2026: warming via
# demo.py left the bench at 0/610 cache hits, because the two drive the model
# through different convolution configurations and so compile different
# kernels. The bench path is the one that matters, since webcam_demo/server.py
# prepares its input identically, so warming this warms both.
echo "running the bench once to populate the JIT kernel cache"
echo "(slow on a fresh machine, about two minutes, first time only)"
bash "$HERE/../bench/run_bench.sh" --quick --320-only || true

# -----------------------------------------------------------------------------
say "STAGE 6  environment for every new terminal"
# -----------------------------------------------------------------------------
MARK="# --- hackathon yolov4 environment ---"
# Instances set up before 6 Nov 2026 carry the block under an earlier marker.
# Its activate line is the same, so look for that too and do not append twice.
ACTIVATE="source \"$DST/python_env/bin/activate\""
if ! grep -qF -e "$MARK" -e "$ACTIVATE" "$HOME/.bashrc" 2>/dev/null; then
  cat >> "$HOME/.bashrc" <<BASHEOF

$MARK
export PATH="\$HOME/.local/bin:\$PATH"
source "$DST/python_env/bin/activate"
export TT_METAL_HOME="$DST"
export PYTHONPATH="$DST"
cd "$DST"
BASHEOF
  echo "appended environment to ~/.bashrc"
else
  echo "~/.bashrc already carries the environment"
fi

# The editor's Python extension types an activate command for its own chosen
# interpreter into every new terminal, after .bashrc has run. On the event
# instances that interpreter was tt-forge-venv, whose ttnn build imports but
# fails when it compiles kernels. Point the editor at python_env and stop it
# activating anything in terminals. Observed and fixed by hand on 5 Nov 2026.
python3 - "$DST" <<'PYEOF'
import json, pathlib, sys
p = pathlib.Path.home() / ".local/share/code-server/User/settings.json"
p.parent.mkdir(parents=True, exist_ok=True)
try:
    s = json.loads(p.read_text()) if p.exists() and p.read_text().strip() else {}
except ValueError:
    print(f"could not parse {p}; set python.terminal.activateEnvironment to false by hand")
    sys.exit(0)
s["python.terminal.activateEnvironment"] = False
s["python.defaultInterpreterPath"] = sys.argv[1] + "/python_env/bin/python"
p.write_text(json.dumps(s, indent=2) + "\n")
print(f"editor settings: terminals stay in python_env ({p})")
PYEOF
echo "reload the browser tab, then open a new terminal, for this to take effect"

# -----------------------------------------------------------------------------
say "DONE"
# -----------------------------------------------------------------------------
cat <<'SUMMARY'
Verify, from ~/tt-metal:

  # detector on real images, writes to yolov4_predictions/
  python -m pytest -x "models/demos/yolov4/demo.py" \
    -k "test_yolov4 and not coco and not dp"

  # numerical check against the PyTorch reference
  python -m pytest -x -q \
    "models/demos/yolov4/tests/pcc/test_ttnn_yolov4.py::test_yolov4[0-pretrained_weight_true-0]"

  # per-frame timing, stage by stage
  bash ~/challenge/bench/run_bench.sh --quick --320-only

Expected, measured 2026-09-24 on a Wormhole n150 L:
  demo at 320x320   giraffe 1.0, zebra 1.0 (giraffe.jpg; dinning.png is also run)
  demo at 640x640   giraffe 0.969, zebra weak (0.028, 0.06)
  PCC test          boxes pass at 0.99; confidences report 0.0208 and FAIL. This
                    is expected and explained in README.md. Do not chase it.
  bench at 320      69.122 ms a frame, 14.5 frames a second

To use your own images, drop JPEGs into models/demos/yolov4/resources/ and
re-run the demo. Every image in that folder is run, in name order, and each
lands in yolov4_predictions/ with its detection count in the log.

If the demo reports PASSED, check it actually found something. It reports
success even when it detects nothing, so open the output images rather than
trusting the exit code.

Take your measurements with the browser tab in the foreground. Chrome throttles
background tabs, which shows up as a JPEG encode of a second or more.
SUMMARY
