#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

# =============================================================================
# start_webcam_demo.sh
#
# One command to get the live webcam detection demo running on a fresh
# Tenstorrent Console VSCode Server instance. Written for someone who has not
# seen this stack before.
#
#   Date      2026-09-23
#   Version   v0.1
#
# WHAT IT DOES
#   1. Checks you are on an instance with a card and a built tt-metal.
#   2. Runs the base setup if it has not been done: fetches the model code and
#      weights, applies the patches. Skipped automatically on a second run.
#   3. Starts the detection server and tells you what to click.
#
# WHAT YOU DO
#       bash start_webcam_demo.sh
#   then accept the port-forwarding prompt in VS Code, or open the PORTS tab
#   beside TERMINAL and click the forwarded address. Click "Start camera".
#
# HOW LONG
#   First run on a fresh instance: about 10 to 15 minutes, nearly all of it
#   downloading the model repository and compiling kernels. After that, under a
#   minute.
#
# OPTIONS
#   --skip-setup     assume the base setup is already done, go straight to the
#                    server. Use this if you know the machine is prepared.
#   --port N         serve on a different port (default 8000)
#
# IF SOMETHING GOES WRONG
#   Every failure below prints what it expected and what it found. Nothing here
#   modifies the tt-metal build itself, so a bad run costs you nothing but time.
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$(cd "$HERE/.." && pwd)"
DST="$HOME/tt-metal"
PORT="${PORT:-8000}"
SKIP_SETUP=0

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-setup) SKIP_SETUP=1; shift ;;
    --port)       PORT="$2"; shift 2 ;;
    -h|--help)    sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)            echo "unknown option: $1"; exit 2 ;;
  esac
done

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31mSTOP: %s\033[0m\n' "$*"; exit 1; }

# -----------------------------------------------------------------------------
step "1/5  Checking this instance"
# -----------------------------------------------------------------------------
[ -f "$HERE/server.py" ]  || fail "server.py is not next to this script, in $HERE"
[ -f "$HERE/index.html" ] || fail "index.html is not next to this script, in $HERE.
Both files must sit in the same folder as this script. In VS Code use
File > New File, paste, and save into $HERE."

[ -d "$DST/build/lib" ] || fail "No built tt-metal at $DST/build/lib.
This script expects the Tenstorrent Console 'development environment' image,
which ships a built tt-metal in your home directory. If you are somewhere else,
you will have to build or install tt-metal first."

[ -f "$DST/python_env/bin/activate" ] || fail "No python_env at $DST/python_env.
The image normally provides one. Run 'ls $DST' and see what is there."

export PATH="$HOME/.local/bin:$PATH"
if command -v tt-smi >/dev/null 2>&1; then
  echo "  tt-smi found. Run 'tt-smi' on its own later if you want to see the board."
else
  echo "  note: tt-smi not on PATH. Not fatal, but check the card is present."
fi
echo "  tt-metal build : $DST/build/lib"
echo "  demo files     : $HERE"
echo "  port           : $PORT"

# -----------------------------------------------------------------------------
step "2/5  Is the model already set up?"
# -----------------------------------------------------------------------------
NEED_SETUP=0
[ -d "$DST/models/demos/yolov4" ] || NEED_SETUP=1
[ -f "$DST/models/demos/yolov4/tests/pcc/yolov4.pth" ] || NEED_SETUP=1
grep -q "hackathon patch" "$DST/models/tt_cnn/tt/builder.py" 2>/dev/null || NEED_SETUP=1
grep -q "hackathon demo patch" "$DST/models/demos/yolov4/demo.py" 2>/dev/null || NEED_SETUP=1

if [ "$SKIP_SETUP" = "1" ]; then
  echo "  --skip-setup given, not checking further"
elif [ "$NEED_SETUP" = "0" ]; then
  echo "  yes: model code, weights and both patches are in place"
else
  echo "  no: something is missing, running the base setup"
  [ -x "$PKG/setup_yolov4_on_ttmetal.sh" ] || [ -f "$PKG/setup_yolov4_on_ttmetal.sh" ] || fail \
"setup_yolov4_on_ttmetal.sh was not found at $PKG.

webcam_demo/ has been separated from the folder above it. Everything should
be together under ~/challenge, which is how your instance was prepared.
setup/setup_yolov4_on_ttmetal.sh rebuilds the lot."

  step "3/5  Base setup (this is the slow part, 10 to 15 minutes)"
  bash "$PKG/setup_yolov4_on_ttmetal.sh"
fi

# -----------------------------------------------------------------------------
step "4/5  Environment"
# -----------------------------------------------------------------------------
# These three lines are the whole trick. Use the tt-metal that ships in the
# image, and its own virtual environment, which already carries torch,
# torchvision, OpenCV and the rest. Do not pip install ttnn, torch or
# torchvision into it: the build is matched to the SFPI toolchain on this
# machine and a stray install breaks the card with an undefined C++ symbol that
# gives no hint of the cause.
# shellcheck disable=SC1091
source "$DST/python_env/bin/activate"
export TT_METAL_HOME="$DST"
export PYTHONPATH="$DST"
cd "$DST"

python - <<'PYEOF'
import sys
try:
    import ttnn, torch, cv2  # noqa: F401
except Exception as exc:
    sys.exit(f"environment is not usable: {exc!r}")
print(f"  python   : {sys.version.split()[0]}")
print(f"  torch    : {torch.__version__}")
print(f"  opencv   : {cv2.__version__}")
PYEOF

# -----------------------------------------------------------------------------
step "5/5  Starting the detection server"
# -----------------------------------------------------------------------------
cat <<NOTE

  Startup takes a minute or so. It opens the card, loads the weights, builds the
  model, then runs one frame to compile the kernels. Wait for:

      ready on port $PORT

  Then, in VS Code:
    - accept the port-forwarding prompt if one appears, or
    - open the PORTS tab beside TERMINAL and click the forwarded address.

  On the page, click "Start camera" and allow access when the browser asks.

  Chrome only grants the camera over https or localhost. The VS Code forwarded
  address is https, so it works. A raw http:// address will be refused.

  Ctrl+C here stops the server.

NOTE

PORT="$PORT" exec python "$HERE/server.py"
