#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Patch models/demos/yolov4/demo.py: two imports that no longer resolve.

  Date      2026-09-23
  Version   v1.0

FIXES
    ImportError while importing test module '.../models/demos/yolov4/demo.py'
    cannot import name 'disable_persistent_kernel_cache'
        from 'models.common.utility_functions'

WHY, ITEM 1
    `disable_persistent_kernel_cache` has been removed from tt-metal entirely.
    Kernel caching is managed internally now, which is what the "JIT cache
    stats" telemetry in the run logs reports. Current demos, for example
    models/demos/vision/detection/rtdetr/demo.py, do not call it.

    `run_for_wormhole_b0` does still exist, at line 1096 of the same module, so
    that half of the import stays. Rather than edit the two call sites at lines
    40 and 135, this patch defines a no-op of the same name, so nothing else in
    the file changes.

WHY, ITEM 2
    `fiftyone` is imported at module level but used only at line 172, inside the
    COCO variants. An absent fiftyone therefore blocked the plain image demo
    too. It is now optional. If you later want test_yolov4_coco you will need
    `pip install fiftyone`, which pulls in MongoDB and is not small.

NOT FIXED HERE, ON PURPOSE
    conf_thresh (0.3) and nms_thresh (0.4) are hard-coded at two sites each.
    They are exactly the knobs a hackathon team needs to move, so they should
    become command-line arguments rather than being patched. See README.md.

    The demo also reports PASSED when it detects nothing at all, because it
    asserts nothing about its output. That cost an hour of this session.
    The baseline script handed to teams must assert at least one detection.

SCOPE
    One file, two insertions, both in the import block. Idempotent, and
    restores from demo.py.orig if run twice.

Usage:
    python3 02_demo_imports.py <tt-metal-root>
    e.g.  python3 02_demo_imports.py ~/tt-metal
"""

import sys
from pathlib import Path

MARKER = "# --- hackathon demo patch"

ANCHOR_FIFTYONE = "import fiftyone\n"
BLOCK_FIFTYONE = """# --- hackathon demo patch: fiftyone is only needed by the coco variants ---
try:
    import fiftyone
except ImportError:
    fiftyone = None
"""

ANCHOR_UTILS = (
    "from models.common.utility_functions import "
    "disable_persistent_kernel_cache, run_for_wormhole_b0\n"
)
BLOCK_UTILS = """from models.common.utility_functions import run_for_wormhole_b0

# --- hackathon demo patch: removed from tt-metal; caching is internal now ---
def disable_persistent_kernel_cache():
    pass

"""

def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        print(f"error: give the tt-metal root, e.g. python3 {sys.argv[0]} ~/tt-metal")
        return 2

    root = Path(sys.argv[1]).expanduser()
    target = root / "models" / "demos" / "yolov4" / "demo.py"
    if not target.is_file():
        print(f"error: not found: {target}")
        print("Copy models/demos/yolov4 from the model source into tt-metal first.")
        return 1

    backup = target.with_suffix(".py.orig")
    text = backup.read_text() if backup.exists() else target.read_text()

    for name, anchor in (
        ("fiftyone import", ANCHOR_FIFTYONE),
        ("utility_functions import", ANCHOR_UTILS),
    ):
        if text.count(anchor) != 1:
            print(f"error: anchor for {name} found {text.count(anchor)} times, expected 1.")
            print("demo.py differs from the branch this patch was written against.")
            return 1

    if not backup.exists():
        backup.write_text(text)
        print(f"backup written: {backup}")

    text = text.replace(ANCHOR_FIFTYONE, BLOCK_FIFTYONE, 1)
    text = text.replace(ANCHOR_UTILS, BLOCK_UTILS, 1)
    target.write_text(text)
    print(f"patched: {target}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
