#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Patch models/demos/yolov4/tt/model_preprocessing.py: the per-block accuracy
tests cannot construct their models.

  Date      2026-10-14
  Version   v1.0

FIXES
    Every test under models/demos/yolov4/tests/pcc/ except the whole-model one
    fails before it runs anything. Three independent defects, each hiding the
    next, found by fixing them one at a time on a Wormhole n150 on 14 Oct 2026:

      1. Unsupported parameter: mesh_mapper
      2. TypeError: _create_dsN_model_parameters() missing 1 required
         positional argument: 'resolution'
      3. KeyError: 'resolution' at models/demos/yolov4/tt/head.py:178

WHY, ITEM 1
    custom_preprocessor (line 14) takes mesh_mapper=None. The whole-model
    builder, create_yolov4_model_parameters, wraps it:

        _, weights_mesh_mapper, _ = get_mesh_mappers(device)
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper)

    The seven per-block builders pass it raw, so the mesh plumbing reaches a
    function that does not expect it. This patch makes them do what the
    whole-model builder already does.

WHY, ITEM 2
    create_ds1 calls _create_ds1_model_parameters(conv_args, resolution).
    ds2 to ds5 and neck call theirs with conv_args alone, while every one of
    those private helpers requires resolution. Fixed by passing it.

WHY, ITEM 3
    Every builder sets parameters["resolution"] except the head one, and
    head.py reads self.parameters.resolution at line 178. Fixed by setting it.

    downsample1 is the only builder that was complete in all three respects,
    which is consistent with a suite where the first test is the one people run.

WHAT THIS DOES NOT FIX
    With the above applied, six of the seven produce a correlation figure and
    one does not. Measured at 320x320 on random-noise input, which is what the
    tests feed:

        downsample1  0.9202      downsample4  0.4507
        downsample2  0.9728      downsample5  RuntimeError, TT_THROW at
        downsample3  0.8824                   tt_metal/impl/program/program.cpp:1868
        neck         0.9806      head         0.9821

    Five of those sit where low-precision convolution weights on noise would
    put them, well below thresholds that were evidently set under different
    conditions. downsample4 at 0.45 and downsample5 throwing are real problems
    in those blocks, not stale thresholds, and are reported upstream rather
    than patched here.

    So treat this suite as a way to see which block your change moved, against
    the baseline above, and not as a pass or fail gate.

SAFETY
    The whole-model path is untouched. Verified on 14 Oct 2026 after applying:
    the whole-model accuracy test still passes, the bench reports 65.768 ms a
    frame against 66.047 before, and the JIT kernel cache reports 610/610 hits,
    meaning the whole-model graph compiles to byte-identical kernels.
"""

import re
import sys
from pathlib import Path

PER_BLOCK = ("ds1", "ds2", "ds3", "ds4", "ds5", "neck", "head")

# item 1: wrap the preprocessor, exactly as the whole-model builder does
MESH_PATTERN = re.compile(
    r"(def create_(?:ds[1-5]|neck|head)_model_parameters\("
    r"model: yolov4\.Yolov4, input_tensor: torch\.Tensor, resolution, device\):\n)"
    r"(    parameters = preprocess_model_parameters\(\n"
    r"        initialize_model=lambda: model,\n)"
    r"        custom_preprocessor=custom_preprocessor,\n"
)

def mesh_sub(m):
    return (
        m.group(1)
        + "    _, weights_mesh_mapper, _ = get_mesh_mappers(device)\n"
        + m.group(2)
        + "        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),\n"
    )

# item 3: the head builder never records the resolution it was given
HEAD_PATTERN = re.compile(
    r"(def create_head_model_parameters\(.*?\n)(.*?)(    parameters\.conv_args = \{\}\n)",
    re.S,
)

def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "tt-metal"
    target = root / "models/demos/yolov4/tt/model_preprocessing.py"
    if not target.is_file():
        print(f"error: not found: {target}")
        print("Copy models/demos/yolov4 from the model source into tt-metal first.")
        return 1

    # Always transform from the pristine original, so a second run is a no-op
    # rather than a patch applied on top of a patch.
    backup = target.with_suffix(".py.orig")
    if backup.exists():
        text = backup.read_text()
    else:
        text = target.read_text()
        backup.write_text(text)
        print(f"backup written: {backup}")

    # Refuse on anything that is not the file this was written against, rather
    # than half-applying and leaving a worse mess than we found.
    n_raw = text.count("        custom_preprocessor=custom_preprocessor,\n")
    if n_raw != len(PER_BLOCK):
        print(f"error: found {n_raw} unwrapped preprocessors, expected {len(PER_BLOCK)}.")
        print("model_preprocessing.py differs from the commit this patch was written against.")
        return 1

    text, n_mesh = MESH_PATTERN.subn(mesh_sub, text)
    if n_mesh != len(PER_BLOCK):
        print(f"error: matched {n_mesh} per-block builders, expected {len(PER_BLOCK)}.")
        return 1

    n_res = 0
    for name in ("ds2", "ds3", "ds4", "ds5", "neck"):
        old = f"    _create_{name}_model_parameters(parameters.conv_args)\n"
        new = f"    _create_{name}_model_parameters(parameters.conv_args, resolution)\n"
        if old in text:
            text = text.replace(old, new, 1)
            n_res += 1
    if n_res != 5:
        print(f"error: fixed {n_res} helper calls, expected 5.")
        return 1

    m = HEAD_PATTERN.search(text)
    if not m:
        print("error: could not locate create_head_model_parameters.")
        return 1
    if 'parameters["resolution"] = resolution' in m.group(2):
        n_head = 0
    else:
        text = text[: m.start(3)] + '    parameters["resolution"] = resolution\n' + text[m.start(3) :]
        n_head = 1

    target.write_text(text)
    print(f"patched: {target}")
    print(f"  {n_mesh} preprocessors wrapped, {n_res} helper calls fixed, "
          f"{n_head} resolution assignment added")
    return 0

if __name__ == "__main__":
    sys.exit(main())
