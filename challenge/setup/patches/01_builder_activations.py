#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Patch models/tt_cnn/tt/builder.py: activations tt-metal can no longer fuse.

  Date      2026-09-23
  Version   v1.0  (supersedes the v1/v2 iterations from the live session)

FIXES TWO ERRORS
    RuntimeError: TT_THROW @ .../matmul/device/utilities/matmul_utilities.hpp:332
    Unsupported UnaryOpType for fused activation: UnaryOpType::HARDMISH
    Unsupported UnaryOpType for fused activation: UnaryOpType::LEAKY_RELU

WHY
    Current tt-metal will fuse only ten activations into a conv2d/matmul
    epilogue. From the switch in matmul_utilities.hpp:
        GELU, GELU_TANH, TANH, SILU, RELU6, SIGMOID,
        HARDSIGMOID, HARDTANH, SELU, SOFTPLUS
    YOLOv4 asks for HARDMISH (72 call sites, downsample1..downsample5) and
    LEAKY_RELU (35 call sites, neck and head). Neither is in that set. The
    branch predates the restriction.

    Both still exist as standalone unary operations:
        ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp
            line  97  DECLARE_UNARY_OP(hardmish)
            line 122  DECLARE_UNARY_OP_WITH_FAST_AND_APPROXIMATE_MODE(mish)
    so this patch stops asking the convolution to fuse them and applies the same
    activation to the convolution's output. Mathematically equivalent, so the
    PCC tests stay a real check. Slower, because it is a separate pass rather
    than a fused epilogue.

    Substituting a fusable activation such as SILU would also have run, and
    would have silently changed the model's numerics. Not acceptable in a
    challenge about measuring what you gave up.

THE LEAKY_RELU SLOPE, AND WHY IT IS HARD-CODED
    ttnn.UnaryWithParam exposes only `op_type` to Python. Neither `params` nor
    `param` is readable:
        >>> a = ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1)
        >>> [x for x in dir(a) if not x.startswith('_')]
        ['op_type']
    So the slope cannot be recovered from the object. An earlier version of this
    patch fell back to 0.01, a tenfold error across 35 layers, and the symptom
    was subtle and expensive: box regression still matched the reference at PCC
    0.99, but the confidence head produced nothing, the demo detected zero
    objects, and pytest reported PASSED.

    All 35 call sites in YOLOv4 pass 0.1, so 0.1 is correct here. It is
    model-specific. If this patch is reused for another model, check its slope.

    The inability to read the parameter back is arguably a gap in tt-metal worth
    reporting.

SCOPE
    One file, three insertions. Covers every convolution in the model, because
    all of them go through TtConv2d and none call ttnn.conv2d directly.
    Idempotent, and restores from builder.py.orig if run twice.

Usage:
    python3 01_builder_activations.py <tt-metal-root>
    e.g.  python3 01_builder_activations.py ~/tt-metal
"""

import sys
from pathlib import Path

MARKER = "# --- hackathon patch: activations tt-metal can no longer fuse ---"

ANCHOR_IMPORT = "from ttnn.model_preprocessing import Conv2dArgs\n\nimport ttnn\n"

BLOCK_IMPORT = ANCHOR_IMPORT + f'''
{MARKER}
# Fusable set read from the switch in
#   ttnn/cpp/ttnn/operations/matmul/device/utilities/matmul_utilities.hpp
# Anything outside it is applied after the convolution instead.
import dataclasses as _dc

_FUSABLE = {{
    "GELU", "GELU_TANH", "TANH", "SILU", "RELU6", "SIGMOID",
    "HARDSIGMOID", "HARDTANH", "SELU", "SOFTPLUS",
}}

# ttnn.UnaryWithParam does not expose its parameter to Python, so any slope has
# to be supplied here. 0.1 is what all 35 YOLOv4 call sites pass.
_POST = {{
    "HARDMISH":   lambda x, p: ttnn.hardmish(x),
    "MISH":       lambda x, p: ttnn.mish(x),
    "LEAKY_RELU": lambda x, p: ttnn.leaky_relu(x, p[0] if p else 0.1),
    "RELU":       lambda x, p: ttnn.relu(x),
}}

def _act_params(act):
    for attr in ("params", "param"):
        v = getattr(act, attr, None)
        if v is None:
            continue
        try:
            return list(v)
        except TypeError:
            return [v]
    return []

def _split_unfusable_activation(cfg):
    """Return (configuration, post_activation_fn).

    If the configuration asks for an activation this tt-metal cannot fuse, clear
    it from the configuration and hand back a callable to apply afterwards.
    """
    act = getattr(cfg, "activation", None)
    if act is None:
        return cfg, None
    ot = getattr(act, "op_type", None)
    name = getattr(ot, "name", None) or str(ot).rsplit(".", 1)[-1]
    if name in _FUSABLE:
        return cfg, None
    fn = _POST.get(name)
    if fn is None:
        raise RuntimeError(
            f"activation {{name}} is not fusable in this tt-metal and has no "
            f"post-op mapping in the hackathon patch. Add it to _POST."
        )
    params = _act_params(act)
    return _dc.replace(cfg, activation=None), (lambda x: fn(x, params))
'''

ANCHOR_INIT = (
    "        self.configuration = configuration\n"
    "        self.conv2d_config = to_conv2d_config(configuration)\n"
)
BLOCK_INIT = (
    "        configuration, self.post_activation = _split_unfusable_activation(configuration)\n"
    + ANCHOR_INIT
)

ANCHOR_CALL = (
    "            x, (h_out, w_out) = self._apply_channel_slicing(x)\n"
    "\n"
    "        if return_output_dim:\n"
)
BLOCK_CALL = (
    "            x, (h_out, w_out) = self._apply_channel_slicing(x)\n"
    "\n"
    "        if self.post_activation is not None:\n"
    "            x = self.post_activation(x)\n"
    "\n"
    "        if return_output_dim:\n"
)

def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        print(f"error: give the tt-metal root, e.g. python3 {sys.argv[0]} ~/tt-metal")
        return 2

    root = Path(sys.argv[1]).expanduser()
    target = root / "models" / "tt_cnn" / "tt" / "builder.py"
    if not target.is_file():
        print(f"error: not found: {target}")
        print("Copy models/tt_cnn from the model source into tt-metal first.")
        return 1

    backup = target.with_suffix(".py.orig")
    text = backup.read_text() if backup.exists() else target.read_text()

    for name, anchor in (
        ("import block", ANCHOR_IMPORT),
        ("TtConv2d.__init__", ANCHOR_INIT),
        ("TtConv2d.__call__", ANCHOR_CALL),
    ):
        if text.count(anchor) != 1:
            print(f"error: anchor for {name} found {text.count(anchor)} times, expected 1.")
            print("builder.py differs from the branch this patch was written against.")
            print("Do not force it. Report the mismatch instead.")
            return 1

    if not backup.exists():
        backup.write_text(text)
        print(f"backup written: {backup}")

    text = text.replace(ANCHOR_IMPORT, BLOCK_IMPORT, 1)
    text = text.replace(ANCHOR_INIT, BLOCK_INIT, 1)
    text = text.replace(ANCHOR_CALL, BLOCK_CALL, 1)
    target.write_text(text)
    print(f"patched: {target}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
