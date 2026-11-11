# claude_fix, step 1: WP-1 device split probe

Guide: `docs/GUIDE_SPEED_ACCURACY_MEMORY.md`, work package WP-1. No change to `challenge/my_submission.py`.

`tools/probe_device_split.py` measures in one card run (about 3 minutes):

| Phase | Measures | Decides |
|---|---|---|
| A | trace only, trace + D2H read, full production chain of milestone 8 | WP-4 (read on CQ1) is worth it if `read_ms` >= 0.15 |
| C | traced cost of genboxes + output epilogue vs. the raw-head replacement | the WP-3 prize (`prize_ms`); also saves raw head tensors for a laptop parity check |
| B | `ttnn.max` / `ttnn.argmax` over 80 classes, untraced | whether `trunc_confs` is admissible |
| D | copy-into-preallocated APIs, conv double-buffer defaults | how WP-4 and WP-5a get written |

## Run on the instance

Nothing else may use the card (no webcam demo, no evaluate.py).

```bash
cd ~/SEMICON-Hackathon && git fetch origin && git checkout claude_fix && git pull
source ~/tt-metal/python_env/bin/activate
export TT_METAL_HOME=~/tt-metal PYTHONPATH=~/tt-metal
python ~/SEMICON-Hackathon/tools/probe_device_split.py \
    --out-dir ~/SEMICON-Hackathon/experiments/probes 2>&1 | tee /tmp/probe.log
```

## Send back

The `==== summary ====` block at the end of `/tmp/probe.log`, and the file
`experiments/probes/probe_device_split_<stamp>.json` (commit it, or paste it).
If a phase printed `FAILED`, the traceback is in the JSON under `tracebacks`.
The two `.npz` files (about 5 MB) are only needed later for WP-3; keep them on the instance.
