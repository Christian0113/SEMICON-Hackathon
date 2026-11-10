# Raw measurement output

Every number in the brief and in the top-level README comes from a file in
here. Nothing is retyped, so if a figure looks wrong, check it here first.

| file | what it is |
|---|---|
| `bench_320.json` | One frame, stage by stage, 320x320, 300 measured frames after 20 discarded. Produced by `bench/run_bench.sh`. |
| `board.txt` | A reduced `tt-smi` record for that session: board type, board ID, firmware bundle version and driver version, so the numbers have a board attached to them. The full dump is not published. The file itself lists what was removed. |
| `env.txt` | Python, torch, opencv and the tt-metal commit for that session. |
| `loop_results_20260924_093510/` | Nine files from `bench/run_loop.sh`: the whole frame on the traced path at 320 with one command queue, 320 with two, and 640 with one. The 640 file's `validated` field reads `FAILED: 0.7678787216744429`, so its timings describe a model that is not computing the right answer. The same figure reproduced on 26 October 2026. |
| `rig_cost.json` | What the practice setup costs: round trip, network, and the work on the machine, summarised per run. |
| `roundtrip_*.jsonl` | The raw per-frame records behind `rig_cost.json`, one JSON object per line, as the demo page reported them. |

## Two fields will not match your own runs

`path` and `host` were generalised before these files were published, so your
own output will differ from ours in those two places and nowhere else. Each
file says so in `path_note` and `host_note` beside the value. Every measured
number is byte-identical to the run that produced it.

The three `.txt` logs in `loop_results_20260924_093510/` are plain text and
have no note field, so their one edit is recorded here: a directory name in a
file path was shortened to `hackathon_yolov4`. Nothing else in them changed.

## Reproducing it

```bash
bash bench/run_bench.sh              # bench_320.json, board.txt, env.txt
python3 webcam_demo/summarise_roundtrip.py roundtrip_*.jsonl --out rig_cost.json
```

The second one needs records from the demo page first. Run the demo, press
Start, let it run, press Stop.
