#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Turn the browser's per-frame records into a citable measurement.

    Date      2026-09-24
    Version   v0.1
    Measured  2026-09-24, output in ../data/rig_cost.json

WHY THIS EXISTS
    The cost of the practice rig, meaning the laptop camera, the JPEG encode
    and the internet in both directions, used to be visible only as a figure on
    the demo page, which is not something you can put in a change log.
    server.py now appends a record per frame to a JSONL file and this reads that
    file, so the rig's cost arrives the same way every other number in the
    challenge does: written to disk, from a named machine, reproducible.

HOW THE SUBTRACTION WORKS, AND WHY IT IS SOUND
    Two clocks are involved and they do not agree with each other. The browser
    is on a laptop, the server is in a data centre, and nothing keeps them in
    step. So no timestamp from one is ever compared with a timestamp from the
    other. Only durations are used, each measured entirely within one machine:

        rtt_ms            browser: from just before the JPEG is sent to just
                          after the reply has been read and parsed
        server_total_ms   server:  from the first byte of the request arriving
                          to the answer being ready to write

    network_ms = rtt_ms - server_total_ms

    Subtracting one duration from another is unaffected by the offset between
    the clocks. It is affected by relative drift, but over the few hundred
    milliseconds of one frame that is far below the resolution being reported.

    What network_ms contains: the request crossing the network, the reply
    crossing back, TLS and HTTP framing on both, and the browser's own parse.
    What it does not contain: capture_ms and encode_ms, which the page times
    directly and which happen before the stopwatch starts. Those are reported
    separately and added back for the "cost of the rig" total, because a
    machine with the camera on the board pays neither.

WHAT TO DO WITH THE ANSWER
    onmachine_ms is what the challenge scores. Everything else this prints is
    the rig, and exists to be shown on screen and then excluded.

HOW TO RUN
    python summarise_roundtrip.py roundtrip_20260924_101500.jsonl
    python summarise_roundtrip.py roundtrip_*.jsonl --out rig_cost.json

    Records are grouped by the "run" label the page assigns. Changing the
    frames-in-flight setting starts a new run, because it changes what the
    queue wait means and the two must not be pooled.
"""

import argparse
import glob
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone

# Fields worth a percentile table, in the order a frame encounters them.
FIELDS = [
    ("capture_ms", "grab the frame from the camera", "rig"),
    ("encode_ms", "compress it to a JPEG", "rig"),
    ("rtt_ms", "round trip, browser clock", "total"),
    ("network_ms", "internet both ways, derived", "rig"),
    ("server_total_ms", "everything the server process spent", "server"),
    ("read_ms", "read the JPEG off the socket", "server"),
    ("wait_ms", "queued behind other frames", "queue"),
    ("decode_ms", "unpack the JPEG", "scored"),
    ("work_ms", "detect and work out the boxes", "scored"),
    ("onmachine_ms", "decode + work, the scored number", "scored"),
]

def summarise(values):
    """Nearest-rank percentiles, matching bench_stages.py and bench_levers.py."""
    s = sorted(v for v in values if v is not None)
    n = len(s)
    if not n:
        return None

    def pct(p):
        return s[min(n - 1, max(0, int(round(p / 100.0 * n)) - 1))]

    return {
        "n": n,
        "mean": round(statistics.fmean(s), 3),
        "median": round(statistics.median(s), 3),
        "p95": round(pct(95), 3),
        "p99": round(pct(99), 3),
        "min": round(s[0], 3),
        "max": round(s[-1], 3),
        "stdev": round(statistics.stdev(s), 3) if n > 1 else 0.0,
    }

def load(paths):
    rows, bad = [], 0
    for path in paths:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    bad += 1
    return rows, bad

def report(run, rows, args):
    out = {"run": run, "frames": len(rows)}

    drop = args.warmup
    if drop and len(rows) > drop:
        rows = rows[drop:]
        out["warmup_discarded"] = drop
    out["frames_summarised"] = len(rows)

    depths = {r.get("max_in_flight") for r in rows}
    out["max_in_flight"] = sorted(d for d in depths if d is not None)
    out["stats"] = {f: summarise([r.get(f) for r in rows]) for f, _, _ in FIELDS}

    # A negative network time means the two measurements do not bracket each
    # other the way this script assumes. Report it rather than hide it.
    neg = [r for r in rows if isinstance(r.get("network_ms"), (int, float))
           and r["network_ms"] < 0]
    out["negative_network_frames"] = len(neg)

    st = out["stats"]

    def mean(f):
        return st[f]["mean"] if st.get(f) else None

    def med(f):
        return st[f]["median"] if st.get(f) else None

    rig = None
    rig_med = None
    if mean("network_ms") is not None:
        rig = mean("network_ms") + (mean("capture_ms") or 0) + (mean("encode_ms") or 0)
        out["rig_cost_ms_mean"] = round(rig, 3)
        rig_med = med("network_ms") + (med("capture_ms") or 0) + (med("encode_ms") or 0)
        out["rig_cost_ms_median"] = round(rig_med, 3)
    if mean("onmachine_ms") is not None and rig is not None:
        out["rig_multiple_of_scored"] = round(rig / mean("onmachine_ms"), 2)

    print()
    print("=" * 74)
    print(f"  {run}")
    print(f"  {out['frames_summarised']} frames, frames in flight "
          f"{out['max_in_flight']}")
    print("=" * 74)
    print(f"  {'':<34}{'mean':>9}{'median':>9}{'p95':>9}{'max':>9}")
    last_kind = None
    for field, label, kind in FIELDS:
        s = st.get(field)
        if s is None:
            continue
        if kind != last_kind:
            print(f"  -- {kind}")
            last_kind = kind
        print(f"  {label:<34}{s['mean']:>9.2f}{s['median']:>9.2f}"
              f"{s['p95']:>9.2f}{s['max']:>9.2f}   {field}")

    print()
    print("  the subtraction, on means:")
    if mean("rtt_ms") is not None and mean("server_total_ms") is not None:
        print(f"    rtt                      {mean('rtt_ms'):9.2f}")
        print(f"    - server_total          -{mean('server_total_ms'):9.2f}")
        print(f"    = network               ={mean('network_ms'):9.2f}")
        print(f"    + capture + encode      +"
              f"{(mean('capture_ms') or 0) + (mean('encode_ms') or 0):9.2f}")
        print(f"    = cost of the rig       ={rig:9.2f}   "
              f"a real machine pays none of this")
        if rig_med is not None:
            print(f"    same on medians         ={rig_med:9.2f}   "
                  f"quote this one")
            print()
            print("    The JPEG encode is prone to occasional stalls of a "
                  "second or more when the")
            print("    browser tab loses focus, which drags the mean a long "
                  "way from the median.")
            print("    Measure with the tab in the foreground, and quote the "
                  "median.")
    if out.get("rig_multiple_of_scored"):
        print(f"\n    scored, on the machine   {mean('onmachine_ms'):9.2f}")
        print(f"    the rig is {out['rig_multiple_of_scored']}x that")

    if out["negative_network_frames"]:
        print(f"\n  WARNING: {out['negative_network_frames']} frames had a "
              f"negative network time.")
        print("  The browser's round trip came out shorter than the server's own")
        print("  total, which should be impossible. Check that server.py and")
        print("  index.html are the instrumented versions before quoting this.")

    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", help="one or more roundtrip_*.jsonl files")
    ap.add_argument("--warmup", type=int, default=20,
                    help="frames to discard at the start of each run")
    ap.add_argument("--min-frames", type=int, default=30,
                    help="runs shorter than this are listed but not summarised")
    ap.add_argument("--out", default=None, help="JSON output path")
    args = ap.parse_args()

    paths = []
    for p in args.paths:
        paths.extend(sorted(glob.glob(p)) or [p])
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        sys.exit("no such file: " + ", ".join(missing))

    rows, bad = load(paths)
    if not rows:
        sys.exit("no records found. Did the page post any telemetry?")
    print(f"read {len(rows)} records from {len(paths)} file(s)"
          + (f", {bad} unparseable lines skipped" if bad else ""))

    by_run = defaultdict(list)
    for r in rows:
        by_run[r.get("run") or "unlabelled"].append(r)

    runs = []
    for run, rs in sorted(by_run.items()):
        if len(rs) < args.min_frames:
            print(f"  skipping {run}: only {len(rs)} frames "
                  f"(--min-frames {args.min_frames})")
            continue
        runs.append(report(run, rs, args))

    if not runs:
        sys.exit("\nno run was long enough to summarise. Let it run longer.")

    payload = {
        "summarised_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_files": paths,
        "method": ("network_ms = rtt_ms (browser clock) - server_total_ms "
                   "(server clock). Durations only, never timestamps, so the "
                   "two clocks need not agree."),
        "runs": runs,
    }
    out = args.out or (f"roundtrip_summary_"
                       f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n  written to {os.path.abspath(out)}")
    print()
    print("  Quote the run with frames in flight = 1. At higher depths the")
    print("  queue wait is a property of this page rather than of the rig, and")
    print("  the round trip carries it.")

if __name__ == "__main__":
    main()
