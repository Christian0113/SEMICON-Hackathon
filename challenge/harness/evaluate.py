# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Run a team's submission over the hold-out photographs, time it, score it.

The same code scores every team, including the unmodified detector, which is
run through this identically to give the reference point.

    python evaluate.py --submission team_blue/submission.py \
                       --images ~/challenge/testset/images \
                       --annotations ~/challenge/testset/instances.json \
                       --out results_blue.json

Add --selftest to exercise the whole harness with a built-in stub that needs no
card, which is how you check the plumbing before the event.
"""
from __future__ import annotations
import argparse, importlib.util, json, os, platform, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cocomap                      # noqa: E402
import submission_api as api        # noqa: E402

WARMUP_FRAMES = 8

# How far the slowest timed repeat may sit above the fastest before the run is
# called into question, in percent. Repeated timings of identical work scatter
# upwards from a floor, so a few percent is ordinary. Measured on the n150:
# three runs over the same 800 photographs spread 14%, and a run over the 200
# practice photographs on 2026-10-26 produced a slowest reading 6.1 times the
# fastest, with host-side suppression inflated by the same factor. The second
# case is something else competing for the host CPU, and it is worth saying so
# out loud rather than leaving it in a percentage nobody reads.
SPREAD_WARN_PERCENT = 50

# Below this many seconds a run is too short for a percentage to mean anything:
# ordinary scheduler jitter on a run lasting a fraction of a second is easily
# 50%, and the stub selftest would warn every time for no reason.
SPREAD_WARN_MIN_SECONDS = 1.0


# --------------------------------------------------------------------------
def load_submission(path):
    spec = importlib.util.spec_from_file_location("team_submission", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["team_submission"] = mod
    spec.loader.exec_module(mod)
    for name in ("Submission", "submission", "TeamSubmission"):
        obj = getattr(mod, name, None)
        if isinstance(obj, type):
            return obj()
    raise SystemExit(f"{path} does not define a class called Submission")


class _PowerSampler(threading.Thread):
    """Sample the card's power while the run happens. Records nothing at all if
    tt-smi is unavailable, rather than guessing."""

    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        # Not named _stop: threading.Thread already has a private _stop()
        # which join() calls, and shadowing it makes join() try to call
        # this Event. That raised on a real instance.
        self.interval, self.samples, self._halt = interval, [], threading.Event()

    def run(self):
        while not self._halt.is_set():
            try:
                out = subprocess.run(["tt-smi", "-s"], capture_output=True,
                                     text=True, timeout=5).stdout
                w = json.loads(out)["device_info"][0]["telemetry"]["power"]
                self.samples.append(float(w))
            except Exception:
                pass
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=3)
        if not self.samples:
            return {"watts_mean": None, "watts_max": None, "samples": 0,
                    "note": "TBD - tt-smi gave no readings, power not recorded"}
        return {"watts_mean": round(sum(self.samples) / len(self.samples), 2),
                "watts_max": round(max(self.samples), 2),
                "samples": len(self.samples)}


# --------------------------------------------------------------------------
def _scale_against_truth(gt, by_name, by_path, names):
    """Catch boxes left in the size the model ran at, using the answers.

    The check in submission_api compares boxes against each other and needs a
    few hundred before it can say anything. Here the correct answers are to
    hand, so the same mistake shows up on a handful of photographs: compare how
    much of its frame a submitted box covers against how much the real objects
    cover. Boxes stuck at the run resolution are far smaller on any photograph
    larger than that.
    """
    import statistics as st
    truth = {}
    for a in gt["annotations"]:
        truth.setdefault(a["image_id"], []).append(a["bbox"])
    d_ratio, t_ratio = [], []
    for n in names:
        W, H = by_name[n]["width"], by_name[n]["height"]
        dim = max(W, H)
        for d in by_path.get(n, []):
            b = d.get("bbox")
            if isinstance(b, (list, tuple)) and len(b) == 4:
                d_ratio.append(max(b[0] + b[2], b[1] + b[3]) / dim)
        for b in truth.get(by_name[n]["id"], []):
            t_ratio.append(max(b[0] + b[2], b[1] + b[3]) / dim)
    if len(d_ratio) < 5 or len(t_ratio) < 5:
        return []
    md, mt = st.median(d_ratio), st.median(t_ratio)
    if mt > 0 and md < 0.5 * mt:
        return [f"your boxes reach {md:.0%} of the frame where the real objects reach "
                f"{mt:.0%}. They are probably still in the size you ran the model at "
                f"rather than the original photograph's. Use scale_boxes()."]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission")
    ap.add_argument("--images")
    ap.add_argument("--annotations")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--team", default="unnamed")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="confidence for the plain readout only; the score uses all of them")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N, for testing")
    ap.add_argument("--repeats", type=int, default=3,
                    help="how many times to time run(); the fastest one is the "
                         "reported speed, because interference can only slow a run down")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        a.submission = a.submission or os.path.join(HERE, "submission_stub.py")
        a.team = a.team if a.team != "unnamed" else "selftest"

    for need, what in ((a.images, "--images"), (a.annotations, "--annotations"),
                       (a.submission, "--submission")):
        if not need:
            raise SystemExit(f"{what} is required")

    # Everything the submission is handed must be absolute, because we change
    # directory below and a relative path would then point somewhere else.
    for f in ("images", "annotations", "out", "submission"):
        if getattr(a, f):
            setattr(a, f, os.path.abspath(os.path.expanduser(getattr(a, f))))

    gt = json.load(open(a.annotations))
    by_name = {i["file_name"]: i for i in gt["images"]}
    names = sorted(by_name)
    if a.limit:
        names = names[:a.limit]
        keep = {by_name[n]["id"] for n in names}
        gt = {**gt,
              "images": [i for i in gt["images"] if i["id"] in keep],
              "annotations": [x for x in gt["annotations"] if x["image_id"] in keep]}
    paths = [os.path.join(a.images, n) for n in names]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"{len(missing)} photographs are not in {a.images}, "
                         f"first is {os.path.basename(missing[0])}")
    print(f"  {len(paths)} photographs from {a.images}")

    sub = load_submission(a.submission)

    device = None
    try:
        import ttnn                                        # noqa: F401
        # The detector needs a larger L1 small allocation than the default, and
        # bench_stages.py opens the card the same way. Without it the model
        # fails to build with an allocation error that says nothing useful.
        try:
            from models.demos.yolov4.common import YOLOV4_L1_SMALL_SIZE as L1
        except Exception:
            L1 = 10960   # the value in models/demos/yolov4/common.py
        # A submission may ask for device options it cannot set later.
        # trace_region_size and num_command_queues are fixed at open time, so
        # without this a team could not capture a trace through the harness at
        # all, which is the single largest speed-up the model has.
        opts = dict(getattr(sub, "DEVICE_OPTIONS", None) or {})
        opts.pop("device_id", None)
        opts.pop("l1_small_size", None)
        try:
            device = ttnn.open_device(device_id=0, l1_small_size=L1, **opts)
        except TypeError:
            if opts:
                print(f"  this ttnn does not accept {sorted(opts)}; "
                      f"opening without them")
            try:
                device = ttnn.open_device(device_id=0, l1_small_size=L1)
            except TypeError:
                device = ttnn.open_device(device_id=0)
            opts = {}
        asked = "".join(f", {k}={v}" for k, v in sorted(opts.items()))
        print(f"  card opened (l1_small_size={L1}{asked})")
    except Exception as e:
        print(f"  no card available ({type(e).__name__}), passing device=None")

    # The model's own load_torch_model() looks for its weights at the relative
    # path models/demos/yolov4/tests/pcc/yolov4.pth, so it only works when the
    # working directory is TT_METAL_HOME. bench/run_bench.sh does `cd` there
    # before running. Do the same here, so a submission works wherever it was
    # launched from and nobody has to know this.
    cwd = os.getcwd()
    tt_home = os.environ.get("TT_METAL_HOME") or os.path.expanduser("~/tt-metal")
    if os.path.isdir(tt_home):
        os.chdir(tt_home)
        print(f"  working directory set to {tt_home} for the model's weights")

    t0 = time.perf_counter()
    sub.setup(device)
    t_setup = time.perf_counter() - t0
    print(f"  setup {t_setup:.1f}s")

    t0 = time.perf_counter()
    sub.warmup(paths[:WARMUP_FRAMES])
    t_warm = time.perf_counter() - t0
    print(f"  warmup {t_warm:.1f}s over {min(WARMUP_FRAMES, len(paths))} frames")

    # run() is timed more than once. Anything else happening on the host or the
    # card adds time to a run and nothing ever takes time away, so repeated
    # measurements of identical work scatter upwards from a floor. The floor is
    # the number we want. Measured on the n150 on 2026-10-26, three consecutive
    # runs over the same 800 photographs gave 88.352, 87.579 and 99.790 ms a
    # frame, so a single reading can be 14% pessimistic. A run over the 200
    # practice photographs the same day spread 6.1 times, which is why a large
    # spread is called out below rather than only printed.
    n_rep = max(1, a.repeats)
    power = _PowerSampler(); power.start()
    times, out = [], None
    for i in range(n_rep):
        print(f"  timing the run... ({i + 1} of {n_rep})")
        t0 = time.perf_counter()
        out_i = sub.run(list(paths))
        times.append(time.perf_counter() - t0)
        print(f"    {1000 * times[-1] / len(paths):.3f} ms a frame")
        if out is None or times[-1] == min(times):
            out = out_i
    power_stats = power.stop()
    elapsed = min(times)
    spread = 100 * (max(times) - elapsed) / elapsed if n_rep > 1 else 0.0
    disturbed = (n_rep > 1 and spread > SPREAD_WARN_PERCENT
                 and elapsed >= SPREAD_WARN_MIN_SECONDS)
    if n_rep > 1:
        print(f"  fastest of {n_rep}: {1000 * elapsed / len(paths):.3f} ms a frame "
              f"(slowest was {spread:.1f}% above it)")
    if disturbed:
        print(f"  DISTURBED: the slowest of {n_rep} readings was {spread:.1f}% above "
              f"the fastest.")
        print( "    Timings that far apart on identical work mean something else "
               "had the host")
        print( "    or the card while this ran. The fastest reading is still the "
               "honest one,")
        print( "    but all of them may be inflated. Close anything else running "
               "on this")
        print( "    machine, including a camera demo or a second terminal, and "
               "measure again.")
    try:
        sub.teardown()
    finally:
        os.chdir(cwd)
        if device is not None:
            try:
                import ttnn
                ttnn.close_device(device)
            except Exception:
                pass

    # ---- shape the output and check it before scoring ---------------------
    if not isinstance(out, dict):
        raise SystemExit("run() must return a dict mapping path -> list of detections")
    by_path = {os.path.basename(k): v for k, v in out.items()}
    absent = [n for n in names if n not in by_path]
    if absent:
        print(f"  WARNING: {len(absent)} photographs got no entry at all "
              f"(scored as zero detections)")
    sizes = {n: (by_name[n]["width"], by_name[n]["height"]) for n in names}
    problems = api.check_detections({n: by_path.get(n, []) for n in names}, sizes)
    problems += _scale_against_truth(gt, by_name, by_path, names)
    for p in problems:
        print(f"  PROBLEM: {p}")

    dets = []
    for n in names:
        iid = by_name[n]["id"]
        for d in by_path.get(n, []):
            dets.append({"image_id": iid, "category_id": int(d["category_id"]),
                         "bbox": [float(v) for v in d["bbox"]], "score": float(d["score"])})

    # ---- score ------------------------------------------------------------
    mAP, per_class = cocomap.evaluate(gt, dets)
    plain = cocomap.plain_readout(gt, dets, conf=a.conf)
    fps = len(paths) / elapsed if elapsed > 0 else 0.0

    board = None
    try:
        info = json.loads(subprocess.run(["tt-smi", "-s"], capture_output=True,
                                         text=True, timeout=10).stdout)["device_info"][0]
        board = {"board_type": info["board_info"]["board_type"],
                 "board_id": info["board_info"]["board_id"],
                 "fw_bundle_version": info["firmwares"]["fw_bundle_version"]}
    except Exception:
        board = {"note": "TBD - tt-smi unavailable, board not recorded"}

    res = {
        "team": a.team,
        "submission": os.path.abspath(a.submission),
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "board": board,
        "images": len(paths),
        "detections_returned": len(dets),
        "photographs_with_no_entry": len(absent),
        "timing": {"run_seconds": round(elapsed, 3),
                   "fps": round(fps, 2),
                   "ms_per_frame": round(1000 * elapsed / len(paths), 3),
                   "repeats": n_rep,
                   "ms_per_frame_each_run": [round(1000 * t / len(paths), 3)
                                             for t in times],
                   "reported": "fastest of the repeats",
                   "spread_percent": round(spread, 1),
                   "disturbed": disturbed,
                   "setup_seconds": round(t_setup, 2),
                   "warmup_seconds": round(t_warm, 2)},
        "accuracy": {"mAP_50": round(mAP, 4)},
        "plain_readout": plain,
        "power": power_stats,
        "problems": problems,
    }
    json.dump(res, open(a.out, "w"), indent=2)

    print()
    print(f"  team            : {res['team']}")
    print(f"  images          : {res['images']}")
    print(f"  total time      : {res['timing']['run_seconds']} s")
    print(f"  speed           : {res['timing']['fps']} frames per second "
          f"({res['timing']['ms_per_frame']} ms a frame)")
    print(f"  accuracy (mAP)  : {res['accuracy']['mAP_50']}")
    print()
    print(f"  plain readout at confidence {plain['conf']}")
    print(f"    objects really present                {plain['objects_present']}")
    print(f"    objects found                         {plain['objects_found']}"
          f"   ({100*plain['recall']:.1f}%)")
    print(f"    things reported that were not there   {plain['false_detections']}")
    print(f"    of those, on the empty photographs    "
          f"{plain['false_detections_on_empty_images']}")
    if power_stats.get("watts_mean") is not None:
        print(f"  power           : {power_stats['watts_mean']} W mean, "
              f"{power_stats['watts_max']} W peak over {power_stats['samples']} samples")
        print(f"  frames per watt : {fps / power_stats['watts_mean']:.2f}")
    else:
        print(f"  power           : not recorded ({power_stats['note']})")
    print(f"\n  written to {a.out}")


if __name__ == "__main__":
    main()
