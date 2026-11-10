#!/usr/bin/env bash
# Score the current branch's submission on the practice set and log the result.
#
#   bash tools/run_eval.sh                 # tag = current branch name
#   bash tools/run_eval.sh L01-decoders8   # custom tag (e.g. for a sweep point)
#   LEVERS="L02=0" bash tools/run_eval.sh L02-off
#
# Writes results/<tag>.json and appends one line to results/LOG.md.
# Uses the supplied harness in ~/challenge (the copy the jury checksums).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRANCH="$(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
COMMIT="$(git -C "$REPO" rev-parse --short HEAD)"
TAG="${1:-$BRANCH}"; shift || true
TAG="${TAG//\//_}"
SUB="${SUB:-$REPO/challenge/harness/my_submission.py}"
CH="${CHALLENGE:-$HOME/challenge}"
OUT="$REPO/results/$TAG.json"
mkdir -p "$REPO/results"

if [ -n "$(git -C "$REPO" status --porcelain -- "$SUB")" ]; then
  echo "WARNING: $SUB has uncommitted changes; the log will name commit $COMMIT anyway."
fi
if pgrep -f 'server.py|evaluate.py|pytest|run_bench|run_loop' >/dev/null; then
  echo "WARNING: other card or benchmark jobs are running; the timing may be DISTURBED:"
  pgrep -af 'server.py|evaluate.py|pytest|run_bench|run_loop' || true
fi

python "$CH/harness/evaluate.py" --submission "$SUB" \
  --images "$CH/practice/images" --annotations "$CH/practice/instances.json" \
  --team "$TAG" --out "$OUT" "$@"

python - "$OUT" "$REPO/results/LOG.md" "$TAG" "$BRANCH" "$COMMIT" "${LEVERS:-}" <<'PY'
import json, sys, os, datetime
out, log, tag, branch, commit, levers = sys.argv[1:7]
r = json.load(open(out)); t = r["timing"]
new = not os.path.exists(log)
with open(log, "a") as f:
    if new:
        f.write("# Results log\n\nOne line per scored practice run (`tools/run_eval.sh`). "
                "Baseline: fast reference 17.803 ms, mAP 0.464, 25176 detections.\n\n"
                "| date (UTC) | tag | branch @ commit | LEVERS env | ms/frame | each run | mAP | detections | disturbed | problems |\n"
                "|---|---|---|---|---|---|---|---|---|---|\n")
    f.write(f"| {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M} | {tag} | {branch} @ {commit} | {levers or '-'} | "
            f"{t['ms_per_frame']} | {t['ms_per_frame_each_run']} | {r['accuracy']['mAP_50']} | "
            f"{r['detections_returned']} | {t['disturbed']} | {len(r['problems'])} |\n")
print(f"\n  logged to {log}: {t['ms_per_frame']} ms/frame, mAP {r['accuracy']['mAP_50']}, "
      f"{r['detections_returned']} detections")
PY
echo "  Commit the result:  git add results/ && git commit -m \"[$TAG] practice run\""
