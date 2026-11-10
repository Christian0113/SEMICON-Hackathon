#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# jury_hour30.sh  --  score one team's submission at hour 30, in one go
#
#     bash ~/challenge/jury_hour30.sh <team-name> <submission.py>
#
# Run by the jury after the team has unlocked the photographs. It checks the
# harness and the answers file, checks the machine is quiet, runs a 20-photo
# smoke test, runs the scored evaluation (three timed repeats, fastest counts),
# and prints the scorecard fields. Everything is saved in ~/jury_results/.
# The individual steps are in JURY_HOUR30.md, for running by hand.
set -uo pipefail

[ $# -eq 2 ] || { echo "usage: bash $0 <team-name> <submission.py>"; exit 2; }
TEAM="$1"
[ -f "$2" ] || { echo "submission not found: $2"; exit 1; }
SUB="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
C="$HOME/challenge"; T="$C/testset"
STAMP=$(date +%H%M); mkdir -p "$HOME/jury_results"; OUT="$HOME/jury_results/${TEAM}_$STAMP"
source "$HOME/tt-metal/python_env/bin/activate" 2>/dev/null || true
export TT_METAL_HOME="$HOME/tt-metal" PYTHONPATH="$HOME/tt-metal"

say() { printf '\n========== %s ==========\n' "$*"; }
log() { tee -a "$OUT.log"; }
echo "team $TEAM  submission $SUB  host $(hostname)  start $(date -Is)" > "$OUT.log"

say "1/5  harness and answers are the ones supplied" | log
( cd "$C" && md5sum -c - <<'EOF'
6a1e5b687b65e2b293f04a1bde924be8  harness/evaluate.py
46ab17c85eceeaae9b041c201adfcc14  harness/cocomap.py
c3bee45903158e1eaa4d471f5619cc59  harness/submission_api.py
be6753591179fee8128755c5e2f7b9fd  testset/instances.json
EOF
) 2>&1 | log
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "STOP: scoring code or answers changed on this machine. Score this team on the reference instance." | log; exit 1; }
N=$(find "$T/images" -type f 2>/dev/null | wc -l | tr -d ' ')
echo "photographs: $N" | log
[ "$N" = "800" ] || { echo "STOP: expected 800 photographs. Has the team run unlock_hour30.sh?" | log; exit 1; }

say "2/5  machine is quiet" | log
uptime | log
BUSY=$(pgrep -af 'server.py|evaluate.py|pytest|run_bench|run_loop' | grep -v "$$" || true)
if [ -n "$BUSY" ]; then
  echo "$BUSY" | log
  echo "Other jobs are running. Ask the team to stop them, then press Enter (Ctrl-C to abort)."
  read -r _
fi

say "3/5  smoke test, 20 photographs, not scored" | log
( cd "$C" && python harness/evaluate.py --submission "$SUB" \
    --images "$T/images" --annotations "$T/instances.json" \
    --limit 20 --repeats 1 --team "$TEAM-smoke" --out "/tmp/smoke_$TEAM.json" ) 2>&1 | tail -15 | log
[ "${PIPESTATUS[0]}" -eq 0 ] && [ -f "/tmp/smoke_$TEAM.json" ] \
  || { echo "STOP: the smoke test failed. The team may use its fix window, then run this again." | log; exit 1; }

say "4/5  scored run, three timed repeats over 800 photographs, about four minutes" | log
uptime | log
( cd "$C" && python harness/evaluate.py --submission "$SUB" \
    --images "$T/images" --annotations "$T/instances.json" \
    --team "$TEAM" --out "$OUT.json" ) 2>&1 | log
uptime | log
[ -f "$OUT.json" ] || { echo "STOP: the scored run did not write a result." | log; exit 1; }

say "5/5  for the scorecard" | log
python - "$OUT.json" <<'PYEOF' 2>&1 | log
import json, sys
r = json.load(open(sys.argv[1])); t = r["timing"]; p = r["plain_readout"]
print("images      ", r["images"], " (must be 800)")
print("ms a frame  ", t["ms_per_frame"], "  fps", t["fps"])
print("each run    ", t["ms_per_frame_each_run"], "  spread %", t["spread_percent"], "  DISTURBED", t["disturbed"])
print("mAP         ", r["accuracy"]["mAP_50"], "  (as supplied: 0.467)")
print("found       ", p["objects_found"], "of", p["objects_present"])
print("false       ", p["false_detections"], "  on empty photographs", p["false_detections_on_empty_images"], " (as supplied: 14)")
print("problems    ", r["problems"] or "none")
if t["disturbed"]:
    print("\nDISTURBED: run this script again later, when the machine is quieter.")
    print("At most two scored runs per team; the faster counts. Both are kept.")
PYEOF
echo "end $(date -Is)" >> "$OUT.log"
echo; echo "saved: $OUT.json  and  $OUT.log"
echo "Before the instance is released, download ~/jury_results from the Explorer."
