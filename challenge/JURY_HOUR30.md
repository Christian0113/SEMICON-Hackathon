# Jury commands, hour 30

For the jury. Open this file in VSCode, copy each block into the terminal in
order, and wait for each to finish before the next. Use one terminal for the
whole team, because block 1 sets names that the later blocks use.

The team unlocks the photographs first, with the password read out in the room:

    bash ~/challenge/unlock_hour30.sh <password>

Nothing in this file is secret. Teams may read it.

---

## Score a team in one command

First check the script has not been changed. Both checksums must match the
ones printed on the jury sheet. If either differs, use the printed copy.

```
md5sum ~/challenge/jury_hour30.sh ~/challenge/JURY_HOUR30.md
```

Then, with the team's name and its submission file:

```
bash ~/challenge/jury_hour30.sh team-name ~/challenge/their_submission.py
```

It runs blocks 2 to 6 below in order, stops with a `STOP:` line if a check
fails, pauses if other jobs are running, and saves the result and a log in
`~/jury_results/`. About five minutes. If it prints `DISTURBED`, follow block 7.
Then do block 8.

The blocks below are the same steps, for running by hand if the script fails.

---

## 0. Check this file has not been changed

Compare the result with the checksum printed on the jury sheet. If it differs,
stop and use the printed copy of these commands.

```
md5sum ~/challenge/JURY_HOUR30.md
```

## 1. Name the team and its submission

Change the two values, then copy the block.

```
TEAM=team-name
SUB=$HOME/challenge/their_submission.py
mkdir -p ~/jury_results && ls -l "$SUB"
```

## 2. Check the harness and the answers are the ones we supplied

Every line must end in `OK`, and the count must be 800. If anything says
`FAILED`, stop: the scoring code or the answers file on this machine has been
changed. Score this team on the reference instance instead.

```
cd ~/challenge && md5sum -c - <<'EOF'
6a1e5b687b65e2b293f04a1bde924be8  harness/evaluate.py
46ab17c85eceeaae9b041c201adfcc14  harness/cocomap.py
c3bee45903158e1eaa4d471f5619cc59  harness/submission_api.py
be6753591179fee8128755c5e2f7b9fd  testset/instances.json
EOF
ls ~/challenge/testset/images | wc -l
```

## 3. Check the machine is quiet

The first command should print nothing. If it lists a camera demo, a bench or
another evaluation, ask the team to stop it. Note the load average (the three
numbers after "load average") on the scorecard.

```
pgrep -af 'server.py|evaluate.py|pytest|run_bench|run_loop'
uptime
```

## 4. Smoke test, 20 photographs, about a minute

Catches a crash or a wrong class mapping before the long run. Not scored.

```
cd ~/challenge && python harness/evaluate.py --submission "$SUB" \
  --images ~/challenge/testset/images --annotations ~/challenge/testset/instances.json \
  --limit 20 --repeats 1 --team "$TEAM-smoke" --out /tmp/smoke_$TEAM.json 2>&1 | tail -25
```

## 5. The scored run, about four minutes

Times the submission three times over all 800 photographs and reports the
fastest. The load average before and after, the host and the time are saved
with the result.

```
STAMP=$(date +%H%M); OUT=~/jury_results/${TEAM}_$STAMP
{ echo "team $TEAM  submission $SUB  host $(hostname)  start $(date -Is)"; uptime; } > $OUT.log
cd ~/challenge && python harness/evaluate.py --submission "$SUB" \
  --images ~/challenge/testset/images --annotations ~/challenge/testset/instances.json \
  --team "$TEAM" --out $OUT.json 2>&1 | tee -a $OUT.log
{ echo "end $(date -Is)"; uptime; } >> $OUT.log
```

## 6. Read back the result for the scorecard

```
python - "$OUT.json" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1])); t = r["timing"]; p = r["plain_readout"]
print("images      ", r["images"], " (must be 800)")
print("ms a frame  ", t["ms_per_frame"], "  fps", t["fps"])
print("each run    ", t["ms_per_frame_each_run"], "  spread %", t["spread_percent"], "  DISTURBED", t["disturbed"])
print("mAP         ", r["accuracy"]["mAP_50"], "  (as supplied: 0.467)")
print("found       ", p["objects_found"], "of", p["objects_present"])
print("false       ", p["false_detections"], "  on empty photographs", p["false_detections_on_empty_images"], " (as supplied: 14)")
print("problems    ", r["problems"] or "none")
PYEOF
tail -2 $OUT.log
```

## 7. If the run was disturbed

If block 6 shows `DISTURBED True`, or the load average in block 3 or at the end
of the log was high, run block 3 and block 5 again later, when the machine is
quieter. At most two scored runs per team. The faster of the two counts, and
both stay in `~/jury_results/`.

## 8. Keep the evidence

Before the team's instance is released, download the folder `~/jury_results`
from the VSCode Explorer (right-click, Download) to the jury laptop.
