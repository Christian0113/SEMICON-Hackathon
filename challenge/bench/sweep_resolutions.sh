#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# sweep_resolutions.sh -- evaluate YOLOv4 accuracy and speed across candidate resolutions
#
# Usage:
#   bash challenge/bench/sweep_resolutions.sh [practice|easy10|testset]
#
set -euo pipefail

DATASET="${1:-practice}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SUBMISSION="$ROOT_DIR/challenge/my_submission.py"

case "$DATASET" in
  practice)
    IMAGES="$ROOT_DIR/challenge/practice/images"
    ANNOTATIONS="$ROOT_DIR/challenge/practice/instances.json"
    ;;
  easy10)
    IMAGES="$ROOT_DIR/challenge/easy10/images"
    ANNOTATIONS="$ROOT_DIR/challenge/easy10/instances.json"
    ;;
  testset)
    IMAGES="$ROOT_DIR/challenge/testset/images"
    ANNOTATIONS="$ROOT_DIR/challenge/testset/instances.json"
    ;;
  *)
    echo "Unknown dataset: $DATASET. Choose practice, easy10, or testset."
    exit 1
    ;;
esac

OUT_DIR="$ROOT_DIR/challenge/runs/resolution_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

RESOLUTIONS=("320,320" "384,384" "416,416" "480,480" "512,512" "640,640")

echo "=========================================================="
echo "Starting Resolution Sweep on $DATASET ($IMAGES)"
echo "Output directory: $OUT_DIR"
echo "Submission: $SUBMISSION"
echo "=========================================================="

SUMMARY_FILE="$OUT_DIR/summary.txt"
printf "%-12s | %-12s | %-12s | %-10s | %-12s\n" "Resolution" "Latency (ms)" "Throughput" "mAP@50" "Detections" > "$SUMMARY_FILE"
printf "%s\n" "----------------------------------------------------------------------" >> "$SUMMARY_FILE"

for RES in "${RESOLUTIONS[@]}"; do
  RES_TAG="${RES//,/_}"
  RUN_JSON="$OUT_DIR/run_${RES_TAG}.json"
  echo ""
  echo ">>> Testing Resolution: $RES ($RES_TAG) <<<"

  YOLO_RESOLUTION="$RES" python3 "$ROOT_DIR/challenge/harness/evaluate.py" \
    --submission "$SUBMISSION" \
    --images "$IMAGES" \
    --annotations "$ANNOTATIONS" \
    --repeats 1 \
    --out "$RUN_JSON" || {
      echo "  [WARN] Evaluation failed for resolution $RES"
      continue
    }

  python3 -c "
import json
with open('$RUN_JSON') as f:
    d = json.load(f)
t = d.get('timing', {})
lat = t.get('best_repeat_ms_per_frame', t.get('mean_ms_per_frame', 0.0))
fps = 1000.0 / lat if lat > 0 else 0.0
m_ap = d.get('mAP', {}).get('mAP_50', 0.0)
dets = d.get('plain_readout', {}).get('detections_found', 0)
line = f'$RES:<12 | {lat:8.3f} ms | {fps:6.1f} FPS | {m_ap:8.4f} | {dets:6d}'
print('Result: ' + line)
with open('$SUMMARY_FILE', 'a') as sf:
    sf.write(f'${RES:<12} | {lat:8.3f} ms | {fps:6.1f} FPS | {m_ap:8.4f} | {dets:6d}\n')
" || true
done

echo ""
echo "=========================================================="
echo "Resolution Sweep Complete! Summary:"
echo "=========================================================="
cat "$SUMMARY_FILE"
