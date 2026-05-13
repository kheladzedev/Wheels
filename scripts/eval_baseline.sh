#!/usr/bin/env bash
# Run eval_keypoints.py against a trained checkpoint and persist the
# report under outputs/eval/.
#
# Parameterised via env vars so the same wrapper handles
# wheel_baseline_v1 today and wheel_real_v1 after the QA retrain.
#
# Defaults target the wheel_baseline_v1 run:
#
#   MODEL  runs/pose/wheel_baseline_v1/weights/best.pt
#   DATA   configs/pose_dataset.yaml
#   OUT    outputs/eval/wheel_baseline_v1.json
#
# Override examples:
#   MODEL=runs/pose/wheel_real_v1/weights/best.pt \
#     DATA=configs/pose_dataset_real_v1.yaml \
#     OUT=outputs/eval/wheel_real_v1.json \
#     ./scripts/eval_baseline.sh
#
# Extra CLI args are forwarded to eval_keypoints.py (e.g. --device mps).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-./.venv/bin/python}"
MODEL="${MODEL:-runs/pose/wheel_baseline_v1/weights/best.pt}"
DATA="${DATA:-configs/pose_dataset.yaml}"
OUT="${OUT:-outputs/eval/wheel_baseline_v1.json}"
SPLIT="${SPLIT:-val}"
DEVICE="${DEVICE:-cpu}"
WORST_N="${WORST_N:-10}"

if [[ ! -x "$PY" ]]; then
    echo "ERROR: $PY not found. Create the venv first (see README)." >&2
    exit 2
fi
if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: model checkpoint not found: $MODEL" >&2
    echo "Hint: train one with src/train_yolo.py or override MODEL=..." >&2
    exit 2
fi
if [[ ! -f "$DATA" ]]; then
    echo "ERROR: dataset config not found: $DATA" >&2
    exit 2
fi

mkdir -p "$(dirname "$OUT")"

echo "==> eval $MODEL on $DATA split=$SPLIT -> $OUT"
"$PY" src/eval_keypoints.py \
    --model "$MODEL" \
    --data "$DATA" \
    --split "$SPLIT" \
    --device "$DEVICE" \
    --output "$OUT" \
    --worst-n "$WORST_N" \
    "$@"

echo
echo "OK — report at $OUT"
