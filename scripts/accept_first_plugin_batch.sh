#!/usr/bin/env bash
# Accept the first real Android-plugin batch into the ML pipeline.
#
# Runs validation -> incoming preview -> YOLO-pose conversion (with the
# quality gate enforced) -> dataset validation -> YOLO-pose preview.
# Fails fast on the first non-zero exit so a bad batch cannot silently
# leak into training.
#
# Explicit non-goals: this script does NOT train, does NOT run model
# inference, does NOT make the accept / reject decision. That decision
# is the human's job after visually inspecting the previews this script
# generates. See docs/FIRST_PLUGIN_BATCH_ACCEPTANCE.md.
#
# Usage:
#   ./scripts/accept_first_plugin_batch.sh
#   ./scripts/accept_first_plugin_batch.sh path/to/incoming_batch
#
# Defaults:
#   SOURCE_ROOT   data/incoming/android_plugin_real
#   DATASET_ROOT  data/wheel_pose_dataset
#   SOURCE_NAME   android_plugin_first_real_batch
#   PREVIEW_COUNT 20

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-./.venv/bin/python}"
SOURCE_ROOT="${1:-${SOURCE_ROOT:-data/incoming/android_plugin_real}}"
DATASET_ROOT="${DATASET_ROOT:-data/wheel_pose_dataset}"
SOURCE_NAME="${SOURCE_NAME:-android_plugin_first_real_batch}"
PREVIEW_COUNT="${PREVIEW_COUNT:-20}"

if [[ ! -x "$PY" ]]; then
    echo "ERROR: $PY not found. Create the venv first (see README)." >&2
    exit 2
fi

if [[ ! -d "$SOURCE_ROOT" ]]; then
    echo "ERROR: source root not found: $SOURCE_ROOT" >&2
    echo "Drop the plugin batch under that path or pass a different one as \$1." >&2
    exit 2
fi

run() {
    echo
    echo "==> $*"
    "$@"
}

echo "VSBL — first real plugin batch acceptance"
echo "  source root:  $SOURCE_ROOT"
echo "  dataset root: $DATASET_ROOT"
echo "  source name:  $SOURCE_NAME"
echo "  preview N:    $PREVIEW_COUNT"
echo
echo "NOTE: this script does NOT train and does NOT run inference."
echo "      Training only after a human has inspected the previews"
echo "      and explicitly marked the batch ACCEPT_FOR_TRAINING."

run "$PY" src/check_keypoint_incoming.py \
    --source-root "$SOURCE_ROOT"

run "$PY" src/preview_keypoint_annotations.py \
    --source-root "$SOURCE_ROOT" \
    --count "$PREVIEW_COUNT"

run "$PY" src/convert_keypoint_incoming_to_yolo_pose.py \
    --source-root "$SOURCE_ROOT" \
    --dataset-root "$DATASET_ROOT" \
    --source-name "$SOURCE_NAME" \
    --overwrite \
    --fail-on-quality-gate

run "$PY" src/check_yolo_pose_dataset.py \
    --dataset-root "$DATASET_ROOT"

run "$PY" src/preview_yolo_pose_labels.py \
    --dataset-root "$DATASET_ROOT" \
    --split train \
    --count "$PREVIEW_COUNT"

CONVERSION_REPORT="$DATASET_ROOT/metadata/conversion_report.json"

echo
echo "OK — automated acceptance steps passed."
echo
echo "Next step is MANUAL:"
echo "  1. Open the incoming preview:"
echo "       outputs/keypoint_preview/"
echo "  2. Open the YOLO-pose preview:"
echo "       outputs/pose_label_preview/train/"
echo "  3. Read the conversion report:"
echo "       $CONVERSION_REPORT"
echo "  4. Per docs/FIRST_PLUGIN_BATCH_ACCEPTANCE.md, decide:"
echo "       ACCEPT_FOR_TRAINING | REJECT_NEEDS_PLUGIN_FIX | ACCEPT_ONLY_AS_DEBUG"
echo
echo "Do not train until a human has signed off on the previews."
