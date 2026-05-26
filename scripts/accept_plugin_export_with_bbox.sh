#!/usr/bin/env bash
# Validate a raw Unreal/plugin export that is expected to contain real BBox/WheelBBox.
#
# This script does not train, does not run model inference, and does not mark
# the batch ACCEPT_FOR_TRAINING. It prepares validation artifacts for human
# review and leaves metadata/acceptance_status.json in DEBUG_ONLY state.
#
# Usage:
#   ./scripts/accept_plugin_export_with_bbox.sh /path/to/raw/export

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-./.venv/bin/python}"
SOURCE_ROOT="${1:-}"
INCOMING_ROOT="${INCOMING_ROOT:-data/incoming/android_plugin_real}"
DATASET_ROOT="${DATASET_ROOT:-data/wheel_pose_dataset}"
SOURCE_NAME="${SOURCE_NAME:-plugin_real_bbox_trial}"
PREVIEW_COUNT="${PREVIEW_COUNT:-50}"
BBOX_AUDIT_ROOT="${BBOX_AUDIT_ROOT:-outputs/unreal_bbox_audit}"

if [[ -z "$SOURCE_ROOT" ]]; then
    echo "Usage: $0 /path/to/raw/export" >&2
    exit 2
fi

if [[ ! -x "$PY" ]]; then
    echo "ERROR: Python venv not found or not executable: $PY" >&2
    exit 2
fi

if [[ ! -d "$SOURCE_ROOT" ]]; then
    echo "ERROR: source root not found: $SOURCE_ROOT" >&2
    exit 2
fi

run() {
    echo
    echo "==> $*"
    "$@"
}

echo "VSBL — plugin export with real BBox validation"
echo "  raw source:    $SOURCE_ROOT"
echo "  incoming root: $INCOMING_ROOT"
echo "  dataset root:  $DATASET_ROOT"
echo "  source name:   $SOURCE_NAME"
echo "  preview N:     $PREVIEW_COUNT"
echo
echo "NOTE: no training, no inference, no automatic ACCEPT_FOR_TRAINING."

run "$PY" scripts/import_unreal_export.py \
    --source-root "$SOURCE_ROOT" \
    --out-root "$INCOMING_ROOT" \
    --source-name "$SOURCE_NAME" \
    --overwrite

run "$PY" src/check_keypoint_incoming.py \
    --source-root "$INCOMING_ROOT"

run "$PY" src/preview_keypoint_annotations.py \
    --source-root "$INCOMING_ROOT" \
    --count "$PREVIEW_COUNT"

run "$PY" src/convert_keypoint_incoming_to_yolo_pose.py \
    --source-root "$INCOMING_ROOT" \
    --dataset-root "$DATASET_ROOT" \
    --source-name "$SOURCE_NAME" \
    --overwrite

run "$PY" src/check_yolo_pose_dataset.py \
    --dataset-root "$DATASET_ROOT"

run "$PY" src/preview_yolo_pose_labels.py \
    --dataset-root "$DATASET_ROOT" \
    --split train \
    --count "$PREVIEW_COUNT"

run "$PY" scripts/audit_unreal_bbox_quality.py \
    --source-root "$INCOMING_ROOT" \
    --out-dir "$BBOX_AUDIT_ROOT" \
    --max-samples "$PREVIEW_COUNT"

echo
echo "Validation artifacts:"
echo "  Incoming preview:     outputs/keypoint_preview"
echo "  YOLO-pose preview:    outputs/pose_label_preview/train"
echo "  BBox contact sheet:   $BBOX_AUDIT_ROOT/contact_sheet.jpg"
echo "  Acceptance status:    $INCOMING_ROOT/metadata/acceptance_status.json"
echo
echo "Result remains DEBUG_ONLY until human preview explicitly accepts geometry."
