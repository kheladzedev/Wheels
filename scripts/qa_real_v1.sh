#!/usr/bin/env bash
# Launch the manual_keypoint_annotator in QA mode over the real_v1
# auto-drafts. Writes accepted/edited annotations into a NEW directory
# (annotations_qa/) so the original auto-drafts stay untouched and the
# QA pass can be re-run incrementally without losing progress.
#
# Keys (inside the OpenCV window — see annotator source for the full list):
#   y / Enter    accept current wheel as-is
#   drag         move a keypoint
#   click in bbox + d    drop wheel
#   e            clear all keypoints, re-click from scratch
#   n            next image (skips remaining wheels on current one)
#   q            quit (progress is saved per-image as you go)
#
# Re-running this script picks up where you left off (it skips images
# that already have a QA JSON unless you pass --rerun).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-./.venv/bin/python}"
IMAGES_DIR="${IMAGES_DIR:-data/incoming/real_v1/images}"
ANNOTATIONS_DIR="${ANNOTATIONS_DIR:-data/incoming/real_v1/annotations_qa}"
PREFILL_DIR="${PREFILL_DIR:-data/incoming/real_v1/annotations}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/incoming/real_v1_qa}"

if [[ ! -x "$PY" ]]; then
    echo "ERROR: $PY not found. Create the venv first (see README)." >&2
    exit 2
fi
if [[ ! -d "$IMAGES_DIR" ]]; then
    echo "ERROR: images dir not found: $IMAGES_DIR" >&2
    exit 2
fi
if [[ ! -d "$PREFILL_DIR" ]]; then
    echo "ERROR: prefill dir not found: $PREFILL_DIR" >&2
    echo "Hint: run src/auto_annotate_wheels.py first." >&2
    exit 2
fi

mkdir -p "$ANNOTATIONS_DIR"

echo "==> manual QA pass"
echo "    images:       $IMAGES_DIR"
echo "    prefill from: $PREFILL_DIR"
echo "    write to:     $ANNOTATIONS_DIR"
echo "    bundle:       $OUTPUT_ROOT"
echo

exec "$PY" src/manual_keypoint_annotator.py \
    --images-dir "$IMAGES_DIR" \
    --annotations-dir "$ANNOTATIONS_DIR" \
    --output-root "$OUTPUT_ROOT" \
    --prefill-from "$PREFILL_DIR" \
    "$@"
