#!/usr/bin/env bash
# Prepare synthetic/model-based data inputs for wheel-pose experiments.
#
# This does not require Unreal to be running. It:
#   1. Expands/refreshes the Sketchfab GLB seed under data/sketchfab_cars.
#   2. Validates the existing UE synthetic incoming batch.
#   3. Converts it to YOLO-pose layout.
#   4. Builds a small preview set for visual QA.

set -euo pipefail

cd "$(dirname "$0")/.."

MAX_MODELS="${MAX_MODELS:-72}"
TARGET_TOTAL="${TARGET_TOTAL:-300}"
MAX_MB="${MAX_MB:-150}"
WORKERS="${WORKERS:-1}"
RETRY_SLEEP="${RETRY_SLEEP:-60}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-3}"
DOWNLOAD_DELAY="${DOWNLOAD_DELAY:-1.5}"
MAX_DOWNLOAD_SECONDS="${MAX_DOWNLOAD_SECONDS:-180}"
FROM_EXISTING_MANIFESTS="${FROM_EXISTING_MANIFESTS:-0}"
CANDIDATE_OFFSET="${CANDIDATE_OFFSET:-0}"
CANDIDATE_LIMIT="${CANDIDATE_LIMIT:-}"
SHUFFLE_CANDIDATES="${SHUFFLE_CANDIDATES:-0}"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"
CAR_BODY_ONLY="${CAR_BODY_ONLY:-1}"
SORT_CANDIDATES="${SORT_CANDIDATES:-small-first}"
MAX_FACE_COUNT="${MAX_FACE_COUNT:-}"
MAX_VERTEX_COUNT="${MAX_VERTEX_COUNT:-}"
SKIP_FAILED_WITHIN_HOURS="${SKIP_FAILED_WITHIN_HOURS:-24}"
MAX_CONSECUTIVE_FAILURES="${MAX_CONSECUTIVE_FAILURES:-5}"

echo "[1/4] Sketchfab seed: target_total=${TARGET_TOTAL} max_per_query=${MAX_MODELS}"
fetch_args=(
  --max "${MAX_MODELS}" \
  --target-total "${TARGET_TOTAL}" \
  --max-mb "${MAX_MB}" \
  --workers "${WORKERS}" \
  --retry-sleep "${RETRY_SLEEP}" \
  --download-retries "${DOWNLOAD_RETRIES}" \
  --download-delay "${DOWNLOAD_DELAY}" \
  --max-download-seconds "${MAX_DOWNLOAD_SECONDS}" \
  --clean-rejected-existing \
  --candidate-offset "${CANDIDATE_OFFSET}" \
  --sort-candidates "${SORT_CANDIDATES}" \
  --skip-failed-within-hours "${SKIP_FAILED_WITHIN_HOURS}" \
  --max-consecutive-failures "${MAX_CONSECUTIVE_FAILURES}" \
  --output-dir data/sketchfab_cars
)

if [[ -n "${CANDIDATE_LIMIT}" ]]; then
  fetch_args+=(--candidate-limit "${CANDIDATE_LIMIT}")
fi

if [[ "${SHUFFLE_CANDIDATES}" == "1" ]]; then
  fetch_args+=(--shuffle-candidates --shuffle-seed "${SHUFFLE_SEED}")
fi

if [[ "${CAR_BODY_ONLY}" == "1" ]]; then
  fetch_args+=(--car-body-only)
fi

if [[ -n "${MAX_FACE_COUNT}" ]]; then
  fetch_args+=(--max-face-count "${MAX_FACE_COUNT}")
fi

if [[ -n "${MAX_VERTEX_COUNT}" ]]; then
  fetch_args+=(--max-vertex-count "${MAX_VERTEX_COUNT}")
fi

if [[ "${FROM_EXISTING_MANIFESTS}" == "1" ]]; then
  fetch_args+=(--from-existing-manifests)
else
  fetch_args+=(
    --query "car"
    --query "vehicle"
    --query "sports car"
    --query "classic car"
    --query "race car"
    --query "sedan"
    --query "coupe"
    --query "hatchback"
    --query "wagon"
    --query "suv"
    --query "offroad car"
    --query "pickup truck"
    --query "truck"
    --query "van"
    --query "bus"
    --query "police car"
    --query "taxi car"
    --query "porsche"
    --query "bmw car"
    --query "toyota car"
    --query "nissan car"
    --query "ford car"
    --query "chevrolet car"
    --query "mercedes car"
    --query "audi car"
    --query "volkswagen car"
    --query "car wheel"
  )
fi

./.venv/bin/python src/fetch_sketchfab_cars.py "${fetch_args[@]}"

echo "[2/4] Validate UE incoming batch"
./.venv/bin/python src/check_keypoint_incoming.py \
  --source-root data/incoming/ue_synthetic

echo "[3/4] Convert UE incoming -> YOLO-pose"
./.venv/bin/python src/convert_keypoint_incoming_to_yolo_pose.py \
  --source-root data/incoming/ue_synthetic \
  --dataset-root data/wheel_pose_dataset_ue_synthetic \
  --source-name ue_synthetic_v1 \
  --overwrite

./.venv/bin/python src/check_yolo_pose_dataset.py \
  --dataset-root data/wheel_pose_dataset_ue_synthetic

echo "[4/4] Render annotation previews"
./.venv/bin/python src/preview_keypoint_annotations.py \
  --source-root data/incoming/ue_synthetic \
  --count 6 \
  --output-root outputs/ue_synthetic_preview_check

echo "[done] Synthetic status: docs/SYNTHETIC_DATA_STATUS.md"
