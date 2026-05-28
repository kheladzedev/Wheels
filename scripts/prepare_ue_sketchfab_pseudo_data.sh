#!/usr/bin/env bash
# Build a pseudo-labeled YOLO-pose dataset from Unreal-rendered Sketchfab images.
#
# Typical usage after Unreal Editor + UnrealMCP is running:
#
#   RUN_MCP_IMPORT=1 \
#   UE_RENDER_IMAGES_DIR=outputs/ue_sketchfab_renders/images \
#   ./scripts/prepare_ue_sketchfab_pseudo_data.sh
#
# Pass UE_RENDER_SCRIPT=scripts/ue/render_sketchfab_cars.py to render the
# imported meshes through CameraCapture before pseudo-labeling.

set -euo pipefail

cd "$(dirname "$0")/.."

RUN_MCP_IMPORT="${RUN_MCP_IMPORT:-0}"
UE_RENDER_SCRIPT="${UE_RENDER_SCRIPT:-}"
UE_RENDER_IMAGES_DIR="${UE_RENDER_IMAGES_DIR:-outputs/ue_sketchfab_renders/images}"
SOURCE_NAME="${SOURCE_NAME:-ue_sketchfab_pseudo}"
INCOMING_ROOT="${INCOMING_ROOT:-data/incoming/${SOURCE_NAME}}"
DATASET_ROOT="${DATASET_ROOT:-data/wheel_pose_dataset_${SOURCE_NAME}}"
MODEL="${MODEL:-runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt}"
CONF="${CONF:-0.25}"
DEVICE="${DEVICE:-mps}"
MAX_DET="${MAX_DET:-20}"

if [[ "${RUN_MCP_IMPORT}" == "1" ]]; then
  echo "[1/5] Import Sketchfab GLBs into Unreal"
  ./.venv/bin/python scripts/ue/_send.py exec_file scripts/ue/import_sketchfab_glbs.py
else
  echo "[1/5] Skip MCP import (RUN_MCP_IMPORT=${RUN_MCP_IMPORT})"
fi

if [[ -n "${UE_RENDER_SCRIPT}" ]]; then
  echo "[2/5] Run UE render script: ${UE_RENDER_SCRIPT}"
  ./.venv/bin/python scripts/ue/_send.py exec_file "${UE_RENDER_SCRIPT}"
else
  echo "[2/5] Skip UE render script (UE_RENDER_SCRIPT empty)"
fi

if [[ ! -d "${UE_RENDER_IMAGES_DIR}" ]]; then
  echo "ERROR: render image directory does not exist: ${UE_RENDER_IMAGES_DIR}" >&2
  exit 2
fi

echo "[3/5] Pseudo-label render images -> incoming"
./.venv/bin/python src/pseudo_label_images_to_incoming.py \
  --images-dir "${UE_RENDER_IMAGES_DIR}" \
  --output-root "${INCOMING_ROOT}" \
  --model "${MODEL}" \
  --source-name "${SOURCE_NAME}" \
  --conf "${CONF}" \
  --device "${DEVICE}" \
  --max-det "${MAX_DET}" \
  --overwrite

./.venv/bin/python src/check_keypoint_incoming.py \
  --source-root "${INCOMING_ROOT}"

echo "[4/5] Convert incoming -> YOLO-pose"
./.venv/bin/python src/convert_keypoint_incoming_to_yolo_pose.py \
  --source-root "${INCOMING_ROOT}" \
  --dataset-root "${DATASET_ROOT}" \
  --source-name "${SOURCE_NAME}" \
  --overwrite

echo "[5/5] Validate YOLO-pose dataset"
./.venv/bin/python src/check_yolo_pose_dataset.py \
  --dataset-root "${DATASET_ROOT}"

echo "[done] incoming=${INCOMING_ROOT} dataset=${DATASET_ROOT}"
