#!/usr/bin/env bash
# One-command orchestration for the remaining handoff work.
#
# Safe default: run validation/readiness only.
#
# Full path once UnrealMCP is running:
#   RUN_FETCH=1 RUN_OBJAVERSE=1 RUN_UE=1 ./scripts/finish_project_today.sh
#
# Optional:
#   RUN_TRAIN=1 TRAIN_DATA=configs/pose_dataset_ue_synthetic_pseudo_from_champion.yaml

set -euo pipefail

cd "$(dirname "$0")/.."

RUN_FETCH="${RUN_FETCH:-0}"
RUN_OBJAVERSE="${RUN_OBJAVERSE:-0}"
RUN_UE="${RUN_UE:-0}"
RUN_TRAIN="${RUN_TRAIN:-0}"
RUN_FINAL_CHECKS="${RUN_FINAL_CHECKS:-1}"
WAIT_FOR_MCP="${WAIT_FOR_MCP:-0}"
MCP_WAIT_TIMEOUT="${MCP_WAIT_TIMEOUT:-1800}"
MCP_WAIT_INTERVAL="${MCP_WAIT_INTERVAL:-10}"

TARGET_TOTAL="${TARGET_TOTAL:-300}"
RATE_LIMIT_SLEEP="${RATE_LIMIT_SLEEP:-900}"
FETCH_MAX_ROUNDS="${FETCH_MAX_ROUNDS:-0}"
OBJAVERSE_MAX_MB="${OBJAVERSE_MAX_MB:-45}"
OBJAVERSE_MAX_DOWNLOADS="${OBJAVERSE_MAX_DOWNLOADS:-66}"
OBJAVERSE_SHUFFLE_SEED="${OBJAVERSE_SHUFFLE_SEED:-527}"

SOURCE_NAME="${SOURCE_NAME:-ue_sketchfab_pseudo}"
UE_RENDER_SCRIPT="${UE_RENDER_SCRIPT:-scripts/ue/render_sketchfab_cars.py}"
UE_RENDER_IMAGES_DIR="${UE_RENDER_IMAGES_DIR:-outputs/ue_sketchfab_renders/images}"
INCOMING_ROOT="${INCOMING_ROOT:-data/incoming/${SOURCE_NAME}}"
DATASET_ROOT="${DATASET_ROOT:-data/wheel_pose_dataset_${SOURCE_NAME}}"

TRAIN_DATA="${TRAIN_DATA:-configs/pose_dataset_ue_synthetic_pseudo_from_champion.yaml}"
TRAIN_MODEL="${TRAIN_MODEL:-runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt}"
TRAIN_NAME="${TRAIN_NAME:-wheel_finish_smoke}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-20}"
TRAIN_BATCH="${TRAIN_BATCH:-8}"
TRAIN_DEVICE="${TRAIN_DEVICE:-mps}"

echo "[0/5] Current readiness"
./.venv/bin/python src/project_readiness.py || true

if [[ "${RUN_FETCH}" == "1" ]]; then
  echo "[1/5] Fetch Sketchfab GLBs until target=${TARGET_TOTAL}"
  TARGET_TOTAL="${TARGET_TOTAL}" \
  RATE_LIMIT_SLEEP="${RATE_LIMIT_SLEEP}" \
  MAX_ROUNDS="${FETCH_MAX_ROUNDS}" \
  ./scripts/fetch_sketchfab_until_target.sh
else
  echo "[1/5] Skip Sketchfab fetch (RUN_FETCH=${RUN_FETCH})"
fi

if [[ "${RUN_OBJAVERSE}" == "1" ]]; then
  echo "[1b/5] Fill model pool from Objaverse fallback until target=${TARGET_TOTAL}"
  ./.venv/bin/python src/fetch_objaverse_cars.py \
    --output-dir data/sketchfab_cars \
    --target-total "${TARGET_TOTAL}" \
    --max-downloads "${OBJAVERSE_MAX_DOWNLOADS}" \
    --max-mb "${OBJAVERSE_MAX_MB}" \
    --shuffle-seed "${OBJAVERSE_SHUFFLE_SEED}"
else
  echo "[1b/5] Skip Objaverse fallback (RUN_OBJAVERSE=${RUN_OBJAVERSE})"
fi

if [[ "${RUN_UE}" == "1" ]]; then
  if [[ "${WAIT_FOR_MCP}" == "1" ]]; then
    echo "[2/5] Wait for UnrealMCP"
    TIMEOUT_SECONDS="${MCP_WAIT_TIMEOUT}" \
    INTERVAL_SECONDS="${MCP_WAIT_INTERVAL}" \
    ./scripts/wait_for_unreal_mcp.sh
  fi
  echo "[2/5] Run UE import/render/pseudo/convert/check"
  RUN_MCP_IMPORT=1 \
  UE_RENDER_SCRIPT="${UE_RENDER_SCRIPT}" \
  UE_RENDER_IMAGES_DIR="${UE_RENDER_IMAGES_DIR}" \
  SOURCE_NAME="${SOURCE_NAME}" \
  INCOMING_ROOT="${INCOMING_ROOT}" \
  DATASET_ROOT="${DATASET_ROOT}" \
  ./scripts/prepare_ue_sketchfab_pseudo_data.sh
else
  echo "[2/5] Skip UE/MCP pipeline (RUN_UE=${RUN_UE})"
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  echo "[3/5] Train YOLO-pose experiment"
  ./.venv/bin/python src/train_yolo.py \
    --data "${TRAIN_DATA}" \
    --model "${TRAIN_MODEL}" \
    --epochs "${TRAIN_EPOCHS}" \
    --imgsz 640 \
    --batch "${TRAIN_BATCH}" \
    --device "${TRAIN_DEVICE}" \
    --project runs/pose \
    --name "${TRAIN_NAME}"
else
  echo "[3/5] Skip training (RUN_TRAIN=${RUN_TRAIN})"
fi

if [[ "${RUN_FINAL_CHECKS}" == "1" ]]; then
  echo "[4/5] Final lightweight checks"
  ./.venv/bin/python -m py_compile \
    src/fetch_sketchfab_cars.py \
    src/fetch_objaverse_cars.py \
    src/pseudo_label_images_to_incoming.py \
    src/project_readiness.py \
    scripts/ue/import_sketchfab_glbs.py \
    scripts/ue/render_sketchfab_cars.py
  ./.venv/bin/python -m pytest \
    tests/test_fetch_sketchfab_cars.py \
    tests/test_fetch_objaverse_cars.py \
    tests/test_pseudo_label_images_to_incoming.py \
    tests/test_project_readiness.py
else
  echo "[4/5] Skip final checks (RUN_FINAL_CHECKS=${RUN_FINAL_CHECKS})"
fi

echo "[5/5] Final readiness"
./.venv/bin/python src/project_readiness.py
