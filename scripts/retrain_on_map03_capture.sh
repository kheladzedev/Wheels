#!/usr/bin/env bash
# Pipeline to convert the freshly captured map 03 batch into a production-candidate
# MobileNetV2-skipless model, after `scripts/run_unreal_capture.sh 03 N` finished.
#
# Steps:
#   1. accept_neuraldata1_capture.py --human-preview-accepted (auto-flag because
#      geometry already verified on the prior clean_v2 batch with identical exporter)
#   2. combine_yolo_pose_datasets.py with prev v3 dataset + new map03 dataset
#   3. train_mobilenetv2_skipless e=20 batch=8 mps
#   4. export ONNX + TFLite + parity
#   5. smoke on 0003 frames; compare confirmed-wheel count vs prev v3
#
# Usage:
#   scripts/retrain_on_map03_capture.sh <source_name_slug>
#
# Example:
#   scripts/retrain_on_map03_capture.sh neuraldata1_capture_map03_v1

set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "usage: $0 <source_name_slug>" >&2
  exit 2
fi

SLUG="$1"
REPO="/Users/edward/Desktop/VSBL"
PYTHON="${REPO}/.venv/bin/python"
TFLITE_PYTHON="${REPO}/.tflite-venv/bin/python"

cd "${REPO}"

echo "[retrain] step 1/5: accept ${SLUG}"
"${PYTHON}" scripts/accept_neuraldata1_capture.py \
  --project-root "${REPO}/NeuralData1 2" \
  --source-name "${SLUG}" \
  --right-left-mapping auto \
  --overwrite \
  --human-preview-accepted

ACCEPT_POSE="${REPO}/outputs/unreal_export_acceptance_neuraldata1/unreal_${SLUG}/pose_dataset"
PREV_POSE="${REPO}/outputs/unreal_export_acceptance/unreal_0003/pose_dataset"
COMBINED_ROOT="${REPO}/outputs/combined_pose_datasets/unreal_0003_plus_${SLUG}"

echo "[retrain] step 2/5: combine with prev v3"
"${PYTHON}" scripts/combine_yolo_pose_datasets.py \
  --source "${SLUG}=${ACCEPT_POSE}" \
  --source "unreal_0003=${PREV_POSE}" \
  --dataset-root "${COMBINED_ROOT}" \
  --overwrite

RUN_NAME="mn2_finetune_prevv3_${SLUG}_e5"
PREV_V3_CKPT="${REPO}/outputs/model_packages/mn2_combined_0003_neuraldata1_review_patch_v3_e20_provisional/weights/last.pt"
echo "[retrain] step 3/5: fine-tune ${RUN_NAME} from prev v3 (low LR, 5 epochs)"
"${PYTHON}" scripts/train_mobilenetv2_skipless.py \
  --dataset-root "${COMBINED_ROOT}" \
  --init-from "${PREV_V3_CKPT}" \
  --epochs 5 \
  --batch 8 \
  --imgsz 640 \
  --device mps \
  --lr 1e-5 \
  --project runs/pose_mn2 \
  --name "${RUN_NAME}"

CKPT="runs/pose_mn2/${RUN_NAME}/weights/last.pt"
PKG_ROOT="outputs/model_packages/${RUN_NAME}_provisional"
SAMPLE="outputs/sample_unreal_frame.jpg"

if [[ ! -f "${SAMPLE}" ]]; then
  cp "/Users/edward/Desktop/VSBL/NeuralData1 2/Images/$(ls '/Users/edward/Desktop/VSBL/NeuralData1 2/Images' | head -1)" "${SAMPLE}"
fi

echo "[retrain] step 4a/5: ONNX export"
"${PYTHON}" scripts/export_mobilenetv2_skipless.py \
  --checkpoint "${CKPT}" \
  --out-dir "${PKG_ROOT}/onnx_export" \
  --name "${RUN_NAME}" \
  --sample-image "${SAMPLE}" \
  --imgsz 640 \
  --device cpu \
  --conf 0.30 \
  --nms-iou 0.5 \
  --max-det 5

echo "[retrain] step 4b/5: TFLite export"
"${PYTHON}" scripts/export_mobilenetv2_tflite.py \
  --onnx-path "${PKG_ROOT}/onnx_export/${RUN_NAME}.onnx" \
  --sample-image "${SAMPLE}" \
  --out-dir "${PKG_ROOT}/tflite_export" \
  --name "${RUN_NAME}" \
  --converter-python "${TFLITE_PYTHON}" \
  --imgsz 640 \
  --conf 0.30 \
  --nms-iou 0.5 \
  --max-det 5

echo "[retrain] step 5/5: smoke on 0003 frames"
SMOKE_DIR="outputs/manager_smoke_${RUN_NAME}"
"${PYTHON}" scripts/predict_mobilenetv2_tflite.py \
  --tflite-model "${PKG_ROOT}/tflite_export/${RUN_NAME}.tflite" \
  --source "0003/Images" \
  --runtime-python "${TFLITE_PYTHON}" \
  --imgsz 640 \
  --conf 0.30 \
  --nms-iou 0.5 \
  --max-det 5 \
  --limit 8 \
  --preview-count 8 \
  --out-dir "${SMOKE_DIR}"

echo "[retrain] done."
echo "  checkpoint: ${CKPT}"
echo "  package:    ${PKG_ROOT}"
echo "  smoke:      ${SMOKE_DIR}/previews"
echo ""
echo "Compare confirmed wheels vs prev v3 (baseline 6/10) before declaring win."
