#!/usr/bin/env bash
# Launch NeuralData Unreal project in standalone game mode to drive
# CameraCaptureWheels. Waits until the project's Images/ directory has at
# least the requested number of frames, then kills the editor.
#
# Usage:
#   scripts/run_unreal_capture.sh <map_short> <target_frame_count> [extra UE flags...]
#
# Example:
#   scripts/run_unreal_capture.sh standartWheelsRoom_capture_clean_v2 600
#
# Notes:
# - Append mode: CameraCaptureWheels continues numbering from the existing
#   highest index in NeuralData1\ 2/Images/. Pre-existing files stay.
# - Stop condition is a file count threshold, not a duration.
# - macOS only. Spawns UnrealEditor.app via its binary; not the headless -Cmd.

set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "usage: $0 <map_short> <target_frame_count> [extra UE flags...]" >&2
  exit 2
fi

MAP_SHORT="$1"
TARGET="$2"
shift 2

REPO="/Users/edward/Desktop/VSBL"
PROJECT="${REPO}/NeuralData1 2"
UPROJECT="${PROJECT}/NeuralData.uproject"
IMAGES_DIR="${PROJECT}/Images"
UE_BIN="/Users/Shared/Epic Games/UE_5.7/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor"
MAP_LONG="/Game/Wheels/maps/${MAP_SHORT}"

if [[ ! -x "${UE_BIN}" ]]; then
  echo "error: UnrealEditor binary not found at: ${UE_BIN}" >&2
  exit 1
fi
if [[ ! -f "${UPROJECT}" ]]; then
  echo "error: uproject not found at: ${UPROJECT}" >&2
  exit 1
fi

mkdir -p "${IMAGES_DIR}"
START_COUNT=$(find "${IMAGES_DIR}" -maxdepth 1 -type f -name "*.jpg" | wc -l | tr -d ' ')
echo "[run_unreal_capture] starting frame count: ${START_COUNT}"
echo "[run_unreal_capture] target frame count:   ${TARGET}"
echo "[run_unreal_capture] map:                  ${MAP_LONG}"

LOG_FILE="${REPO}/outputs/unreal_control/capture_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "${LOG_FILE}")"

# Launch UE in standalone game mode in the background.
"${UE_BIN}" \
  "${UPROJECT}" \
  "${MAP_LONG}" \
  -game \
  -ResX=1920 -ResY=1080 \
  -windowed \
  -nosplash \
  "$@" \
  > "${LOG_FILE}" 2>&1 &

UE_PID=$!
echo "[run_unreal_capture] UE PID: ${UE_PID}"
echo "[run_unreal_capture] log:    ${LOG_FILE}"

cleanup() {
  if kill -0 "${UE_PID}" 2>/dev/null; then
    echo "[run_unreal_capture] killing UE PID ${UE_PID}"
    kill "${UE_PID}" 2>/dev/null || true
    sleep 2
    if kill -0 "${UE_PID}" 2>/dev/null; then
      kill -9 "${UE_PID}" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT INT TERM

# Poll Images/ until we reach the target. Print progress every iteration so the
# operator can follow along.
LAST_REPORT=0
while true; do
  if ! kill -0 "${UE_PID}" 2>/dev/null; then
    echo "[run_unreal_capture] UE process exited unexpectedly; see log" >&2
    exit 1
  fi
  CURR=$(find "${IMAGES_DIR}" -maxdepth 1 -type f -name "*.jpg" | wc -l | tr -d ' ')
  if (( CURR != LAST_REPORT )); then
    echo "[run_unreal_capture] Images/ count: ${CURR} / ${TARGET}"
    LAST_REPORT=$CURR
  fi
  if (( CURR >= TARGET )); then
    echo "[run_unreal_capture] reached target ${TARGET}; shutting down UE"
    break
  fi
  sleep 5
done

cleanup
trap - EXIT

# Final counts
FINAL_IMG=$(find "${IMAGES_DIR}" -maxdepth 1 -type f -name "*.jpg" | wc -l | tr -d ' ')
FINAL_KP=$(find "${PROJECT}/keyPoint" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
echo "[run_unreal_capture] done. Images=${FINAL_IMG}, keyPoint dirs=${FINAL_KP}"
