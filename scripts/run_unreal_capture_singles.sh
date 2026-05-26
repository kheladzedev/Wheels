#!/usr/bin/env bash
# Multiple independent capture runs of CameraCaptureWheels.
#
# Unlike run_unreal_capture_loop.sh which calls run_unreal_capture.sh as a
# child and shares state, this script launches each capture as an isolated
# UE process with explicit SaveGames + trace-server cleanup between runs.
# That matches the observation that a single fresh UE launch reliably writes
# 100 frames, while back-to-back UE relaunches inside a loop hit some shared
# state we cannot easily flush.
#
# Usage:
#   scripts/run_unreal_capture_singles.sh <map_short> <num_runs>

set -uo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "usage: $0 <map_short> <num_runs>" >&2
  exit 2
fi

MAP_SHORT="$1"
NUM_RUNS="$2"

REPO="/Users/edward/Desktop/VSBL"
PROJECT="${REPO}/NeuralData1 2"
IMAGES_DIR="${PROJECT}/Images"
SAVES_DIR="${PROJECT}/Saved/SaveGames"

for ((i=1; i<=NUM_RUNS; i++)); do
  BEFORE=$(find "${IMAGES_DIR}" -maxdepth 1 -type f -name "*.jpg" 2>/dev/null | wc -l | tr -d ' ')
  TARGET=$((BEFORE + 100))
  echo "[singles] run ${i}/${NUM_RUNS}: before=${BEFORE} target=${TARGET}"

  # Drop SaveGame + trace state so the next UE launch behaves like the first.
  rm -rf "${SAVES_DIR}"/* 2>/dev/null || true
  pkill -9 -f "UnrealTraceServer" 2>/dev/null || true
  rm -f /tmp/UnrealTraceServer.pid 2>/dev/null || true
  sleep 2

  "${REPO}/scripts/run_unreal_capture.sh" "${MAP_SHORT}" "${TARGET}" \
    >> "${REPO}/outputs/unreal_control/singles_$(date +%Y%m%d).log" 2>&1

  AFTER=$(find "${IMAGES_DIR}" -maxdepth 1 -type f -name "*.jpg" 2>/dev/null | wc -l | tr -d ' ')
  DELTA=$((AFTER - BEFORE))
  echo "[singles] run ${i} done: ${BEFORE} -> ${AFTER} (+${DELTA})"

  pkill -9 -f "UnrealEditor.*${MAP_SHORT}" 2>/dev/null || true
  sleep 8

  if (( DELTA < 30 )); then
    echo "[singles] run ${i} produced ${DELTA} frames; not aborting, retry next"
  fi
done

FINAL=$(find "${IMAGES_DIR}" -maxdepth 1 -type f -name "*.jpg" 2>/dev/null | wc -l | tr -d ' ')
echo "[singles] final total: ${FINAL}"
