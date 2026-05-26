#!/usr/bin/env bash
# Run CameraCaptureWheels in a loop because each play through the spline only
# yields ~100 frames before the exporter halts. Each invocation appends to the
# project's Images/ folder (UE picks the next free index), so total frames grow
# monotonically across runs.
#
# Usage:
#   scripts/run_unreal_capture_loop.sh <map_short> <runs> [target_per_run]
#
# Example (15 runs × ~100 frames ≈ 1500 total):
#   scripts/run_unreal_capture_loop.sh standartWheelsRoom 15

set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "usage: $0 <map_short> <runs> [target_per_run]" >&2
  exit 2
fi

MAP_SHORT="$1"
RUNS="$2"
TARGET_PER_RUN="${3:-200}"

REPO="/Users/edward/Desktop/VSBL"
IMAGES_DIR="${REPO}/NeuralData1 2/Images"

START_TOTAL=$(find "${IMAGES_DIR}" -maxdepth 1 -type f -name "*.jpg" 2>/dev/null | wc -l | tr -d ' ')
echo "[loop] start total: ${START_TOTAL}"
echo "[loop] runs: ${RUNS}, target_per_run: ${TARGET_PER_RUN}"

for (( i=1; i<=RUNS; i++ )); do
  BEFORE=$(find "${IMAGES_DIR}" -maxdepth 1 -type f -name "*.jpg" 2>/dev/null | wc -l | tr -d ' ')
  CUMULATIVE_TARGET=$((BEFORE + TARGET_PER_RUN))
  echo "[loop] run ${i}/${RUNS}: before=${BEFORE} target_after=${CUMULATIVE_TARGET}"

  # Sub-call's stuck-detection is its job, but we cap with a soft per-run timeout.
  # run_unreal_capture.sh polls and kills UE when target reached or on signal.
  # Each spline play yields ~100 frames then UE idles → we kill via stuck.
  (
    "${REPO}/scripts/run_unreal_capture.sh" "${MAP_SHORT}" "${CUMULATIVE_TARGET}" 2>&1
  ) &
  RUN_PID=$!

  # Watch for "stuck" (no growth for 30s) and kill the run script.
  prev=$BEFORE
  last_change=$(date +%s)
  while kill -0 $RUN_PID 2>/dev/null; do
    curr=$(find "${IMAGES_DIR}" -maxdepth 1 -type f -name "*.jpg" 2>/dev/null | wc -l | tr -d ' ')
    if (( curr != prev )); then
      last_change=$(date +%s)
      prev=$curr
    fi
    now=$(date +%s)
    if (( curr >= CUMULATIVE_TARGET )); then
      break
    fi
    if (( now - last_change > 180 )); then
      echo "[loop] run ${i} stuck at ${curr}, killing"
      # Kill the run script's whole tree.
      pkill -P $RUN_PID 2>/dev/null || true
      kill $RUN_PID 2>/dev/null || true
      pkill -9 -f "UnrealEditor.*${MAP_SHORT}" 2>/dev/null || true
      break
    fi
    sleep 3
  done

  wait $RUN_PID 2>/dev/null || true

  AFTER=$(find "${IMAGES_DIR}" -maxdepth 1 -type f -name "*.jpg" 2>/dev/null | wc -l | tr -d ' ')
  DELTA=$((AFTER - BEFORE))
  echo "[loop] run ${i} done: ${BEFORE} -> ${AFTER} (+${DELTA})"

  # Belt and braces — make sure no UE is left running before next iteration.
  # First a polite SIGTERM, give UE time to release shared memory / lock files,
  # then SIGKILL if anything lingers, then clear lingering trace server state
  # that otherwise blocks the next UE launch.
  pkill -f "UnrealEditor.*${MAP_SHORT}" 2>/dev/null || true
  sleep 5
  pkill -9 -f "UnrealEditor.*${MAP_SHORT}" 2>/dev/null || true
  pkill -9 -f "UnrealTraceServer" 2>/dev/null || true
  rm -f /tmp/UnrealTraceServer.pid 2>/dev/null || true
  echo "[loop] run ${i} cooldown..."
  sleep 10

  if (( DELTA < 10 )); then
    echo "[loop] run ${i} produced <10 frames; aborting loop"
    break
  fi
done

END_TOTAL=$(find "${IMAGES_DIR}" -maxdepth 1 -type f -name "*.jpg" 2>/dev/null | wc -l | tr -d ' ')
echo "[loop] final total: ${END_TOTAL} (+ $((END_TOTAL - START_TOTAL)) over loop)"
