#!/usr/bin/env bash
# Keep extending data/sketchfab_cars until TARGET_TOTAL car-body GLBs exist.
#
# Designed for unattended cooldown-aware runs. The underlying fetcher returns:
#   0  normal run
#   75 Sketchfab temporary block / HTTP 429
#   76 too many consecutive per-file failures

set -euo pipefail

cd "$(dirname "$0")/.."

TARGET_TOTAL="${TARGET_TOTAL:-300}"
OUTPUT_DIR="${OUTPUT_DIR:-data/sketchfab_cars}"
MAX_MODELS="${MAX_MODELS:-200}"
MAX_MB="${MAX_MB:-45}"
CANDIDATE_LIMIT="${CANDIDATE_LIMIT:-24}"
CANDIDATE_OFFSET="${CANDIDATE_OFFSET:-0}"
OFFSET_STEP="${OFFSET_STEP:-24}"
MAX_OFFSET="${MAX_OFFSET:-480}"
RATE_LIMIT_SLEEP="${RATE_LIMIT_SLEEP:-900}"
FAILURE_SLEEP="${FAILURE_SLEEP:-30}"
DOWNLOAD_DELAY="${DOWNLOAD_DELAY:-0.5}"
MAX_DOWNLOAD_SECONDS="${MAX_DOWNLOAD_SECONDS:-10}"
SKIP_FAILED_WITHIN_HOURS="${SKIP_FAILED_WITHIN_HOURS:-24}"
MAX_CONSECUTIVE_FAILURES="${MAX_CONSECUTIVE_FAILURES:-4}"
MAX_FACE_COUNT="${MAX_FACE_COUNT:-250000}"
MAX_ROUNDS="${MAX_ROUNDS:-0}"

count_glbs() {
  find "${OUTPUT_DIR}" -maxdepth 1 -name '*.glb' | wc -l | tr -d ' '
}

round=0
while true; do
  current="$(count_glbs)"
  echo "[loop] round=${round} current=${current}/${TARGET_TOTAL} offset=${CANDIDATE_OFFSET}"
  if [[ "${current}" -ge "${TARGET_TOTAL}" ]]; then
    echo "[loop] target reached"
    exit 0
  fi
  if [[ "${MAX_ROUNDS}" != "0" && "${round}" -ge "${MAX_ROUNDS}" ]]; then
    echo "[loop] max rounds reached (${MAX_ROUNDS}); current=${current}/${TARGET_TOTAL}"
    exit 0
  fi

  set +e
  ./.venv/bin/python src/fetch_sketchfab_cars.py \
    --output-dir "${OUTPUT_DIR}" \
    --target-total "${TARGET_TOTAL}" \
    --max "${MAX_MODELS}" \
    --max-mb "${MAX_MB}" \
    --workers 1 \
    --retry-sleep 1 \
    --download-retries 0 \
    --download-delay "${DOWNLOAD_DELAY}" \
    --max-download-seconds "${MAX_DOWNLOAD_SECONDS}" \
    --from-existing-manifests \
    --car-body-only \
    --sort-candidates small-first \
    --max-face-count "${MAX_FACE_COUNT}" \
    --skip-failed-within-hours "${SKIP_FAILED_WITHIN_HOURS}" \
    --max-consecutive-failures "${MAX_CONSECUTIVE_FAILURES}" \
    --candidate-offset "${CANDIDATE_OFFSET}" \
    --candidate-limit "${CANDIDATE_LIMIT}"
  rc="$?"
  set -e

  case "${rc}" in
    0)
      CANDIDATE_OFFSET=0
      ;;
    75)
      echo "[loop] Sketchfab temporary block; sleeping ${RATE_LIMIT_SLEEP}s"
      sleep "${RATE_LIMIT_SLEEP}"
      ;;
    76)
      CANDIDATE_OFFSET=$((CANDIDATE_OFFSET + OFFSET_STEP))
      if [[ "${CANDIDATE_OFFSET}" -gt "${MAX_OFFSET}" ]]; then
        CANDIDATE_OFFSET=0
      fi
      echo "[loop] bad candidate segment; next offset=${CANDIDATE_OFFSET}; sleeping ${FAILURE_SLEEP}s"
      sleep "${FAILURE_SLEEP}"
      ;;
    *)
      echo "[loop] fetcher failed with rc=${rc}" >&2
      exit "${rc}"
      ;;
  esac

  round=$((round + 1))
done
