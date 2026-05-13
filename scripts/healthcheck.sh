#!/usr/bin/env bash
# VSBL repo healthcheck — fast, no GPU, no training.
#
# Verifies the four invariants the project relies on at any commit:
#   1. pytest suite is green.
#   2. The plugin synthetic generator works end-to-end.
#   3. The plugin incoming validator accepts its own output.
#   4. The plugin previewer renders successfully.
#
# Run from the repo root. Exits non-zero on the first failure with the
# command that failed echoed for context.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="./.venv/bin/python"
PYTEST="./.venv/bin/pytest"

if [[ ! -x "$PY" ]]; then
    echo "ERROR: $PY not found. Create the venv first (see README)." >&2
    exit 2
fi

run() {
    echo
    echo "==> $*"
    "$@"
}

run "$PYTEST" -q

run "$PY" src/create_sample_keypoint_incoming.py --count 20 --overwrite

run "$PY" src/check_keypoint_incoming.py \
    --source-root data/incoming/android_plugin

run "$PY" src/preview_keypoint_annotations.py \
    --source-root data/incoming/android_plugin --count 5

echo
echo "OK — healthcheck passed."
