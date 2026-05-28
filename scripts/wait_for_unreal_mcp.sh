#!/usr/bin/env bash
# Wait until the local UnrealMCP TCP helper starts accepting commands.

set -euo pipefail

cd "$(dirname "$0")/.."

TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-10}"
STARTED_AT="$(date +%s)"

echo "[mcp-wait] waiting for UnrealMCP on 127.0.0.1:55557 timeout=${TIMEOUT_SECONDS}s"
while true; do
  if ./.venv/bin/python scripts/ue/_send.py exec_code 'print("ue_mcp_ping")' >/tmp/vsbl_unreal_mcp_ping.log 2>&1; then
    echo "[mcp-wait] UnrealMCP reachable"
    cat /tmp/vsbl_unreal_mcp_ping.log
    exit 0
  fi

  now="$(date +%s)"
  elapsed=$((now - STARTED_AT))
  if [[ "${elapsed}" -ge "${TIMEOUT_SECONDS}" ]]; then
    echo "[mcp-wait] timed out after ${elapsed}s" >&2
    cat /tmp/vsbl_unreal_mcp_ping.log >&2 || true
    exit 75
  fi

  echo "[mcp-wait] not ready after ${elapsed}s; sleeping ${INTERVAL_SECONDS}s"
  sleep "${INTERVAL_SECONDS}"
done
