#!/usr/bin/env python3
"""Tiny helper to send commands to the UnrealMCP TCP plugin (port 55557).

Usage:
    python3 scripts/ue/_send.py exec_file <path.py>
    python3 scripts/ue/_send.py exec_code  '<inline python source>'
    python3 scripts/ue/_send.py cmd        '{"type":"...","params":{...}}'
"""

from __future__ import annotations
import json
import os
import socket
import sys
from pathlib import Path

HOST, PORT = "127.0.0.1", 55557
TIMEOUT_S = float(os.environ.get("VSBL_MCP_TIMEOUT", "1800"))


def send(payload: dict) -> dict:
    s = socket.socket()
    s.settimeout(TIMEOUT_S)
    s.connect((HOST, PORT))
    s.send(json.dumps(payload).encode())
    data = b""
    while True:
        chunk = s.recv(16384)
        if not chunk:
            break
        data += chunk
        try:
            json.loads(data.decode())
            break
        except Exception:
            continue
    s.close()
    return json.loads(data.decode())


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    mode, arg = argv[1], argv[2]
    if mode == "exec_file":
        payload = {
            "type": "execute_python",
            "params": {"file": str(Path(arg).resolve())},
        }
    elif mode == "exec_code":
        payload = {"type": "execute_python", "params": {"code": arg}}
    elif mode == "cmd":
        payload = json.loads(arg)
    else:
        print(f"unknown mode {mode}")
        return 2
    resp = send(payload)
    print(json.dumps(resp, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
