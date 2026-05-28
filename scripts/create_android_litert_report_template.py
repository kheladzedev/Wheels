"""Write a fill-in Android LiteRT device report template.

The output is intentionally named `*.template.json` so it cannot be
mistaken for the real production-gate input
`data/incoming/android_litert_device_report.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_TFLITE = Path("outputs/production_audit/tflite_export/best_float32.tflite")
DEFAULT_OUT = Path("outputs/production_audit/android_litert_device_report.template.json")


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_template(artifact: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_type": "android_litert_device_validation",
        "test_session_id": "FILL_ME",
        "test_app_version": "FILL_ME",
        "test_date_utc": "FILL_ME_YYYY-MM-DD",
        "device": {
            "model": "FILL_ME",
            "manufacturer": "FILL_ME",
            "android_version": "FILL_ME",
            "soc": "FILL_ME",
            "is_emulator": False,
        },
        "runtime": "LiteRT",
        "artifact": {
            "path": str(artifact),
            "sha256": sha256_file(artifact) or "FILL_ME",
            "format": "tflite_float32",
        },
        "input": {
            "shape": [1, 640, 640, 3],
            "dtype": "float32",
            "profile": "zero_float32_smoke",
        },
        "latency_ms": {
            "runs": 30,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
        },
        "memory_mb": {
            "peak": 0.0,
        },
        "output": {
            "shape": [1, 14, 8400],
            "finite": True,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
        },
        "notes": (
            "Replace FILL_ME and measured numeric fields with values from the "
            "target Android app/device run. Then save as "
            "data/incoming/android_litert_device_report.json and run "
            "src/validate_android_litert_report.py."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_TFLITE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    template = build_template(args.artifact)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(template, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"template={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
