"""Write a fill-in AR-device holdout provenance template.

The output is intentionally named `*.template.json` so it cannot be
mistaken for the real production-gate input
`data/incoming/ar_device_holdout/metadata/provenance.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path("outputs/production_audit/ar_device_holdout_provenance.template.json")


def build_template() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_type": "android_ar_device_human_labelled",
        "label_type": "human_reviewed",
        "capture_device": "FILL_ME",
        "review_status": "accepted",
        "capture_app_version": "FILL_ME",
        "capture_date_utc": "FILL_ME_YYYY-MM-DD",
        "annotator": "FILL_ME",
        "reviewer": "FILL_ME",
        "notes": (
            "Replace FILL_ME values with real AR-device holdout provenance. "
            "Then save as data/incoming/ar_device_holdout/metadata/provenance.json "
            "and run src/run_production_evidence_intake.py."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(build_template(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"template={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
