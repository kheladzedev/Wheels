#!/usr/bin/env python3
"""Import CSV web-floor annotations into a manifest/config pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from web_floor_annotation_import import import_web_floor_csv_annotations


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config-out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = import_web_floor_csv_annotations(
        csv_path=args.csv,
        image_root=args.image_root,
        dataset_root=args.dataset_root,
        config_out=args.config_out,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "manifest": str(args.dataset_root / "manifest.json"),
                "config": str(args.config_out),
                "items": len(manifest["items"]),
                "wheels": sum(len(item["wheels"]) for item in manifest["items"]),
                "fixture_only": manifest["fixture_only"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
