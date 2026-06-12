#!/usr/bin/env python3
"""Audit whether a web-floor dataset is ready for production training."""

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

from web_floor_real_data_gate import WebFloorRealDataGateConfig, audit_web_floor_real_data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/pose_dataset_web_floor_fixture.yaml"))
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--min-frames", type=int, default=50)
    parser.add_argument("--min-wheels", type=int, default=80)
    parser.add_argument("--required-split", action="append", dest="required_splits")
    parser.add_argument("--allow-fixture", action="store_true")
    parser.add_argument("--no-provenance-required", action="store_true")
    parser.add_argument("--allow-unknown-distance-mode", action="store_true")
    parser.add_argument("--min-distance-span", type=float, default=0.5)
    parser.add_argument("--min-angle-span-rad", type=float, default=0.05)
    parser.add_argument("--fail-on-not-ready", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gate = WebFloorRealDataGateConfig(
        min_frames=args.min_frames,
        min_wheels=args.min_wheels,
        required_splits=tuple(args.required_splits or ["train", "holdout"]),
        require_non_fixture=not args.allow_fixture,
        require_provenance=not args.no_provenance_required,
        require_known_distance_mode=not args.allow_unknown_distance_mode,
        min_distance_span=args.min_distance_span,
        min_angle_span_rad=args.min_angle_span_rad,
    )
    report = audit_web_floor_real_data(args.config, gate=gate, output_json=args.output_json)
    print(json.dumps(report, indent=2))
    if args.fail_on_not_ready and not report["production_data_ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
