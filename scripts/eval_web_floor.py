#!/usr/bin/env python3
"""Evaluate the web floor fixture readiness gate."""

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

from evaluate_web_floor import evaluate_web_floor_fixture


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/pose_dataset_web_floor_fixture.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/web_floor_network/train_fixture/web_floor_fixture_checkpoint.pt"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/web_floor_network/eval_fixture/web_floor_eval.json"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate_web_floor_fixture(
        config=args.config,
        checkpoint=args.checkpoint,
        output_json=args.output_json,
        device=args.device,
    )
    print(json.dumps({
        "output_json": str(args.output_json),
        "pipeline_ready": report["pipeline_ready"],
        "trained_model_ready": report["trained_model_ready"],
        "production_ready": report["production_ready"],
        "floor_mae": report["floor_metrics"]["mae"],
        "runtime_requirements": report["runtime_requirements"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
