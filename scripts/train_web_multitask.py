#!/usr/bin/env python3
"""Train/dry-run the web multi-task wheel + floor model."""

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

from web_floor_training import WebFloorTrainConfig, run_fixture_training


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/pose_dataset_web_floor_fixture.yaml"))
    parser.add_argument("--stage", choices=("2d", "floor", "joint", "recon"), default="floor")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/web_floor_network/train_fixture"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "auto"))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument(
        "--enable-reconstruction-loss",
        action="store_true",
        help="Offline experiment flag only; runtime remains direct/no-depth/no-RANSAC.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = WebFloorTrainConfig(
        config=args.config,
        stage=args.stage,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        out_dir=args.out_dir,
        device=args.device,
        lr=args.lr,
        seed=args.seed,
        pretrained=args.pretrained,
        enable_reconstruction_loss=args.enable_reconstruction_loss,
    )
    metrics = run_fixture_training(config)
    print(json.dumps(metrics, indent=2))
    print(f"Metrics: {args.out_dir / 'metrics.json'}")
    print(f"Checkpoint: {metrics['checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
