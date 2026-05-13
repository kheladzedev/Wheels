"""Train a YOLO-pose detector on the wheel + 3-keypoint dataset.

The model emits one class (`wheel`) with 3 keypoints per instance:
rim_left, rim_right, disc_bottom — see docs/ANNOTATION_JSON_FORMAT.md.

Usage:
    python src/train_yolo.py --data configs/dataset.yaml --epochs 50 \\
        --project runs/pose --name wheel_baseline

On Apple Silicon pass --device mps. On a CUDA box pass --device 0.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO-pose for AR wheel fitting")
    p.add_argument("--data", type=Path, default=Path("configs/dataset.yaml"))
    p.add_argument(
        "--model",
        default="yolo11n-pose.pt",
        help="Base weights to fine-tune from. Must be a -pose variant for "
        "keypoint training (e.g. yolo11n-pose.pt, yolo11s-pose.pt).",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument(
        "--device",
        default=None,
        help="'mps' on Apple Silicon, '0' on CUDA, 'cpu' otherwise",
    )
    p.add_argument(
        "--project",
        type=Path,
        default=Path("runs/pose"),
        help="Where to store run outputs. Resolved to an absolute path so Ultralytics "
        "does not redirect it under its global settings dir (~/runs/...).",
    )
    p.add_argument(
        "--name", default="wheel_baseline", help="Run name (subdir inside --project)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset config not found: {args.data}")

    if "pose" not in str(args.model).lower():
        raise ValueError(
            f"--model={args.model!r} does not look like a YOLO-pose checkpoint. "
            "Pose training requires a -pose variant (e.g. yolo11n-pose.pt). "
            "If you really want to train a detect model, edit this guard out."
        )

    project_abs = args.project.expanduser().resolve()
    project_abs.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project_abs),
        name=args.name,
    )
    metrics = model.val(data=str(args.data), device=args.device)
    print(metrics)
    print(f"\nRun outputs: {project_abs / args.name}")


if __name__ == "__main__":
    main()
