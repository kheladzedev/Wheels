"""Train a YOLO-pose detector on the wheel + 3-keypoint dataset.

The model emits one class (`wheel`) with 3 keypoints per instance:
a, b, c_disc_bottom under the confirmed floor-ray contract.

Usage:
    python src/train_yolo.py --data configs/pose_dataset.yaml --epochs 50 \\
        --project runs/pose --name wheel_baseline

On Apple Silicon pass --device mps. On a CUDA box pass --device 0.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

from check_yolo_pose_dataset import SPLITS, check_split


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO-pose for AR wheel fitting")
    p.add_argument("--data", type=Path, default=Path("configs/pose_dataset.yaml"))
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
    return p.parse_args(argv)


def resolve_dataset_root(data_yaml: Path) -> Path:
    with data_yaml.open("r", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh) or {}
    raw = spec.get("path")
    if raw is None:
        raise SystemExit(f"ERROR: {data_yaml} has no 'path' entry")
    p = Path(raw)
    if p.is_absolute():
        return p
    candidates = [
        data_yaml.parent / p,
        Path.cwd() / p,
        data_yaml.parent.parent / p,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return candidates[0].resolve()


def validate_training_dataset_or_raise(data_yaml: Path) -> Path:
    """Run the confirmed floor-ray dataset gate before starting training."""
    dataset_root = resolve_dataset_root(data_yaml)
    if not dataset_root.is_dir():
        raise SystemExit(f"ERROR: dataset root does not exist: {dataset_root}")

    errors: list[str] = []
    stats = [check_split(dataset_root, split, errors) for split in SPLITS]
    if all(s["images"] == 0 for s in stats):
        errors.append(f"{dataset_root}: no images found in any split")

    if errors:
        shown = "\n".join(f"  - {line}" for line in errors[:20])
        overflow = "" if len(errors) <= 20 else f"\n  ... and {len(errors) - 20} more"
        raise SystemExit(
            "ERROR: dataset failed the confirmed floor-ray training preflight.\n"
            "Run src/check_yolo_pose_dataset.py for the full report. First problems:\n"
            f"{shown}{overflow}"
        )
    return dataset_root


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset config not found: {args.data}")

    if "pose" not in str(args.model).lower():
        raise ValueError(
            f"--model={args.model!r} does not look like a YOLO-pose checkpoint. "
            "Pose training requires a -pose variant (e.g. yolo11n-pose.pt). "
            "If you really want to train a detect model, edit this guard out."
        )

    dataset_root = validate_training_dataset_or_raise(args.data)

    project_abs = args.project.expanduser().resolve()
    project_abs.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    print(f"Dataset preflight passed: {dataset_root}")
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project_abs),
        name=args.name,
    )
    metrics = model.val(
        data=str(args.data),
        device=args.device,
        project=str(project_abs),
        name=f"{args.name}_val",
    )
    print(metrics)
    print(f"\nRun outputs: {project_abs / args.name}")


if __name__ == "__main__":
    main()
