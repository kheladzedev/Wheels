"""Train wheel_real_v1_soft with heavier augmentation. Two variants:
--model n  -> yolo11n-pose (fast, current baseline architecture)
--model s  -> yolo11s-pose (more capacity)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["n", "s"], default="n")
    p.add_argument("--name", required=True)
    p.add_argument("--epochs", type=int, default=80)
    args = p.parse_args()

    weights = "yolo11n-pose.pt" if args.model == "n" else "yolo11s-pose.pt"
    model = YOLO(weights)
    model.train(
        data="configs/pose_dataset_real_v1_soft.yaml",
        epochs=args.epochs,
        device="mps",
        project="runs/pose",
        name=args.name,
        # Augmentation (Ultralytics keys)
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,
        hsv_h=0.020,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.10,
        scale=0.5,
        fliplr=0.5,
        degrees=10.0,
        # Pose head training stability
        patience=30,
        cos_lr=True,
    )


if __name__ == "__main__":
    main()
