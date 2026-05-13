"""DEPRECATED. Generates bbox-only YOLO labels with classes {wheel=0, rim=1}.

The pipeline migrated to YOLO-pose (single class `wheel` + 3 keypoints) after
the AR spec landed — this script's labels are no longer compatible with
configs/dataset.yaml and check_dataset.py. Running it followed by
check_dataset.py will fail field-count validation.

Use `create_sample_incoming.py` + `convert_incoming_to_yolo.py` instead.
That flow produces pose-format labels and matches the production ingestion
path.

This file is kept only for reference / future re-use if a bbox-only smoke
path is ever needed again.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

SPLITS = ("train", "val")
CLASS_WHEEL = 0
CLASS_RIM = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a mock wheel/rim dataset")
    p.add_argument("--dataset-root", type=Path, default=Path("data/wheel_dataset"))
    p.add_argument("--train-count", type=int, default=50)
    p.add_argument("--val-count", type=int, default=10)
    p.add_argument("--image-width", type=int, default=640)
    p.add_argument("--image-height", type=int, default=480)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, delete an existing dataset root before generating.",
    )
    return p.parse_args()


def ensure_dirs(root: Path) -> None:
    for split in SPLITS:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def random_background(rng: random.Random, w: int, h: int) -> np.ndarray:
    """Light gray-ish background with a hint of variation."""
    base = rng.randint(190, 230)
    img = np.full((h, w, 3), base, dtype=np.uint8)
    # Add a subtle horizon line for visual variety.
    horizon = rng.randint(int(h * 0.55), int(h * 0.75))
    ground_shade = max(base - rng.randint(20, 50), 0)
    img[horizon:, :] = ground_shade
    return img


def draw_car_body(
    img: np.ndarray,
    rng: random.Random,
    body_x1: int,
    body_y1: int,
    body_x2: int,
    body_y2: int,
) -> None:
    """Draw a rectangular silhouette of the car body. No label is emitted for it."""
    color = (
        rng.randint(40, 200),
        rng.randint(40, 200),
        rng.randint(40, 200),
    )
    cv2.rectangle(img, (body_x1, body_y1), (body_x2, body_y2), color, thickness=-1)
    # Cabin trapezoid on top for a tiny bit of realism.
    cabin_top_y = body_y1 - rng.randint(20, 50)
    cabin_pad = rng.randint(20, 60)
    pts = np.array(
        [
            [body_x1 + cabin_pad, body_y1],
            [body_x2 - cabin_pad, body_y1],
            [body_x2 - cabin_pad - 10, cabin_top_y],
            [body_x1 + cabin_pad + 10, cabin_top_y],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(img, [pts], color)


def xyxy_to_yolo(
    x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """Convert pixel xyxy to YOLO-normalized cx, cy, w, h, clamped to [0, 1]."""
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return (
        min(max(cx, 0.0), 1.0),
        min(max(cy, 0.0), 1.0),
        min(max(w, 0.0), 1.0),
        min(max(h, 0.0), 1.0),
    )


def generate_one(
    rng: random.Random, img_w: int, img_h: int
) -> tuple[np.ndarray, list[str]]:
    """Generate one mock image and its YOLO label lines."""
    img = random_background(rng, img_w, img_h)
    labels: list[str] = []

    # Car body geometry.
    body_w = rng.randint(int(img_w * 0.5), int(img_w * 0.75))
    body_h = rng.randint(int(img_h * 0.2), int(img_h * 0.3))
    body_x1 = rng.randint(20, img_w - body_w - 20)
    body_y1 = rng.randint(int(img_h * 0.4), int(img_h * 0.55))
    body_x2 = body_x1 + body_w
    body_y2 = body_y1 + body_h

    draw_car_body(img, rng, body_x1, body_y1, body_x2, body_y2)

    # 2 or 4 wheels along the bottom edge of the body.
    n_wheels = rng.choice([2, 4])
    wheel_radius = rng.randint(22, 38)
    # Wheels sit so their center is slightly below the body's bottom edge.
    wheel_cy = body_y2 + int(wheel_radius * 0.3)

    # Spread wheel centers evenly inside the body span, with margin.
    margin = wheel_radius + 5
    xs_start = body_x1 + margin
    xs_end = body_x2 - margin
    if n_wheels == 2:
        wheel_xs = [xs_start, xs_end]
    else:
        step = (xs_end - xs_start) / 3
        wheel_xs = [int(xs_start + step * i) for i in range(4)]

    for cx in wheel_xs:
        # Wheel (tire) — dark circle.
        cv2.circle(img, (cx, wheel_cy), wheel_radius, (30, 30, 30), thickness=-1)
        wx1, wy1 = cx - wheel_radius, wheel_cy - wheel_radius
        wx2, wy2 = cx + wheel_radius, wheel_cy + wheel_radius
        ycx, ycy, yw, yh = xyxy_to_yolo(wx1, wy1, wx2, wy2, img_w, img_h)
        labels.append(f"{CLASS_WHEEL} {ycx:.6f} {ycy:.6f} {yw:.6f} {yh:.6f}")

        # Rim — lighter circle inside the wheel.
        rim_radius = int(wheel_radius * rng.uniform(0.5, 0.65))
        rim_color = (
            rng.randint(160, 220),
            rng.randint(160, 220),
            rng.randint(160, 220),
        )
        cv2.circle(img, (cx, wheel_cy), rim_radius, rim_color, thickness=-1)
        rx1, ry1 = cx - rim_radius, wheel_cy - rim_radius
        rx2, ry2 = cx + rim_radius, wheel_cy + rim_radius
        rcx, rcy, rw, rh = xyxy_to_yolo(rx1, ry1, rx2, ry2, img_w, img_h)
        labels.append(f"{CLASS_RIM} {rcx:.6f} {rcy:.6f} {rw:.6f} {rh:.6f}")

    return img, labels


def write_split(
    root: Path, split: str, count: int, rng: random.Random, img_w: int, img_h: int
) -> tuple[int, int]:
    """Generate `count` samples for one split. Returns (n_images, n_label_lines)."""
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split

    total_labels = 0
    for i in range(count):
        img, labels = generate_one(rng, img_w, img_h)
        name = f"mock_{split}_{i:04d}"
        cv2.imwrite(str(images_dir / f"{name}.jpg"), img)
        (labels_dir / f"{name}.txt").write_text(
            "\n".join(labels) + "\n", encoding="utf-8"
        )
        total_labels += len(labels)
    return count, total_labels


def main() -> int:
    args = parse_args()
    root: Path = args.dataset_root

    if root.exists() and any(root.iterdir()):
        if not args.overwrite:
            print(f"Dataset root already exists and is not empty: {root}")
            print(
                "Pass --overwrite to delete and regenerate, or choose a different --dataset-root."
            )
            return 1
        shutil.rmtree(root)

    ensure_dirs(root)
    rng = random.Random(args.seed)

    train_imgs, train_labels = write_split(
        root, "train", args.train_count, rng, args.image_width, args.image_height
    )
    val_imgs, val_labels = write_split(
        root, "val", args.val_count, rng, args.image_width, args.image_height
    )

    print()
    print("Mock dataset generated.")
    print(f"  Root:           {root}")
    print(f"  Image size:     {args.image_width}x{args.image_height}")
    print(f"  Train images:   {train_imgs}")
    print(f"  Val images:     {val_imgs}")
    print(
        f"  Total labels:   {train_labels + val_labels} lines "
        f"(train={train_labels}, val={val_labels})"
    )
    print("  Classes:        0=wheel, 1=rim")
    print()
    print(
        "NOTE: this is a synthetic smoke-test dataset, not training data for "
        "a real model."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
