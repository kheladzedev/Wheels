"""Render YOLO-pose labels onto random images for visual inspection.

Built for the plugin-converter output (`data/wheel_pose_dataset/`). Mirrors
`preview_labels.py` but uses the A / B / C naming and writes to a separate
output directory so it doesn't clobber the legacy previews.

Keypoint semantics (2026-05-14 revision — see docs/KEYPOINT_SPEC.md):
    * A (green) — left floor-ray point. Screen-space raycast source
      onto the floor plane for the wheel-plane base. **Not** a
      metal-rim edge.
    * B (yellow) — right floor-ray point. Mirror of A. **Not** a
      metal-rim edge.
    * C (red) — lowest visible point of the metal rim / disc.

Visual labels remain the short tags "A" / "B" / "c_disc_bottom" so the
overlay matches the YOLO-pose label-line layout 1:1.

Usage:
    python src/preview_yolo_pose_labels.py \\
        --dataset-root data/wheel_pose_dataset --split train --count 10
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_NAMES = {0: "wheel"}
CLASS_COLORS = {0: (0, 165, 255)}  # BGR — orange

# Keypoint order matches the converter: a, b, c_disc_bottom.
KEYPOINT_NAMES = ("a", "b", "c_disc_bottom")
KEYPOINT_COLORS = (
    (0, 255, 0),  # a — green
    (0, 255, 255),  # b — yellow
    (0, 0, 255),  # c_disc_bottom — red
)
N_KEYPOINTS = 3
FIELDS_PER_LINE = 5 + N_KEYPOINTS * 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preview YOLO-pose labels (A/B/C) on random images"
    )
    p.add_argument("--dataset-root", required=True, type=Path)
    p.add_argument("--split", default="train", choices=("train", "val"))
    p.add_argument("--count", type=int, default=10, help="How many samples to render")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/pose_label_preview"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def yolo_to_xyxy(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    x1 = int(round((cx - w / 2.0) * img_w))
    y1 = int(round((cy - h / 2.0) * img_h))
    x2 = int(round((cx + w / 2.0) * img_w))
    y2 = int(round((cy + h / 2.0) * img_h))
    return x1, y1, x2, y2


def draw_labels(img: np.ndarray, label_path: Path) -> tuple[int, int]:
    """Draw all wheels + their A/B/C keypoints. Returns (n_wheels, n_keypoints)."""
    if not label_path.exists():
        return 0, 0
    img_h, img_w = img.shape[:2]
    n_wheels = 0
    n_kps = 0
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) != FIELDS_PER_LINE:
            # check_yolo_pose_dataset.py is the authoritative validator;
            # silently skip malformed lines here so preview still renders.
            continue
        try:
            cls_id = int(parts[0])
            cx, cy, w, h = (float(x) for x in parts[1:5])
            kp_vals = parts[5:]
        except ValueError:
            continue

        x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        name = CLASS_NAMES.get(cls_id, str(cls_id))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            name,
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        n_wheels += 1

        for i in range(N_KEYPOINTS):
            try:
                kx_n = float(kp_vals[i * 3])
                ky_n = float(kp_vals[i * 3 + 1])
                vis = int(float(kp_vals[i * 3 + 2]))
            except ValueError:
                continue
            if vis == 0:
                continue
            kx = int(round(kx_n * img_w))
            ky = int(round(ky_n * img_h))
            kp_color = KEYPOINT_COLORS[i]
            # Filled for visible (vis=2), hollow for occluded (vis=1).
            cv2.circle(img, (kx, ky), 5, kp_color, -1 if vis == 2 else 2)
            tag = KEYPOINT_NAMES[i] + ("?" if vis == 1 else "")
            cv2.putText(
                img,
                tag,
                (kx + 6, ky - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                kp_color,
                1,
                cv2.LINE_AA,
            )
            n_kps += 1

    return n_wheels, n_kps


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    images_dir = args.dataset_root / "images" / args.split
    labels_dir = args.dataset_root / "labels" / args.split

    if not images_dir.is_dir():
        print(f"ERROR: images dir not found: {images_dir}")
        return 2
    if not labels_dir.is_dir():
        print(f"ERROR: labels dir not found: {labels_dir}")
        return 2

    all_images = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    if not all_images:
        print(f"ERROR: no images found in {images_dir}")
        return 1

    rng = random.Random(args.seed)
    sample = rng.sample(all_images, k=min(args.count, len(all_images)))

    out_dir = args.out_dir / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sample:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Skip (unreadable): {img_path}")
            continue
        n_wheels, n_kps = draw_labels(img, labels_dir / f"{img_path.stem}.txt")
        out_path = out_dir / f"{img_path.stem}_pose_labels.jpg"
        cv2.imwrite(str(out_path), img)
        print(f"{img_path.name}: {n_wheels} wheel(s), {n_kps} kp(s) -> {out_path}")

    print(f"\nDone. Preview dir: {out_dir}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
