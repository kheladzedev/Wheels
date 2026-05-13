"""Sanity-check the wheel YOLO-pose dataset before training.

Verifies the layout described in docs/DATASET_SPEC.md:
  - images/{train,val} and labels/{train,val} exist
  - every image has a corresponding label file
  - every label line has the right number of fields for pose format:
      5 (bbox) + 3 * 3 (keypoints) = 14 fields
  - class_id is 0 (single class `wheel`)
  - bbox coordinates are in [0, 1]
  - keypoint coordinates are in [0, 1] when visibility > 0
  - visibility flags are in {0, 1, 2}

Exits with code 0 if everything is OK, non-zero otherwise.

Usage:
    python src/check_dataset.py --dataset-root data/wheel_dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_CLASS_IDS = {0}
SPLITS = ("train", "val")
N_KEYPOINTS = 3
FIELDS_PER_LINE = 5 + N_KEYPOINTS * 3  # class + bbox(4) + (x,y,v) per kp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate wheel YOLO-pose dataset layout")
    p.add_argument("--dataset-root", required=True, type=Path)
    return p.parse_args()


def list_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def validate_label_file(label_path: Path) -> list[str]:
    problems: list[str] = []
    try:
        text = label_path.read_text(encoding="utf-8")
    except OSError as e:
        return [f"{label_path}: cannot read ({e})"]

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != FIELDS_PER_LINE:
            problems.append(
                f"{label_path}:{lineno}: expected {FIELDS_PER_LINE} fields "
                f"(class + bbox + {N_KEYPOINTS}*(x,y,v)), got {len(parts)}"
            )
            continue
        try:
            cls_id = int(parts[0])
            bbox_vals = [float(x) for x in parts[1:5]]
            kp_vals = parts[5:]
        except ValueError:
            problems.append(f"{label_path}:{lineno}: non-numeric field")
            continue

        if cls_id not in ALLOWED_CLASS_IDS:
            problems.append(
                f"{label_path}:{lineno}: class_id={cls_id} not in {sorted(ALLOWED_CLASS_IDS)}"
            )

        for name, val in zip(("cx", "cy", "w", "h"), bbox_vals):
            if not 0.0 <= val <= 1.0:
                problems.append(
                    f"{label_path}:{lineno}: bbox {name}={val} not in [0,1]"
                )

        for i in range(N_KEYPOINTS):
            try:
                kx = float(kp_vals[i * 3])
                ky = float(kp_vals[i * 3 + 1])
                vis = int(float(kp_vals[i * 3 + 2]))
            except ValueError:
                problems.append(f"{label_path}:{lineno}: kp{i} non-numeric")
                continue
            if vis not in (0, 1, 2):
                problems.append(
                    f"{label_path}:{lineno}: kp{i} visibility={vis} not in 0/1/2"
                )
            if vis > 0:
                if not 0.0 <= kx <= 1.0:
                    problems.append(f"{label_path}:{lineno}: kp{i}.x={kx} not in [0,1]")
                if not 0.0 <= ky <= 1.0:
                    problems.append(f"{label_path}:{lineno}: kp{i}.y={ky} not in [0,1]")
    return problems


def check_split(root: Path, split: str, errors: list[str]) -> dict:
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    stats = {
        "split": split,
        "images": 0,
        "labels": 0,
        "missing_labels": 0,
        "empty_labels": 0,
    }

    if not images_dir.is_dir():
        errors.append(f"Missing directory: {images_dir}")
        return stats
    if not labels_dir.is_dir():
        errors.append(f"Missing directory: {labels_dir}")
        return stats

    images = list_images(images_dir)
    label_files = {p.stem for p in labels_dir.glob("*.txt")}

    stats["images"] = len(images)
    stats["labels"] = len(label_files)

    for img in images:
        label_path = labels_dir / f"{img.stem}.txt"
        if not label_path.exists():
            errors.append(f"Missing label for image: {img}")
            stats["missing_labels"] += 1
            continue
        if label_path.stat().st_size == 0:
            stats["empty_labels"] += 1
        errors.extend(validate_label_file(label_path))

    image_stems = {p.stem for p in images}
    orphan_labels = label_files - image_stems
    for stem in sorted(orphan_labels):
        errors.append(f"Label without image: {labels_dir / (stem + '.txt')}")

    return stats


def main() -> int:
    args = parse_args()
    root: Path = args.dataset_root

    print(f"Dataset root: {root}")
    if not root.is_dir():
        print(f"ERROR: dataset root does not exist: {root}")
        return 2

    errors: list[str] = []
    all_stats = [check_split(root, split, errors) for split in SPLITS]

    print()
    print("Split   Images   Labels   Missing labels   Empty labels")
    print("-----   ------   ------   --------------   ------------")
    for s in all_stats:
        print(
            f"{s['split']:<5}   {s['images']:>6}   {s['labels']:>6}"
            f"   {s['missing_labels']:>14}   {s['empty_labels']:>12}"
        )

    print()
    if errors:
        print(f"FAILED — {len(errors)} problem(s):")
        for line in errors[:50]:
            print(f"  - {line}")
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more")
        return 1

    if all(s["images"] == 0 for s in all_stats):
        print("FAILED — no images found in any split")
        return 1

    print("OK — dataset layout looks valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
