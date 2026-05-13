"""Render plugin-format keypoint annotations onto sampled frames.

Reads a batch produced in the on-disk contract from
`docs/KEYPOINT_DATASET_FORMAT.md` and draws each wheel's bbox + A/B/C
keypoints onto the matching image. Output goes to JPEGs in
`<output-root>/<stem>_preview.jpg`.

Keypoint semantics (2026-05-14 revision — see docs/KEYPOINT_SPEC.md):
    * A (green) — left floor-ray point. Screen-space raycast source
      onto the floor plane. **Not** a metal-rim edge.
    * B (yellow) — right floor-ray point. Same role on the right.
      **Not** a metal-rim edge.
    * C (red) — lowest visible point of the metal rim / disc.

Sister scripts:
  - `src/check_keypoint_incoming.py` — validates the batch.
  - `src/create_sample_keypoint_incoming.py` — synthesises a batch.
  - `src/preview_labels.py` — sister tool for the YOLO-pose canonical layout.

Usage:
    python src/preview_keypoint_annotations.py \\
        --source-root data/incoming/android_plugin --count 10
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# BGR. Orange for the bbox so it never collides with the green/yellow/red
# keypoint markers in either channel.
BBOX_COLOR_BGR = (0, 165, 255)
POINT_A_COLOR_BGR = (0, 255, 0)
POINT_B_COLOR_BGR = (0, 255, 255)
POINT_C_COLOR_BGR = (0, 0, 255)
LABEL_OFFSET = (6, -6)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preview plugin-format keypoint annotations"
    )
    p.add_argument("--source-root", required=True, type=Path)
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/keypoint_preview"),
    )
    return p.parse_args(argv)


def _list_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _draw_one(img: np.ndarray, annotation: dict) -> int:
    """Overlay bbox + A/B/C markers for every wheel in the annotation.

    Returns the number of wheels drawn. Annotation is trusted (validator is
    a separate tool); malformed entries are silently skipped here so the
    preview doesn't blow up on partial batches.
    """
    n = 0
    for wheel in annotation.get("wheels", []):
        bbox = wheel.get("bbox_xyxy")
        points = wheel.get("points") or {}
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        try:
            x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
        except (TypeError, ValueError):
            continue
        cv2.rectangle(img, (x1, y1), (x2, y2), BBOX_COLOR_BGR, 2)

        for key, color, label in (
            ("a", POINT_A_COLOR_BGR, "A"),
            ("b", POINT_B_COLOR_BGR, "B"),
            ("c_disc_bottom", POINT_C_COLOR_BGR, "C"),
        ):
            pt = points.get(key)
            if not (isinstance(pt, list) and len(pt) == 2):
                continue
            try:
                px, py = int(round(float(pt[0]))), int(round(float(pt[1])))
            except (TypeError, ValueError):
                continue
            cv2.circle(img, (px, py), 5, color, -1, lineType=cv2.LINE_AA)
            cv2.putText(
                img,
                label,
                (px + LABEL_OFFSET[0], py + LABEL_OFFSET[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_root: Path = args.source_root
    images_dir = source_root / "images"
    annos_dir = source_root / "annotations"

    if not images_dir.is_dir():
        print(f"ERROR: missing images directory: {images_dir}")
        return 1
    if not annos_dir.is_dir():
        print(f"ERROR: missing annotations directory: {annos_dir}")
        return 1

    images = _list_images(images_dir)
    if not images:
        print(f"No images found in {images_dir}")
        return 0

    rng = random.Random(args.seed)
    pairs: list[tuple[Path, Path]] = []
    for img_path in images:
        anno = annos_dir / f"{img_path.stem}.json"
        if anno.exists():
            pairs.append((img_path, anno))
        else:
            print(f"WARNING: skipping {img_path.name} — no annotation {anno.name}")

    if not pairs:
        print("No matched (image, annotation) pairs to preview.")
        return 0

    sample = rng.sample(pairs, k=min(args.count, len(pairs)))
    args.output_root.mkdir(parents=True, exist_ok=True)

    n_done = 0
    for img_path, anno_path in sample:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"WARNING: unreadable image, skipping: {img_path}")
            continue
        try:
            annotation = json.loads(anno_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"WARNING: bad JSON, skipping: {anno_path} ({e})")
            continue
        _draw_one(img, annotation)
        out_path = args.output_root / f"{img_path.stem}_preview.jpg"
        cv2.imwrite(str(out_path), img)
        n_done += 1

    print(f"Previewed {n_done} images, wrote to {args.output_root}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
