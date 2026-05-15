"""Compare before/after wheel-detection overlays on the 8 manual_real images.

Reads two prediction overlay directories and produces:
  1. A printed table of per-image detection counts (before, after, delta).
  2. A side-by-side image strip per source: before | after, saved to
     outputs/manual_real_compare/.

Detection counts are inferred from the overlay filenames matching the
source filenames in data/manual_real/images/. The count itself isn't
recoverable from a saved overlay — Ultralytics burns labels into the
image. So we pass counts in as a CLI dict (printed by the inference
script when it ran).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data" / "manual_real" / "images"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--before-dir", required=True, type=Path)
    p.add_argument("--after-dir", required=True, type=Path)
    p.add_argument(
        "--counts-json",
        required=True,
        type=Path,
        help="JSON file mapping {source_filename: {before: int, after: int}}.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "outputs" / "manual_real_compare",
    )
    return p.parse_args()


def stack_side_by_side(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    h_b, w_b = before.shape[:2]
    h_a, w_a = after.shape[:2]
    h = max(h_b, h_a)
    scale_b = h / h_b
    scale_a = h / h_a
    before_r = cv2.resize(before, (int(w_b * scale_b), h))
    after_r = cv2.resize(after, (int(w_a * scale_a), h))
    # 8-pixel divider strip
    div = np.full((h, 8, 3), 64, dtype=np.uint8)
    return np.hstack([before_r, div, after_r])


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = json.loads(args.counts_json.read_text(encoding="utf-8"))

    print(f"{'image':<60} {'before':>8} {'after':>8} {'delta':>8}")
    print("-" * 86)
    total_b = total_a = 0
    improved_count = 0

    for src_path in sorted(SRC.iterdir()):
        if src_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        # Ultralytics writes overlays with .jpg suffix regardless of source ext.
        before_path = args.before_dir / f"{src_path.stem}.jpg"
        after_path = args.after_dir / f"{src_path.stem}.jpg"
        if not before_path.exists() or not after_path.exists():
            print(f"  SKIP {src_path.name}: missing overlay")
            continue

        before = cv2.imread(str(before_path))
        after = cv2.imread(str(after_path))
        side = stack_side_by_side(before, after)
        out_path = args.out_dir / f"{src_path.stem}_compare.jpg"
        cv2.imwrite(str(out_path), side)

        c = counts.get(src_path.name) or counts.get(src_path.stem) or {}
        b = int(c.get("before", -1))
        a = int(c.get("after", -1))
        delta = a - b if b >= 0 and a >= 0 else 0
        total_b += max(b, 0)
        total_a += max(a, 0)
        if b > a:
            improved_count += 1
        print(f"{src_path.name:<60} {b:>8} {a:>8} {delta:>+8}")

    print("-" * 86)
    print(f"{'TOTAL':<60} {total_b:>8} {total_a:>8} {total_a - total_b:>+8}")
    print(
        f"\nimages with fewer detections after: {improved_count} / "
        f"{sum(1 for p in SRC.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})}"
    )
    print(f"side-by-side strips in: {args.out_dir}")


if __name__ == "__main__":
    main()
