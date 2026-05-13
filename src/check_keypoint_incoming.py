"""Validate an Android-plugin keypoint batch against the on-disk contract.

The contract lives in `docs/KEYPOINT_DATASET_FORMAT.md`. This validator
treats that document as authoritative — if the document and the code
disagree, fix the code.

Exit codes:
  0  — no errors (warnings allowed)
  1  — at least one ERROR
  2  — source root missing or doesn't have the required subdirectories

Usage:
    python src/check_keypoint_incoming.py --source-root data/incoming/android_plugin
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REQUIRED_POINT_KEYS = frozenset({"a", "b", "c_disc_bottom"})
BBOX_POINT_TOLERANCE_PX = 5.0
MAX_PROBLEMS_TO_PRINT = 20


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate a plugin keypoint batch")
    p.add_argument("--source-root", required=True, type=Path)
    return p.parse_args(argv)


def _is_number(v: object) -> bool:
    """True for ints/floats, but not bool (bool is a subclass of int)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_pair_of_numbers(v: object) -> bool:
    return isinstance(v, list) and len(v) == 2 and all(_is_number(x) for x in v)


def _is_bbox_xyxy(v: object) -> bool:
    return isinstance(v, list) and len(v) == 4 and all(_is_number(x) for x in v)


def _list_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _list_annotation_stems(annos_dir: Path) -> set[str]:
    return {p.stem for p in annos_dir.glob("*.json")}


def _validate_wheel(
    wheel: object,
    idx: int,
    image_w: int,
    image_h: int,
    src: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Append errors/warnings for one wheel. `src` is the json path-as-string."""
    prefix = f"{src} wheel[{idx}]"

    if not isinstance(wheel, dict):
        errors.append(f"{prefix}: not a dict")
        return

    bbox = wheel.get("bbox_xyxy")
    if not _is_bbox_xyxy(bbox):
        errors.append(f"{prefix}.bbox_xyxy: missing or not a list of 4 numbers")
        # No point continuing this wheel; bbox is load-bearing for in-bbox check.
        return
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if not (x1 < x2 and y1 < y2):
        errors.append(f"{prefix}.bbox_xyxy: requires x1<x2 and y1<y2, got {bbox}")
        return

    # Bbox outside the image is a warning, not an error.
    if x1 < 0 or y1 < 0 or x2 >= image_w or y2 >= image_h:
        warnings.append(
            f"{prefix}.bbox_xyxy: outside image [0,{image_w})x[0,{image_h}): {bbox}"
        )

    points = wheel.get("points")
    if not isinstance(points, dict):
        errors.append(f"{prefix}.points: missing or not a dict")
        return

    actual_keys = set(points.keys())
    missing_keys = REQUIRED_POINT_KEYS - actual_keys
    extra_keys = actual_keys - REQUIRED_POINT_KEYS
    if missing_keys:
        errors.append(f"{prefix}.points: missing key(s) {sorted(missing_keys)}")
    if extra_keys:
        errors.append(f"{prefix}.points: unexpected key(s) {sorted(extra_keys)}")

    for key in REQUIRED_POINT_KEYS:
        if key not in points:
            continue
        pt = points[key]
        if not _is_pair_of_numbers(pt):
            errors.append(
                f"{prefix}.points.{key}: must be a list of 2 numbers, got {pt!r}"
            )
            continue
        px, py = float(pt[0]), float(pt[1])
        if not (0 <= px <= image_w and 0 <= py <= image_h):
            errors.append(
                f"{prefix}.points.{key}: ({px}, {py}) outside image "
                f"[0,{image_w}]x[0,{image_h}]"
            )
            continue
        if not (
            x1 - BBOX_POINT_TOLERANCE_PX <= px <= x2 + BBOX_POINT_TOLERANCE_PX
            and y1 - BBOX_POINT_TOLERANCE_PX <= py <= y2 + BBOX_POINT_TOLERANCE_PX
        ):
            warnings.append(
                f"{prefix}.points.{key}: ({px}, {py}) outside bbox "
                f"[{x1},{y1},{x2},{y2}] with {BBOX_POINT_TOLERANCE_PX}px slack"
            )


def _validate_one(
    image_path: Path,
    anno_path: Path,
    errors: list[str],
    warnings: list[str],
) -> int:
    """Validate a single (image, annotation) pair. Returns number of wheels seen.

    Pulls image dimensions via cv2.imread to check the in-bounds rule. An
    unreadable image is an ERROR — the validator can't honour the contract's
    "coordinates lie in image bounds" rule without the dimensions.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        errors.append(f"{image_path}: unreadable image")
        return 0
    image_h, image_w = img.shape[:2]

    try:
        payload = json.loads(anno_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        errors.append(f"{anno_path}: invalid JSON ({e})")
        return 0
    except OSError as e:
        errors.append(f"{anno_path}: cannot read ({e})")
        return 0

    if not isinstance(payload, dict):
        errors.append(f"{anno_path}: top-level JSON must be an object")
        return 0

    src = str(anno_path)

    frame_id = payload.get("frame_id")
    if not isinstance(frame_id, str):
        errors.append(f"{src}.frame_id: missing or not a string")
    elif frame_id != image_path.stem:
        errors.append(f"{src}.frame_id: expected {image_path.stem!r}, got {frame_id!r}")

    image_field = payload.get("image")
    if not isinstance(image_field, str):
        errors.append(f"{src}.image: missing or not a string")
    elif image_field != image_path.name:
        errors.append(f"{src}.image: expected {image_path.name!r}, got {image_field!r}")

    wheels = payload.get("wheels")
    if not isinstance(wheels, list):
        errors.append(f"{src}.wheels: missing or not a list")
        return 0

    for i, wheel in enumerate(wheels):
        _validate_wheel(wheel, i, image_w, image_h, src, errors, warnings)
    return len(wheels)


def _print_problems(label: str, problems: Iterable[str]) -> None:
    """Print up to MAX_PROBLEMS_TO_PRINT entries prefixed with the label."""
    items = list(problems)
    for line in items[:MAX_PROBLEMS_TO_PRINT]:
        print(f"{label}: {line}")
    overflow = len(items) - MAX_PROBLEMS_TO_PRINT
    if overflow > 0:
        print(f"{label}: ... and {overflow} more")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_root: Path = args.source_root

    if not source_root.is_dir():
        print(f"ERROR: source root does not exist or is not a directory: {source_root}")
        return 2

    images_dir = source_root / "images"
    annos_dir = source_root / "annotations"
    if not images_dir.is_dir():
        print(f"ERROR: missing required directory: {images_dir}")
        return 2
    if not annos_dir.is_dir():
        print(f"ERROR: missing required directory: {annos_dir}")
        return 2

    images = _list_images(images_dir)
    anno_stems = _list_annotation_stems(annos_dir)
    image_stems = {p.stem for p in images}

    errors: list[str] = []
    warnings: list[str] = []
    total_wheels = 0

    for image_path in images:
        anno_path = annos_dir / f"{image_path.stem}.json"
        if not anno_path.exists():
            errors.append(f"{image_path}: missing annotation {anno_path.name}")
            continue
        total_wheels += _validate_one(image_path, anno_path, errors, warnings)

    # Orphan annotations (json with no matching image) are a warning per spec.
    for stem in sorted(anno_stems - image_stems):
        warnings.append(f"{annos_dir / (stem + '.json')}: no matching image")

    print(f"Source root:     {source_root}")
    print(f"Images:          {len(images)}")
    print(f"Annotations:     {len(anno_stems)}")
    print(f"Wheels (total):  {total_wheels}")
    print(f"Errors:          {len(errors)}")
    print(f"Warnings:        {len(warnings)}")

    if errors:
        _print_problems("ERROR", errors)
    if warnings:
        _print_problems("WARNING", warnings)

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
