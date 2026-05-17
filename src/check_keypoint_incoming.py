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
MIN_FLOOR_RAY_REL_Y = 0.80
MIN_DISC_BOTTOM_REL_Y = 0.50
MIN_AB_SEPARATION_RATIO = 0.50
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

    parsed_points: dict[str, tuple[float, float]] = {}
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
        parsed_points[key] = (px, py)
        if not (
            x1 - BBOX_POINT_TOLERANCE_PX <= px <= x2 + BBOX_POINT_TOLERANCE_PX
            and y1 - BBOX_POINT_TOLERANCE_PX <= py <= y2 + BBOX_POINT_TOLERANCE_PX
        ):
            warnings.append(
                f"{prefix}.points.{key}: ({px}, {py}) outside bbox "
                f"[{x1},{y1},{x2},{y2}] with {BBOX_POINT_TOLERANCE_PX}px slack"
            )

    if REQUIRED_POINT_KEYS <= parsed_points.keys():
        _validate_floorray_geometry(prefix, x1, y1, x2, y2, parsed_points, errors)


def _validate_floorray_geometry(
    prefix: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    points: dict[str, tuple[float, float]],
    errors: list[str],
) -> None:
    """Enforce the 2026-05-13 floor-ray A/B annotation semantics."""
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    if bbox_w <= 0 or bbox_h <= 0:
        return

    ax, ay = points["a"]
    bx, by = points["b"]
    _cx, cy = points["c_disc_bottom"]
    rel_y_a = (ay - y1) / bbox_h
    rel_y_b = (by - y1) / bbox_h
    rel_y_c = (cy - y1) / bbox_h
    ab_sep_ratio = abs(bx - ax) / bbox_w

    if ax >= bx:
        errors.append(
            f"{prefix}.points: expected a.x < b.x for left/right floor-ray anchors"
        )
    if rel_y_a < MIN_FLOOR_RAY_REL_Y:
        errors.append(
            f"{prefix}.points.a: floor-ray point too high in bbox "
            f"(rel_y={rel_y_a:.3f}, expected >= {MIN_FLOOR_RAY_REL_Y:.2f})"
        )
    if rel_y_b < MIN_FLOOR_RAY_REL_Y:
        errors.append(
            f"{prefix}.points.b: floor-ray point too high in bbox "
            f"(rel_y={rel_y_b:.3f}, expected >= {MIN_FLOOR_RAY_REL_Y:.2f})"
        )
    if ab_sep_ratio < MIN_AB_SEPARATION_RATIO:
        errors.append(
            f"{prefix}.points: A/B anchors too close "
            f"(separation={ab_sep_ratio:.3f}, expected >= {MIN_AB_SEPARATION_RATIO:.2f})"
        )
    if rel_y_c <= MIN_DISC_BOTTOM_REL_Y:
        errors.append(
            f"{prefix}.points.c_disc_bottom: disc-bottom point too high in bbox "
            f"(rel_y={rel_y_c:.3f}, expected > {MIN_DISC_BOTTOM_REL_Y:.2f})"
        )
    if not cy < min(ay, by):
        errors.append(
            f"{prefix}.points.c_disc_bottom: expected C above A/B in image "
            f"(c.y={cy}, a.y={ay}, b.y={by})"
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
