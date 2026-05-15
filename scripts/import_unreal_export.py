"""Import a raw Unreal/plugin export into the VSBL incoming-dataset format.

Mapping (per plugin-author confirmation 2026-05-15) is treated as the
source of truth for this batch::

    Right  -> points.a
    Left   -> points.b
    Center -> points.c_disc_bottom

Drop policy::

    - object dropped if all three points are (0, 0)
    - object dropped if any required point is missing
    - object dropped if any required point is outside the image bounds
    - object dropped if the keyPoint .txt cannot be parsed

bbox_xyxy is computed around the three accepted points with a configurable
margin (default ``80 px``) and clipped to image bounds. If clipping
collapses the bbox to zero area, the object is dropped.

Outputs the on-disk contract in ``docs/KEYPOINT_DATASET_FORMAT.md``::

    <out-root>/
      images/<frame_id>.jpg
      annotations/<frame_id>.json
      metadata/source_info.json
      metadata/import_report.json

Ground metadata (``DeltaZ/Roll/Pitch/FOV``) is recorded in
``metadata/import_report.json`` — never injected into the annotation
JSON, since the annotation contract is strict.

Usage::

    python scripts/import_unreal_export.py \\
        --source-root ~/Downloads/0001 \\
        --out-root data/incoming/android_plugin_real \\
        --overwrite
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2

# Reuse the tolerant parser + classifier from the inspector. Adding scripts/
# to sys.path keeps the import working under direct `python scripts/...` runs.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import inspect_unreal_export as ix  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
DEFAULT_MARGIN_PX = 80

DROP_ALL_ZERO = "all_zero"
DROP_OUT_OF_BOUNDS = "out_of_bounds"
DROP_MISSING_POINTS = "missing_points"
DROP_PARSE_ERROR = "parse_error"
DROP_INVALID_BBOX = "invalid_bbox_after_clip"

DROP_REASONS: tuple[str, ...] = (
    DROP_ALL_ZERO,
    DROP_OUT_OF_BOUNDS,
    DROP_MISSING_POINTS,
    DROP_PARSE_ERROR,
    DROP_INVALID_BBOX,
)


@dataclass
class ImportSummary:
    images_found: int = 0
    images_imported: int = 0
    keypoint_object_files_found: int = 0
    valid_wheels: int = 0
    drop_counts: dict[str, int] = field(
        default_factory=lambda: {k: 0 for k in DROP_REASONS}
    )
    ground_meta_parsed: int = 0
    ground_meta: dict[str, dict[str, float]] = field(default_factory=dict)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Import a raw Unreal/plugin export into VSBL incoming-dataset "
            "format (Right->a, Left->b, Center->c_disc_bottom)."
        )
    )
    p.add_argument("--source-root", required=True, type=Path)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--margin", type=int, default=DEFAULT_MARGIN_PX)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def build_bbox_from_points(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    image_w: int,
    image_h: int,
    margin: int,
) -> Optional[tuple[float, float, float, float]]:
    """Return ``(x1, y1, x2, y2)`` covering all 3 points + margin, clipped to image.

    Returns ``None`` if the clipped bbox would collapse (zero area or
    inverted ordering).
    """
    xs = [a[0], b[0], c[0]]
    ys = [a[1], b[1], c[1]]
    x1 = min(xs) - margin
    y1 = min(ys) - margin
    x2 = max(xs) + margin
    y2 = max(ys) + margin

    # `check_keypoint_incoming` treats coordinates in the half-open range
    # [0, image_w). Clip to the last valid pixel index so a bbox sitting
    # exactly on the far edge doesn't produce a spurious warning.
    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(image_w - 1), x2)
    y2 = min(float(image_h - 1), y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _drop(summary: ImportSummary, reason: str) -> None:
    summary.drop_counts[reason] = summary.drop_counts.get(reason, 0) + 1


def _try_build_wheel(
    text: str,
    image_w: int,
    image_h: int,
    margin: int,
    summary: ImportSummary,
) -> Optional[dict]:
    """Parse one keyPoint .txt body and emit the wheel dict if valid.

    Returns ``None`` if the object is dropped — the appropriate counter is
    incremented in ``summary`` before returning.
    """
    pts = ix.parse_keypoint_text(text)

    missing = [n for n in ix.POINT_NAMES if n not in pts]
    if missing:
        _drop(summary, DROP_MISSING_POINTS)
        return None

    zeros = [
        n
        for n in ix.POINT_NAMES
        if abs(pts[n][0]) <= ix.ZERO_EPS and abs(pts[n][1]) <= ix.ZERO_EPS
    ]
    if len(zeros) == len(ix.POINT_NAMES):
        _drop(summary, DROP_ALL_ZERO)
        return None
    if zeros:
        # Per plugin-author confirmation: (0, 0) means invisible -> drop the
        # whole object (we cannot emit only 2 of 3 — the contract forbids it).
        _drop(summary, DROP_OUT_OF_BOUNDS)
        return None

    oob = [
        n
        for n in ix.POINT_NAMES
        if not (0 <= pts[n][0] <= image_w - 1 and 0 <= pts[n][1] <= image_h - 1)
    ]
    if oob:
        _drop(summary, DROP_OUT_OF_BOUNDS)
        return None

    right = pts["Right"]
    left = pts["Left"]
    center = pts["Center"]
    bbox = build_bbox_from_points(right, left, center, image_w, image_h, margin)
    if bbox is None:
        _drop(summary, DROP_INVALID_BBOX)
        return None

    return {
        "bbox_xyxy": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
        "points": {
            "a": [float(right[0]), float(right[1])],
            "b": [float(left[0]), float(left[1])],
            "c_disc_bottom": [float(center[0]), float(center[1])],
        },
    }


def _read_image_size(path: Path) -> Optional[tuple[int, int]]:
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        sz = ix.read_jpeg_size(path)
        if sz is not None:
            return sz
    img = cv2.imread(str(path))
    if img is None:
        return None
    return img.shape[1], img.shape[0]


def _prepare_out_root(out_root: Path, overwrite: bool) -> None:
    if out_root.exists() and any(out_root.iterdir()):
        if not overwrite:
            raise SystemExit(
                f"ERROR: out-root already exists and is not empty: {out_root}\n"
                "Pass --overwrite to delete and regenerate."
            )
        shutil.rmtree(out_root)
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "annotations").mkdir(parents=True, exist_ok=True)
    (out_root / "metadata").mkdir(parents=True, exist_ok=True)


def run(args: argparse.Namespace) -> int:
    src = args.source_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()

    images_dir = src / "Images"
    ground_dir = src / "Ground"
    kp_root = src / "keyPoint"
    if not images_dir.is_dir():
        print(f"ERROR: missing directory {images_dir}", file=sys.stderr)
        return 2

    _prepare_out_root(out_root, args.overwrite)

    summary = ImportSummary()
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    summary.images_found = len(images)

    for img_path in images:
        frame_id = img_path.stem
        size = _read_image_size(img_path)
        if size is None:
            # We can't satisfy the validator without dimensions — skip outright.
            continue
        image_w, image_h = size

        wheels: list[dict] = []
        kp_dir = kp_root / frame_id
        if kp_dir.is_dir():
            for kp_file in sorted(kp_dir.iterdir(), key=lambda p: p.name):
                if kp_file.suffix.lower() != ".txt" or not kp_file.is_file():
                    continue
                summary.keypoint_object_files_found += 1
                try:
                    text = kp_file.read_text(errors="replace")
                except OSError:
                    _drop(summary, DROP_PARSE_ERROR)
                    continue
                wheel = _try_build_wheel(text, image_w, image_h, args.margin, summary)
                if wheel is not None:
                    wheels.append(wheel)
                    summary.valid_wheels += 1

        ground_path = ground_dir / f"{frame_id}.txt"
        if ground_path.is_file():
            try:
                gtxt = ground_path.read_text(errors="replace")
                gmeta = ix.parse_ground_text(gtxt)
                if gmeta is not None:
                    summary.ground_meta_parsed += 1
                    summary.ground_meta[frame_id] = {
                        "delta_z": gmeta.delta_z,
                        "roll": gmeta.roll,
                        "pitch": gmeta.pitch,
                        "fov": gmeta.fov,
                    }
            except OSError:
                pass

        out_image = out_root / "images" / img_path.name
        shutil.copy2(img_path, out_image)

        annotation = {
            "frame_id": frame_id,
            "image": img_path.name,
            "wheels": wheels,
        }
        (out_root / "annotations" / f"{frame_id}.json").write_text(
            json.dumps(annotation, indent=2), encoding="utf-8"
        )
        summary.images_imported += 1

    _write_metadata(out_root, src, args, summary)
    _print_summary(summary)
    return 0


def _write_metadata(
    out_root: Path,
    src: Path,
    args: argparse.Namespace,
    summary: ImportSummary,
) -> None:
    source_info = {
        "source_name": "unreal_0001_trial",
        "source_format": "raw_unreal_plugin_export",
        "source_root": str(src),
        "captured_at": None,
        "imported_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "mapping": {
            "Right": "a",
            "Left": "b",
            "Center": "c_disc_bottom",
        },
        "mapping_basis": "plugin_author_confirmation",
        "bbox_strategy": (
            f"axis-aligned bbox around 3 keypoints + {args.margin}px margin, "
            "clipped to image bounds"
        ),
        "drop_policy": (
            "drop wheel if any required point is (0,0), missing, or outside "
            "image bounds; or if the clipped bbox has zero area"
        ),
        "not_yet_training_approved": True,
        "requires_human_preview": True,
        "image_count": summary.images_imported,
    }
    (out_root / "metadata" / "source_info.json").write_text(
        json.dumps(source_info, indent=2), encoding="utf-8"
    )

    import_report = {
        "source_root": str(src),
        "out_root": str(out_root),
        "imported_at": source_info["imported_at"],
        "margin_px": args.margin,
        "images_found": summary.images_found,
        "images_imported": summary.images_imported,
        "keypoint_object_files_found": summary.keypoint_object_files_found,
        "valid_wheels": summary.valid_wheels,
        "drop_counts": summary.drop_counts,
        "ground_meta_parsed": summary.ground_meta_parsed,
        "ground_meta": summary.ground_meta,
    }
    (out_root / "metadata" / "import_report.json").write_text(
        json.dumps(import_report, indent=2), encoding="utf-8"
    )


def _print_summary(summary: ImportSummary) -> None:
    print(f"Images found:                  {summary.images_found}")
    print(f"Images imported:               {summary.images_imported}")
    print(f"keyPoint files found:          {summary.keypoint_object_files_found}")
    print(f"Valid wheels written:          {summary.valid_wheels}")
    print("Drops:")
    for k in DROP_REASONS:
        print(f"  {k}: {summary.drop_counts.get(k, 0)}")
    print(f"Ground metadata parsed:        {summary.ground_meta_parsed}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
