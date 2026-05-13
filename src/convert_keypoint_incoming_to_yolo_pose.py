"""Convert an Android-plugin batch into a YOLO-pose dataset.

Input layout (the contract emitted by the plugin, see
docs/KEYPOINT_DATASET_FORMAT.md):

    data/incoming/android_plugin/
      images/<stem>.jpg | .jpeg | .png | .bmp | .webp
      annotations/<stem>.json
      metadata/                       # free-form, not read here

Each annotation:

    {
      "frame_id": "frame_0001",
      "image": "frame_0001.jpg",
      "wheels": [
        {
          "bbox_xyxy": [x1, y1, x2, y2],
          "points": {
            "a":             [x, y],
            "b":             [x, y],
            "c_disc_bottom": [x, y]
          }
        }
      ]
    }

Output layout (consumed by configs/pose_dataset.yaml):

    data/wheel_pose_dataset/
      images/{train,val}/<source_name>__<stem>.<ext>
      labels/{train,val}/<source_name>__<stem>.txt
      metadata/split_manifest.json
      metadata/conversion_report.json

YOLO-pose label format, one line per wheel, normalized to [0, 1] except v:

    <class_id> <cx> <cy> <w> <h> <a_x> <a_y> <a_v> <b_x> <b_y> <b_v> <c_x> <c_y> <c_v>

Plugin annotations carry no occlusion flag — every kept keypoint is emitted
with `v=2` (visible). Wheels that fail validation (bbox order, image-bounds,
missing point) are dropped with a warning logged into the report; the image
is still kept in the split (with an empty/partial label file) so downstream
consumers can see what was skipped.

Usage:
    python src/convert_keypoint_incoming_to_yolo_pose.py \\
        --source-root data/incoming/android_plugin \\
        --dataset-root data/wheel_pose_dataset \\
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val")
WHEEL_CLASS_ID = 0
N_KEYPOINTS = 3
KEYPOINT_KEYS: tuple[str, str, str] = ("a", "b", "c_disc_bottom")
KEYPOINT_VISIBILITY = 2  # plugin spec: every emitted point is visible

DROP_REASON_KEYS: tuple[str, ...] = (
    "invalid_json",
    "missing_annotation",
    "unreadable_image",
    "wheels_not_list",
)

# Defaults for the per-batch quality gate. These were chosen for a *real*
# plugin batch:
#   - 5% skipped images (decode failures, missing annotations, malformed JSON)
#     is plausible noise; more than that and somebody pushed a broken batch.
#   - 10% warnings ratio (per source image) catches batches where many wheels
#     have invalid bboxes / out-of-image keypoints / missing point keys.
# Tunable via --max-skip-ratio / --max-warning-ratio on the CLI.
DEFAULT_MAX_SKIP_RATIO = 0.05
DEFAULT_MAX_WARNING_RATIO = 0.10


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Android-plugin annotations to YOLO-pose format"
    )
    p.add_argument(
        "--source-root",
        required=True,
        type=Path,
        help="Path to data/incoming/<source_name>/",
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/wheel_pose_dataset"),
        help="Where to write the canonical YOLO-pose dataset.",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of images sent to the val split (0..1).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, delete an existing non-empty dataset root before writing.",
    )
    p.add_argument(
        "--source-name",
        default=None,
        help="Override the source slug used in output filenames. "
        "Defaults to the source-root directory name.",
    )
    p.add_argument(
        "--max-skip-ratio",
        type=float,
        default=DEFAULT_MAX_SKIP_RATIO,
        help=(
            "Quality gate: fraction of source images allowed to be skipped "
            "(missing annotation, invalid JSON, unreadable image, malformed "
            "'wheels'). Above this, the gate fails. Default: "
            f"{DEFAULT_MAX_SKIP_RATIO}."
        ),
    )
    p.add_argument(
        "--max-warning-ratio",
        type=float,
        default=DEFAULT_MAX_WARNING_RATIO,
        help=(
            "Quality gate: per-source-image fraction of warnings allowed "
            "(invalid bbox order, out-of-image points, missing point keys, "
            "etc.). Above this, the gate fails. Default: "
            f"{DEFAULT_MAX_WARNING_RATIO}."
        ),
    )
    p.add_argument(
        "--fail-on-quality-gate",
        action="store_true",
        help=(
            "If set, the converter exits with code 1 when the quality gate "
            "is not passed. The conversion report (including the "
            "quality_gate section) is still written. Without the flag the "
            "converter only prints a WARNING and exits 0."
        ),
    )
    return p.parse_args(argv)


def evaluate_quality_gate(
    source_images: int,
    skipped_images: int,
    warnings_count: int,
    max_skip_ratio: float,
    max_warning_ratio: float,
) -> dict:
    """Compute the quality-gate decision + the structured report block.

    Pulled out so the tests can exercise the gate directly without a full
    converter run. The shape matches what lands under `quality_gate` in
    `conversion_report.json`.

    Ratios are taken over `source_images`. When the batch has 0 source
    images (degenerate — converter would have errored before this point)
    ratios are reported as 0.0 and the gate trivially passes.
    """
    if source_images <= 0:
        skipped_ratio = 0.0
        warnings_ratio = 0.0
    else:
        skipped_ratio = skipped_images / source_images
        warnings_ratio = warnings_count / source_images

    reasons: list[str] = []
    if skipped_ratio > max_skip_ratio:
        reasons.append(
            f"skipped_ratio={skipped_ratio:.4f} > max_skip_ratio={max_skip_ratio:.4f} "
            f"({skipped_images}/{source_images} images skipped)"
        )
    if warnings_ratio > max_warning_ratio:
        reasons.append(
            f"warnings_ratio={warnings_ratio:.4f} > "
            f"max_warning_ratio={max_warning_ratio:.4f} "
            f"({warnings_count} warnings over {source_images} source images)"
        )

    return {
        "skipped_ratio": skipped_ratio,
        "warnings_ratio": warnings_ratio,
        "quality_gate": {
            "max_skip_ratio": max_skip_ratio,
            "max_warning_ratio": max_warning_ratio,
            "passed": len(reasons) == 0,
            "reasons": reasons,
        },
    }


def list_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def read_image_size(path: Path) -> tuple[int, int] | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    return w, h


def assign_splits(
    images: list[Path],
    val_ratio: float,
    seed: int,
) -> dict[Path, str]:
    rng = random.Random(seed)
    shuffled = images[:]
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_ratio))
    val_set = set(shuffled[:n_val])
    return {p: ("val" if p in val_set else "train") for p in images}


def validate_and_convert_bbox(
    bbox_xyxy: object,
    img_w: int,
    img_h: int,
    image_name: str,
) -> tuple[tuple[float, float, float, float] | None, str | None]:
    """Validate one bbox; return (yolo_norm | None, warning | None).

    Returned yolo_norm = (cx, cy, w, h) normalized to [0, 1].
    Boxes partially outside the image are clipped with a warning. Fully
    out-of-image boxes are dropped.
    """
    if not isinstance(bbox_xyxy, list) or len(bbox_xyxy) != 4:
        return (
            None,
            f"{image_name}: bbox_xyxy must be a list of 4 numbers, got {bbox_xyxy!r}",
        )
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    except (TypeError, ValueError):
        return (
            None,
            f"{image_name}: bbox_xyxy contains non-numeric values: {bbox_xyxy!r}",
        )
    if x2 <= x1 or y2 <= y1:
        return (
            None,
            f"{image_name}: invalid bbox order (x2<=x1 or y2<=y1): {bbox_xyxy!r}",
        )

    warning: str | None = None
    cx1, cy1 = max(x1, 0.0), max(y1, 0.0)
    cx2, cy2 = min(x2, float(img_w)), min(y2, float(img_h))
    if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
        warning = (
            f"{image_name}: bbox {bbox_xyxy} outside image {img_w}x{img_h}, "
            f"clipped to [{cx1}, {cy1}, {cx2}, {cy2}]"
        )
    if cx2 <= cx1 or cy2 <= cy1:
        return (
            None,
            f"{image_name}: bbox fully outside image after clipping: {bbox_xyxy!r}",
        )

    cx = ((cx1 + cx2) / 2.0) / img_w
    cy = ((cy1 + cy2) / 2.0) / img_h
    w = (cx2 - cx1) / img_w
    h = (cy2 - cy1) / img_h
    return (cx, cy, w, h), warning


def validate_and_convert_points(
    points: object,
    img_w: int,
    img_h: int,
    image_name: str,
) -> tuple[list[tuple[float, float, int]] | None, list[str]]:
    """Validate exactly {a, b, c_disc_bottom}. Return (a, b, c) normalized or None.

    Points outside the image are clipped to image bounds with a warning. The
    plugin contract does not carry occlusion — every kept point is emitted
    with v=2.
    """
    warnings: list[str] = []
    if not isinstance(points, dict):
        return None, [
            f"{image_name}: points must be a dict with keys {list(KEYPOINT_KEYS)}, "
            f"got {type(points).__name__}"
        ]
    missing = [k for k in KEYPOINT_KEYS if k not in points]
    if missing:
        return None, [f"{image_name}: points missing keys: {missing}"]
    extra = [k for k in points if k not in KEYPOINT_KEYS]
    if extra:
        warnings.append(f"{image_name}: points has unexpected keys (ignored): {extra}")

    out: list[tuple[float, float, int]] = []
    for key in KEYPOINT_KEYS:
        xy = points[key]
        if not isinstance(xy, list) or len(xy) != 2:
            return None, [
                *warnings,
                f"{image_name}: points.{key} must be a list of 2 numbers, got {xy!r}",
            ]
        try:
            x, y = float(xy[0]), float(xy[1])
        except (TypeError, ValueError):
            return None, [
                *warnings,
                f"{image_name}: points.{key} contains non-numeric values: {xy!r}",
            ]
        cx, cy = min(max(x, 0.0), float(img_w)), min(max(y, 0.0), float(img_h))
        if (cx, cy) != (x, y):
            warnings.append(
                f"{image_name}: point {key} {xy} outside image {img_w}x{img_h}, "
                f"clipped to [{cx}, {cy}]"
            )
        out.append((cx / img_w, cy / img_h, KEYPOINT_VISIBILITY))
    return out, warnings


def ensure_output_dirs(root: Path) -> None:
    for split in SPLITS:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)


def format_label_line(
    cls_id: int,
    bbox: tuple[float, float, float, float],
    keypoints: list[tuple[float, float, int]],
) -> str:
    cx, cy, w, h = bbox
    parts = [str(cls_id), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
    for kx, ky, v in keypoints:
        parts.extend([f"{kx:.6f}", f"{ky:.6f}", str(v)])
    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_root: Path = args.source_root
    dataset_root: Path = args.dataset_root

    if not 0.0 <= args.val_ratio <= 1.0:
        print(f"ERROR: --val-ratio must be in [0, 1], got {args.val_ratio}")
        return 2

    images_in = source_root / "images"
    annos_in = source_root / "annotations"
    if not images_in.is_dir():
        print(f"ERROR: missing directory {images_in}")
        return 2
    if not annos_in.is_dir():
        print(f"ERROR: missing directory {annos_in}")
        return 2

    if dataset_root.exists() and any(dataset_root.iterdir()):
        if not args.overwrite:
            print(
                f"ERROR: dataset root already exists and is not empty: {dataset_root}"
            )
            print(
                "Pass --overwrite to delete and regenerate, "
                "or pick a different --dataset-root."
            )
            return 1
        shutil.rmtree(dataset_root)

    ensure_output_dirs(dataset_root)
    source_name = args.source_name or source_root.name

    images = list_images(images_in)
    if not images:
        print(f"ERROR: no images found in {images_in}")
        return 1

    split_assignment = assign_splits(images, args.val_ratio, args.seed)

    warnings: list[str] = []
    skipped: list[dict] = []
    split_manifest: dict[str, list[str]] = {"train": [], "val": []}
    converted = 0
    wheels_total = 0

    for img_path in images:
        size = read_image_size(img_path)
        if size is None:
            skipped.append(
                {
                    "image": str(img_path),
                    "reason": "unreadable image",
                    "reason_key": "unreadable_image",
                }
            )
            continue
        img_w, img_h = size

        anno_path = annos_in / f"{img_path.stem}.json"
        if not anno_path.exists():
            skipped.append(
                {
                    "image": str(img_path),
                    "reason": "missing annotation JSON",
                    "reason_key": "missing_annotation",
                }
            )
            continue

        try:
            anno = json.loads(anno_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            skipped.append(
                {
                    "image": str(img_path),
                    "reason": f"invalid JSON: {e}",
                    "reason_key": "invalid_json",
                }
            )
            continue

        wheels = anno.get("wheels", [])
        if not isinstance(wheels, list):
            skipped.append(
                {
                    "image": str(img_path),
                    "reason": "'wheels' is not a list",
                    "reason_key": "wheels_not_list",
                }
            )
            continue

        label_lines: list[str] = []
        for wheel in wheels:
            if not isinstance(wheel, dict):
                warnings.append(
                    f"{img_path.name}: wheel entry is not a dict (got "
                    f"{type(wheel).__name__}), dropped"
                )
                continue
            yolo_bbox, bbox_warn = validate_and_convert_bbox(
                wheel.get("bbox_xyxy"), img_w, img_h, img_path.name
            )
            if bbox_warn:
                warnings.append(bbox_warn)
            if yolo_bbox is None:
                continue

            yolo_kps, kp_warns = validate_and_convert_points(
                wheel.get("points"), img_w, img_h, img_path.name
            )
            warnings.extend(kp_warns)
            if yolo_kps is None:
                warnings.append(f"{img_path.name}: dropping wheel — invalid points")
                continue

            label_lines.append(format_label_line(WHEEL_CLASS_ID, yolo_bbox, yolo_kps))
            wheels_total += 1

        split = split_assignment[img_path]
        out_stem = f"{source_name}__{img_path.stem}"
        out_image = (
            dataset_root / "images" / split / f"{out_stem}{img_path.suffix.lower()}"
        )
        out_label = dataset_root / "labels" / split / f"{out_stem}.txt"

        shutil.copy2(img_path, out_image)
        out_label.write_text(
            ("\n".join(label_lines) + "\n") if label_lines else "",
            encoding="utf-8",
        )

        split_manifest[split].append(out_image.name)
        converted += 1

    drop_reasons: dict[str, int] = {k: 0 for k in DROP_REASON_KEYS}
    for entry in skipped:
        key = entry.get("reason_key")
        assert key in drop_reasons, (
            f"Unknown reason_key {key!r} in skipped entry. Add it to DROP_REASON_KEYS."
        )
        drop_reasons[key] += 1

    manifest_path = dataset_root / "metadata" / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_root": str(source_root),
                "source_name": source_name,
                "val_ratio": args.val_ratio,
                "seed": args.seed,
                "keypoint_keys": list(KEYPOINT_KEYS),
                "files": split_manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    gate_block = evaluate_quality_gate(
        source_images=len(images),
        skipped_images=len(skipped),
        warnings_count=len(warnings),
        max_skip_ratio=args.max_skip_ratio,
        max_warning_ratio=args.max_warning_ratio,
    )

    report = {
        "source_root": str(source_root),
        "source_name": source_name,
        # Goal §2 field names. Legacy aliases (`converted`, `skipped`) kept
        # below for backward compat with downstream consumers + existing
        # tests that read those keys.
        "source_images": len(images),
        "converted_images": converted,
        "skipped_images": len(skipped),
        "skipped_ratio": gate_block["skipped_ratio"],
        "warnings_count": len(warnings),
        "warnings_ratio": gate_block["warnings_ratio"],
        "quality_gate": gate_block["quality_gate"],
        # Legacy aliases — do not remove without coordinating with consumers.
        "converted": converted,
        "skipped": len(skipped),
        "train": len(split_manifest["train"]),
        "val": len(split_manifest["val"]),
        "wheels": wheels_total,
        "skipped_details": skipped,
        "drop_reasons": drop_reasons,
        "warnings": warnings,
    }
    report_path = dataset_root / "metadata" / "conversion_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(f"Source root:       {source_root}")
    print(f"Source name slug:  {source_name}")
    print(f"Source images:     {len(images)}")
    print(f"Converted:         {converted}")
    print(f"  train:           {len(split_manifest['train'])}")
    print(f"  val:             {len(split_manifest['val'])}")
    print(f"Wheels (lines):    {wheels_total}")
    print(f"Skipped:           {len(skipped)}")
    if DROP_REASON_KEYS:
        print("Drop reasons:")
        label_width = max(len(k) for k in DROP_REASON_KEYS) + 1
        for key in DROP_REASON_KEYS:
            print(f"  {(key + ':').ljust(label_width)} {drop_reasons[key]}")
    if warnings:
        print(
            f"Warnings:          {len(warnings)} (see metadata/conversion_report.json)"
        )
        for w in warnings[:5]:
            print(f"  - {w}")
        if len(warnings) > 5:
            print(f"  ... and {len(warnings) - 5} more")

    qg = report["quality_gate"]
    print()
    print("Quality gate:")
    print(
        f"  skipped_ratio:    {report['skipped_ratio']:.4f} "
        f"(max {qg['max_skip_ratio']:.4f})"
    )
    print(
        f"  warnings_ratio:   {report['warnings_ratio']:.4f} "
        f"(max {qg['max_warning_ratio']:.4f})"
    )
    print(f"  passed:           {qg['passed']}")
    if not qg["passed"]:
        for reason in qg["reasons"]:
            print(f"    - {reason}")

    print()
    print(f"Metadata:          {manifest_path}")
    print(f"                   {report_path}")

    if converted == 0:
        print(
            "\nERROR: nothing converted — check inputs and skipped_details in the report."
        )
        return 1

    if not qg["passed"]:
        if args.fail_on_quality_gate:
            print(
                "\nERROR: quality gate not passed. "
                "Conversion report saved; failing because "
                "--fail-on-quality-gate was set."
            )
            return 1
        print(
            "\nWARNING: quality gate not passed. Continuing because "
            "--fail-on-quality-gate was not set; re-run with the flag to "
            "make this fatal."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
