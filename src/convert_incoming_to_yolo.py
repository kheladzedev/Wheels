"""Convert an incoming batch into the canonical YOLO-pose dataset.

Input layout (one batch per source):
    data/incoming/<source_name>/
      images/        # *.jpg | *.jpeg | *.png | *.bmp | *.webp
      annotations/   # <image_stem>.json — see docs/ANNOTATION_JSON_FORMAT.md
      metadata/      # optional, free-form; not read by this script

Output layout (consumed by configs/dataset.yaml):
    data/wheel_dataset/
      images/{train,val}/<source_name>__<image_stem>.<ext>
      labels/{train,val}/<source_name>__<image_stem>.txt
      metadata/split_manifest.json
      metadata/conversion_report.json

YOLO-pose label format (one line per wheel, normalized to [0,1] except v):
    <class_id> <cx> <cy> <w> <h> <kp0_x> <kp0_y> <v0> <kp1_x> <kp1_y> <v1> <kp2_x> <kp2_y> <v2>

The train/val split is currently a random per-image split. For production
this is wrong — frames of the same car/scene must not be in both splits.
See docs/REAL_DATA_INGESTION.md §6.

Usage:
    python src/convert_incoming_to_yolo.py \\
        --source-root data/incoming/manual_sample \\
        --dataset-root data/wheel_dataset \\
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAME_TO_ID = {"wheel": 0}
SPLITS = ("train", "val")
N_KEYPOINTS = 3
KEYPOINT_NAMES = ("rim_left", "rim_right", "disc_bottom")

# Machine-readable drop-reason keys. This tuple is authoritative: any reason_key
# used in a skipped.append() call MUST appear here, or main() will assert and
# refuse to write the report. Order is also the print order in main()'s summary.
DROP_REASON_KEYS: tuple[str, ...] = (
    "image_too_small",
    "invalid_json",
    "missing_annotation",
    "objects_not_list",
    "unreadable_image",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert incoming annotations to YOLO-pose format"
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
        default=Path("data/wheel_dataset"),
        help="Where to write the canonical YOLO dataset",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of images sent to the val split (0..1)",
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
        "--min-side",
        type=int,
        default=480,
        help="Drop images whose longer side (max(width, height)) is below this "
        "threshold, per docs/REAL_DATA_INGESTION.md §5. Set to 0 to disable "
        "the filter. Default: 480.",
    )
    p.add_argument(
        "--scene-regex",
        default=None,
        help="Regex applied to image stems to extract a scene/group key. The "
        "first capture group is the key. All images with the same key go to "
        "the same split — required for video frames and multi-shot photo "
        "sessions to avoid train/val leakage. Example for stems like "
        "'scene_001_frame_42': '^(scene_\\d+)_.*$'. "
        "If omitted, falls back to random per-image split (unsafe for "
        "video / multi-shot data — see docs/REAL_DATA_INGESTION.md §6).",
    )
    return p.parse_args()


def assign_splits(
    images: list[Path],
    val_ratio: float,
    seed: int,
    scene_regex: str | None,
) -> tuple[dict[Path, str], dict]:
    """Return (image_path -> split, strategy_info_for_manifest).

    Strategy:
      - If scene_regex is None: random per-image split.
      - Else: extract scene key per image, shuffle scene keys, assign whole
        scenes by val_ratio (computed over scenes, not images).

    Images whose stem doesn't match the regex are treated as their own
    singleton "scene" — safe default that doesn't silently mix them in.
    """
    rng = random.Random(seed)

    if scene_regex is None:
        shuffled = images[:]
        rng.shuffle(shuffled)
        n_val = int(round(len(shuffled) * val_ratio))
        val_set = set(shuffled[:n_val])
        assignment = {p: ("val" if p in val_set else "train") for p in images}
        return assignment, {
            "split_strategy": "random_per_image",
            "split_strategy_note": (
                "Random per-image split is unsafe when consecutive frames or "
                "multiple shots of the same car appear in the batch. Pass "
                "--scene-regex for video frames or multi-shot photo sessions."
            ),
        }

    pattern = re.compile(scene_regex)
    scene_of: dict[Path, str] = {}
    unmatched: list[str] = []
    for p in images:
        m = pattern.match(p.stem)
        if m and m.groups():
            scene_of[p] = m.group(1)
        else:
            scene_of[p] = f"__singleton__{p.stem}"
            unmatched.append(p.stem)

    scenes = sorted({k for k in scene_of.values()})
    rng.shuffle(scenes)
    n_val_scenes = int(round(len(scenes) * val_ratio))
    val_scenes = set(scenes[:n_val_scenes])
    assignment = {p: ("val" if scene_of[p] in val_scenes else "train") for p in images}
    return assignment, {
        "split_strategy": "scene_regex",
        "scene_regex": scene_regex,
        "n_scenes": len(scenes),
        "n_val_scenes": n_val_scenes,
        "unmatched_stems": unmatched,
        "split_strategy_note": (
            f"Images grouped by regex {scene_regex!r}. {len(unmatched)} stem(s) "
            "did not match and were treated as singleton scenes."
        ),
    }


def list_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def read_image_size(path: Path) -> tuple[int, int] | None:
    """Return (width, height) or None if the image can't be decoded."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    return w, h


def validate_and_convert_bbox(
    bbox_xyxy: list[float],
    img_w: int,
    img_h: int,
    image_name: str,
) -> tuple[tuple[float, float, float, float] | None, str | None]:
    """Validate one bbox; return (yolo_norm | None, warning | None).

    Returned yolo_norm = (cx, cy, w, h) normalized to [0, 1].
    Boxes that extend outside the image are clipped and a warning is emitted.
    Returns also the clipped pixel-space corners so keypoint validation can
    refer back to the actual region used.
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


def validate_and_convert_keypoints(
    keypoints: list[dict] | None,
    img_w: int,
    img_h: int,
    image_name: str,
) -> tuple[list[tuple[float, float, int]] | None, list[str]]:
    """Validate exactly N_KEYPOINTS keypoints. Return list of (x_norm, y_norm, v) or None.

    Keypoints with visibility==0 are emitted as (0, 0, 0) per YOLO-pose convention.
    Out-of-image keypoints with visibility>0 are clipped to image bounds with a warning.
    """
    warnings: list[str] = []
    if not isinstance(keypoints, list) or len(keypoints) != N_KEYPOINTS:
        return None, [
            f"{image_name}: keypoints must be a list of {N_KEYPOINTS} entries, "
            f"got {len(keypoints) if isinstance(keypoints, list) else type(keypoints).__name__}"
        ]

    out: list[tuple[float, float, int]] = []
    for idx, kp in enumerate(keypoints):
        name = KEYPOINT_NAMES[idx]
        try:
            vis = int(kp.get("visibility", 0))
        except (TypeError, ValueError, AttributeError):
            return None, [f"{image_name}: keypoint {name} has non-integer visibility"]
        if vis not in (0, 1, 2):
            return None, [
                f"{image_name}: keypoint {name} visibility must be 0/1/2, got {vis}"
            ]

        if vis == 0:
            out.append((0.0, 0.0, 0))
            continue

        xy = kp.get("xy") if isinstance(kp, dict) else None
        if not isinstance(xy, list) or len(xy) != 2:
            return None, [
                f"{image_name}: keypoint {name} xy must be a list of 2 numbers"
            ]
        try:
            x, y = float(xy[0]), float(xy[1])
        except (TypeError, ValueError):
            return None, [
                f"{image_name}: keypoint {name} xy contains non-numeric values"
            ]

        cx, cy = min(max(x, 0.0), float(img_w)), min(max(y, 0.0), float(img_h))
        if (cx, cy) != (x, y):
            warnings.append(
                f"{image_name}: keypoint {name} {xy} outside image {img_w}x{img_h}, "
                f"clipped to [{cx}, {cy}]"
            )
        out.append((cx / img_w, cy / img_h, vis))

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


def main() -> int:
    args = parse_args()
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
                "Pass --overwrite to delete and regenerate, or pick a different --dataset-root."
            )
            return 1
        shutil.rmtree(dataset_root)

    ensure_output_dirs(dataset_root)
    source_name = args.source_name or source_root.name

    images = list_images(images_in)
    if not images:
        print(f"ERROR: no images found in {images_in}")
        return 1

    try:
        split_assignment, split_strategy_info = assign_splits(
            images,
            args.val_ratio,
            args.seed,
            args.scene_regex,
        )
    except re.error as e:
        print(f"ERROR: invalid --scene-regex {args.scene_regex!r}: {e}")
        return 2

    def split_for(p: Path) -> str:
        return split_assignment[p]

    warnings: list[str] = []
    skipped: list[dict] = []
    per_class_counts: dict[str, int] = {name: 0 for name in CLASS_NAME_TO_ID}
    split_manifest: dict[str, list[str]] = {"train": [], "val": []}
    converted = 0

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

        if args.min_side > 0 and max(img_w, img_h) < args.min_side:
            skipped.append(
                {
                    "image": str(img_path),
                    "reason": (
                        f"image too small: max(w,h)={max(img_w, img_h)} "
                        f"< min_side={args.min_side}"
                    ),
                    "reason_key": "image_too_small",
                    "size": [img_w, img_h],
                }
            )
            continue

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

        objects = anno.get("objects", [])
        if not isinstance(objects, list):
            skipped.append(
                {
                    "image": str(img_path),
                    "reason": "'objects' is not a list",
                    "reason_key": "objects_not_list",
                }
            )
            continue

        label_lines: list[str] = []
        for obj in objects:
            class_name = obj.get("class_name")
            if class_name not in CLASS_NAME_TO_ID:
                warnings.append(
                    f"{img_path.name}: unknown class_name {class_name!r}, skipped"
                )
                continue

            yolo_bbox, bbox_warn = validate_and_convert_bbox(
                obj.get("bbox_xyxy"), img_w, img_h, img_path.name
            )
            if bbox_warn:
                warnings.append(bbox_warn)
            if yolo_bbox is None:
                continue

            yolo_kps, kp_warns = validate_and_convert_keypoints(
                obj.get("keypoints"), img_w, img_h, img_path.name
            )
            warnings.extend(kp_warns)
            if yolo_kps is None:
                warnings.append(f"{img_path.name}: dropping wheel — invalid keypoints")
                continue

            cls_id = CLASS_NAME_TO_ID[class_name]
            label_lines.append(format_label_line(cls_id, yolo_bbox, yolo_kps))
            per_class_counts[class_name] += 1

        split = split_for(img_path)
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

    manifest_path = dataset_root / "metadata" / "split_manifest.json"
    manifest_payload = {
        "source_root": str(source_root),
        "source_name": source_name,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "keypoint_names": list(KEYPOINT_NAMES),
        "files": split_manifest,
    }
    manifest_payload.update(split_strategy_info)
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    drop_reasons: dict[str, int] = {k: 0 for k in DROP_REASON_KEYS}
    for entry in skipped:
        key = entry.get("reason_key")
        assert key in drop_reasons, (
            f"Unknown reason_key {key!r} in skipped entry. Add it to DROP_REASON_KEYS."
        )
        drop_reasons[key] += 1

    report = {
        "source_root": str(source_root),
        "source_name": source_name,
        "source_images": len(images),
        "converted": converted,
        "train": len(split_manifest["train"]),
        "val": len(split_manifest["val"]),
        "skipped": len(skipped),
        "skipped_details": skipped,
        "drop_reasons": drop_reasons,
        "object_counts_by_class": per_class_counts,
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
    print(f"Skipped:           {len(skipped)}")
    print("Drop reasons:")
    label_width = max(len(k) for k in DROP_REASON_KEYS) + 1
    for key in DROP_REASON_KEYS:
        print(f"  {(key + ':').ljust(label_width)} {drop_reasons[key]}")
    print(f"Objects by class:  {per_class_counts}")
    if warnings:
        print(
            f"Warnings:          {len(warnings)} (see metadata/conversion_report.json)"
        )
        for w in warnings[:5]:
            print(f"  - {w}")
        if len(warnings) > 5:
            print(f"  ... and {len(warnings) - 5} more")
    print()
    print(f"Metadata:          {manifest_path}")
    print(f"                   {report_path}")

    if converted == 0:
        print(
            "\nERROR: nothing converted — check inputs and skipped_details in the report."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
