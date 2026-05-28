"""Filter UE geometry-derived incoming annotations before training.

The raw UE geometry exporter intentionally keeps draft projected
wheel/tire/rim bounds. This filter creates a stricter incoming batch by
dropping suspicious wheels/frames and writing a structured QA report.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-nonblack-frac", type=float, default=0.02)
    parser.add_argument("--min-bbox-side-frac", type=float, default=0.015)
    parser.add_argument("--max-bbox-area-frac", type=float, default=0.08)
    parser.add_argument("--max-bbox-width-frac", type=float, default=0.55)
    parser.add_argument("--max-bbox-height-frac", type=float, default=0.50)
    parser.add_argument("--max-wheels-per-frame", type=int, default=6)
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _image_stats(path: Path) -> tuple[int, int, float] | None:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    h, w = image.shape[:2]
    rgb = image[:, :, :3] if getattr(image, "ndim", 0) == 3 else image
    if getattr(rgb, "ndim", 0) == 3:
        lum = rgb.mean(axis=2)
    else:
        lum = rgb
    nonblack_frac = float((lum > 10).sum()) / float(max(w * h, 1))
    return w, h, nonblack_frac


def _find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        path = images_dir / f"{stem}{ext}"
        if path.is_file():
            return path
    return None


def wheel_drop_reason(
    wheel: object,
    *,
    img_w: int,
    img_h: int,
    min_side_frac: float,
    max_area_frac: float,
    max_width_frac: float,
    max_height_frac: float,
) -> str | None:
    if not isinstance(wheel, dict):
        return "wheel_not_object"
    bbox = wheel.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return "bad_bbox"
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return "bad_bbox"
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return "bad_bbox"
    width_frac = width / float(max(img_w, 1))
    height_frac = height / float(max(img_h, 1))
    area_frac = width_frac * height_frac
    if width_frac < min_side_frac or height_frac < min_side_frac:
        return "bbox_too_small"
    if area_frac > max_area_frac:
        return "bbox_area_too_large"
    if width_frac > max_width_frac:
        return "bbox_width_too_large"
    if height_frac > max_height_frac:
        return "bbox_height_too_large"
    points = wheel.get("points")
    if not isinstance(points, dict):
        return "missing_points"
    for key in ("a", "b", "c_disc_bottom"):
        value = points.get(key)
        if not isinstance(value, list) or len(value) != 2:
            return f"missing_point_{key}"
    return None


def filter_batch(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root
    output_root = args.output_root
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    for sub in ("images", "annotations", "metadata"):
        (output_root / sub).mkdir(parents=True, exist_ok=True)

    source_images = source_root / "images"
    source_annotations = source_root / "annotations"
    report: dict[str, Any] = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "source_frames": 0,
        "kept_frames": 0,
        "source_wheels": 0,
        "kept_wheels": 0,
        "dropped_frames": Counter(),
        "dropped_wheels": Counter(),
        "kept": [],
        "dropped": [],
        "thresholds": {
            "min_nonblack_frac": args.min_nonblack_frac,
            "min_bbox_side_frac": args.min_bbox_side_frac,
            "max_bbox_area_frac": args.max_bbox_area_frac,
            "max_bbox_width_frac": args.max_bbox_width_frac,
            "max_bbox_height_frac": args.max_bbox_height_frac,
            "max_wheels_per_frame": args.max_wheels_per_frame,
        },
    }

    for ann_path in sorted(source_annotations.glob("*.json")):
        report["source_frames"] += 1
        frame_id = ann_path.stem
        image_path = _find_image(source_images, frame_id)
        if image_path is None:
            report["dropped_frames"]["missing_image"] += 1
            report["dropped"].append({"frame_id": frame_id, "reason": "missing_image"})
            continue
        stats = _image_stats(image_path)
        if stats is None:
            report["dropped_frames"]["unreadable_image"] += 1
            report["dropped"].append({"frame_id": frame_id, "reason": "unreadable_image"})
            continue
        img_w, img_h, nonblack_frac = stats
        if nonblack_frac < args.min_nonblack_frac:
            report["dropped_frames"]["black_or_empty_image"] += 1
            report["dropped"].append(
                {
                    "frame_id": frame_id,
                    "reason": "black_or_empty_image",
                    "nonblack_frac": nonblack_frac,
                }
            )
            continue
        payload = _load_json(ann_path)
        if payload is None:
            report["dropped_frames"]["invalid_json"] += 1
            report["dropped"].append({"frame_id": frame_id, "reason": "invalid_json"})
            continue
        wheels = payload.get("wheels")
        if not isinstance(wheels, list):
            report["dropped_frames"]["wheels_not_list"] += 1
            report["dropped"].append({"frame_id": frame_id, "reason": "wheels_not_list"})
            continue
        report["source_wheels"] += len(wheels)
        kept_wheels = []
        for wheel in wheels:
            reason = wheel_drop_reason(
                wheel,
                img_w=img_w,
                img_h=img_h,
                min_side_frac=args.min_bbox_side_frac,
                max_area_frac=args.max_bbox_area_frac,
                max_width_frac=args.max_bbox_width_frac,
                max_height_frac=args.max_bbox_height_frac,
            )
            if reason is None:
                kept_wheels.append(wheel)
            else:
                report["dropped_wheels"][reason] += 1
        if not kept_wheels:
            report["dropped_frames"]["no_wheels_after_filter"] += 1
            report["dropped"].append({"frame_id": frame_id, "reason": "no_wheels_after_filter"})
            continue
        if len(kept_wheels) > args.max_wheels_per_frame:
            report["dropped_frames"]["too_many_wheels_after_filter"] += 1
            report["dropped"].append(
                {
                    "frame_id": frame_id,
                    "reason": "too_many_wheels_after_filter",
                    "wheels": len(kept_wheels),
                }
            )
            continue
        out_image = output_root / "images" / image_path.name
        out_annotation = output_root / "annotations" / ann_path.name
        shutil.copy2(image_path, out_image)
        filtered_payload = dict(payload)
        filtered_payload["wheels"] = kept_wheels
        filtered_payload["_qa_filter"] = {
            "source_root": str(source_root),
            "source_wheels": len(wheels),
            "kept_wheels": len(kept_wheels),
            "nonblack_frac": nonblack_frac,
        }
        out_annotation.write_text(
            json.dumps(filtered_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        report["kept_frames"] += 1
        report["kept_wheels"] += len(kept_wheels)
        report["kept"].append(
            {
                "frame_id": frame_id,
                "wheels": len(kept_wheels),
                "source_wheels": len(wheels),
            }
        )

    for key in ("dropped_frames", "dropped_wheels"):
        report[key] = dict(report[key])
    (output_root / "metadata" / "qa_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = filter_batch(args)
    print(f"Source frames: {report['source_frames']}")
    print(f"Kept frames:   {report['kept_frames']}")
    print(f"Source wheels: {report['source_wheels']}")
    print(f"Kept wheels:   {report['kept_wheels']}")
    print(f"Dropped frames: {report['dropped_frames']}")
    print(f"Dropped wheels: {report['dropped_wheels']}")
    print(f"Report: {args.output_root / 'metadata/qa_report.json'}")
    return 0 if report["kept_frames"] > 0 and report["kept_wheels"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
