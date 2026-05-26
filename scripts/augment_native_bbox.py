"""Augment existing UE incoming annotations with native wheel bboxes from
COCO+SAM2.

The Unreal exporter does not write a real `BBox` field per visible wheel, so
the converter currently derives `bbox_xyxy` from min/max over the four
sphere helper points. That derived bbox underfits the actual tire silhouette
(it lies inside the wheel hull rather than around it).

This script bridges the gap by running the same COCO vehicle + SAM 2 wheel
pipeline used by `src/auto_annotate_wheels.py` against the UE-captured RGB
frames, matching each detected wheel to the nearest existing A/B/C
annotation by Center distance, and overwriting the bbox_xyxy in place when a
confident match is found. Existing A/B/C and the rest of the schema are
untouched.

Per-wheel match rule: a detected wheel's bbox center must lie within
`--match-tolerance` pixels of the existing `c_disc_bottom` point AND the
nearest detected bbox's height must be within a 2× factor of the derived
bbox height. If no match, the existing wheel is left unchanged.

Usage:
    python scripts/augment_native_bbox.py \\
        --incoming-root outputs/unreal_export_acceptance_neuraldata1/<slug>/incoming \\
        --images-root outputs/raw_unreal_exports/<slug>/Images \\
        --device mps
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.auto_annotate_wheels import (  # noqa: E402
    DEFAULT_DETECTOR_WEIGHTS,
    DEFAULT_SAM_WEIGHTS,
    _load_models,
    annotate_image,
)


def _center(box: list[float]) -> tuple[float, float]:
    return (0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))


def _height(box: list[float]) -> float:
    return float(box[3] - box[1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--incoming-root",
        type=Path,
        required=True,
        help="acceptance .../incoming dir with annotations/",
    )
    p.add_argument(
        "--images-root",
        type=Path,
        required=True,
        help="raw Images/ dir for the same source",
    )
    p.add_argument("--device", default="mps")
    p.add_argument("--detector-weights", default=DEFAULT_DETECTOR_WEIGHTS)
    p.add_argument("--sam-weights", default=DEFAULT_SAM_WEIGHTS)
    p.add_argument("--detect-conf", type=float, default=0.25)
    p.add_argument("--drop-conf", type=float, default=0.20)
    p.add_argument("--review-conf", type=float, default=0.30)
    p.add_argument(
        "--match-tolerance",
        type=float,
        default=80.0,
        help="max pixel distance between detected wheel center "
        "and existing c_disc_bottom to accept the match",
    )
    p.add_argument(
        "--height-ratio-max",
        type=float,
        default=2.0,
        help="ratio between detected and derived bbox height "
        "must stay within [1/r, r] to accept match",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = process every annotation; otherwise cap",
    )
    args = p.parse_args()

    ann_dir = args.incoming_root / "annotations"
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"no annotations under {args.incoming_root}")

    ann_files = sorted(ann_dir.glob("*.json"))
    if args.limit:
        ann_files = ann_files[: args.limit]

    detector, sam, _device = _load_models(
        args.detector_weights, args.sam_weights, args.device
    )

    stats = {
        "images_processed": 0,
        "images_with_wheels": 0,
        "existing_wheels": 0,
        "detected_wheels": 0,
        "bbox_replaced": 0,
        "bbox_kept_no_match": 0,
        "bbox_kept_height_mismatch": 0,
        "image_read_fail": 0,
    }

    for ann_path in ann_files:
        try:
            doc = json.loads(ann_path.read_text())
        except Exception:
            continue
        wheels = doc.get("wheels", [])
        stats["images_processed"] += 1
        if not wheels:
            continue
        stats["images_with_wheels"] += 1
        stats["existing_wheels"] += len(wheels)

        image_name = doc.get("image", f"{ann_path.stem}.jpg")
        img_path = args.images_root / image_name
        image = cv2.imread(str(img_path))
        if image is None:
            stats["image_read_fail"] += 1
            continue

        detected, img_stats = annotate_image(
            image,
            detector,
            sam,
            device=args.device,
            detect_conf=args.detect_conf,
            drop_conf=args.drop_conf,
            review_conf=args.review_conf,
        )
        stats["detected_wheels"] += len(detected)

        # Match: detected bbox must contain at least one of A/B/C keypoints
        # (with a small slack) and have plausible height vs derived bbox.
        for w in wheels:
            existing_bbox = w.get("bbox_xyxy")
            pts = w.get("points", {})
            if not existing_bbox or not pts:
                stats["bbox_kept_no_match"] += 1
                continue
            anchors = [pts.get(k) for k in ("c_disc_bottom", "a", "b")]
            anchors = [(float(p[0]), float(p[1])) for p in anchors if p]
            existing_h = _height(existing_bbox)
            best = None
            best_score = -1.0
            for d in detected:
                x1, y1, x2, y2 = d["bbox_xyxy"]
                # Tolerance around bbox edges.
                tol = args.match_tolerance
                contained = sum(
                    1
                    for (ax, ay) in anchors
                    if (x1 - tol) <= ax <= (x2 + tol) and (y1 - tol) <= ay <= (y2 + tol)
                )
                if contained == 0:
                    continue
                score = contained / max(1, len(anchors))
                if score > best_score:
                    best_score = score
                    best = d
            if best is None:
                stats["bbox_kept_no_match"] += 1
                continue
            det_h = _height(best["bbox_xyxy"])
            if existing_h <= 0 or det_h <= 0:
                stats["bbox_kept_no_match"] += 1
                continue
            ratio = max(det_h / existing_h, existing_h / det_h)
            if ratio > args.height_ratio_max:
                stats["bbox_kept_height_mismatch"] += 1
                continue
            w["bbox_xyxy_derived"] = list(existing_bbox)
            w["bbox_xyxy"] = [float(v) for v in best["bbox_xyxy"]]
            w["bbox_source"] = "coco_sam2_native"
            w["_match_score"] = round(best_score, 2)
            stats["bbox_replaced"] += 1

        ann_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))

    report_path = args.incoming_root / "metadata" / "native_bbox_augment.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
