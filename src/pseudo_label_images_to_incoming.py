"""Pseudo-label rendered images into the plugin keypoint incoming format.

This is the handoff bridge for the MCP/Unreal path:

  1. UnrealMCP renders imported Sketchfab cars to an image directory.
  2. This script runs the current YOLO-pose champion on those renders.
  3. Valid predictions are written as plugin-format incoming annotations:
     ``images/<stem>.<ext>`` + ``annotations/<stem>.json``.

The labels are model-generated, not engine ground truth. They are marked
with ``_draft`` / ``_needs_review`` provenance so they can be used for
triage, QA, and self-training experiments without pretending they are
direct floor-ray labels from Unreal geometry.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_MODEL = Path("runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt")
DEFAULT_OUTPUT = Path("data/incoming/ue_sketchfab_pseudo")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--images-dir", required=True, type=Path)
    p.add_argument("--output-root", default=DEFAULT_OUTPUT, type=Path)
    p.add_argument("--model", default=DEFAULT_MODEL, type=Path)
    p.add_argument("--source-name", default="ue_sketchfab_pseudo")
    p.add_argument("--conf", type=float, default=0.40)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--max-det", type=int, default=20)
    p.add_argument("--device", default="mps")
    p.add_argument("--min-bbox-side", type=float, default=8.0)
    p.add_argument("--include-empty", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def iter_image_paths(images_dir: Path) -> list[Path]:
    exts = {ext.lower() for ext in IMAGE_EXTS}
    return sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts
    )


def _clip_bbox_xyxy(
    bbox: np.ndarray, image_w: int, image_h: int
) -> list[float] | None:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    x1 = max(0.0, min(x1, image_w - 1.0))
    y1 = max(0.0, min(y1, image_h - 1.0))
    x2 = max(0.0, min(x2, image_w - 1.0))
    y2 = max(0.0, min(y2, image_h - 1.0))
    if not (x1 < x2 and y1 < y2):
        return None
    return [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]


def _points_inside_image(kp_xy: np.ndarray, image_w: int, image_h: int) -> bool:
    if kp_xy.shape[0] < 3 or kp_xy.shape[1] < 2:
        return False
    pts = kp_xy[:3, :2]
    if not np.all(np.isfinite(pts)):
        return False
    if np.any(pts <= 0.5):
        return False
    return bool(
        pts[:, 0].min() >= 0.0
        and pts[:, 0].max() < image_w
        and pts[:, 1].min() >= 0.0
        and pts[:, 1].max() < image_h
    )


def build_wheel_annotation(
    bbox_xyxy: np.ndarray,
    keypoints_xy: np.ndarray,
    confidence: float,
    *,
    image_w: int,
    image_h: int,
    min_bbox_side: float,
) -> dict | None:
    bbox = _clip_bbox_xyxy(bbox_xyxy, image_w, image_h)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    if (x2 - x1) < min_bbox_side or (y2 - y1) < min_bbox_side:
        return None
    if not _points_inside_image(keypoints_xy, image_w, image_h):
        return None

    pts = keypoints_xy[:3, :2]
    return {
        "bbox_xyxy": bbox,
        "points": {
            "a": [round(float(pts[0, 0]), 3), round(float(pts[0, 1]), 3)],
            "b": [round(float(pts[1, 0]), 3), round(float(pts[1, 1]), 3)],
            "c_disc_bottom": [
                round(float(pts[2, 0]), 3),
                round(float(pts[2, 1]), 3),
            ],
        },
        "_draft": True,
        "_needs_review": True,
        "_review_reasons": ["model_pseudo_label_from_render"],
        "_pseudo_conf": round(float(confidence), 4),
    }


def _prepare_output(root: Path, *, overwrite: bool) -> None:
    if root.exists():
        if not overwrite:
            raise SystemExit(f"output root exists; pass --overwrite: {root}")
        shutil.rmtree(root)
    for sub in ("images", "annotations", "metadata"):
        (root / sub).mkdir(parents=True, exist_ok=True)


def _write_source_info(
    output_root: Path,
    *,
    args: argparse.Namespace,
    images_seen: int,
    frames_written: int,
    wheels_written: int,
) -> None:
    info = {
        "source_name": args.source_name,
        "annotation_method": "yolo_pose_champion_pseudo_label_from_render",
        "_warning": "NOT_ENGINE_GROUND_TRUTH_REQUIRES_REVIEW",
        "images_dir": str(args.images_dir),
        "model": str(args.model),
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "min_bbox_side": args.min_bbox_side,
        "include_empty": args.include_empty,
        "images_seen": images_seen,
        "frames_written": frames_written,
        "wheels_written": wheels_written,
    }
    (output_root / "metadata" / "source_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.images_dir.is_dir():
        print(f"ERROR: missing images dir: {args.images_dir}", file=sys.stderr)
        return 2
    if not args.model.is_file():
        print(f"ERROR: missing model: {args.model}", file=sys.stderr)
        return 2

    image_paths = iter_image_paths(args.images_dir)
    if not image_paths:
        print(f"ERROR: no images under {args.images_dir}", file=sys.stderr)
        return 2

    _prepare_output(args.output_root, overwrite=args.overwrite)

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    frames_written = 0
    wheels_written = 0

    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"WARNING: unreadable image skipped: {image_path}", file=sys.stderr)
            continue
        image_h, image_w = img.shape[:2]
        results = model.predict(
            source=str(image_path),
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
        )

        wheels: list[dict] = []
        if results:
            result = results[0]
            if result.boxes is not None and result.keypoints is not None:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                kp_xy = result.keypoints.xy.cpu().numpy()
                for i in range(len(boxes_xyxy)):
                    wheel = build_wheel_annotation(
                        boxes_xyxy[i],
                        kp_xy[i],
                        float(confs[i]),
                        image_w=image_w,
                        image_h=image_h,
                        min_bbox_side=args.min_bbox_side,
                    )
                    if wheel is not None:
                        wheels.append(wheel)

        if not wheels and not args.include_empty:
            continue

        dst_img = args.output_root / "images" / image_path.name
        shutil.copy2(image_path, dst_img)
        annotation = {
            "frame_id": image_path.stem,
            "image": image_path.name,
            "wheels": wheels,
            "_draft": True,
            "_warning": "MODEL_PSEUDO_LABEL_FROM_RENDER_NOT_ENGINE_GROUND_TRUTH",
        }
        (args.output_root / "annotations" / f"{image_path.stem}.json").write_text(
            json.dumps(annotation, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        frames_written += 1
        wheels_written += len(wheels)

    _write_source_info(
        args.output_root,
        args=args,
        images_seen=len(image_paths),
        frames_written=frames_written,
        wheels_written=wheels_written,
    )
    print(
        f"[pseudo] images_seen={len(image_paths)} "
        f"frames_written={frames_written} wheels_written={wheels_written}"
    )
    print(f"[pseudo] next: ./.venv/bin/python src/check_keypoint_incoming.py --source-root {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
