"""Check PyTorch-vs-exported model drift over multiple sample images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_model import (  # noqa: E402
    DEFAULT_BBOX_ATOL,
    DEFAULT_CONF,
    DEFAULT_CONF_ATOL,
    DEFAULT_IOU,
    DEFAULT_KP_ATOL,
    DEFAULT_MAX_DET,
    IMAGE_EXTS,
    compare_detections,
    infer_one,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt-model", required=True, type=Path)
    parser.add_argument("--exported-model", required=True, type=Path)
    parser.add_argument(
        "--exported-task",
        choices=("auto", "detect", "segment", "classify", "pose", "obb"),
        default="auto",
        help=(
            "Task hint when loading the exported artifact. Use 'pose' for "
            "TFLite/LiteRT files when Ultralytics cannot infer the task."
        ),
    )
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--max-det", type=int, default=DEFAULT_MAX_DET)
    parser.add_argument("--bbox-atol", type=float, default=DEFAULT_BBOX_ATOL)
    parser.add_argument("--kp-atol", type=float, default=DEFAULT_KP_ATOL)
    parser.add_argument("--conf-atol", type=float, default=DEFAULT_CONF_ATOL)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def iter_images(root: Path, limit: int) -> list[Path]:
    images = [
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    images.sort()
    if limit > 0:
        images = images[:limit]
    return images


def main() -> int:
    args = parse_args()
    if not args.pt_model.is_file():
        raise FileNotFoundError(args.pt_model)
    if not args.exported_model.is_file():
        raise FileNotFoundError(args.exported_model)
    if not args.images_dir.is_dir():
        raise FileNotFoundError(args.images_dir)

    from ultralytics import YOLO  # noqa: PLC0415

    pt_model = YOLO(str(args.pt_model))
    exported_kwargs = {}
    if args.exported_task != "auto":
        exported_kwargs["task"] = args.exported_task
    exported_model = YOLO(str(args.exported_model), **exported_kwargs)
    images = iter_images(args.images_dir, args.limit)
    if not images:
        raise RuntimeError(f"No images found under {args.images_dir}")

    samples = []
    matched_count = 0
    max_bbox = 0.0
    max_kp = 0.0
    max_conf = 0.0
    for image in images:
        pt_result = infer_one(
            pt_model,
            image,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
        )
        exported_result = infer_one(
            exported_model,
            image,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
        )
        report = compare_detections(
            pt_result,
            exported_result,
            bbox_atol=args.bbox_atol,
            kp_atol=args.kp_atol,
            conf_atol=args.conf_atol,
        )
        if report["matched"]:
            matched_count += 1
        max_bbox = max(max_bbox, float(report["max_bbox_drift_px"]))
        max_kp = max(max_kp, float(report["max_kp_drift_px"]))
        max_conf = max(max_conf, float(report["max_conf_drift"]))
        samples.append({"image": str(image), **report})

    payload = {
        "pt_model": str(args.pt_model),
        "exported_model": str(args.exported_model),
        "exported_task": args.exported_task,
        "images_dir": str(args.images_dir),
        "device": args.device,
        "thresholds": {
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "bbox_atol": args.bbox_atol,
            "kp_atol": args.kp_atol,
            "conf_atol": args.conf_atol,
        },
        "samples_checked": len(samples),
        "samples_matched": matched_count,
        "ok": matched_count == len(samples),
        "max_bbox_drift_px": max_bbox,
        "max_kp_drift_px": max_kp,
        "max_conf_drift": max_conf,
        "samples": samples,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(
        f"checked={len(samples)} matched={matched_count} "
        f"max_bbox={max_bbox:.3f}px max_kp={max_kp:.3f}px "
        f"max_conf={max_conf:.3f}"
    )
    if args.out is not None:
        print(f"report={args.out}")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
