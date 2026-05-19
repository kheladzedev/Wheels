"""Run MobileNetV2-skipless wheel pose inference and export AR JSON.

This is the practical image/folder inference path for MobileNetV2 checkpoints.
It emits the same AR-team confirmed schema as the YOLO inference path:

    {frame_id, wheels[].{bbox_xyxy, confidence, points.{a,b,c_disc_bottom}}}

The current baseline checkpoint is still provisional because it was trained on
the dirty 0003 export; this script makes it inspectable and integrable, not
production-approved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.models.mobilenetv2_skipless_pose import (  # noqa: E402
    MobileNetV2SkiplessPose,
    decode_predictions,
)
from src.postprocess_wheels import build_ar_payload, to_confirmed_schema  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KEYPOINT_NAMES = ("a", "b", "c_disc_bottom")
POINT_ORDER = ("a", "b", "c_disc_bottom")
COLOR_BBOX = (0, 255, 255)
COLOR_KP = {
    "a": (0, 0, 255),
    "b": (255, 0, 0),
    "c_disc_bottom": (0, 255, 0),
}
DISPLAY_KP_NAMES = {"a": "A", "b": "B", "c_disc_bottom": "C"}
MODEL_STATUS = "provisional_0003_not_production"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MobileNetV2-skipless pose inference on image(s)"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=5)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/mobilenetv2_infer/default"),
    )
    parser.add_argument("--preview-count", type=int, default=40)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on sorted input images for quick diagnostic runs.",
    )
    return parser.parse_args(argv)


def load_model(checkpoint_path: Path, device: torch.device) -> MobileNetV2SkiplessPose:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = MobileNetV2SkiplessPose(pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.train(False)
    return model


def iter_image_paths(source: Path, limit: int | None = None) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"unsupported image extension: {source}")
        images = [source]
    elif source.is_dir():
        images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    else:
        raise FileNotFoundError(f"source not found: {source}")

    if limit is not None:
        images = images[:limit]
    if not images:
        raise ValueError(f"no images found in source: {source}")
    return images


def image_to_tensor(image_bgr: np.ndarray, imgsz: int) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    image_np = image_rgb.astype(np.float32) / 255.0
    return torch.from_numpy(image_np).permute(2, 0, 1).contiguous()


def _xyxy_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), dtype=torch.float32)

    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area_a = ((a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0))
    area_b = ((b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0))
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


def nms_boxes(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> list[int]:
    if boxes.numel() == 0:
        return []
    order = torch.argsort(scores, descending=True)
    keep: list[int] = []
    while int(order.numel()) > 0:
        current = int(order[0].item())
        keep.append(current)
        if int(order.numel()) == 1:
            break
        rest = order[1:]
        ious = _xyxy_iou(boxes[current].unsqueeze(0), boxes[rest]).squeeze(0)
        order = rest[ious <= iou_threshold]
    return keep


def decode_model_detections(
    model: MobileNetV2SkiplessPose,
    image: torch.Tensor,
    device: torch.device,
    conf: float,
    nms_iou: float,
    max_det: int,
    imgsz: int,
) -> list[dict[str, Any]]:
    with torch.no_grad():
        preds = model(image.unsqueeze(0).to(device))
        decoded = decode_predictions(preds)

    scores = decoded["cls_prob"][0].detach().cpu()
    boxes = decoded["bbox_xyxy"][0].detach().cpu()
    keypoints = decoded["kpt_xy"][0].detach().cpu()
    visibility = decoded["vis_prob"][0].detach().cpu()

    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, imgsz)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, imgsz)
    keypoints[..., 0] = keypoints[..., 0].clamp(0, imgsz)
    keypoints[..., 1] = keypoints[..., 1].clamp(0, imgsz)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    keep_mask = (scores >= conf) & (widths > 1.0) & (heights > 1.0)
    if not bool(keep_mask.any()):
        return []

    kept_idx = torch.where(keep_mask)[0]
    kept_boxes = boxes[kept_idx]
    kept_scores = scores[kept_idx]
    nms_keep = nms_boxes(kept_boxes, kept_scores, nms_iou)[:max_det]

    detections: list[dict[str, Any]] = []
    for local_idx in nms_keep:
        src_idx = int(kept_idx[local_idx].item())
        detections.append(
            {
                "score": float(scores[src_idx]),
                "bbox_xyxy": _float_list(boxes[src_idx]),
                "points": {
                    name: _float_list(keypoints[src_idx, i])
                    for i, name in enumerate(KEYPOINT_NAMES)
                },
                "visibility": {
                    name: float(visibility[src_idx, i])
                    for i, name in enumerate(KEYPOINT_NAMES)
                },
            }
        )
    return detections


def scale_detections_to_original(
    detections: list[dict[str, Any]],
    *,
    original_width: int,
    original_height: int,
    imgsz: int,
) -> list[dict[str, Any]]:
    scale_x = float(original_width) / float(imgsz)
    scale_y = float(original_height) / float(imgsz)
    scaled: list[dict[str, Any]] = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        bbox = [
            _clip_float(x1 * scale_x, 0.0, float(original_width)),
            _clip_float(y1 * scale_y, 0.0, float(original_height)),
            _clip_float(x2 * scale_x, 0.0, float(original_width)),
            _clip_float(y2 * scale_y, 0.0, float(original_height)),
        ]
        points: dict[str, list[float]] = {}
        for name in POINT_ORDER:
            px, py = det["points"][name]
            points[name] = [
                _clip_float(px * scale_x, 0.0, float(original_width)),
                _clip_float(py * scale_y, 0.0, float(original_height)),
            ]
        scaled.append(
            {
                "score": float(det["score"]),
                "bbox_xyxy": bbox,
                "points": points,
                "visibility": {
                    name: float(det.get("visibility", {}).get(name, 1.0))
                    for name in POINT_ORDER
                },
            }
        )
    return scaled


def detections_to_confirmed_payload(
    detections: list[dict[str, Any]],
    *,
    frame_id: str,
    conf: float,
) -> dict[str, Any]:
    legacy_detections = []
    for det in detections:
        legacy_detections.append(
            {
                "class_name": "wheel",
                "bbox": det["bbox_xyxy"],
                "confidence": float(det["score"]),
                "keypoints": [
                    {
                        "xy": det["points"][name],
                        # The MobileNetV2 head always predicts the full A/B/C
                        # contract for each kept wheel; geometry guards below
                        # still reject unsafe confirmed-schema payloads.
                        "visibility": 2,
                        "confidence": float(det["visibility"].get(name, det["score"])),
                    }
                    for name in POINT_ORDER
                ],
            }
        )

    legacy_payload = build_ar_payload(
        legacy_detections,
        conf_threshold=conf,
        frame_id=frame_id,
        timestamp=None,
    )
    confirmed_payload = to_confirmed_schema(legacy_payload)
    assert "frame_id" in confirmed_payload
    _assert_confirmed_schema_closed(confirmed_payload)
    return confirmed_payload


def predict_image(
    model: MobileNetV2SkiplessPose,
    image_path: Path,
    *,
    device: torch.device,
    imgsz: int,
    conf: float,
    nms_iou: float,
    max_det: int,
) -> tuple[dict[str, Any], int, np.ndarray]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"failed to read image: {image_path}")
    original_height, original_width = image_bgr.shape[:2]
    image_tensor = image_to_tensor(image_bgr, imgsz)

    model_detections = decode_model_detections(
        model,
        image_tensor,
        device=device,
        conf=conf,
        nms_iou=nms_iou,
        max_det=max_det,
        imgsz=imgsz,
    )
    scaled_detections = scale_detections_to_original(
        model_detections,
        original_width=original_width,
        original_height=original_height,
        imgsz=imgsz,
    )
    confirmed = detections_to_confirmed_payload(
        scaled_detections,
        frame_id=image_path.stem,
        conf=conf,
    )
    return confirmed, len(model_detections), image_bgr


def write_preview(image_bgr: np.ndarray, payload: dict[str, Any], out_path: Path) -> None:
    canvas = image_bgr.copy()
    for index, wheel in enumerate(payload.get("wheels", [])):
        x1, y1, x2, y2 = (int(round(v)) for v in wheel["bbox_xyxy"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_BBOX, 2)
        cv2.putText(
            canvas,
            f"W{index} {wheel['confidence']:.2f}",
            (x1, max(y1 - 7, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_BBOX,
            1,
            cv2.LINE_AA,
        )
        for name in POINT_ORDER:
            x, y = (int(round(v)) for v in wheel["points"][name])
            color = COLOR_KP[name]
            cv2.drawMarker(
                canvas,
                (x, y),
                color,
                markerType=cv2.MARKER_CROSS,
                markerSize=14,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                DISPLAY_KP_NAMES[name],
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), canvas):
        raise ValueError(f"failed to write preview: {out_path}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    image_paths = iter_image_paths(args.source, limit=args.limit)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = args.out_dir / "previews"
    predictions_jsonl = args.out_dir / "predictions.jsonl"
    thresholds = {
        "conf": float(args.conf),
        "nms_iou": float(args.nms_iou),
        "max_det": int(args.max_det),
    }

    prediction_count = 0
    raw_detection_count = 0
    confirmed_dropped_count = 0
    empty_prediction_count = 0
    preview_count = 0
    frame_index: list[dict[str, Any]] = []

    with predictions_jsonl.open("w", encoding="utf-8") as jsonl_fh:
        for image_path in image_paths:
            payload, raw_count, image_bgr = predict_image(
                model,
                image_path,
                device=device,
                imgsz=args.imgsz,
                conf=args.conf,
                nms_iou=args.nms_iou,
                max_det=args.max_det,
            )

            out_json = args.out_dir / f"{image_path.stem}.json"
            out_json.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            jsonl_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

            confirmed_count = len(payload["wheels"])
            prediction_count += confirmed_count
            raw_detection_count += raw_count
            confirmed_dropped_count += raw_count - confirmed_count
            if confirmed_count == 0:
                empty_prediction_count += 1

            preview_path: Path | None = None
            if preview_count < args.preview_count:
                preview_path = previews_dir / f"{image_path.stem}_mn2_pred.jpg"
                write_preview(image_bgr, payload, preview_path)
                preview_count += 1

            frame_index.append(
                {
                    "frame_id": payload["frame_id"],
                    "source_image": str(image_path),
                    "json": str(out_json),
                    "preview": str(preview_path) if preview_path else None,
                    "raw_detection_count": raw_count,
                    "confirmed_wheel_count": confirmed_count,
                    "confirmed_dropped_count": raw_count - confirmed_count,
                }
            )

    summary = {
        "checkpoint": str(args.checkpoint),
        "source": str(args.source),
        "out_dir": str(args.out_dir),
        "model_status": MODEL_STATUS,
        "imgsz": int(args.imgsz),
        "device": str(device),
        "thresholds": thresholds,
        "image_count": len(image_paths),
        "prediction_count": prediction_count,
        "raw_detection_count": raw_detection_count,
        "confirmed_dropped_count": confirmed_dropped_count,
        "empty_prediction_count": empty_prediction_count,
        "preview_count": preview_count,
        "predictions_jsonl": str(predictions_jsonl),
        "frame_index": frame_index,
    }
    (args.out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _assert_confirmed_schema_closed(payload: dict[str, Any]) -> None:
    allowed_top = {"frame_id", "wheels"}
    if set(payload) != allowed_top:
        raise AssertionError(f"confirmed payload keys changed: {sorted(payload)}")
    for idx, wheel in enumerate(payload["wheels"]):
        allowed_wheel = {"bbox_xyxy", "confidence", "points"}
        if set(wheel) != allowed_wheel:
            raise AssertionError(
                f"confirmed wheel[{idx}] keys changed: {sorted(wheel)}"
            )
        if set(wheel["points"]) != set(POINT_ORDER):
            raise AssertionError(
                f"confirmed wheel[{idx}] points changed: {sorted(wheel['points'])}"
            )


def _clip_float(value: float, lower: float, upper: float) -> float:
    return float(min(max(value, lower), upper))


def _float_list(values: torch.Tensor) -> list[float]:
    return [float(v) for v in values.tolist()]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print(f"Images:              {summary['image_count']}")
    print(f"Raw detections:      {summary['raw_detection_count']}")
    print(f"Confirmed wheels:    {summary['prediction_count']}")
    print(f"Dropped by geometry: {summary['confirmed_dropped_count']}")
    print(f"Empty predictions:   {summary['empty_prediction_count']}")
    print(f"Model status:        {summary['model_status']}")
    print(f"Predictions JSONL:   {summary['predictions_jsonl']}")
    print(f"Run summary:         {args.out_dir / 'run_summary.json'}")
    if summary["preview_count"]:
        print(f"Previews:            {args.out_dir / 'previews'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
