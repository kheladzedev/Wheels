"""Diagnostic eval + preview for MobileNetV2-skipless pose checkpoints.

This is intentionally lightweight: it evaluates the checkpoint against the
YOLO-pose dataset emitted by the Unreal acceptance pipeline and writes enough
metrics/previews to decide whether the model learned useful bbox + A/B/C
geometry. It is not a production inference path.
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
sys.path.insert(0, str(REPO / "scripts"))

from src.models.mobilenetv2_skipless_pose import (  # noqa: E402
    N_KEYPOINTS,
    MobileNetV2SkiplessPose,
    decode_predictions,
)
from train_mobilenetv2_skipless import YoloPoseDataset  # noqa: E402


KEYPOINT_NAMES = ("a", "b", "c_disc_bottom")
KEYPOINT_COLORS = (
    (0, 0, 255),  # a, red in BGR
    (255, 0, 0),  # b, blue in BGR
    (0, 255, 0),  # c, green in BGR
)
GT_BBOX_COLOR = (0, 180, 0)
PRED_BBOX_COLOR = (0, 255, 255)
MATCH_IOU = 0.5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MobileNetV2-skipless pose checkpoint on YOLO-pose data"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=5)
    parser.add_argument("--preview-count", type=int, default=40)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/mobilenetv2_eval/mn2_0003_provisional_e5_val"),
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


def decode_single_image(
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
                "keypoints": _keypoint_dict(keypoints[src_idx]),
                "visibility": {
                    name: float(visibility[src_idx, i])
                    for i, name in enumerate(KEYPOINT_NAMES)
                },
            }
        )
    return detections


def gt_to_records(
    bboxes: torch.Tensor, keypoints: torch.Tensor, visibility: torch.Tensor
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for i in range(bboxes.shape[0]):
        records.append(
            {
                "bbox_xyxy": _float_list(bboxes[i]),
                "keypoints": _keypoint_dict(keypoints[i]),
                "visibility": {
                    name: float(visibility[i, j])
                    for j, name in enumerate(KEYPOINT_NAMES)
                },
            }
        )
    return records


def match_predictions(
    gt: list[dict[str, Any]],
    preds: list[dict[str, Any]],
    iou_threshold: float = MATCH_IOU,
) -> list[dict[str, Any]]:
    if not gt or not preds:
        return []
    gt_boxes = torch.tensor([g["bbox_xyxy"] for g in gt], dtype=torch.float32)
    pred_boxes = torch.tensor([p["bbox_xyxy"] for p in preds], dtype=torch.float32)
    ious = _xyxy_iou(gt_boxes, pred_boxes)

    pairs: list[tuple[float, int, int]] = []
    for gt_idx in range(ious.shape[0]):
        for pred_idx in range(ious.shape[1]):
            iou = float(ious[gt_idx, pred_idx])
            if iou >= iou_threshold:
                pairs.append((iou, gt_idx, pred_idx))
    pairs.sort(reverse=True)

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[dict[str, Any]] = []
    for iou, gt_idx, pred_idx in pairs:
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        matches.append(
            {
                "gt_index": gt_idx,
                "pred_index": pred_idx,
                "iou": iou,
                "keypoint_error_px": _keypoint_errors(gt[gt_idx], preds[pred_idx]),
            }
        )
    return matches


def _keypoint_errors(gt: dict[str, Any], pred: dict[str, Any]) -> dict[str, float]:
    errors: dict[str, float] = {}
    for name in KEYPOINT_NAMES:
        gx, gy = gt["keypoints"][name]
        px, py = pred["keypoints"][name]
        errors[name] = float(((px - gx) ** 2 + (py - gy) ** 2) ** 0.5)
    return errors


def _float_list(values: torch.Tensor) -> list[float]:
    return [float(v) for v in values.tolist()]


def _keypoint_dict(values: torch.Tensor) -> dict[str, list[float]]:
    return {name: _float_list(values[i]) for i, name in enumerate(KEYPOINT_NAMES)}


def draw_preview(
    image_path: Path,
    imgsz: int,
    gt: list[dict[str, Any]],
    preds: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    out_path: Path,
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read image for preview: {image_path}")
    canvas = cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)

    matched_pred_indices = {m["pred_index"] for m in matches}
    matched_gt_indices = {m["gt_index"] for m in matches}

    for idx, item in enumerate(gt):
        _draw_box(canvas, item["bbox_xyxy"], GT_BBOX_COLOR, f"GT {idx}")
        _draw_keypoints(canvas, item["keypoints"], marker="circle")
        if idx not in matched_gt_indices:
            _draw_box(canvas, item["bbox_xyxy"], (0, 80, 0), "GT miss", thickness=1)

    for idx, item in enumerate(preds):
        label = f"P {idx} {item['score']:.2f}"
        color = PRED_BBOX_COLOR if idx in matched_pred_indices else (0, 140, 255)
        _draw_box(canvas, item["bbox_xyxy"], color, label)
        _draw_keypoints(canvas, item["keypoints"], marker="cross")

    cv2.putText(
        canvas,
        "GT=green boxes/circles, P=yellow boxes/crosses",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def _draw_box(
    image: np.ndarray,
    box: list[float],
    color: tuple[int, int, int],
    label: str,
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        image,
        label,
        (x1, max(y1 - 6, 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_keypoints(
    image: np.ndarray,
    keypoints: dict[str, list[float]],
    marker: str,
) -> None:
    for idx, name in enumerate(KEYPOINT_NAMES):
        x, y = (int(round(v)) for v in keypoints[name])
        color = KEYPOINT_COLORS[idx]
        if marker == "cross":
            cv2.drawMarker(
                image,
                (x, y),
                color,
                markerType=cv2.MARKER_CROSS,
                markerSize=14,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
        else:
            cv2.circle(image, (x, y), 5, color, -1)
        cv2.putText(
            image,
            name,
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    dataset = YoloPoseDataset(args.dataset_root, args.split, imgsz=args.imgsz)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = args.out_dir / "previews"
    predictions_path = args.out_dir / "predictions.jsonl"

    image_count = len(dataset)
    gt_wheel_count = 0
    prediction_count = 0
    matched_count = 0
    false_positive_empty_label_count = 0
    empty_label_image_count = 0
    matched_ious: list[float] = []
    keypoint_errors: dict[str, list[float]] = {name: [] for name in KEYPOINT_NAMES}
    preview_written = 0

    with predictions_path.open("w", encoding="utf-8") as pred_file:
        for idx in range(len(dataset)):
            image, bboxes, keypoints, visibility = dataset[idx]
            gt = gt_to_records(bboxes, keypoints, visibility)
            preds = decode_single_image(
                model,
                image,
                device=device,
                conf=args.conf,
                nms_iou=args.nms_iou,
                max_det=args.max_det,
                imgsz=args.imgsz,
            )
            matches = match_predictions(gt, preds)

            gt_wheel_count += len(gt)
            prediction_count += len(preds)
            matched_count += len(matches)
            if not gt:
                empty_label_image_count += 1
                false_positive_empty_label_count += len(preds)

            for match in matches:
                matched_ious.append(float(match["iou"]))
                for name, value in match["keypoint_error_px"].items():
                    keypoint_errors[name].append(float(value))

            image_path = dataset.image_paths[idx]
            row = {
                "image": str(image_path),
                "label": str(dataset._label_path_for(image_path)),
                "gt": gt,
                "predictions": preds,
                "matches": matches,
            }
            pred_file.write(json.dumps(row, sort_keys=True) + "\n")

            if preview_written < args.preview_count:
                draw_preview(
                    image_path,
                    imgsz=args.imgsz,
                    gt=gt,
                    preds=preds,
                    matches=matches,
                    out_path=previews_dir / f"{image_path.stem}_mn2_eval.jpg",
                )
                preview_written += 1

    precision = matched_count / prediction_count if prediction_count else 0.0
    recall = matched_count / gt_wheel_count if gt_wheel_count else 0.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    mean_keypoint_error = {
        name: float(np.mean(values)) if values else 0.0
        for name, values in keypoint_errors.items()
    }

    report = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "imgsz": args.imgsz,
        "device": str(device),
        "conf": args.conf,
        "nms_iou": args.nms_iou,
        "max_det": args.max_det,
        "match_iou": MATCH_IOU,
        "image_count": image_count,
        "gt_wheel_count": gt_wheel_count,
        "prediction_count": prediction_count,
        "matched_count": matched_count,
        "precision": float(precision),
        "recall": float(recall),
        "mean_iou": mean_iou,
        "mean_keypoint_error_px": mean_keypoint_error,
        "empty_label_image_count": empty_label_image_count,
        "false_positive_empty_label_count": false_positive_empty_label_count,
        "preview_count": preview_written,
        "previews_dir": str(previews_dir),
        "predictions_jsonl": str(predictions_path),
        "provisional_data_note": "0003 remains provisional until the data-quality gate passes",
    }
    return report


def write_reports(report: dict[str, Any], out_dir: Path) -> None:
    json_path = out_dir / "eval_report.json"
    md_path = out_dir / "eval_report.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# MobileNetV2 Diagnostic Eval",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Dataset: `{report['dataset_root']}` (`{report['split']}`)",
        f"- Images: {report['image_count']}",
        f"- GT wheels: {report['gt_wheel_count']}",
        f"- Predictions: {report['prediction_count']}",
        f"- Matched: {report['matched_count']}",
        f"- Precision: {report['precision']:.4f}",
        f"- Recall: {report['recall']:.4f}",
        f"- Mean IoU: {report['mean_iou']:.4f}",
        f"- False positives on empty labels: {report['false_positive_empty_label_count']}",
        "",
        "## Mean Keypoint Error",
        "",
    ]
    for name in KEYPOINT_NAMES:
        lines.append(f"- {name}: {report['mean_keypoint_error_px'][name]:.2f}px")
    lines += [
        "",
        "## Artifacts",
        "",
        f"- Predictions: `{report['predictions_jsonl']}`",
        f"- Previews: `{report['previews_dir']}`",
        "",
        "Note: this is a provisional diagnostic eval, not production approval.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate(args)
    write_reports(report, args.out_dir)
    print(f"Eval report: {args.out_dir / 'eval_report.md'}")
    print(f"Previews:    {report['previews_dir']}")
    print(
        "metrics: "
        f"precision={report['precision']:.4f} "
        f"recall={report['recall']:.4f} "
        f"mean_iou={report['mean_iou']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
