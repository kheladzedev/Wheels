"""Run MobileNetV2 TFLite/LiteRT runtime smoke inference.

This script exercises the deployable `.tflite` artifact instead of the
PyTorch checkpoint. TensorFlow stays outside the main VSBL `.venv`: the TFLite
interpreter is invoked through `--runtime-python`, typically
`.tflite-venv/bin/python`.

The emitted per-image JSON is the same confirmed AR schema as
`predict_mobilenetv2_skipless.py`:

    {frame_id, wheels[].{bbox_xyxy, confidence, points.{a,b,c_disc_bottom}}}
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from export_mobilenetv2_tflite import (  # noqa: E402
    dependency_status,
    normalize_tflite_outputs,
    tflite_raw_outputs,
)
from predict_mobilenetv2_skipless import (  # noqa: E402
    MODEL_STATUS,
    detections_to_confirmed_payload,
    image_to_tensor,
    iter_image_paths,
    nms_boxes,
    scale_detections_to_original,
    write_preview,
)
from src.models.mobilenetv2_skipless_pose import (  # noqa: E402
    FEATURE_STRIDE,
    N_KEYPOINTS,
    decode_predictions,
)


OUTPUT_NAMES = ("cls", "bbox", "kpt", "vis")
KEYPOINT_NAMES = ("a", "b", "c_disc_bottom")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run confirmed-schema inference through a TFLite/LiteRT model"
    )
    parser.add_argument("--tflite-model", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--runtime-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=5)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/mobilenetv2_tflite_runtime/default"),
    )
    parser.add_argument("--preview-count", type=int, default=40)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on sorted input images for quick diagnostic runs.",
    )
    return parser.parse_args(argv)


def expected_output_shapes(imgsz: int) -> dict[str, list[int]]:
    if imgsz % FEATURE_STRIDE != 0:
        raise ValueError(f"imgsz must be divisible by {FEATURE_STRIDE}: {imgsz}")
    grid = imgsz // FEATURE_STRIDE
    return {
        "cls": [1, 1, grid, grid],
        "bbox": [1, 4, grid, grid],
        "kpt": [1, N_KEYPOINTS * 2, grid, grid],
        "vis": [1, N_KEYPOINTS, grid, grid],
    }


def _np_to_torch(raw: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {name: torch.from_numpy(value) for name, value in raw.items()}


def decode_tflite_detections(
    raw: dict[str, torch.Tensor],
    *,
    imgsz: int,
    conf: float,
    nms_iou: float,
    max_det: int,
) -> list[dict[str, Any]]:
    decoded = decode_predictions(raw)
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


def predict_image(
    *,
    tflite_model: Path,
    runtime_python: Path,
    image_path: Path,
    out_dir: Path,
    imgsz: int,
    conf: float,
    nms_iou: float,
    max_det: int,
) -> tuple[dict[str, Any], int, np.ndarray, dict[str, Any]]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"failed to read image: {image_path}")

    image_tensor = image_to_tensor(image_bgr, imgsz)
    runtime_dir = out_dir / "runtime_outputs"
    logs_dir = out_dir / "logs"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    input_npy = runtime_dir / f"{image_path.stem}_input.npy"
    output_npz = runtime_dir / f"{image_path.stem}_tflite_outputs.npz"
    runtime_log = logs_dir / f"{image_path.stem}_tflite_runtime.log"
    np.save(input_npy, image_tensor.unsqueeze(0).numpy())

    runtime = tflite_raw_outputs(
        converter_python=runtime_python,
        tflite_path=tflite_model,
        sample_tensor_path=input_npy,
        output_npz_path=output_npz,
        log_path=runtime_log,
    )
    frame_runtime = {
        "input_npy": str(input_npy),
        "output_npz": str(output_npz),
        "runtime_log": str(runtime_log),
        "runtime_returncode": int(runtime.returncode),
    }
    if runtime.returncode != 0 or not output_npz.is_file():
        frame_runtime["error"] = "TFLite runtime invocation failed"
        payload = {"frame_id": image_path.stem, "wheels": []}
        return payload, 0, image_bgr, frame_runtime

    mapping_report, tflite_np = normalize_tflite_outputs(
        output_npz,
        expected_output_shapes(imgsz),
    )
    frame_runtime["output_mapping"] = mapping_report
    if not mapping_report["mapped"]:
        frame_runtime["error"] = "TFLite output mapping failed"
        payload = {"frame_id": image_path.stem, "wheels": []}
        return payload, 0, image_bgr, frame_runtime

    raw_detections = decode_tflite_detections(
        _np_to_torch(tflite_np),
        imgsz=imgsz,
        conf=conf,
        nms_iou=nms_iou,
        max_det=max_det,
    )
    h, w = image_bgr.shape[:2]
    scaled = scale_detections_to_original(
        raw_detections,
        original_width=w,
        original_height=h,
        imgsz=imgsz,
    )
    payload = detections_to_confirmed_payload(
        scaled,
        frame_id=image_path.stem,
        conf=conf,
    )
    return payload, len(raw_detections), image_bgr, frame_runtime


def _write_report(summary: dict[str, Any], out_dir: Path) -> None:
    lines = [
        "# MobileNetV2 TFLite/LiteRT Runtime Smoke Report",
        "",
        f"- Status: **{summary['status']}**",
        f"- TFLite: `{summary['tflite_model']}`",
        f"- Runtime python: `{summary['runtime_python']}`",
        f"- Model status: **{summary['model_status']}**",
        f"- Images: {summary['image_count']}",
        f"- Raw detections: {summary['raw_detection_count']}",
        f"- Confirmed wheels: {summary['prediction_count']}",
        f"- Runtime failures: {summary['runtime_failure_count']}",
        "",
        "Note: this remains a runtime smoke check for a provisional model, not "
        "Android production approval.",
        "",
    ]
    (out_dir / "runtime_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.tflite_model.is_file():
        raise FileNotFoundError(f"TFLite model not found: {args.tflite_model}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    deps = dependency_status(args.runtime_python)
    image_paths = iter_image_paths(args.source, limit=args.limit)
    predictions_jsonl = args.out_dir / "predictions.jsonl"
    previews_dir = args.out_dir / "previews"

    prediction_count = 0
    raw_detection_count = 0
    confirmed_dropped_count = 0
    empty_prediction_count = 0
    runtime_failure_count = 0
    preview_count = 0
    frame_index: list[dict[str, Any]] = []

    with predictions_jsonl.open("w", encoding="utf-8") as jsonl_fh:
        for image_path in image_paths:
            payload, raw_count, image_bgr, runtime_info = predict_image(
                tflite_model=args.tflite_model,
                runtime_python=args.runtime_python,
                image_path=image_path,
                out_dir=args.out_dir,
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
            if runtime_info.get("error"):
                runtime_failure_count += 1

            preview_path: Path | None = None
            if preview_count < args.preview_count:
                preview_path = previews_dir / f"{image_path.stem}_mn2_tflite_pred.jpg"
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
                    "runtime": runtime_info,
                }
            )

    summary = {
        "status": "PASS" if runtime_failure_count == 0 else "RUNTIME_FAILED",
        "tflite_model": str(args.tflite_model),
        "runtime_python": str(args.runtime_python),
        "source": str(args.source),
        "out_dir": str(args.out_dir),
        "model_status": MODEL_STATUS,
        "imgsz": int(args.imgsz),
        "thresholds": {
            "conf": float(args.conf),
            "nms_iou": float(args.nms_iou),
            "max_det": int(args.max_det),
        },
        "runtime_dependencies": deps,
        "image_count": len(image_paths),
        "prediction_count": prediction_count,
        "raw_detection_count": raw_detection_count,
        "confirmed_dropped_count": confirmed_dropped_count,
        "empty_prediction_count": empty_prediction_count,
        "runtime_failure_count": runtime_failure_count,
        "preview_count": preview_count,
        "predictions_jsonl": str(predictions_jsonl),
        "frame_index": frame_index,
    }
    (args.out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_report(summary, args.out_dir)
    return summary


def _float_list(values: torch.Tensor) -> list[float]:
    return [float(v) for v in values.tolist()]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run(args)
    print(f"Status:              {summary['status']}")
    print(f"Images:              {summary['image_count']}")
    print(f"Raw detections:      {summary['raw_detection_count']}")
    print(f"Confirmed wheels:    {summary['prediction_count']}")
    print(f"Dropped by geometry: {summary['confirmed_dropped_count']}")
    print(f"Runtime failures:    {summary['runtime_failure_count']}")
    print(f"Model status:        {summary['model_status']}")
    print(f"Predictions JSONL:   {summary['predictions_jsonl']}")
    print(f"Run summary:         {args.out_dir / 'run_summary.json'}")
    print(f"Runtime report:      {args.out_dir / 'runtime_report.md'}")
    if summary["preview_count"]:
        print(f"Previews:            {args.out_dir / 'previews'}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
