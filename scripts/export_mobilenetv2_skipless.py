"""Export MobileNetV2-skipless pose checkpoints to ONNX with parity checks.

This exports the raw model graph only. Decode, thresholding, NMS, geometry
guards, and confirmed AR JSON formatting remain runtime-side, matching
``scripts/predict_mobilenetv2_skipless.py``.
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
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from predict_mobilenetv2_skipless import (  # noqa: E402
    decode_model_detections,
    image_to_tensor,
    load_model,
    write_preview,
)
from src.models.mobilenetv2_skipless_pose import (  # noqa: E402
    FEATURE_STRIDE,
    MobileNetV2SkiplessPose,
    decode_predictions,
)


OUTPUT_NAMES = ("cls", "bbox", "kpt", "vis")
DEFAULT_MODEL_STATUS = "provisional_0003_not_production"


class OnnxExportWrapper(nn.Module):
    """Return tuple outputs so ONNX names are stable."""

    def __init__(self, model: MobileNetV2SkiplessPose) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out = self.model(images)
        return out["cls"], out["bbox"], out["kpt"], out["vis"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MobileNetV2-skipless checkpoint to ONNX"
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-image", required=True, type=Path)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=5)
    parser.add_argument("--bbox-atol", type=float, default=2.0)
    parser.add_argument("--kpt-atol", type=float, default=3.0)
    parser.add_argument("--conf-atol", type=float, default=0.05)
    parser.add_argument("--raw-atol", type=float, default=1e-4)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args(argv)


def export_onnx(
    model: MobileNetV2SkiplessPose,
    out_path: Path,
    *,
    imgsz: int,
    opset: int,
    device: torch.device,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = OnnxExportWrapper(model).to(device)
    wrapper.train(False)
    dummy = torch.zeros(1, 3, imgsz, imgsz, dtype=torch.float32, device=device)
    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        input_names=["images"],
        output_names=list(OUTPUT_NAMES),
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    if not out_path.is_file():
        raise FileNotFoundError(f"ONNX export did not create file: {out_path}")


def load_sample_tensor(image_path: Path, imgsz: int) -> tuple[torch.Tensor, np.ndarray]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"failed to read sample image: {image_path}")
    return image_to_tensor(image_bgr, imgsz), image_bgr


def pytorch_raw_outputs(
    model: MobileNetV2SkiplessPose,
    image: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        out = model(image.unsqueeze(0).to(device))
    return {name: out[name].detach().cpu() for name in OUTPUT_NAMES}


def onnx_raw_outputs(onnx_path: Path, image: torch.Tensor) -> dict[str, torch.Tensor]:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - exercised only without dep
        raise RuntimeError("onnxruntime is required for parity checks") from exc

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = {session.get_inputs()[0].name: image.unsqueeze(0).numpy()}
    outputs = session.run(None, inputs)
    return {
        name: torch.from_numpy(np.asarray(value))
        for name, value in zip(OUTPUT_NAMES, outputs)
    }


def raw_parity_report(
    pt_raw: dict[str, torch.Tensor],
    onnx_raw: dict[str, torch.Tensor],
    *,
    raw_atol: float,
) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    failures: list[str] = []
    max_abs = 0.0
    for name in OUTPUT_NAMES:
        pt = pt_raw[name].float()
        ox = onnx_raw[name].float()
        if tuple(pt.shape) != tuple(ox.shape):
            failures.append(f"{name} shape differs: pt={tuple(pt.shape)} onnx={tuple(ox.shape)}")
            outputs[name] = {
                "pt_shape": list(pt.shape),
                "onnx_shape": list(ox.shape),
                "max_abs_diff": None,
            }
            continue
        diff = torch.abs(pt - ox)
        output_max = float(diff.max().item()) if diff.numel() else 0.0
        max_abs = max(max_abs, output_max)
        outputs[name] = {
            "pt_shape": list(pt.shape),
            "onnx_shape": list(ox.shape),
            "max_abs_diff": output_max,
        }
        if output_max > raw_atol:
            failures.append(f"{name} max_abs_diff {output_max:.6g} > raw_atol {raw_atol:.6g}")
    return {
        "matched": not failures,
        "max_abs_diff": max_abs,
        "outputs": outputs,
        "failures": failures,
    }


def _detections_from_raw(
    raw: dict[str, torch.Tensor],
    *,
    conf: float,
    nms_iou: float,
    max_det: int,
    imgsz: int,
) -> list[dict[str, Any]]:
    class RawModel(nn.Module):
        def __init__(self, raw_outputs: dict[str, torch.Tensor]) -> None:
            super().__init__()
            self.raw_outputs = raw_outputs

        def forward(self, _images: torch.Tensor) -> dict[str, torch.Tensor]:
            return {name: value.clone() for name, value in self.raw_outputs.items()}

    return decode_model_detections(
        RawModel(raw),
        torch.zeros(3, imgsz, imgsz, dtype=torch.float32),
        device=torch.device("cpu"),
        conf=conf,
        nms_iou=nms_iou,
        max_det=max_det,
        imgsz=imgsz,
    )


def decoded_parity_report(
    pt_raw: dict[str, torch.Tensor],
    onnx_raw: dict[str, torch.Tensor],
    *,
    conf: float,
    nms_iou: float,
    max_det: int,
    imgsz: int,
    bbox_atol: float,
    kpt_atol: float,
    conf_atol: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pt_dets = _detections_from_raw(
        pt_raw, conf=conf, nms_iou=nms_iou, max_det=max_det, imgsz=imgsz
    )
    onnx_dets = _detections_from_raw(
        onnx_raw, conf=conf, nms_iou=nms_iou, max_det=max_det, imgsz=imgsz
    )
    failures: list[str] = []
    max_bbox = 0.0
    max_kpt = 0.0
    max_conf = 0.0
    if len(pt_dets) != len(onnx_dets):
        failures.append(f"detection count differs: pt={len(pt_dets)} onnx={len(onnx_dets)}")

    for idx, (pt, ox) in enumerate(zip(pt_dets, onnx_dets)):
        bbox_diff = float(
            np.max(np.abs(np.asarray(pt["bbox_xyxy"]) - np.asarray(ox["bbox_xyxy"])))
        )
        conf_diff = abs(float(pt["score"]) - float(ox["score"]))
        kpt_diff = 0.0
        for name in ("a", "b", "c_disc_bottom"):
            kpt_diff = max(
                kpt_diff,
                float(
                    np.max(
                        np.abs(np.asarray(pt["points"][name]) - np.asarray(ox["points"][name]))
                    )
                ),
            )
        max_bbox = max(max_bbox, bbox_diff)
        max_conf = max(max_conf, conf_diff)
        max_kpt = max(max_kpt, kpt_diff)
        if bbox_diff > bbox_atol:
            failures.append(f"detection {idx} bbox drift {bbox_diff:.4f}px > {bbox_atol:.4f}px")
        if kpt_diff > kpt_atol:
            failures.append(f"detection {idx} keypoint drift {kpt_diff:.4f}px > {kpt_atol:.4f}px")
        if conf_diff > conf_atol:
            failures.append(f"detection {idx} conf drift {conf_diff:.6f} > {conf_atol:.6f}")

    return (
        {
            "matched": not failures,
            "n_pytorch": len(pt_dets),
            "n_onnx": len(onnx_dets),
            "max_bbox_drift_px": max_bbox,
            "max_keypoint_drift_px": max_kpt,
            "max_conf_drift": max_conf,
            "failures": failures,
        },
        onnx_dets,
    )


def _raw_to_pred_dict(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: raw[name] for name in OUTPUT_NAMES}


def _decoded_tensor_report(raw: dict[str, torch.Tensor]) -> dict[str, list[int]]:
    decoded = decode_predictions(_raw_to_pred_dict(raw), stride=FEATURE_STRIDE)
    return {name: list(value.shape) for name, value in decoded.items()}


def _detections_to_preview_payload(
    detections: list[dict[str, Any]],
    frame_id: str,
) -> dict[str, Any]:
    wheels = []
    for det in detections:
        wheels.append(
            {
                "bbox_xyxy": [float(v) for v in det["bbox_xyxy"]],
                "confidence": float(det["score"]),
                "points": {
                    name: [float(v) for v in det["points"][name]]
                    for name in ("a", "b", "c_disc_bottom")
                },
            }
        )
    return {"frame_id": frame_id, "wheels": wheels}


def write_reports(report: dict[str, Any], out_dir: Path) -> None:
    (out_dir / "export_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# MobileNetV2 ONNX Export Report",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- ONNX: `{report['onnx_path']}`",
        f"- Model status: **{report['model_status']}**",
        f"- Raw parity: **{report['raw_parity']['matched']}** "
        f"(max abs diff {report['raw_parity']['max_abs_diff']:.6g})",
        f"- Decoded parity: **{report['decoded_parity']['matched']}**",
        f"- PyTorch detections: {report['decoded_parity']['n_pytorch']}",
        f"- ONNX detections: {report['decoded_parity']['n_onnx']}",
        f"- Max bbox drift: {report['decoded_parity']['max_bbox_drift_px']:.4f}px",
        f"- Max keypoint drift: {report['decoded_parity']['max_keypoint_drift_px']:.4f}px",
        f"- Max confidence drift: {report['decoded_parity']['max_conf_drift']:.6f}",
        "",
        "## Failures",
        "",
    ]
    failures = report["raw_parity"]["failures"] + report["decoded_parity"]["failures"]
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- none")
    lines += [
        "",
        "Note: this ONNX artifact is still provisional if the checkpoint was trained "
        "on dirty 0003 data. It is an export/parity artifact, not production approval.",
        "",
    ]
    (out_dir / "export_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if args.imgsz % FEATURE_STRIDE != 0:
        raise ValueError(f"--imgsz must be divisible by {FEATURE_STRIDE}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device)
    onnx_path = args.out_dir / f"{args.name}.onnx"
    export_onnx(model, onnx_path, imgsz=args.imgsz, opset=args.opset, device=device)

    image_tensor, sample_bgr = load_sample_tensor(args.sample_image, args.imgsz)
    pt_raw = pytorch_raw_outputs(model, image_tensor, device)
    onnx_raw = onnx_raw_outputs(onnx_path, image_tensor)
    raw_parity = raw_parity_report(pt_raw, onnx_raw, raw_atol=args.raw_atol)
    decoded_parity, onnx_dets = decoded_parity_report(
        pt_raw,
        onnx_raw,
        conf=args.conf,
        nms_iou=args.nms_iou,
        max_det=args.max_det,
        imgsz=args.imgsz,
        bbox_atol=args.bbox_atol,
        kpt_atol=args.kpt_atol,
        conf_atol=args.conf_atol,
    )

    preview_path = args.out_dir / f"{args.name}_onnx_pred.jpg"
    write_preview(
        cv2.resize(sample_bgr, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR),
        _detections_to_preview_payload(onnx_dets, args.sample_image.stem),
        preview_path,
    )

    report = {
        "checkpoint": str(args.checkpoint),
        "onnx_path": str(onnx_path),
        "sample_image": str(args.sample_image),
        "preview": str(preview_path),
        "imgsz": args.imgsz,
        "opset": args.opset,
        "device": str(device),
        "thresholds": {
            "conf": args.conf,
            "nms_iou": args.nms_iou,
            "max_det": args.max_det,
            "bbox_atol": args.bbox_atol,
            "kpt_atol": args.kpt_atol,
            "conf_atol": args.conf_atol,
            "raw_atol": args.raw_atol,
        },
        "model_status": DEFAULT_MODEL_STATUS,
        "raw_output_shapes": {name: list(value.shape) for name, value in pt_raw.items()},
        "decoded_output_shapes": _decoded_tensor_report(pt_raw),
        "raw_parity": raw_parity,
        "decoded_parity": decoded_parity,
    }
    write_reports(report, args.out_dir)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    print(f"ONNX:           {report['onnx_path']}")
    print(f"Export report:  {args.out_dir / 'export_report.md'}")
    print(f"Preview:        {report['preview']}")
    print(f"Raw parity:     {report['raw_parity']['matched']}")
    print(f"Decoded parity: {report['decoded_parity']['matched']}")
    return 0 if report["raw_parity"]["matched"] and report["decoded_parity"]["matched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
