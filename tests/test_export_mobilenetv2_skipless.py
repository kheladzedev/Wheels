"""Tests for MobileNetV2 ONNX export/parity tooling."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import export_mobilenetv2_skipless as exporter  # noqa: E402
from src.models.mobilenetv2_skipless_pose import MobileNetV2SkiplessPose  # noqa: E402


def _write_checkpoint(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = MobileNetV2SkiplessPose(pretrained=False)
    torch.save({"model_state_dict": model.state_dict(), "epoch": 1}, path)
    return path


def _write_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((80, 96, 3), dtype=np.uint8)
    image[:, :] = (40, 80, 120)
    assert cv2.imwrite(str(path), image)
    return path


def test_raw_parity_report_detects_matching_outputs() -> None:
    pt = {
        "cls": torch.zeros(1, 1, 2, 2),
        "bbox": torch.ones(1, 4, 2, 2),
        "kpt": torch.ones(1, 6, 2, 2) * 2,
        "vis": torch.ones(1, 3, 2, 2) * 3,
    }
    ox = {name: value.clone() for name, value in pt.items()}

    report = exporter.raw_parity_report(pt, ox, raw_atol=1e-6)

    assert report["matched"] is True
    assert report["max_abs_diff"] == 0.0
    assert set(report["outputs"]) == {"cls", "bbox", "kpt", "vis"}


def test_decoded_parity_handles_empty_detections() -> None:
    raw = {
        "cls": torch.full((1, 1, 2, 2), -20.0),
        "bbox": torch.zeros(1, 4, 2, 2),
        "kpt": torch.zeros(1, 6, 2, 2),
        "vis": torch.zeros(1, 3, 2, 2),
    }

    report, detections = exporter.decoded_parity_report(
        raw,
        {name: value.clone() for name, value in raw.items()},
        conf=0.99,
        nms_iou=0.5,
        max_det=5,
        imgsz=64,
        bbox_atol=2.0,
        kpt_atol=3.0,
        conf_atol=0.05,
    )

    assert report["matched"] is True
    assert report["n_pytorch"] == 0
    assert report["n_onnx"] == 0
    assert detections == []


def test_export_main_writes_onnx_report_and_preview(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "weights" / "last.pt")
    sample = _write_image(tmp_path / "sample.jpg")
    out_dir = tmp_path / "export"

    rc = exporter.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--sample-image",
            str(sample),
            "--imgsz",
            "64",
            "--device",
            "cpu",
            "--conf",
            "0.99",
            "--nms-iou",
            "0.5",
            "--max-det",
            "3",
            "--out-dir",
            str(out_dir),
            "--name",
            "tiny_mn2",
        ]
    )

    assert rc == 0
    onnx_path = out_dir / "tiny_mn2.onnx"
    report_path = out_dir / "export_report.json"
    md_path = out_dir / "export_report.md"
    preview_path = out_dir / "tiny_mn2_onnx_pred.jpg"
    assert onnx_path.exists()
    assert report_path.exists()
    assert md_path.exists()
    assert preview_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["onnx_path"] == str(onnx_path)
    assert report["raw_parity"]["matched"] is True
    assert report["decoded_parity"]["matched"] is True
    assert set(report["raw_output_shapes"]) == {"cls", "bbox", "kpt", "vis"}
    assert "MobileNetV2 ONNX Export Report" in md_path.read_text(encoding="utf-8")
