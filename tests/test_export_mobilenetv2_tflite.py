"""Tests for guarded MobileNetV2 TFLite/LiteRT export tooling."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import export_mobilenetv2_tflite as tflite_exporter  # noqa: E402


def _write_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((80, 96, 3), dtype=np.uint8)
    image[:, :] = (30, 60, 90)
    assert cv2.imwrite(str(path), image)
    return path


def _args(tmp_path: Path) -> argparse.Namespace:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake-onnx")
    return argparse.Namespace(
        onnx_path=onnx_path,
        checkpoint=None,
        sample_image=_write_image(tmp_path / "sample.jpg"),
        out_dir=tmp_path / "tflite",
        name="tiny_mn2",
        converter_python=Path(sys.executable),
        imgsz=64,
        input_name="images",
        opset=17,
        conf=0.99,
        nms_iou=0.5,
        max_det=5,
        bbox_atol=2.0,
        kpt_atol=3.0,
        conf_atol=0.05,
        raw_atol=1e-6,
        keep_saved_model=False,
    )


def _raw_outputs() -> dict[str, torch.Tensor]:
    return {
        "cls": torch.full((1, 1, 2, 2), -20.0),
        "bbox": torch.zeros(1, 4, 2, 2),
        "kpt": torch.zeros(1, 6, 2, 2),
        "vis": torch.zeros(1, 3, 2, 2),
    }


def test_module_available_uses_converter_python() -> None:
    assert tflite_exporter.module_available(Path(sys.executable), "json") is True
    assert (
        tflite_exporter.module_available(
            Path(sys.executable), "vsbl_module_that_should_not_exist"
        )
        is False
    )


def test_normalize_tflite_outputs_maps_nchw_and_nhwc(tmp_path: Path) -> None:
    npz_path = tmp_path / "outputs.npz"
    np.savez(
        npz_path,
        output_0=np.zeros((1, 2, 2, 1), dtype=np.float32),
        output_1=np.zeros((1, 4, 2, 2), dtype=np.float32),
        output_2=np.zeros((1, 2, 2, 6), dtype=np.float32),
        output_3=np.zeros((1, 3, 2, 2), dtype=np.float32),
    )

    report, mapped = tflite_exporter.normalize_tflite_outputs(
        npz_path,
        {
            "cls": [1, 1, 2, 2],
            "bbox": [1, 4, 2, 2],
            "kpt": [1, 6, 2, 2],
            "vis": [1, 3, 2, 2],
        },
    )

    assert report["mapped"] is True
    assert {name: list(value.shape) for name, value in mapped.items()} == {
        "cls": [1, 1, 2, 2],
        "bbox": [1, 4, 2, 2],
        "kpt": [1, 6, 2, 2],
        "vis": [1, 3, 2, 2],
    }


def test_run_writes_blocked_report_when_tensorflow_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    monkeypatch.setattr(
        tflite_exporter,
        "dependency_status",
        lambda python: {
            "converter_python": str(python),
            "python_exists": True,
            "tensorflow": False,
            "onnx2tf_module": False,
            "onnx2tf_command": False,
            "ready": False,
            "setup_command": tflite_exporter.setup_command(python),
        },
    )

    report = tflite_exporter.run(args)

    assert report["status"] == "BLOCKED_MISSING_TENSORFLOW"
    assert report["model_status"] == tflite_exporter.DEFAULT_MODEL_STATUS
    assert (args.out_dir / "tflite_export_report.json").exists()
    assert (args.out_dir / "tflite_export_report.md").exists()
    saved = json.loads(
        (args.out_dir / "tflite_export_report.json").read_text(encoding="utf-8")
    )
    assert saved["status"] == "BLOCKED_MISSING_TENSORFLOW"
    assert ".tflite-venv/bin/python" in saved["dependencies"]["setup_command"]
    assert "BLOCKED_MISSING_TENSORFLOW" in (
        args.out_dir / "tflite_export_report.md"
    ).read_text(encoding="utf-8")


def test_run_writes_pass_report_when_converter_and_runtime_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.keep_saved_model = True

    monkeypatch.setattr(
        tflite_exporter,
        "dependency_status",
        lambda python: {
            "converter_python": str(python),
            "python_exists": True,
            "tensorflow": True,
            "onnx2tf_module": True,
            "onnx2tf_command": False,
            "ready": True,
            "setup_command": tflite_exporter.setup_command(python),
        },
    )

    def fake_onnx_to_saved_model(
        *,
        converter_python: Path,
        onnx_path: Path,
        saved_model_dir: Path,
        input_name: str,
        log_path: Path,
    ) -> subprocess.CompletedProcess[str]:
        assert input_name == "images"
        saved_model_dir.mkdir(parents=True, exist_ok=True)
        (saved_model_dir / "saved_model.pb").write_bytes(b"fake")
        (saved_model_dir / "tiny_mn2_float32.tflite").write_bytes(b"fake-tflite")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ok\n", encoding="utf-8")
        return subprocess.CompletedProcess([], 2, "warning but artifact exists\n")

    def fake_saved_model_to_tflite(
        *,
        converter_python: Path,
        saved_model_dir: Path,
        tflite_path: Path,
        log_path: Path,
    ) -> subprocess.CompletedProcess[str]:
        raise AssertionError("onnx2tf-generated TFLite should be reused first")

    def fake_tflite_runtime(
        *,
        converter_python: Path,
        tflite_path: Path,
        sample_tensor_path: Path,
        output_npz_path: Path,
        log_path: Path,
    ) -> subprocess.CompletedProcess[str]:
        raw = _raw_outputs()
        np.savez(
            output_npz_path,
            output_0=raw["cls"].numpy(),
            output_1=raw["bbox"].numpy(),
            output_2=raw["kpt"].numpy(),
            output_3=raw["vis"].numpy(),
        )
        log_path.write_text("ok\n", encoding="utf-8")
        return subprocess.CompletedProcess([], 0, "ok\n")

    monkeypatch.setattr(
        tflite_exporter, "convert_onnx_to_saved_model", fake_onnx_to_saved_model
    )
    monkeypatch.setattr(
        tflite_exporter, "convert_saved_model_to_tflite", fake_saved_model_to_tflite
    )
    monkeypatch.setattr(tflite_exporter, "tflite_raw_outputs", fake_tflite_runtime)
    monkeypatch.setattr(
        tflite_exporter,
        "onnx_raw_outputs",
        lambda *_args, **_kwargs: _raw_outputs(),
    )

    report = tflite_exporter.run(args)

    assert report["status"] == "PASS"
    assert report["warnings"][0] == (
        "onnx2tf returned a non-zero exit code but saved_model.pb exists; "
        "continuing to TFLite runtime parity"
    )
    assert "Using onnx2tf-generated TFLite artifact" in report["warnings"][1]
    assert report["source_tflite_path"].endswith("tiny_mn2_float32.tflite")
    assert report["raw_parity"]["matched"] is True
    assert report["decoded_parity"]["matched"] is True
    assert Path(report["tflite_path"]).exists()
    assert (args.out_dir / "tflite_export_report.json").exists()
    assert (args.out_dir / "tflite_export_report.md").exists()
