"""Tests for MobileNetV2 TFLite/LiteRT runtime smoke inference."""

from __future__ import annotations

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

import predict_mobilenetv2_tflite as tflite_predictor  # noqa: E402


def _write_image(
    path: Path,
    *,
    width: int = 128,
    height: int = 96,
    color: tuple[int, int, int] = (40, 80, 120),
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = color
    assert cv2.imwrite(str(path), image)
    return path


def _write_tflite(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-tflite")
    return path


def _raw_outputs(*, labelled: bool = True) -> dict[str, np.ndarray]:
    cls = np.full((1, 2, 2, 1), -20.0, dtype=np.float32)
    bbox = np.zeros((1, 2, 2, 4), dtype=np.float32)
    kpt = np.zeros((1, 2, 2, 6), dtype=np.float32)
    vis = np.full((1, 2, 2, 3), 10.0, dtype=np.float32)
    if labelled:
        cls[0, 0, 0, 0] = 10.0
        # Cell center is (16, 16) for imgsz=64 and stride=32.
        # BBox decodes to [8, 8, 56, 56].
        bbox[0, 0, 0, :] = [0.25, 0.25, 1.25, 1.25]
        # A=[12,50], B=[44,50], C=[28,40] satisfy confirmed AR geometry.
        kpt[0, 0, 0, :] = [
            (12.0 - 16.0) / 32.0,
            (50.0 - 16.0) / 32.0,
            (44.0 - 16.0) / 32.0,
            (50.0 - 16.0) / 32.0,
            (28.0 - 16.0) / 32.0,
            (40.0 - 16.0) / 32.0,
        ]
    return {
        "output_0": cls,
        "output_1": bbox,
        "output_2": kpt,
        "output_3": vis,
    }


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    labelled_by_stem: dict[str, bool],
) -> None:
    monkeypatch.setattr(
        tflite_predictor,
        "dependency_status",
        lambda python: {
            "converter_python": str(python),
            "python_exists": True,
            "tensorflow": True,
            "onnx2tf_module": True,
            "onnx2tf_command": False,
            "onnx2tf_command_path": None,
            "converter_modules": {},
            "missing_modules": [],
            "ready": True,
            "setup_command": "",
        },
    )

    def fake_tflite_raw_outputs(
        *,
        converter_python: Path,
        tflite_path: Path,
        sample_tensor_path: Path,
        output_npz_path: Path,
        log_path: Path,
    ) -> subprocess.CompletedProcess[str]:
        stem = sample_tensor_path.name.removesuffix("_input.npy")
        np.savez(
            output_npz_path,
            **_raw_outputs(labelled=labelled_by_stem.get(stem, False)),
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ok\n", encoding="utf-8")
        return subprocess.CompletedProcess([], 0, "ok\n")

    monkeypatch.setattr(tflite_predictor, "tflite_raw_outputs", fake_tflite_raw_outputs)


def test_decode_tflite_detections_from_nhwc_outputs() -> None:
    report, raw_np = tflite_predictor.normalize_tflite_outputs(
        _npz_from_arrays(_raw_outputs(labelled=True)),
        tflite_predictor.expected_output_shapes(64),
    )

    detections = tflite_predictor.decode_tflite_detections(
        {name: torch.from_numpy(value) for name, value in raw_np.items()},
        imgsz=64,
        conf=0.3,
        nms_iou=0.5,
        max_det=5,
    )

    assert report["mapped"] is True
    assert len(detections) == 1
    assert detections[0]["bbox_xyxy"] == pytest.approx([8.0, 8.0, 56.0, 56.0])
    assert detections[0]["points"]["a"] == pytest.approx([12.0, 50.0])
    assert detections[0]["points"]["b"] == pytest.approx([44.0, 50.0])
    assert detections[0]["points"]["c_disc_bottom"] == pytest.approx([28.0, 40.0])


def test_main_single_image_writes_confirmed_json_summary_and_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tflite_model = _write_tflite(tmp_path / "model.tflite")
    image = _write_image(tmp_path / "images" / "frame_001.jpg")
    out_dir = tmp_path / "runtime"
    _patch_runtime(monkeypatch, labelled_by_stem={"frame_001": True})

    exit_code = tflite_predictor.main(
        [
            "--tflite-model",
            str(tflite_model),
            "--source",
            str(image),
            "--runtime-python",
            sys.executable,
            "--imgsz",
            "64",
            "--conf",
            "0.3",
            "--nms-iou",
            "0.5",
            "--max-det",
            "5",
            "--preview-count",
            "1",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    payload = json.loads((out_dir / "frame_001.json").read_text(encoding="utf-8"))
    assert set(payload) == {"frame_id", "wheels"}
    assert payload["frame_id"] == "frame_001"
    assert len(payload["wheels"]) == 1
    wheel = payload["wheels"][0]
    assert set(wheel) == {"bbox_xyxy", "confidence", "points"}
    assert set(wheel["points"]) == {"a", "b", "c_disc_bottom"}
    assert wheel["bbox_xyxy"] == pytest.approx([16.0, 12.0, 112.0, 84.0])
    assert wheel["points"]["a"] == pytest.approx([24.0, 75.0])

    summary = json.loads((out_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert summary["model_status"] == tflite_predictor.MODEL_STATUS
    assert summary["image_count"] == 1
    assert summary["prediction_count"] == 1
    assert summary["raw_detection_count"] == 1
    assert summary["confirmed_dropped_count"] == 0
    assert summary["runtime_failure_count"] == 0
    assert summary["predictions_jsonl"] == str(out_dir / "predictions.jsonl")
    assert summary["frame_index"][0]["runtime"]["runtime_returncode"] == 0
    assert (out_dir / "runtime_report.md").exists()
    assert (out_dir / "previews" / "frame_001_mn2_tflite_pred.jpg").exists()


def test_main_directory_input_handles_empty_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tflite_model = _write_tflite(tmp_path / "model.tflite")
    source = tmp_path / "images"
    _write_image(source / "a.jpg", color=(20, 40, 80))
    _write_image(source / "b.png", color=(80, 40, 20))
    out_dir = tmp_path / "runtime"
    _patch_runtime(monkeypatch, labelled_by_stem={"a": True, "b": False})

    exit_code = tflite_predictor.main(
        [
            "--tflite-model",
            str(tflite_model),
            "--source",
            str(source),
            "--runtime-python",
            sys.executable,
            "--imgsz",
            "64",
            "--conf",
            "0.99",
            "--preview-count",
            "2",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    assert json.loads((out_dir / "b.json").read_text(encoding="utf-8")) == {
        "frame_id": "b",
        "wheels": [],
    }
    summary = json.loads((out_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["image_count"] == 2
    assert summary["prediction_count"] == 1
    assert summary["empty_prediction_count"] == 1
    assert summary["runtime_failure_count"] == 0
    assert len(sorted((out_dir / "previews").glob("*_mn2_tflite_pred.jpg"))) == 2


def _npz_from_arrays(arrays: dict[str, np.ndarray]) -> Path:
    import tempfile

    handle = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    handle.close()
    path = Path(handle.name)
    np.savez(path, **arrays)
    return path
