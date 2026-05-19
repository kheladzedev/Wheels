"""Tests for MobileNetV2 inference/export tooling."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import predict_mobilenetv2_skipless as predictor  # noqa: E402
from src.models.mobilenetv2_skipless_pose import MobileNetV2SkiplessPose  # noqa: E402


def _write_image(
    path: Path,
    *,
    width: int = 128,
    height: int = 96,
    color: tuple[int, int, int] = (20, 40, 80),
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = color
    ok = cv2.imwrite(str(path), image)
    assert ok
    return path


def _write_checkpoint(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = MobileNetV2SkiplessPose(pretrained=False)
    torch.save({"model_state_dict": model.state_dict(), "epoch": 1}, path)
    return path


def _model_space_detection() -> dict:
    return {
        "score": 0.87,
        "bbox_xyxy": [10.0, 10.0, 50.0, 50.0],
        "points": {
            "a": [15.0, 45.0],
            "b": [45.0, 45.0],
            "c_disc_bottom": [32.0, 35.0],
        },
        "visibility": {"a": 0.93, "b": 0.92, "c_disc_bottom": 0.91},
    }


def test_load_model_from_checkpoint_on_cpu(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "weights" / "last.pt")

    model = predictor.load_model(checkpoint, torch.device("cpu"))

    assert isinstance(model, MobileNetV2SkiplessPose)
    assert not model.training


def test_scale_detections_to_original_pixels() -> None:
    scaled = predictor.scale_detections_to_original(
        [_model_space_detection()],
        original_width=128,
        original_height=96,
        imgsz=64,
    )

    det = scaled[0]
    assert det["bbox_xyxy"] == [20.0, 15.0, 100.0, 75.0]
    assert det["points"]["a"] == [30.0, 67.5]
    assert det["points"]["b"] == [90.0, 67.5]
    assert det["points"]["c_disc_bottom"] == [64.0, 52.5]


def test_detections_to_confirmed_payload_shape() -> None:
    scaled = predictor.scale_detections_to_original(
        [_model_space_detection()],
        original_width=128,
        original_height=96,
        imgsz=64,
    )

    payload = predictor.detections_to_confirmed_payload(
        scaled,
        frame_id="frame_001",
        conf=0.3,
    )

    assert set(payload.keys()) == {"frame_id", "wheels"}
    assert payload["frame_id"] == "frame_001"
    assert len(payload["wheels"]) == 1
    wheel = payload["wheels"][0]
    assert set(wheel.keys()) == {"bbox_xyxy", "confidence", "points"}
    assert set(wheel["points"].keys()) == {"a", "b", "c_disc_bottom"}
    assert wheel["bbox_xyxy"] == [20.0, 15.0, 100.0, 75.0]
    assert wheel["confidence"] == pytest.approx(0.87)


def test_main_single_image_writes_confirmed_json_summary_and_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "weights" / "last.pt")
    image = _write_image(tmp_path / "input" / "frame_001.jpg")
    out_dir = tmp_path / "pred"

    monkeypatch.setattr(
        predictor,
        "decode_model_detections",
        lambda *args, **kwargs: [_model_space_detection()],
    )

    exit_code = predictor.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--source",
            str(image),
            "--device",
            "cpu",
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
    payload_path = out_dir / "frame_001.json"
    summary_path = out_dir / "run_summary.json"
    jsonl_path = out_dir / "predictions.jsonl"
    preview_path = out_dir / "previews" / "frame_001_mn2_pred.jpg"

    assert payload_path.exists()
    assert summary_path.exists()
    assert jsonl_path.exists()
    assert preview_path.exists()

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert payload["frame_id"] == "frame_001"
    assert set(payload["wheels"][0]["points"]) == {"a", "b", "c_disc_bottom"}

    jsonl_rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert jsonl_rows == [payload]

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["model_status"] == predictor.MODEL_STATUS
    assert summary["image_count"] == 1
    assert summary["prediction_count"] == 1
    assert summary["raw_detection_count"] == 1
    assert summary["confirmed_dropped_count"] == 0
    assert summary["empty_prediction_count"] == 0
    assert summary["preview_count"] == 1
    assert summary["frame_index"][0]["json"] == str(payload_path)
    assert summary["frame_index"][0]["confirmed_dropped_count"] == 0


def test_main_directory_input_and_empty_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "weights" / "last.pt")
    source = tmp_path / "images"
    _write_image(source / "a.jpg", color=(20, 40, 80))
    _write_image(source / "b.png", color=(80, 40, 20))
    out_dir = tmp_path / "pred"

    monkeypatch.setattr(
        predictor,
        "decode_model_detections",
        lambda *args, **kwargs: [],
    )

    exit_code = predictor.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--source",
            str(source),
            "--device",
            "cpu",
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
    assert json.loads((out_dir / "a.json").read_text(encoding="utf-8")) == {
        "frame_id": "a",
        "wheels": [],
    }
    assert json.loads((out_dir / "b.json").read_text(encoding="utf-8")) == {
        "frame_id": "b",
        "wheels": [],
    }
    previews = sorted((out_dir / "previews").glob("*_mn2_pred.jpg"))
    assert len(previews) == 2

    summary = json.loads((out_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["image_count"] == 2
    assert summary["prediction_count"] == 0
    assert summary["raw_detection_count"] == 0
    assert summary["confirmed_dropped_count"] == 0
    assert summary["empty_prediction_count"] == 2
