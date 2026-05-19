"""Tests for MobileNetV2 diagnostic eval and preview tooling."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import eval_mobilenetv2_skipless as evaler  # noqa: E402
from src.models.mobilenetv2_skipless_pose import MobileNetV2SkiplessPose  # noqa: E402


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((80, 96, 3), dtype=np.uint8)
    image[:, :] = color
    ok = cv2.imwrite(str(path), image)
    assert ok


def _make_eval_dataset(root: Path) -> Path:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    _write_image(root / "images" / "val" / "labelled.jpg", (20, 40, 80))
    (root / "labels" / "val" / "labelled.txt").write_text(
        "0 0.5 0.5 0.5 0.5 0.3 0.5 2 0.7 0.5 2 0.5 0.75 2\n",
        encoding="utf-8",
    )

    _write_image(root / "images" / "val" / "empty.jpg", (80, 40, 20))
    (root / "labels" / "val" / "empty.txt").write_text("", encoding="utf-8")
    return root


def _write_checkpoint(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = MobileNetV2SkiplessPose(pretrained=False)
    torch.save({"model_state_dict": model.state_dict(), "epoch": 1}, path)
    return path


def test_load_model_from_checkpoint_on_cpu(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "weights" / "last.pt")

    model = evaler.load_model(checkpoint, torch.device("cpu"))

    assert isinstance(model, MobileNetV2SkiplessPose)
    assert not model.training


def test_eval_writes_report_predictions_and_previews(tmp_path: Path) -> None:
    dataset_root = _make_eval_dataset(tmp_path / "pose_dataset")
    checkpoint = _write_checkpoint(tmp_path / "weights" / "last.pt")
    out_dir = tmp_path / "eval"

    exit_code = evaler.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--dataset-root",
            str(dataset_root),
            "--split",
            "val",
            "--device",
            "cpu",
            "--imgsz",
            "64",
            "--conf",
            "0.0",
            "--nms-iou",
            "0.5",
            "--max-det",
            "5",
            "--preview-count",
            "2",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    report_path = out_dir / "eval_report.json"
    md_path = out_dir / "eval_report.md"
    predictions_path = out_dir / "predictions.jsonl"
    preview_paths = sorted((out_dir / "previews").glob("*.jpg"))

    assert report_path.exists()
    assert md_path.exists()
    assert predictions_path.exists()
    assert len(preview_paths) == 2

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["checkpoint"] == str(checkpoint)
    assert report["dataset_root"] == str(dataset_root)
    assert report["split"] == "val"
    assert report["image_count"] == 2
    assert report["gt_wheel_count"] == 1
    assert report["empty_label_image_count"] == 1
    assert report["prediction_count"] >= 0
    assert report["false_positive_empty_label_count"] >= 0
    assert set(report["mean_keypoint_error_px"]) == {"a", "b", "c_disc_bottom"}
    assert math.isfinite(report["precision"])
    assert math.isfinite(report["recall"])
    assert math.isfinite(report["mean_iou"])

    lines = predictions_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert {"image", "gt", "predictions", "matches"}.issubset(first)
