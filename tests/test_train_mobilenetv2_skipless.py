"""Tests for the MobileNetV2 real-data training path."""

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

import train_mobilenetv2_skipless as trainer  # noqa: E402


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((80, 96, 3), dtype=np.uint8)
    image[:, :] = color
    ok = cv2.imwrite(str(path), image)
    assert ok


def _make_tiny_dataset(root: Path) -> Path:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    _write_image(root / "images" / "train" / "labelled.jpg", (20, 40, 80))
    (root / "labels" / "train" / "labelled.txt").write_text(
        "0 0.5 0.5 0.5 0.5 0.3 0.5 2 0.7 0.5 2 0.5 0.75 2\n",
        encoding="utf-8",
    )

    _write_image(root / "images" / "val" / "empty.jpg", (80, 40, 20))
    (root / "labels" / "val" / "empty.txt").write_text("", encoding="utf-8")
    return root


def test_yolo_pose_dataset_converts_label_to_pixel_tensors(tmp_path: Path) -> None:
    root = _make_tiny_dataset(tmp_path / "pose_dataset")
    ds = trainer.YoloPoseDataset(root, "train", imgsz=64)

    image, bboxes, keypoints, visibility = ds[0]

    assert image.shape == (3, 64, 64)
    assert image.dtype == torch.float32
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0
    assert bboxes.shape == (1, 4)
    assert keypoints.shape == (1, trainer.N_KEYPOINTS, 2)
    assert visibility.shape == (1, trainer.N_KEYPOINTS)
    assert torch.allclose(bboxes[0], torch.tensor([16.0, 16.0, 48.0, 48.0]))
    assert torch.allclose(
        keypoints[0],
        torch.tensor([[19.2, 32.0], [44.8, 32.0], [32.0, 48.0]]),
        atol=1e-4,
    )
    assert torch.equal(visibility, torch.ones(1, trainer.N_KEYPOINTS))
    assert ds.stats.image_count == 1
    assert ds.stats.labelled_wheel_count == 1
    assert ds.stats.empty_label_count == 0


def test_yolo_pose_dataset_empty_label_is_negative_image(tmp_path: Path) -> None:
    root = _make_tiny_dataset(tmp_path / "pose_dataset")
    ds = trainer.YoloPoseDataset(root, "val", imgsz=64)

    _, bboxes, keypoints, visibility = ds[0]

    assert bboxes.shape == (0, 4)
    assert keypoints.shape == (0, trainer.N_KEYPOINTS, 2)
    assert visibility.shape == (0, trainer.N_KEYPOINTS)
    assert ds.stats.image_count == 1
    assert ds.stats.labelled_wheel_count == 0
    assert ds.stats.empty_label_count == 1


def test_collate_pose_batch_returns_image_batch_and_target_lists(
    tmp_path: Path,
) -> None:
    root = _make_tiny_dataset(tmp_path / "pose_dataset")
    train_item = trainer.YoloPoseDataset(root, "train", imgsz=64)[0]
    val_item = trainer.YoloPoseDataset(root, "val", imgsz=64)[0]

    images, bboxes, keypoints, visibility = trainer.collate_pose_batch(
        [train_item, val_item]
    )

    assert images.shape == (2, 3, 64, 64)
    assert len(bboxes) == 2
    assert len(keypoints) == 2
    assert len(visibility) == 2
    assert bboxes[0].shape == (1, 4)
    assert bboxes[1].shape == (0, 4)


def test_real_data_trainer_smoke_writes_checkpoint_log_and_summary(
    tmp_path: Path,
) -> None:
    root = _make_tiny_dataset(tmp_path / "pose_dataset")
    project = tmp_path / "runs"

    trainer.main(
        [
            "--dataset-root",
            str(root),
            "--epochs",
            "1",
            "--batch",
            "1",
            "--device",
            "cpu",
            "--project",
            str(project),
            "--name",
            "real_smoke",
            "--imgsz",
            "64",
            "--limit-train",
            "1",
            "--limit-val",
            "1",
        ]
    )

    out_dir = project / "real_smoke"
    ckpt = out_dir / "weights" / "last.pt"
    log = out_dir / "train_log.txt"
    summary_path = out_dir / "run_summary.json"

    assert ckpt.exists()
    assert log.exists()
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["mode"] == "real_yolo_pose"
    assert summary["train"]["image_count"] == 1
    assert summary["train"]["labelled_wheel_count"] == 1
    assert summary["val"]["empty_label_count"] == 1
    assert summary["pretrained"] is False
    assert math.isfinite(summary["final_losses"]["train_total"])
    assert math.isfinite(summary["final_losses"]["val_total"])
