"""Tests for train_yolo dataset preflight gates."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import train_yolo


VALID_FLOORRAY_LABEL = (
    "0 0.5 0.5 0.2 0.4 "
    "0.42 0.66 2 0.58 0.66 2 0.5 0.58 2\n"
)
LEGACY_RIM_LABEL = (
    "0 0.5 0.5 0.2 0.4 "
    "0.42 0.45 2 0.58 0.45 2 0.5 0.58 2\n"
)


def _write_dataset(root: Path, label_line: str) -> None:
    for split in ("train", "val"):
        img_dir = root / "images" / split
        label_dir = root / "labels" / split
        img_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        (img_dir / f"{split}_0.jpg").write_bytes(b"not decoded by preflight")
        (label_dir / f"{split}_0.txt").write_text(label_line, encoding="utf-8")


def _write_data_yaml(path: Path, dataset_root: Path) -> None:
    path.write_text(
        "\n".join(
            [
                f"path: {dataset_root}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: wheel",
                "kpt_shape: [3, 3]",
                "flip_idx: [1, 0, 2]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_training_preflight_accepts_valid_floorray_dataset(tmp_path: Path) -> None:
    dataset_root = tmp_path / "ds"
    data_yaml = tmp_path / "data.yaml"
    _write_dataset(dataset_root, VALID_FLOORRAY_LABEL)
    _write_data_yaml(data_yaml, dataset_root)

    resolved = train_yolo.validate_training_dataset_or_raise(data_yaml)

    assert resolved == dataset_root


def test_training_preflight_rejects_legacy_rim_geometry(tmp_path: Path) -> None:
    dataset_root = tmp_path / "ds"
    data_yaml = tmp_path / "data.yaml"
    _write_dataset(dataset_root, LEGACY_RIM_LABEL)
    _write_data_yaml(data_yaml, dataset_root)

    with pytest.raises(SystemExit) as exc:
        train_yolo.validate_training_dataset_or_raise(data_yaml)

    assert "floor-ray" in str(exc.value)


def test_main_aborts_before_yolo_when_dataset_invalid(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_root = tmp_path / "ds"
    data_yaml = tmp_path / "data.yaml"
    _write_dataset(dataset_root, LEGACY_RIM_LABEL)
    _write_data_yaml(data_yaml, dataset_root)

    class _YOLOShouldNotBeCalled:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("YOLO must not be constructed before preflight passes")

    monkeypatch.setattr(train_yolo, "YOLO", _YOLOShouldNotBeCalled)

    with pytest.raises(SystemExit):
        train_yolo.main(
            [
                "--data",
                str(data_yaml),
                "--model",
                "yolo11n-pose.pt",
                "--epochs",
                "1",
                "--project",
                str(tmp_path / "runs"),
            ]
        )


def test_main_keeps_train_and_val_outputs_under_project(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_root = tmp_path / "ds"
    data_yaml = tmp_path / "data.yaml"
    runs_root = tmp_path / "runs"
    _write_dataset(dataset_root, VALID_FLOORRAY_LABEL)
    _write_data_yaml(data_yaml, dataset_root)

    calls: dict[str, dict] = {}

    class _FakeYOLO:
        def __init__(self, model_path: str):
            calls["model_path"] = {"value": model_path}

        def train(self, **kwargs):
            calls["train"] = kwargs

        def val(self, **kwargs):
            calls["val"] = kwargs
            return {"ok": True}

    monkeypatch.setattr(train_yolo, "YOLO", _FakeYOLO)

    train_yolo.main(
        [
            "--data",
            str(data_yaml),
            "--model",
            "yolo11n-pose.pt",
            "--epochs",
            "1",
            "--project",
            str(runs_root),
            "--name",
            "angle_demo",
        ]
    )

    assert calls["train"]["project"] == str(runs_root.resolve())
    assert calls["train"]["name"] == "angle_demo"
    assert calls["val"]["project"] == str(runs_root.resolve())
    assert calls["val"]["name"] == "angle_demo_val"
