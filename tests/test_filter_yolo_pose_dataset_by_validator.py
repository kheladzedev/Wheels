from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import filter_yolo_pose_dataset_by_validator as strict_filter  # noqa: E402
from check_yolo_pose_dataset import validate_label_file  # noqa: E402


VALID_LABEL = (
    "0 0.5 0.5 0.5 0.5 "
    "0.3 0.9 2 0.7 0.9 2 0.5 0.7 2\n"
)
INVALID_LABEL = (
    "0 0.5 0.5 0.5 0.5 "
    "0.45 0.7 2 0.55 0.7 2 0.5 0.55 2\n"
)


def _write_image(path: Path, value: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((24, 32, 3), value, dtype=np.uint8))


def _write_label(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_filter_dataset_drops_only_invalid_label_lines(tmp_path: Path) -> None:
    source = tmp_path / "source"
    out = tmp_path / "strict"

    for split in ("train", "val"):
        _write_image(source / "images" / split / f"{split}_mixed.jpg")
        _write_label(
            source / "labels" / split / f"{split}_mixed.txt",
            VALID_LABEL + INVALID_LABEL,
        )

    summary = strict_filter.build_filtered_dataset(
        source_root=source,
        output_root=out,
        overwrite=False,
    )

    assert summary["ok"] is True
    assert summary["totals"]["source_wheel_labels"] == 4
    assert summary["totals"]["kept_wheel_labels"] == 2
    assert summary["totals"]["dropped_wheel_labels"] == 2
    assert summary["by_split"]["train"]["dropped_wheel_labels"] == 1

    for split in ("train", "val"):
        label_path = out / "labels" / split / f"{split}_mixed.txt"
        assert label_path.read_text(encoding="utf-8") == VALID_LABEL
        assert validate_label_file(label_path) == []

    report = json.loads(
        (out / "metadata" / "strict_filter_report.json").read_text(encoding="utf-8")
    )
    assert report["totals"]["dropped_wheel_labels"] == 2


def test_filter_dataset_writes_empty_label_when_all_lines_invalid(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    out = tmp_path / "strict"

    for split in ("train", "val"):
        _write_image(source / "images" / split / f"{split}_bad.jpg")
        _write_label(source / "labels" / split / f"{split}_bad.txt", INVALID_LABEL)

    summary = strict_filter.build_filtered_dataset(
        source_root=source,
        output_root=out,
        overwrite=False,
    )

    assert summary["totals"]["images_without_valid_labels"] == 2
    assert summary["totals"]["kept_wheel_labels"] == 0
    assert (out / "labels" / "train" / "train_bad.txt").read_text() == ""


def test_filter_dataset_can_write_dataset_config(tmp_path: Path) -> None:
    source = tmp_path / "source"
    out = tmp_path / "strict"
    config = tmp_path / "configs" / "pose_dataset_strict.yaml"

    for split in ("train", "val"):
        _write_image(source / "images" / split / f"{split}_ok.jpg")
        _write_label(source / "labels" / split / f"{split}_ok.txt", VALID_LABEL)

    rc = strict_filter.main(
        [
            "--source-root",
            str(source),
            "--dataset-root",
            str(out),
            "--config-out",
            str(config),
        ]
    )

    assert rc == 0
    assert config.is_file()
    assert f"path: {out}" in config.read_text(encoding="utf-8")
