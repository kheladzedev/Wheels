"""Tests for combining accepted YOLO-pose datasets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import combine_yolo_pose_datasets as combiner  # noqa: E402


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[:, :] = color
    assert cv2.imwrite(str(path), image)


def _make_dataset(root: Path, name: str) -> Path:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    _write_image(root / "images" / "train" / f"{name}_labelled.jpg", (20, 40, 60))
    (root / "labels" / "train" / f"{name}_labelled.txt").write_text(
        "0 0.5 0.5 0.5 0.5 0.3 0.9 2 0.7 0.9 2 0.5 0.7 2\n",
        encoding="utf-8",
    )
    _write_image(root / "images" / "train" / f"{name}_empty.jpg", (60, 40, 20))
    (root / "labels" / "train" / f"{name}_empty.txt").write_text("", encoding="utf-8")

    _write_image(root / "images" / "val" / f"{name}_val_labelled.jpg", (30, 50, 70))
    (root / "labels" / "val" / f"{name}_val_labelled.txt").write_text(
        "0 0.5 0.5 0.5 0.5 0.3 0.9 2 0.7 0.9 2 0.5 0.7 2\n",
        encoding="utf-8",
    )
    return root


def test_combine_keeps_labelled_images_and_prefixes_sources(tmp_path: Path) -> None:
    src_a = _make_dataset(tmp_path / "a", "same")
    src_b = _make_dataset(tmp_path / "b", "same")
    out = tmp_path / "combined"

    rc = combiner.main(
        [
            "--source",
            f"source a={src_a}",
            "--source",
            f"source b={src_b}",
            "--dataset-root",
            str(out),
            "--max-empty-ratio",
            "0",
        ]
    )

    assert rc == 0
    train_images = sorted((out / "images" / "train").glob("*.jpg"))
    train_labels = sorted((out / "labels" / "train").glob("*.txt"))
    assert [p.name for p in train_images] == [
        "source_a__same_labelled.jpg",
        "source_b__same_labelled.jpg",
    ]
    assert [p.name for p in train_labels] == [
        "source_a__same_labelled.txt",
        "source_b__same_labelled.txt",
    ]
    assert (out / "data.yaml").is_file()

    report = json.loads((out / "metadata" / "combine_report.json").read_text())
    assert report["status"] == "provisional_combined_not_production"
    assert report["total_images"] == 4
    assert report["total_wheels"] == 4
    assert report["total_empty_images"] == 0
    assert report["by_source"]["source_a"]["images"] == 2
    assert report["by_source"]["source_b"]["images"] == 2


def test_combine_can_keep_capped_empty_labels(tmp_path: Path) -> None:
    src = _make_dataset(tmp_path / "src", "x")
    out = tmp_path / "combined"

    combiner.main(
        [
            "--source",
            f"src={src}",
            "--dataset-root",
            str(out),
            "--max-empty-ratio",
            "0.5",
        ]
    )

    report = json.loads((out / "metadata" / "combine_report.json").read_text())
    assert report["by_split"]["train"]["images"] == 2
    assert report["by_split"]["train"]["empty_images"] == 1
    assert (out / "labels" / "train" / "src__x_empty.txt").read_text() == ""
