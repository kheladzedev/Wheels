from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.dataset_audit import audit_config, build_audit


def _write_image(path: Path, value: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), np.full((24, 32, 3), value, dtype=np.uint8))
    assert ok


def _write_label(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "0 0.5 0.5 0.2 0.3 0.4 0.5 2 0.6 0.5 2 0.5 0.65 2\n",
        encoding="utf-8",
    )


def _make_config(tmp_path: Path, dataset_root: Path) -> Path:
    config = tmp_path / "configs" / "pose_dataset_test.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        "\n".join(
            [
                f"path: {dataset_root}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: wheel",
                "kpt_shape: [3, 3]",
                "flip_idx: [1, 0, 2]",
            ]
        ),
        encoding="utf-8",
    )
    return config


def test_dataset_audit_passes_clean_dataset(tmp_path):
    root = tmp_path / "dataset"
    for split in ("train", "val"):
        stem = f"{split}_a"
        _write_image(root / "images" / split / f"{stem}.jpg", value=20 if split == "train" else 40)
        _write_label(root / "labels" / split / f"{stem}.txt")
    config = _make_config(tmp_path, root)

    report = audit_config(config, image_sample_limit=10)

    assert report["ok"] is True
    assert report["splits"]["train"]["images"] == 1
    assert report["splits"]["val"]["wheel_labels"] == 1
    assert report["leakage"]["hash_overlap_count"] == 0


def test_dataset_audit_detects_train_val_hash_leakage(tmp_path):
    root = tmp_path / "dataset"
    image = np.full((24, 32, 3), 99, dtype=np.uint8)
    for split in ("train", "val"):
        path = root / "images" / split / f"{split}_different_name.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(path), image)
        _write_label(root / "labels" / split / f"{split}_different_name.txt")
    config = _make_config(tmp_path, root)

    report = audit_config(config, image_sample_limit=10)

    assert report["ok"] is False
    assert "train_val_hash_overlap:1" in report["failures"]


def test_dataset_audit_detects_bad_label_schema(tmp_path):
    root = tmp_path / "dataset"
    for split in ("train", "val"):
        stem = f"{split}_a"
        _write_image(root / "images" / split / f"{stem}.jpg")
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split / f"{stem}.txt").write_text("0 0.5 0.5\n", encoding="utf-8")
    config = _make_config(tmp_path, root)

    report = build_audit([config], image_sample_limit=10)

    assert report["ok"] is False
    assert report["counts"]["failed"] == 1
    assert any("label_errors" in failure for failure in report["reports"][0]["failures"])


def test_dataset_audit_detects_stale_conversion_report_source_count(tmp_path):
    source = tmp_path / "incoming"
    root = tmp_path / "dataset"
    for idx in range(2):
        _write_image(source / "images" / f"frame_{idx}.jpg")
    for split in ("train", "val"):
        stem = f"{split}_a"
        _write_image(root / "images" / split / f"{stem}.jpg")
        _write_label(root / "labels" / split / f"{stem}.txt")
    metadata = root / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "conversion_report.json").write_text(
        (
            '{"source_root": "'
            + str(source)
            + '", "source_images": 1, "converted_images": 2, '
            + '"wheels": 2, "quality_gate": {"passed": true}}'
        ),
        encoding="utf-8",
    )
    config = _make_config(tmp_path, root)

    report = audit_config(config, image_sample_limit=10)

    assert report["ok"] is False
    assert any("conversion_source_image_count_mismatch:1!=2" in f for f in report["failures"])


def test_dataset_audit_detects_stale_conversion_report_dataset_count(tmp_path):
    source = tmp_path / "incoming"
    root = tmp_path / "dataset"
    for split in ("train", "val"):
        stem = f"{split}_a"
        _write_image(source / "images" / f"{stem}.jpg")
        _write_image(root / "images" / split / f"{stem}.jpg")
        _write_label(root / "labels" / split / f"{stem}.txt")
    metadata = root / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "conversion_report.json").write_text(
        (
            '{"source_root": "'
            + str(source)
            + '", "source_images": 2, "converted_images": 1, '
            + '"wheels": 2, "quality_gate": {"passed": true}}'
        ),
        encoding="utf-8",
    )
    config = _make_config(tmp_path, root)

    report = audit_config(config, image_sample_limit=10)

    assert report["ok"] is False
    assert any("conversion_dataset_image_count_mismatch:1!=2" in f for f in report["failures"])
