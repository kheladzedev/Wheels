"""Smoke tests for the legacy YOLO-pose dataset checker."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from check_dataset import (
    FIELDS_PER_LINE,
    main as check_main,
    validate_label_file,
)


def _make_dataset(
    root: Path,
    *,
    splits: tuple[str, ...] = ("train", "val"),
    n_per_split: int = 2,
    label_line: str | None = None,
) -> None:
    default_line = " ".join(
        ["0", "0.5", "0.5", "0.2", "0.4"]
        + ["0.42", "0.66", "2", "0.58", "0.66", "2", "0.5", "0.58", "2"]
    )
    line = default_line if label_line is None else label_line
    for split in splits:
        img_dir = root / "images" / split
        lab_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_split):
            stem = f"sample_{split}_{i}"
            img_path = img_dir / f"{stem}.jpg"
            ok = cv2.imwrite(str(img_path), np.full((48, 64, 3), 128, dtype=np.uint8))
            assert ok
            (lab_dir / f"{stem}.txt").write_text(
                line + ("\n" if line else ""),
                encoding="utf-8",
            )


def _run_check(root: Path) -> int:
    with patch.object(sys, "argv", ["check_dataset.py", "--dataset-root", str(root)]):
        return check_main()


def test_valid_dataset_passes(tmp_path: Path):
    _make_dataset(tmp_path / "ds")
    assert _run_check(tmp_path / "ds") == 0


def test_label_with_wrong_field_count_fails(tmp_path: Path):
    root = tmp_path / "ds"
    short = " ".join(
        ["0", "0.5", "0.5", "0.2", "0.4"]
        + ["0.4", "0.45", "2", "0.6", "0.45", "2"]
    )
    _make_dataset(root, label_line=short)
    assert _run_check(root) == 1


def test_validate_label_file_directly_reports_expected_problems(tmp_path: Path):
    lbl = tmp_path / "bad.txt"
    lbl.write_text(
        "0 0.5 0.5 0.2 0.4 0.4 0.45 2 0.6 0.45 2\n",
        encoding="utf-8",
    )
    problems = validate_label_file(lbl)
    assert any("expected" in p and str(FIELDS_PER_LINE) in p for p in problems)


def test_orphan_label_without_image_fails(tmp_path: Path):
    root = tmp_path / "ds"
    _make_dataset(root)
    stray = root / "labels" / "train" / "stray.txt"
    stray.write_text(
        "0 0.5 0.5 0.2 0.4 0.42 0.66 2 0.58 0.66 2 0.5 0.58 2\n",
        encoding="utf-8",
    )
    assert _run_check(root) == 1


def test_image_without_label_fails(tmp_path: Path):
    root = tmp_path / "ds"
    _make_dataset(root)
    lonely = root / "images" / "val" / "lonely.jpg"
    cv2.imwrite(str(lonely), np.full((48, 64, 3), 128, dtype=np.uint8))
    assert _run_check(root) == 1


def test_label_with_legacy_rim_edge_geometry_fails(tmp_path: Path):
    root = tmp_path / "ds"
    bad = " ".join(
        ["0", "0.5", "0.5", "0.2", "0.4"]
        + ["0.42", "0.45", "2", "0.58", "0.45", "2", "0.5", "0.58", "2"]
    )
    _make_dataset(root, label_line=bad)
    assert _run_check(root) == 1
