"""Tests for the raw Unreal-export acceptance runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import accept_unreal_export as accept  # noqa: E402


def _kp_text(right, left, center, left_top, right_top) -> str:
    return (
        "{\n"
        f'{{name:"Right",XY:{right[0]},{right[1]}\n}},\n'
        f'{{name:"Left",XY:{left[0]},{left[1]}\n}},\n'
        f'{{name:"Center",XY:{center[0]},{center[1]}\n}},\n'
        f'{{name:"LeftTop",XY:{left_top[0]},{left_top[1]}\n}},\n'
        f'{{name:"RightTop",XY:{right_top[0]},{right_top[1]}\n}}\n}}'
    )


def _build_fake_0002_export(root: Path) -> None:
    (root / "Images").mkdir(parents=True)
    (root / "keyPoint" / "0").mkdir(parents=True)
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200
    assert cv2.imwrite(str(root / "Images" / "0.jpg"), img)
    (root / "keyPoint" / "0" / "0.txt").write_text(
        _kp_text(
            (100.0, 420.0),
            (300.0, 420.0),
            (200.0, 330.0),
            (310.0, 120.0),
            (90.0, 120.0),
        ),
        encoding="utf-8",
    )
    (root / "keyPoint" / "0" / "1.txt").write_text(
        _kp_text(
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
        ),
        encoding="utf-8",
    )


def test_accept_unreal_export_runner_end_to_end(tmp_path: Path) -> None:
    source = tmp_path / "0002"
    _build_fake_0002_export(source)
    out_root = tmp_path / "acceptance"

    rc = accept.main(
        [
            "--source-root",
            str(source),
            "--source-name",
            "0002_trial",
            "--out-root",
            str(out_root),
            "--preview-count",
            "1",
            "--overwrite",
        ]
    )

    assert rc == 0
    report_path = out_root / "unreal_0002_trial" / "acceptance_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["technical_status"] == "PASS"
    assert report["review_status"] == "READY_FOR_HUMAN_PREVIEW"
    assert report["training_status"].startswith("NOT_APPROVED_FOR_TRAINING")
    assert report["data_quality_gate"]["passed"] is False
    assert "usable_ratio" in report["data_quality_gate"]["metrics"]
    assert report["raw_export"]["images"] == 1
    assert report["import"]["valid_wheels"] == 1
    assert report["import"]["bbox_strategy_counts"]["top_points"] == 1
    assert report["import"]["drop_counts"]["all_zero"] == 1
    assert report["conversion"]["wheels"] == 1
    assert (out_root / "unreal_0002_trial" / "acceptance_report.md").is_file()


def test_accept_unreal_export_can_fail_on_data_quality_gate(tmp_path: Path) -> None:
    source = tmp_path / "0002"
    _build_fake_0002_export(source)
    out_root = tmp_path / "acceptance"

    rc = accept.main(
        [
            "--source-root",
            str(source),
            "--source-name",
            "0002_trial_fail_gate",
            "--out-root",
            str(out_root),
            "--preview-count",
            "1",
            "--overwrite",
            "--fail-on-data-quality-gate",
        ]
    )

    assert rc == 1
    report_path = (
        out_root / "unreal_0002_trial_fail_gate" / "acceptance_report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["technical_status"] == "PASS"
    assert report["data_quality_gate"]["passed"] is False
