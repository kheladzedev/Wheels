"""Tests for the NeuralData1 Unreal capture acceptance wrapper."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import accept_neuraldata1_capture as capture  # noqa: E402


def _kp_text() -> str:
    return (
        "{\n"
        '{name:"SphereRight",XY:300,420\n},\n'
        '{name:"SphereLeft",XY:100,420\n},\n'
        '{name:"Center",XY:200,330\n},\n'
        '{name:"SphereLeftTop",XY:90,120\n},\n'
        '{name:"SphereRightTop",XY:310,120\n}\n}'
    )


def _make_project(root: Path, *, with_export: bool) -> None:
    root.mkdir(parents=True)
    (root / "NeuralData.uproject").write_text("{}", encoding="utf-8")
    for name in ("Images", "keyPoint", "Depth", "Goal", "Content"):
        (root / name).mkdir()
    (root / "Content" / "legacy.uasset").write_bytes(b"not copied")

    if not with_export:
        return

    (root / "keyPoint" / "0").mkdir()
    image = np.ones((480, 640, 3), dtype=np.uint8) * 200
    assert cv2.imwrite(str(root / "Images" / "0.jpg"), image)
    (root / "keyPoint" / "0" / "0.txt").write_text(
        _kp_text(), encoding="utf-8"
    )
    (root / "Depth" / "0.png").write_bytes(b"depth")
    (root / "Goal" / "0.txt").write_text("goal", encoding="utf-8")


def test_empty_neuraldata1_project_writes_blocked_report(tmp_path: Path) -> None:
    project = tmp_path / "NeuralData1"
    _make_project(project, with_export=False)

    rc = capture.main(
        [
            "--project-root",
            str(project),
            "--source-name",
            "neural_empty",
            "--raw-out-root",
            str(tmp_path / "raw"),
            "--acceptance-out-root",
            str(tmp_path / "acceptance"),
            "--quarantine-root",
            str(tmp_path / "missing_quarantine"),
        ]
    )

    assert rc == 2
    work = tmp_path / "acceptance" / "unreal_neural_empty"
    report = json.loads((work / "capture_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "BLOCKED_NO_CAPTURE_EXPORT"
    assert report["training_allowed"] is False
    assert report["counts"]["images"] == 0
    assert report["counts"]["keypoint_files"] == 0
    assert "Open NeuralData.uproject in Unreal." in report["next_actions"]
    assert not (tmp_path / "raw" / "unreal_neural_empty").exists()


def test_capture_wrapper_copies_only_export_and_runs_acceptance(
    tmp_path: Path,
) -> None:
    project = tmp_path / "NeuralData1"
    _make_project(project, with_export=True)

    rc = capture.main(
        [
            "--project-root",
            str(project),
            "--source-name",
            "neural_smoke",
            "--raw-out-root",
            str(tmp_path / "raw"),
            "--acceptance-out-root",
            str(tmp_path / "acceptance"),
            "--quarantine-root",
            str(tmp_path / "missing_quarantine"),
            "--preview-count",
            "1",
            "--overwrite",
        ]
    )

    assert rc == 0
    raw = tmp_path / "raw" / "unreal_neural_smoke"
    assert (raw / "Images" / "0.jpg").is_file()
    assert (raw / "keyPoint" / "0" / "0.txt").is_file()
    assert (raw / "Depth" / "0.png").is_file()
    assert (raw / "Goal" / "0.txt").is_file()
    assert not (raw / "Content").exists()

    work = tmp_path / "acceptance" / "unreal_neural_smoke"
    capture_report = json.loads(
        (work / "capture_report.json").read_text(encoding="utf-8")
    )
    acceptance_report = json.loads(
        (work / "acceptance_report.json").read_text(encoding="utf-8")
    )
    assert acceptance_report["technical_status"] == "PASS"
    assert acceptance_report["import"]["valid_wheels"] == 1
    assert capture_report["status"] == "READY_FOR_HUMAN_PREVIEW"
    assert capture_report["training_allowed"] is False
    assert capture_report["training_decision"] == (
        "NOT_APPROVED_FOR_TRAINING_UNTIL_HUMAN_PREVIEW_ACCEPTS_GEOMETRY"
    )
    assert str(work / "acceptance_report.json") == capture_report["acceptance_report"]


def test_capture_wrapper_can_record_human_accepted_training_gate(
    tmp_path: Path,
) -> None:
    project = tmp_path / "NeuralData1"
    _make_project(project, with_export=True)

    rc = capture.main(
        [
            "--project-root",
            str(project),
            "--source-name",
            "neural_human_ok",
            "--raw-out-root",
            str(tmp_path / "raw"),
            "--acceptance-out-root",
            str(tmp_path / "acceptance"),
            "--quarantine-root",
            str(tmp_path / "missing_quarantine"),
            "--preview-count",
            "1",
            "--overwrite",
            "--human-preview-accepted",
        ]
    )

    assert rc == 0
    report = json.loads(
        (
            tmp_path
            / "acceptance"
            / "unreal_neural_human_ok"
            / "capture_report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["status"] == "ACCEPT_FOR_TRAINING"
    assert report["training_decision"] == "ACCEPT_FOR_TRAINING"
    assert report["training_allowed"] is True
