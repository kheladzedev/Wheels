from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_mobilenetv2_from_accepted_export as train_guard  # noqa: E402


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _accepted_root(
    tmp_path: Path,
    *,
    plugin_bbox: int = 4,
    synthesized_bbox: int = 0,
    technical_status: str = "PASS",
    data_quality_passed: bool = True,
    conversion_quality_passed: bool = True,
) -> Path:
    root = tmp_path / "acceptance"
    (root / "pose_dataset").mkdir(parents=True)
    _write_json(
        root / "acceptance_report.json",
        {
            "source_name": "unreal_clean_trial",
            "technical_status": technical_status,
            "training_status": "NOT_APPROVED_FOR_TRAINING_UNTIL_HUMAN_PREVIEW_ACCEPTS_GEOMETRY",
            "data_quality_gate": {"passed": data_quality_passed},
            "conversion": {"quality_gate": {"passed": conversion_quality_passed}},
        },
    )
    _write_json(
        root / "incoming" / "metadata" / "import_report.json",
        {
            "bbox_source_counts": {
                "plugin_provided": plugin_bbox,
                "synthesized_by_adapter": synthesized_bbox,
            }
        },
    )
    return root


def test_guard_rejects_without_human_preview(tmp_path: Path) -> None:
    root = _accepted_root(tmp_path)

    args = train_guard.parse_args(
        [
            "--acceptance-root",
            str(root),
            "--dry-run",
        ]
    )

    assert train_guard.run(args) == 2


def test_guard_rejects_synthetic_bbox_without_explicit_review(tmp_path: Path) -> None:
    root = _accepted_root(tmp_path, plugin_bbox=0, synthesized_bbox=4)

    args = train_guard.parse_args(
        [
            "--acceptance-root",
            str(root),
            "--human-preview-accepted",
            "--dry-run",
        ]
    )

    assert train_guard.run(args) == 2


def test_guard_allows_plugin_bbox_after_human_preview_dry_run(tmp_path: Path) -> None:
    root = _accepted_root(tmp_path, plugin_bbox=4, synthesized_bbox=0)

    args = train_guard.parse_args(
        [
            "--acceptance-root",
            str(root),
            "--human-preview-accepted",
            "--dry-run",
            "--epochs",
            "3",
            "--batch",
            "2",
            "--device",
            "cpu",
        ]
    )

    gate = train_guard.validate_training_gate(args)
    assert gate["failures"] == []
    cmd = train_guard.build_train_command(args, gate)
    assert "--dataset-root" in cmd
    assert str(root / "pose_dataset") in cmd
    assert "--pretrained" in cmd
    assert "mn2_unreal_clean_trial_e50" in cmd
    assert train_guard.run(args) == 0


def test_guard_allows_synthetic_bbox_only_with_explicit_review_flag(tmp_path: Path) -> None:
    root = _accepted_root(tmp_path, plugin_bbox=0, synthesized_bbox=4)

    args = train_guard.parse_args(
        [
            "--acceptance-root",
            str(root),
            "--human-preview-accepted",
            "--accept-synthetic-bbox-after-review",
            "--dry-run",
        ]
    )

    assert train_guard.run(args) == 0
