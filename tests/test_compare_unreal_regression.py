"""Tests for Unreal export regression comparison runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import compare_unreal_regression as compare  # noqa: E402


def test_acceptance_slugify_matches_acceptance_runner_prefix() -> None:
    assert compare.acceptance_slugify("0004") == "unreal_0004"
    assert compare.acceptance_slugify("unreal_clean") == "unreal_clean"


def _write_acceptance(
    root: Path,
    *,
    source_name: str,
    technical_status: str = "PASS",
    data_quality_passed: bool = False,
    raw_objects: int = 100,
    valid_wheels: int = 20,
    yolo_wheels: int = 20,
    empty_label_ratio: float = 0.4,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pose_dataset" / "images" / "val").mkdir(parents=True, exist_ok=True)
    (root / "acceptance_report.json").write_text(
        json.dumps(
            {
                "source_name": source_name,
                "technical_status": technical_status,
                "training_status": (
                    "NOT_APPROVED_FOR_TRAINING_UNTIL_HUMAN_PREVIEW_ACCEPTS_GEOMETRY"
                    if data_quality_passed
                    else "NOT_APPROVED_FOR_TRAINING_DATA_QUALITY_GATE_FAILED"
                ),
                "mapping_mode": "screen-sides",
                "mapping_basis": "auto_screen_x_majority",
                "raw_export": {
                    "images": 10,
                    "keypoint_object_files": raw_objects,
                    "counts_by_status": {
                        "VALID_ALL_POINTS_IN_IMAGE": valid_wheels,
                        "EMPTY_ALL_ZERO": raw_objects - valid_wheels,
                    },
                },
                "import": {
                    "valid_wheels": valid_wheels,
                    "bbox_strategy_counts": {"top_points": valid_wheels},
                    "drop_counts": {"all_zero": raw_objects - valid_wheels},
                },
                "conversion": {"wheels": yolo_wheels},
                "data_quality_gate": {
                    "passed": data_quality_passed,
                    "metrics": {"empty_label_image_ratio": empty_label_ratio},
                    "reasons": [] if data_quality_passed else ["dirty export"],
                },
            }
        ),
        encoding="utf-8",
    )
    return root


def _fake_eval(
    batch_name: str,
    acceptance_root: Path,
    args,
    batch_out_dir: Path,
    logs_dir: Path,
):
    report_path = batch_out_dir / "eval" / "eval_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "_report_path": str(report_path),
        "image_count": 5,
        "gt_wheel_count": 4,
        "prediction_count": 4,
        "matched_count": 3,
        "precision": 0.75,
        "recall": 0.75,
        "mean_iou": 0.8,
        "mean_keypoint_error_px": {"a": 3.0, "b": 4.0, "c_disc_bottom": 2.0},
        "false_positive_empty_label_count": 1,
        "previews_dir": str(batch_out_dir / "eval" / "previews"),
        "predictions_jsonl": str(batch_out_dir / "eval" / "predictions.jsonl"),
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report, compare.CommandResult(
        f"{batch_name}_mn2_eval",
        ["eval"],
        0,
        logs_dir / f"{batch_name}_mn2_eval.log",
    )


def _fake_predict(
    batch_name: str,
    acceptance_root: Path,
    args,
    batch_out_dir: Path,
    logs_dir: Path,
):
    summary_path = batch_out_dir / "infer" / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "_summary_path": str(summary_path),
        "out_dir": str(batch_out_dir / "infer"),
        "image_count": 5,
        "raw_detection_count": 6,
        "prediction_count": 3,
        "confirmed_dropped_count": 3,
        "empty_prediction_count": 2,
        "preview_count": 5,
        "model_status": "provisional_0003_not_production",
        "predictions_jsonl": str(batch_out_dir / "infer" / "predictions.jsonl"),
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary, compare.CommandResult(
        f"{batch_name}_mn2_predict",
        ["predict"],
        0,
        logs_dir / f"{batch_name}_mn2_predict.log",
    )


def test_regression_runner_compares_existing_acceptance_roots(
    tmp_path: Path, monkeypatch
) -> None:
    baseline = _write_acceptance(
        tmp_path / "acceptance" / "unreal_0003",
        source_name="unreal_0003",
        raw_objects=100,
        valid_wheels=20,
        yolo_wheels=20,
        empty_label_ratio=0.4,
    )
    candidate = _write_acceptance(
        tmp_path / "acceptance" / "unreal_clean",
        source_name="unreal_clean",
        data_quality_passed=True,
        raw_objects=100,
        valid_wheels=80,
        yolo_wheels=80,
        empty_label_ratio=0.05,
    )
    checkpoint = tmp_path / "weights" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    out_dir = tmp_path / "compare"

    monkeypatch.setattr(compare, "_run_eval", _fake_eval)
    monkeypatch.setattr(compare, "_run_predict", _fake_predict)

    rc = compare.main(
        [
            "--baseline-acceptance-root",
            str(baseline),
            "--candidate-acceptance-root",
            str(candidate),
            "--checkpoint",
            str(checkpoint),
            "--out-dir",
            str(out_dir),
            "--overwrite",
        ]
    )

    assert rc == 0
    report_path = out_dir / "regression_report.json"
    md_path = out_dir / "regression_report.md"
    assert report_path.exists()
    assert md_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["overall_status"] == "accepted"
    assert report["baseline"]["status"] == "provisional"
    assert report["candidate"]["status"] == "accepted"
    delta = report["comparison"]["candidate_minus_baseline"]
    assert delta["valid_imported_wheels"] == 60
    assert delta["empty_label_image_ratio"] == -0.35000000000000003
    assert report["candidate"]["mobilenetv2_inference"]["confirmed_wheel_count"] == 3
    assert "Overall status: **accepted**" in md_path.read_text(encoding="utf-8")


def test_batch_status_fails_when_acceptance_fails(tmp_path: Path) -> None:
    root = _write_acceptance(
        tmp_path / "acceptance" / "bad",
        source_name="bad",
        technical_status="FAIL",
    )
    acceptance = json.loads((root / "acceptance_report.json").read_text())

    summary = compare._batch_summary(
        "bad",
        root,
        acceptance,
        {},
        {},
        [None, None],
    )

    assert summary["status"] == "fail"
    assert summary["technical_status"] == "FAIL"
