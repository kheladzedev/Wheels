"""Tests for MobileNetV2 retrain comparison runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import train_compare_mobilenetv2_baseline as runner  # noqa: E402


def _make_dataset(root: Path) -> Path:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    return root


def _write_checkpoint(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"checkpoint")
    return path


def _eval_report(*, recall: float, precision: float, iou: float, kpt: float) -> dict:
    return {
        "image_count": 5,
        "gt_wheel_count": 4,
        "prediction_count": 4,
        "matched_count": 3,
        "precision": precision,
        "recall": recall,
        "mean_iou": iou,
        "mean_keypoint_error_px": {
            "a": kpt,
            "b": kpt + 0.5,
            "c_disc_bottom": kpt - 0.5,
        },
        "false_positive_empty_label_count": 1,
        "_report_path": "eval_report.json",
        "previews_dir": "eval/previews",
    }


def _infer_summary(*, confirmed: int) -> dict:
    return {
        "image_count": 5,
        "raw_detection_count": confirmed + 1,
        "prediction_count": confirmed,
        "confirmed_dropped_count": 1,
        "empty_prediction_count": 2,
        "preview_count": 5,
        "model_status": "provisional_0003_not_production",
        "_summary_path": "infer/run_summary.json",
        "out_dir": "infer",
        "predictions_jsonl": "infer/predictions.jsonl",
    }


def _ok(name: str) -> runner.CommandResult:
    return runner.CommandResult(name, [name], 0, Path(f"{name}.log"))


def test_model_status_candidate_better() -> None:
    args = runner.parse_args(
        [
            "--dataset-root",
            "dataset",
            "--min-recall-delta",
            "0.02",
            "--min-iou-delta",
            "0.01",
        ]
    )

    status = runner._model_status(
        baseline_eval=runner._eval_summary(
            _eval_report(recall=0.80, precision=0.80, iou=0.86, kpt=4.0)
        ),
        candidate_eval=runner._eval_summary(
            _eval_report(recall=0.84, precision=0.79, iou=0.87, kpt=3.8)
        ),
        command_results=[_ok("eval")],
        args=args,
    )

    assert status == "candidate_better"


def test_model_status_candidate_worse_on_keypoint_regression() -> None:
    args = runner.parse_args(["--dataset-root", "dataset"])

    status = runner._model_status(
        baseline_eval=runner._eval_summary(
            _eval_report(recall=0.80, precision=0.80, iou=0.86, kpt=4.0)
        ),
        candidate_eval=runner._eval_summary(
            _eval_report(recall=0.81, precision=0.80, iou=0.86, kpt=6.0)
        ),
        command_results=[_ok("eval")],
        args=args,
    )

    assert status == "candidate_worse"


def test_runner_compare_only_writes_report(tmp_path: Path, monkeypatch) -> None:
    dataset = _make_dataset(tmp_path / "pose_dataset")
    baseline_ckpt = _write_checkpoint(tmp_path / "baseline" / "last.pt")
    candidate_ckpt = _write_checkpoint(tmp_path / "candidate" / "last.pt")
    out_dir = tmp_path / "compare"

    def fake_eval(label, checkpoint, args, out_dir_arg, logs_dir):
        if label == "baseline":
            report = _eval_report(recall=0.80, precision=0.80, iou=0.86, kpt=4.0)
        else:
            report = _eval_report(recall=0.84, precision=0.79, iou=0.87, kpt=3.8)
        return report, _ok(f"{label}_eval")

    def fake_predict(label, checkpoint, args, out_dir_arg, logs_dir):
        return _infer_summary(confirmed=3 if label == "baseline" else 4), _ok(
            f"{label}_predict"
        )

    monkeypatch.setattr(runner, "_run_eval", fake_eval)
    monkeypatch.setattr(runner, "_run_predict", fake_predict)

    rc = runner.main(
        [
            "--dataset-root",
            str(dataset),
            "--baseline-checkpoint",
            str(baseline_ckpt),
            "--candidate-checkpoint",
            str(candidate_ckpt),
            "--out-dir",
            str(out_dir),
            "--overwrite",
        ]
    )

    assert rc == 0
    report_path = out_dir / "model_comparison_report.json"
    md_path = out_dir / "model_comparison_report.md"
    assert report_path.exists()
    assert md_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "candidate_better"
    assert report["training"]["skipped"] is True
    assert report["candidate"]["checkpoint"] == str(candidate_ckpt)
    assert report["comparison"]["eval_delta_candidate_minus_baseline"]["recall"] == 0.039999999999999925
    assert report["comparison"]["inference_delta_candidate_minus_baseline"][
        "confirmed_wheel_count"
    ] == 1
    assert "Status: **candidate_better**" in md_path.read_text(encoding="utf-8")


def test_runner_trains_when_candidate_checkpoint_not_supplied(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = _make_dataset(tmp_path / "pose_dataset")
    baseline_ckpt = _write_checkpoint(tmp_path / "baseline" / "last.pt")
    trained_ckpt = _write_checkpoint(tmp_path / "trained" / "last.pt")
    out_dir = tmp_path / "compare"

    def fake_train(args, name, out_dir_arg, logs_dir):
        return trained_ckpt, {"checkpoint": str(trained_ckpt), "epochs": 1}, _ok(
            "candidate_train"
        )

    monkeypatch.setattr(runner, "_train_candidate", fake_train)
    monkeypatch.setattr(
        runner,
        "_run_eval",
        lambda label, checkpoint, args, out_dir_arg, logs_dir: (
            _eval_report(recall=0.80, precision=0.80, iou=0.86, kpt=4.0),
            _ok(f"{label}_eval"),
        ),
    )
    monkeypatch.setattr(
        runner,
        "_run_predict",
        lambda label, checkpoint, args, out_dir_arg, logs_dir: (
            _infer_summary(confirmed=3),
            _ok(f"{label}_predict"),
        ),
    )

    rc = runner.main(
        [
            "--dataset-root",
            str(dataset),
            "--baseline-checkpoint",
            str(baseline_ckpt),
            "--epochs",
            "1",
            "--out-dir",
            str(out_dir),
            "--overwrite",
        ]
    )

    assert rc == 0
    report = json.loads(
        (out_dir / "model_comparison_report.json").read_text(encoding="utf-8")
    )
    assert report["training"]["skipped"] is False
    assert report["candidate"]["checkpoint"] == str(trained_ckpt)
    assert report["status"] == "inconclusive"
