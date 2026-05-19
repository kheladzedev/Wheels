"""Train and compare a MobileNetV2 baseline on a candidate YOLO-pose dataset.

This is the step after export regression acceptance. Given a candidate
``pose_dataset`` it can train a fresh MobileNetV2 checkpoint, run eval/inference
for both the current baseline checkpoint and the candidate checkpoint, and write
a conservative model comparison report.

The result is not a production approval. It only answers whether the retrained
checkpoint is a better candidate than ``mn2_0003_kpt_smoothl1_e20`` on the same
validation split and confirmed-schema inference export.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_CHECKPOINT = Path(
    "runs/pose_mn2/mn2_0003_kpt_smoothl1_e20/weights/last.pt"
)
DEFAULT_OUT_ROOT = Path("outputs/mobilenetv2_model_compare")


@dataclass(frozen=True)
class CommandResult:
    name: str
    command: list[str]
    returncode: int
    log_path: Path

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": self.command,
            "returncode": self.returncode,
            "log_path": str(self.log_path),
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train candidate MobileNetV2 and compare against baseline"
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="YOLO-pose dataset root with images/{train,val} and labels/{train,val}.",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=DEFAULT_BASELINE_CHECKPOINT,
    )
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=None,
        help="Optional existing candidate checkpoint; skips training when provided.",
    )
    parser.add_argument("--name", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=5)
    parser.add_argument("--preview-count", type=int, default=40)
    parser.add_argument(
        "--min-recall-delta",
        type=float,
        default=0.02,
        help="Minimum recall improvement to call candidate_better.",
    )
    parser.add_argument(
        "--min-iou-delta",
        type=float,
        default=0.01,
        help="Minimum mean-IoU improvement to call candidate_better.",
    )
    parser.add_argument(
        "--max-kpt-error-regression",
        type=float,
        default=0.50,
        help="Allowed mean A/B/C pixel-error regression when calling candidate_better.",
    )
    return parser.parse_args(argv)


def slugify(raw: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw.strip()).strip("._-")
    return slug or "mn2_candidate"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _run_logged_command(
    name: str,
    command: list[str],
    logs_dir: Path,
) -> CommandResult:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{name}.log"
    print(f"\n==> {name}")
    print(" ".join(command))
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.stdout:
        print("\n".join(proc.stdout.rstrip().splitlines()[-12:]))
    print(f"--> {name}: {'OK' if proc.returncode == 0 else 'FAIL'} (log: {log_path})")
    return CommandResult(name, command, proc.returncode, log_path)


def _default_name(dataset_root: Path) -> str:
    parent = dataset_root.parent.name if dataset_root.name == "pose_dataset" else dataset_root.name
    return f"mn2_{slugify(parent)}_candidate"


def _default_out_dir(name: str, out_dir: Path | None) -> Path:
    if out_dir is not None:
        return out_dir
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUT_ROOT / f"{slugify(name)}_{stamp}"


def _prepare_out_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output dir already exists: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _require_dataset(root: Path) -> None:
    required = [
        root / "images" / "train",
        root / "images" / "val",
        root / "labels" / "train",
        root / "labels" / "val",
    ]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"dataset root is missing required dirs: {missing}")


def _train_candidate(
    args: argparse.Namespace,
    name: str,
    out_dir: Path,
    logs_dir: Path,
) -> tuple[Path, dict[str, Any], CommandResult | None]:
    if args.candidate_checkpoint is not None:
        checkpoint = args.candidate_checkpoint.expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"candidate checkpoint not found: {checkpoint}")
        return checkpoint, {}, None

    project_dir = out_dir / "candidate_train"
    command = [
        sys.executable,
        "scripts/train_mobilenetv2_skipless.py",
        "--dataset-root",
        str(args.dataset_root),
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--device",
        args.device,
        "--imgsz",
        str(args.imgsz),
        "--num-workers",
        str(args.num_workers),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--project",
        str(project_dir),
        "--name",
        name,
    ]
    if args.steps_per_epoch is not None:
        command.extend(["--steps-per-epoch", str(args.steps_per_epoch)])
    if args.limit_train is not None:
        command.extend(["--limit-train", str(args.limit_train)])
    if args.limit_val is not None:
        command.extend(["--limit-val", str(args.limit_val)])
    if args.pretrained:
        command.append("--pretrained")

    result = _run_logged_command("candidate_train", command, logs_dir)
    run_dir = project_dir / name
    summary_path = run_dir / "run_summary.json"
    summary = _load_json(summary_path)
    checkpoint = Path(summary.get("checkpoint") or run_dir / "weights" / "last.pt")
    return checkpoint, summary, result


def _run_eval(
    label: str,
    checkpoint: Path,
    args: argparse.Namespace,
    out_dir: Path,
    logs_dir: Path,
) -> tuple[dict[str, Any], CommandResult]:
    eval_dir = out_dir / label / "eval"
    command = [
        sys.executable,
        "scripts/eval_mobilenetv2_skipless.py",
        "--checkpoint",
        str(checkpoint),
        "--dataset-root",
        str(args.dataset_root),
        "--split",
        args.split,
        "--device",
        args.device,
        "--imgsz",
        str(args.imgsz),
        "--conf",
        str(args.conf),
        "--nms-iou",
        str(args.nms_iou),
        "--max-det",
        str(args.max_det),
        "--preview-count",
        str(args.preview_count),
        "--out-dir",
        str(eval_dir),
    ]
    result = _run_logged_command(f"{label}_eval", command, logs_dir)
    report = _load_json(eval_dir / "eval_report.json")
    if report:
        report["_report_path"] = str(eval_dir / "eval_report.json")
    return report, result


def _run_predict(
    label: str,
    checkpoint: Path,
    args: argparse.Namespace,
    out_dir: Path,
    logs_dir: Path,
) -> tuple[dict[str, Any], CommandResult]:
    infer_dir = out_dir / label / "infer"
    source = args.dataset_root / "images" / args.split
    command = [
        sys.executable,
        "scripts/predict_mobilenetv2_skipless.py",
        "--checkpoint",
        str(checkpoint),
        "--source",
        str(source),
        "--device",
        args.device,
        "--imgsz",
        str(args.imgsz),
        "--conf",
        str(args.conf),
        "--nms-iou",
        str(args.nms_iou),
        "--max-det",
        str(args.max_det),
        "--preview-count",
        str(args.preview_count),
        "--out-dir",
        str(infer_dir),
    ]
    result = _run_logged_command(f"{label}_predict", command, logs_dir)
    summary = _load_json(infer_dir / "run_summary.json")
    if summary:
        summary["_summary_path"] = str(infer_dir / "run_summary.json")
    return summary, result


def _eval_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_count": int(report.get("image_count") or 0),
        "gt_wheel_count": int(report.get("gt_wheel_count") or 0),
        "prediction_count": int(report.get("prediction_count") or 0),
        "matched_count": int(report.get("matched_count") or 0),
        "precision": float(report.get("precision") or 0.0),
        "recall": float(report.get("recall") or 0.0),
        "mean_iou": float(report.get("mean_iou") or 0.0),
        "mean_keypoint_error_px": report.get("mean_keypoint_error_px") or {},
        "false_positive_empty_label_count": int(
            report.get("false_positive_empty_label_count") or 0
        ),
        "report": report.get("_report_path", ""),
        "previews": report.get("previews_dir", ""),
    }


def _infer_summary(summary: dict[str, Any]) -> dict[str, Any]:
    out_dir = summary.get("out_dir")
    return {
        "image_count": int(summary.get("image_count") or 0),
        "raw_detection_count": int(summary.get("raw_detection_count") or 0),
        "confirmed_wheel_count": int(summary.get("prediction_count") or 0),
        "confirmed_dropped_count": int(summary.get("confirmed_dropped_count") or 0),
        "empty_prediction_count": int(summary.get("empty_prediction_count") or 0),
        "preview_count": int(summary.get("preview_count") or 0),
        "model_status": summary.get("model_status", ""),
        "summary": summary.get("_summary_path", ""),
        "predictions": summary.get("predictions_jsonl", ""),
        "previews": str(Path(out_dir) / "previews") if out_dir else "",
    }


def _mean_kpt_error(summary: dict[str, Any]) -> float:
    errors = summary.get("mean_keypoint_error_px") or {}
    values = [float(errors.get(name) or 0.0) for name in ("a", "b", "c_disc_bottom")]
    return sum(values) / len(values)


def _deltas(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "precision": candidate["precision"] - baseline["precision"],
        "recall": candidate["recall"] - baseline["recall"],
        "mean_iou": candidate["mean_iou"] - baseline["mean_iou"],
        "mean_keypoint_error_px": _mean_kpt_error(candidate) - _mean_kpt_error(baseline),
        "prediction_count": candidate["prediction_count"] - baseline["prediction_count"],
        "matched_count": candidate["matched_count"] - baseline["matched_count"],
        "false_positive_empty_label_count": (
            candidate["false_positive_empty_label_count"]
            - baseline["false_positive_empty_label_count"]
        ),
    }


def _infer_deltas(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_detection_count": (
            candidate["raw_detection_count"] - baseline["raw_detection_count"]
        ),
        "confirmed_wheel_count": (
            candidate["confirmed_wheel_count"] - baseline["confirmed_wheel_count"]
        ),
        "confirmed_dropped_count": (
            candidate["confirmed_dropped_count"] - baseline["confirmed_dropped_count"]
        ),
        "empty_prediction_count": (
            candidate["empty_prediction_count"] - baseline["empty_prediction_count"]
        ),
    }


def _model_status(
    *,
    baseline_eval: dict[str, Any],
    candidate_eval: dict[str, Any],
    command_results: list[CommandResult | None],
    args: argparse.Namespace,
) -> str:
    if any(result is not None and not result.ok for result in command_results):
        return "fail"
    if not baseline_eval or not candidate_eval:
        return "fail"

    delta = _deltas(baseline_eval, candidate_eval)
    kpt_regression = delta["mean_keypoint_error_px"]
    recall_gain = delta["recall"]
    iou_gain = delta["mean_iou"]
    precision_drop = delta["precision"] < -0.05

    if (
        recall_gain >= args.min_recall_delta
        and iou_gain >= args.min_iou_delta
        and kpt_regression <= args.max_kpt_error_regression
        and not precision_drop
    ):
        return "candidate_better"
    if (
        recall_gain <= -args.min_recall_delta
        or iou_gain <= -args.min_iou_delta
        or kpt_regression > args.max_kpt_error_regression
        or precision_drop
    ):
        return "candidate_worse"
    return "inconclusive"


def _recommendation(status: str) -> str:
    if status == "candidate_better":
        return (
            "Candidate checkpoint is better on this validation pass. Keep it as "
            "a retraining candidate, then review previews and clean-export quality "
            "before any production decision."
        )
    if status == "candidate_worse":
        return (
            "Candidate checkpoint regressed. Do not promote it; inspect data quality, "
            "targets, and training settings before more epochs."
        )
    if status == "inconclusive":
        return (
            "Candidate checkpoint is not clearly better or worse. Treat it as "
            "experimental and compare again after cleaner data or adjusted training."
        )
    return "Training/eval/inference failed. Fix the failing command before judging models."


def _write_reports(report: dict[str, Any], out_dir: Path) -> None:
    (out_dir / "model_comparison_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "model_comparison_report.md").write_text(
        _format_markdown(report),
        encoding="utf-8",
    )


def _format_markdown(report: dict[str, Any]) -> str:
    be = report["baseline"]["eval"]
    ce = report["candidate"]["eval"]
    bi = report["baseline"]["inference"]
    ci = report["candidate"]["inference"]
    de = report["comparison"]["eval_delta_candidate_minus_baseline"]
    di = report["comparison"]["inference_delta_candidate_minus_baseline"]
    lines = [
        "# MobileNetV2 Retrain Comparison",
        "",
        f"- Status: **{report['status']}**",
        f"- Recommendation: {report['recommendation']}",
        f"- Dataset: `{report['dataset_root']}`",
        f"- Baseline checkpoint: `{report['baseline']['checkpoint']}`",
        f"- Candidate checkpoint: `{report['candidate']['checkpoint']}`",
        "",
        "## Eval",
        "",
        "| Model | GT wheels | Predictions | Matched | Precision | Recall | Mean IoU | Mean A/B/C err | FP empty |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, ev in (("baseline", be), ("candidate", ce)):
        lines.append(
            f"| {label} | {ev['gt_wheel_count']} | {ev['prediction_count']} | "
            f"{ev['matched_count']} | {ev['precision']:.4f} | "
            f"{ev['recall']:.4f} | {ev['mean_iou']:.4f} | "
            f"{_mean_kpt_error(ev):.2f} | {ev['false_positive_empty_label_count']} |"
        )
    lines += [
        "",
        "## Eval Delta Candidate Minus Baseline",
        "",
        f"- Precision: {de['precision']:+.4f}",
        f"- Recall: {de['recall']:+.4f}",
        f"- Mean IoU: {de['mean_iou']:+.4f}",
        f"- Mean A/B/C error px: {de['mean_keypoint_error_px']:+.2f}",
        "",
        "## Inference Export",
        "",
        "| Model | Images | Raw detections | Confirmed wheels | Geometry drops | Empty predictions | Previews |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, inf in (("baseline", bi), ("candidate", ci)):
        lines.append(
            f"| {label} | {inf['image_count']} | {inf['raw_detection_count']} | "
            f"{inf['confirmed_wheel_count']} | {inf['confirmed_dropped_count']} | "
            f"{inf['empty_prediction_count']} | {inf['preview_count']} |"
        )
    lines += [
        "",
        "## Inference Delta Candidate Minus Baseline",
        "",
        f"- Raw detections: {di['raw_detection_count']:+}",
        f"- Confirmed wheels: {di['confirmed_wheel_count']:+}",
        f"- Geometry drops: {di['confirmed_dropped_count']:+}",
        f"- Empty predictions: {di['empty_prediction_count']:+}",
        "",
        "## Artifacts",
        "",
        f"- Baseline eval report: `{be['report']}`",
        f"- Candidate eval report: `{ce['report']}`",
        f"- Baseline inference summary: `{bi['summary']}`",
        f"- Candidate inference summary: `{ci['summary']}`",
        f"- Baseline inference previews: `{bi['previews']}`",
        f"- Candidate inference previews: `{ci['previews']}`",
        "",
        "Note: this report compares checkpoints. It does not override the export "
        "acceptance data-quality gate or approve Android production use.",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    args.dataset_root = args.dataset_root.expanduser()
    args.baseline_checkpoint = args.baseline_checkpoint.expanduser()
    _require_dataset(args.dataset_root)
    if not args.baseline_checkpoint.is_file():
        raise FileNotFoundError(f"baseline checkpoint not found: {args.baseline_checkpoint}")

    name = slugify(args.name or _default_name(args.dataset_root))
    out_dir = _default_out_dir(name, args.out_dir).expanduser()
    _prepare_out_dir(out_dir, args.overwrite)
    logs_dir = out_dir / "logs"

    candidate_checkpoint, train_summary, train_result = _train_candidate(
        args, name, out_dir, logs_dir
    )
    command_results: list[CommandResult | None] = [train_result]

    baseline_eval, baseline_eval_result = _run_eval(
        "baseline", args.baseline_checkpoint, args, out_dir, logs_dir
    )
    command_results.append(baseline_eval_result)
    candidate_eval, candidate_eval_result = _run_eval(
        "candidate", candidate_checkpoint, args, out_dir, logs_dir
    )
    command_results.append(candidate_eval_result)
    baseline_infer, baseline_predict_result = _run_predict(
        "baseline", args.baseline_checkpoint, args, out_dir, logs_dir
    )
    command_results.append(baseline_predict_result)
    candidate_infer, candidate_predict_result = _run_predict(
        "candidate", candidate_checkpoint, args, out_dir, logs_dir
    )
    command_results.append(candidate_predict_result)

    baseline_eval_summary = _eval_summary(baseline_eval)
    candidate_eval_summary = _eval_summary(candidate_eval)
    baseline_infer_summary = _infer_summary(baseline_infer)
    candidate_infer_summary = _infer_summary(candidate_infer)
    status = _model_status(
        baseline_eval=baseline_eval_summary,
        candidate_eval=candidate_eval_summary,
        command_results=command_results,
        args=args,
    )
    report = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "status": status,
        "recommendation": _recommendation(status),
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "device": args.device,
        "thresholds": {
            "conf": args.conf,
            "nms_iou": args.nms_iou,
            "max_det": args.max_det,
            "imgsz": args.imgsz,
        },
        "training": {
            "skipped": train_result is None,
            "summary": train_summary,
            "epochs": args.epochs,
            "batch": args.batch,
            "pretrained": args.pretrained,
        },
        "baseline": {
            "checkpoint": str(args.baseline_checkpoint),
            "eval": baseline_eval_summary,
            "inference": baseline_infer_summary,
        },
        "candidate": {
            "checkpoint": str(candidate_checkpoint),
            "eval": candidate_eval_summary,
            "inference": candidate_infer_summary,
        },
        "comparison": {
            "eval_delta_candidate_minus_baseline": _deltas(
                baseline_eval_summary, candidate_eval_summary
            ),
            "inference_delta_candidate_minus_baseline": _infer_deltas(
                baseline_infer_summary, candidate_infer_summary
            ),
            "decision_thresholds": {
                "min_recall_delta": args.min_recall_delta,
                "min_iou_delta": args.min_iou_delta,
                "max_kpt_error_regression": args.max_kpt_error_regression,
                "max_precision_drop": 0.05,
            },
        },
        "steps": [result.to_json() for result in command_results if result is not None],
    }
    _write_reports(report, out_dir)
    print()
    print(f"Model comparison report: {out_dir / 'model_comparison_report.md'}")
    print(f"Status:                  {status}")
    print(f"Recommendation:          {report['recommendation']}")
    return 0 if status in {"candidate_better", "candidate_worse", "inconclusive"} else 1


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
