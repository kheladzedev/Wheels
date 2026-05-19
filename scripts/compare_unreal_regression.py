"""Compare Unreal export intake + MobileNetV2 regression evidence.

This is the one-command runner for the next clean export:

1. Optionally run official acceptance for a candidate raw Unreal export.
2. Run MobileNetV2 eval and confirmed-schema inference for baseline/candidate.
3. Write a compact regression report with raw/valid/drop counts, eval metrics,
   preview locations, and a conservative status: provisional / accepted / fail.

It intentionally does not make a production-model claim. A clean export that
passes this runner is a retraining candidate, not an Android-ready model.
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
DEFAULT_BASELINE_ACCEPTANCE = Path("outputs/unreal_export_acceptance/unreal_0003")
DEFAULT_ACCEPTANCE_OUT_ROOT = Path("outputs/unreal_export_acceptance")
DEFAULT_OUT_DIR = Path("outputs/unreal_regression_compare")
DEFAULT_CHECKPOINT = Path("runs/pose_mn2/mn2_0003_kpt_smoothl1_e20/weights/last.pt")


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
        description="Compare old/new Unreal exports through acceptance + MobileNetV2"
    )
    parser.add_argument(
        "--baseline-acceptance-root",
        type=Path,
        default=DEFAULT_BASELINE_ACCEPTANCE,
        help="Existing acceptance root for the control batch.",
    )
    parser.add_argument(
        "--baseline-name",
        default=None,
        help="Display name. Defaults to acceptance_report.source_name or folder name.",
    )
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument(
        "--candidate-source-root",
        type=Path,
        help="Raw Unreal export root to accept before comparison.",
    )
    candidate.add_argument(
        "--candidate-acceptance-root",
        type=Path,
        help="Existing candidate acceptance root to compare.",
    )
    parser.add_argument(
        "--candidate-name",
        default=None,
        help="Candidate display/source name. Defaults from source/root folder.",
    )
    parser.add_argument(
        "--acceptance-out-root",
        type=Path,
        default=DEFAULT_ACCEPTANCE_OUT_ROOT,
        help="Where candidate acceptance should be written when --candidate-source-root is used.",
    )
    parser.add_argument("--acceptance-preview-count", type=int, default=30)
    parser.add_argument(
        "--right-left-mapping",
        choices=("auto", "confirmed", "screen-sides"),
        default="auto",
        help="Forwarded to candidate acceptance import.",
    )
    parser.add_argument(
        "--swap-right-left",
        action="store_true",
        help="Forward --swap-right-left to candidate acceptance.",
    )
    parser.add_argument("--acceptance-smoke-train", action="store_true")
    parser.add_argument("--acceptance-device", default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=5)
    parser.add_argument("--preview-count", type=int, default=40)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Comparison output dir. Defaults under outputs/unreal_regression_compare/.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def slugify(raw: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw.strip()).strip("._-")
    return slug or "batch"


def acceptance_slugify(raw: str) -> str:
    slug = slugify(raw)
    return slug if slug.startswith("unreal_") else f"unreal_{slug}"


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


def _acceptance_report_path(root: Path) -> Path:
    return root / "acceptance_report.json"


def _load_acceptance(root: Path) -> dict[str, Any]:
    report_path = _acceptance_report_path(root)
    if not report_path.is_file():
        raise FileNotFoundError(f"missing acceptance report: {report_path}")
    return _load_json(report_path)


def _batch_name(root: Path, report: dict[str, Any], override: str | None) -> str:
    if override:
        return slugify(override)
    reported = report.get("source_name")
    return slugify(str(reported)) if reported else slugify(root.name)


def _default_out_dir(args: argparse.Namespace, candidate_name: str) -> Path:
    if args.out_dir is not None:
        return args.out_dir
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUT_DIR / f"{candidate_name}_vs_unreal_0003_{stamp}"


def _prepare_out_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"output dir already exists: {out_dir}. Pass --overwrite."
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _run_candidate_acceptance(
    args: argparse.Namespace,
    candidate_name: str,
    logs_dir: Path,
) -> tuple[Path, CommandResult]:
    py = sys.executable
    command = [
        py,
        "scripts/accept_unreal_export.py",
        "--source-root",
        str(args.candidate_source_root.expanduser().resolve()),
        "--source-name",
        candidate_name,
        "--out-root",
        str(args.acceptance_out_root),
        "--preview-count",
        str(args.acceptance_preview_count),
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.swap_right_left:
        command.append("--swap-right-left")
    else:
        command.extend(["--right-left-mapping", args.right_left_mapping])
    if args.acceptance_smoke_train:
        command.append("--smoke-train")
        if args.acceptance_device:
            command.extend(["--device", args.acceptance_device])

    result = _run_logged_command("candidate_acceptance", command, logs_dir)
    acceptance_root = (
        REPO_ROOT / args.acceptance_out_root / acceptance_slugify(candidate_name)
    ).resolve()
    return acceptance_root, result


def _run_eval(
    batch_name: str,
    acceptance_root: Path,
    args: argparse.Namespace,
    batch_out_dir: Path,
    logs_dir: Path,
) -> tuple[dict[str, Any], CommandResult | None]:
    dataset_root = acceptance_root / "pose_dataset"
    report_path = batch_out_dir / "eval" / "eval_report.json"
    if not dataset_root.is_dir():
        return {}, None
    command = [
        sys.executable,
        "scripts/eval_mobilenetv2_skipless.py",
        "--checkpoint",
        str(args.checkpoint),
        "--dataset-root",
        str(dataset_root),
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
        str(batch_out_dir / "eval"),
    ]
    result = _run_logged_command(f"{batch_name}_mn2_eval", command, logs_dir)
    report = _load_json(report_path)
    if report:
        report["_report_path"] = str(report_path)
    return report, result


def _run_predict(
    batch_name: str,
    acceptance_root: Path,
    args: argparse.Namespace,
    batch_out_dir: Path,
    logs_dir: Path,
) -> tuple[dict[str, Any], CommandResult | None]:
    source = acceptance_root / "pose_dataset" / "images" / args.split
    summary_path = batch_out_dir / "infer" / "run_summary.json"
    if not source.is_dir():
        return {}, None
    command = [
        sys.executable,
        "scripts/predict_mobilenetv2_skipless.py",
        "--checkpoint",
        str(args.checkpoint),
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
        str(batch_out_dir / "infer"),
    ]
    result = _run_logged_command(f"{batch_name}_mn2_predict", command, logs_dir)
    summary = _load_json(summary_path)
    if summary:
        summary["_summary_path"] = str(summary_path)
    return summary, result


def _status_for_batch(
    acceptance: dict[str, Any],
    command_results: list[CommandResult | None],
) -> str:
    if acceptance.get("technical_status") != "PASS":
        return "fail"
    if any(result is not None and not result.ok for result in command_results):
        return "fail"
    if any(result is None for result in command_results):
        return "fail"
    if bool((acceptance.get("data_quality_gate") or {}).get("passed")):
        return "accepted"
    return "provisional"


def _batch_summary(
    name: str,
    acceptance_root: Path,
    acceptance: dict[str, Any],
    eval_report: dict[str, Any],
    infer_summary: dict[str, Any],
    command_results: list[CommandResult | None],
) -> dict[str, Any]:
    raw = acceptance.get("raw_export") or {}
    imp = acceptance.get("import") or {}
    conv = acceptance.get("conversion") or {}
    dqg = acceptance.get("data_quality_gate") or {}
    dqg_metrics = dqg.get("metrics") or {}
    status = _status_for_batch(acceptance, command_results)
    return {
        "name": name,
        "status": status,
        "acceptance_root": str(acceptance_root),
        "technical_status": acceptance.get("technical_status"),
        "training_status": acceptance.get("training_status"),
        "mapping_mode": acceptance.get("mapping_mode"),
        "mapping_basis": acceptance.get("mapping_basis"),
        "raw_objects": int(raw.get("keypoint_object_files") or 0),
        "raw_images": int(raw.get("images") or 0),
        "raw_status_counts": raw.get("counts_by_status") or {},
        "valid_imported_wheels": int(imp.get("valid_wheels") or 0),
        "bbox_strategy_counts": imp.get("bbox_strategy_counts") or {},
        "drop_counts": imp.get("drop_counts") or {},
        "yolo_wheel_lines": int(conv.get("wheels") or 0),
        "empty_label_image_ratio": float(dqg_metrics.get("empty_label_image_ratio") or 0.0),
        "data_quality_gate": {
            "passed": bool(dqg.get("passed")),
            "metrics": dqg_metrics,
            "reasons": dqg.get("reasons") or [],
        },
        "mobilenetv2_eval": _eval_summary(eval_report),
        "mobilenetv2_inference": _infer_summary(infer_summary),
        "artifacts": {
            "acceptance_report": str(acceptance_root / "acceptance_report.json"),
            "pose_dataset": str(acceptance_root / "pose_dataset"),
            "eval_report": eval_report.get("_report_path", ""),
            "eval_previews": eval_report.get("previews_dir", ""),
            "inference_summary": infer_summary.get("_summary_path", ""),
            "inference_predictions": infer_summary.get("predictions_jsonl", ""),
            "inference_previews": (
                str(Path(infer_summary["out_dir"]) / "previews")
                if infer_summary.get("out_dir")
                else ""
            ),
        },
        "steps": [result.to_json() for result in command_results if result is not None],
    }


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
    }


def _infer_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_count": int(summary.get("image_count") or 0),
        "raw_detection_count": int(summary.get("raw_detection_count") or 0),
        "confirmed_wheel_count": int(summary.get("prediction_count") or 0),
        "confirmed_dropped_count": int(summary.get("confirmed_dropped_count") or 0),
        "empty_prediction_count": int(summary.get("empty_prediction_count") or 0),
        "preview_count": int(summary.get("preview_count") or 0),
        "model_status": summary.get("model_status", ""),
    }


def _numeric_delta(candidate: dict[str, Any], baseline: dict[str, Any], key: str) -> int | float:
    return candidate.get(key, 0) - baseline.get(key, 0)


def _dict_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, int | float]:
    keys = set(candidate) | set(baseline)
    return {
        key: (candidate.get(key, 0) or 0) - (baseline.get(key, 0) or 0)
        for key in sorted(keys)
    }


def _comparison(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_minus_baseline": {
            "raw_objects": _numeric_delta(candidate, baseline, "raw_objects"),
            "valid_imported_wheels": _numeric_delta(
                candidate, baseline, "valid_imported_wheels"
            ),
            "yolo_wheel_lines": _numeric_delta(candidate, baseline, "yolo_wheel_lines"),
            "empty_label_image_ratio": _numeric_delta(
                candidate, baseline, "empty_label_image_ratio"
            ),
            "raw_status_counts": _dict_delta(
                candidate["raw_status_counts"], baseline["raw_status_counts"]
            ),
            "bbox_strategy_counts": _dict_delta(
                candidate["bbox_strategy_counts"], baseline["bbox_strategy_counts"]
            ),
            "drop_counts": _dict_delta(candidate["drop_counts"], baseline["drop_counts"]),
            "mobilenetv2_eval": {
                "precision": _numeric_delta(
                    candidate["mobilenetv2_eval"], baseline["mobilenetv2_eval"], "precision"
                ),
                "recall": _numeric_delta(
                    candidate["mobilenetv2_eval"], baseline["mobilenetv2_eval"], "recall"
                ),
                "mean_iou": _numeric_delta(
                    candidate["mobilenetv2_eval"], baseline["mobilenetv2_eval"], "mean_iou"
                ),
                "prediction_count": _numeric_delta(
                    candidate["mobilenetv2_eval"],
                    baseline["mobilenetv2_eval"],
                    "prediction_count",
                ),
            },
            "mobilenetv2_inference": {
                "raw_detection_count": _numeric_delta(
                    candidate["mobilenetv2_inference"],
                    baseline["mobilenetv2_inference"],
                    "raw_detection_count",
                ),
                "confirmed_wheel_count": _numeric_delta(
                    candidate["mobilenetv2_inference"],
                    baseline["mobilenetv2_inference"],
                    "confirmed_wheel_count",
                ),
                "confirmed_dropped_count": _numeric_delta(
                    candidate["mobilenetv2_inference"],
                    baseline["mobilenetv2_inference"],
                    "confirmed_dropped_count",
                ),
                "empty_prediction_count": _numeric_delta(
                    candidate["mobilenetv2_inference"],
                    baseline["mobilenetv2_inference"],
                    "empty_prediction_count",
                ),
            },
        }
    }


def _overall_status(candidate: dict[str, Any]) -> str:
    return str(candidate["status"])


def _recommendation(overall_status: str) -> str:
    if overall_status == "accepted":
        return (
            "Candidate export passed the data-quality gate. Retrain MobileNetV2 "
            "on it and compare against the current provisional checkpoint before "
            "any production decision."
        )
    if overall_status == "provisional":
        return (
            "Candidate export is technically usable but still provisional. Do not "
            "call it production data; inspect drops/previews and wait for cleaner "
            "export evidence."
        )
    return (
        "Candidate failed the technical/eval/inference pipeline. Do not train on it; "
        "inspect acceptance logs and raw status/drop counts first."
    )


def _process_batch(
    name: str,
    acceptance_root: Path,
    args: argparse.Namespace,
    out_dir: Path,
    logs_dir: Path,
) -> dict[str, Any]:
    acceptance = _load_acceptance(acceptance_root)
    batch_out_dir = out_dir / name
    batch_out_dir.mkdir(parents=True, exist_ok=True)

    eval_report: dict[str, Any] = {}
    infer_summary: dict[str, Any] = {}
    eval_result: CommandResult | None = None
    predict_result: CommandResult | None = None
    if acceptance.get("technical_status") == "PASS":
        eval_report, eval_result = _run_eval(name, acceptance_root, args, batch_out_dir, logs_dir)
        infer_summary, predict_result = _run_predict(
            name, acceptance_root, args, batch_out_dir, logs_dir
        )

    return _batch_summary(
        name=name,
        acceptance_root=acceptance_root,
        acceptance=acceptance,
        eval_report=eval_report,
        infer_summary=infer_summary,
        command_results=[eval_result, predict_result],
    )


def _write_reports(report: dict[str, Any], out_dir: Path) -> None:
    json_path = out_dir / "regression_report.json"
    md_path = out_dir / "regression_report.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(_format_markdown(report), encoding="utf-8")


def _format_markdown(report: dict[str, Any]) -> str:
    baseline = report["baseline"]
    candidate = report["candidate"]
    delta = report["comparison"]["candidate_minus_baseline"]
    lines = [
        "# Unreal Regression Comparison",
        "",
        f"- Overall status: **{report['overall_status']}**",
        f"- Recommendation: {report['recommendation']}",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Thresholds: conf={report['thresholds']['conf']}, "
        f"nms_iou={report['thresholds']['nms_iou']}, "
        f"max_det={report['thresholds']['max_det']}",
        "",
        "## Batch Status",
        "",
        "| Batch | Status | Technical | Training | Raw objects | Valid wheels | YOLO lines | Empty label ratio |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for batch in (baseline, candidate):
        lines.append(
            f"| {batch['name']} | {batch['status']} | {batch['technical_status']} | "
            f"{batch['training_status']} | {batch['raw_objects']} | "
            f"{batch['valid_imported_wheels']} | {batch['yolo_wheel_lines']} | "
            f"{batch['empty_label_image_ratio']:.4f} |"
        )
    lines += [
        "",
        "## Candidate Minus Baseline",
        "",
        f"- Raw objects: {delta['raw_objects']:+}",
        f"- Valid imported wheels: {delta['valid_imported_wheels']:+}",
        f"- YOLO wheel lines: {delta['yolo_wheel_lines']:+}",
        f"- Empty label ratio: {delta['empty_label_image_ratio']:+.4f}",
        "",
        "## MobileNetV2 Eval",
        "",
        "| Batch | GT wheels | Predictions | Matched | Precision | Recall | Mean IoU | A err | B err | C err | FP empty |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for batch in (baseline, candidate):
        ev = batch["mobilenetv2_eval"]
        kpt = ev.get("mean_keypoint_error_px") or {}
        lines.append(
            f"| {batch['name']} | {ev['gt_wheel_count']} | {ev['prediction_count']} | "
            f"{ev['matched_count']} | {ev['precision']:.4f} | {ev['recall']:.4f} | "
            f"{ev['mean_iou']:.4f} | {float(kpt.get('a') or 0.0):.2f} | "
            f"{float(kpt.get('b') or 0.0):.2f} | "
            f"{float(kpt.get('c_disc_bottom') or 0.0):.2f} | "
            f"{ev['false_positive_empty_label_count']} |"
        )
    lines += [
        "",
        "## MobileNetV2 Inference Export",
        "",
        "| Batch | Images | Raw detections | Confirmed wheels | Geometry drops | Empty predictions | Previews | Model status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for batch in (baseline, candidate):
        inf = batch["mobilenetv2_inference"]
        lines.append(
            f"| {batch['name']} | {inf['image_count']} | {inf['raw_detection_count']} | "
            f"{inf['confirmed_wheel_count']} | {inf['confirmed_dropped_count']} | "
            f"{inf['empty_prediction_count']} | {inf['preview_count']} | "
            f"{inf['model_status']} |"
        )
    lines += [
        "",
        "## Drop Counts",
        "",
        "### Baseline",
        "",
        *[f"- `{k}`: {v}" for k, v in sorted(baseline["drop_counts"].items())],
        "",
        "### Candidate",
        "",
        *[f"- `{k}`: {v}" for k, v in sorted(candidate["drop_counts"].items())],
        "",
        "## Artifacts",
        "",
        f"- Baseline eval previews: `{baseline['artifacts']['eval_previews']}`",
        f"- Candidate eval previews: `{candidate['artifacts']['eval_previews']}`",
        f"- Baseline inference previews: `{baseline['artifacts']['inference_previews']}`",
        f"- Candidate inference previews: `{candidate['artifacts']['inference_previews']}`",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    baseline_root = args.baseline_acceptance_root.expanduser()
    baseline_acceptance = _load_acceptance(baseline_root)
    baseline_name = _batch_name(baseline_root, baseline_acceptance, args.baseline_name)

    if args.candidate_source_root is not None:
        candidate_name = acceptance_slugify(
            args.candidate_name or args.candidate_source_root.name
        )
    else:
        candidate_root_for_name = args.candidate_acceptance_root.expanduser()
        candidate_acceptance_for_name = _load_acceptance(candidate_root_for_name)
        candidate_name = _batch_name(
            candidate_root_for_name, candidate_acceptance_for_name, args.candidate_name
        )

    out_dir = _default_out_dir(args, candidate_name).expanduser()
    _prepare_out_dir(out_dir, args.overwrite)
    logs_dir = out_dir / "logs"

    candidate_acceptance_result: CommandResult | None = None
    if args.candidate_source_root is not None:
        candidate_root, candidate_acceptance_result = _run_candidate_acceptance(
            args, candidate_name, logs_dir
        )
    else:
        candidate_root = args.candidate_acceptance_root.expanduser()

    baseline = _process_batch(baseline_name, baseline_root, args, out_dir, logs_dir)
    candidate = _process_batch(candidate_name, candidate_root, args, out_dir, logs_dir)
    if candidate_acceptance_result is not None:
        candidate["steps"].insert(0, candidate_acceptance_result.to_json())
        if not candidate_acceptance_result.ok:
            candidate["status"] = "fail"

    overall = _overall_status(candidate)
    report = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "overall_status": overall,
        "recommendation": _recommendation(overall),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "device": args.device,
        "thresholds": {
            "conf": float(args.conf),
            "nms_iou": float(args.nms_iou),
            "max_det": int(args.max_det),
            "imgsz": int(args.imgsz),
        },
        "baseline": baseline,
        "candidate": candidate,
        "comparison": _comparison(baseline, candidate),
    }
    _write_reports(report, out_dir)
    print()
    print(f"Regression report: {out_dir / 'regression_report.md'}")
    print(f"Overall status:    {overall}")
    print(f"Recommendation:    {report['recommendation']}")
    return 0 if overall in {"accepted", "provisional"} else 1


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
