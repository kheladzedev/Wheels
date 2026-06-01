"""Build a machine-readable inventory of local YOLO-pose model runs.

The production audit needs more than "current champion exists": it
needs lineage, available exported artifacts, training datasets, and eval
evidence for every local run. This script scans `runs/pose/**/args.yaml`
and `outputs/eval/*.json`, then writes JSON + Markdown summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_RUNS_ROOT = Path("runs/pose")
DEFAULT_EVAL_ROOT = Path("outputs/eval")
DEFAULT_DEPLOYMENT_EXPORT_ROOT = Path("outputs/production_audit")
DEFAULT_JSON_OUT = Path("outputs/production_audit/model_inventory.json")
DEFAULT_MD_OUT = Path("docs/MODEL_INVENTORY.md")
WEIGHT_EXTS = (".pt", ".onnx", ".tflite", ".mlmodel")


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _metric(report: dict[str, Any], *keys: str) -> float | None:
    cur: Any = report
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return None


def _rel(path: Path) -> str:
    return str(path).replace("\\", "/")


def discover_runs(runs_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for args_path in sorted(runs_root.rglob("args.yaml")):
        run_dir = args_path.parent
        args = _read_yaml(args_path)
        if args.get("task") != "pose" or args.get("mode") != "train":
            continue
        weights_dir = run_dir / "weights"
        artifacts = sorted(
            path
            for path in weights_dir.glob("*")
            if path.is_file() and path.suffix.lower() in WEIGHT_EXTS
        )
        artifact_payload = [
            {
                "path": _rel(path),
                "kind": path.suffix.lower().lstrip("."),
                "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
            }
            for path in artifacts
        ]
        source_model = str(args.get("model", ""))
        data = str(args.get("data", ""))
        warnings: list[str] = []
        if source_model and source_model.endswith(".pt") and not Path(source_model).is_file():
            warnings.append(f"source_model_missing:{source_model}")
        if data and not Path(data).is_file():
            warnings.append(f"data_config_missing:{data}")
        runs.append(
            {
                "name": str(args.get("name") or run_dir.name),
                "run_dir": _rel(run_dir),
                "args_path": _rel(args_path),
                "source_model": source_model,
                "data": data,
                "epochs": args.get("epochs"),
                "batch": args.get("batch"),
                "imgsz": args.get("imgsz"),
                "device": args.get("device"),
                "seed": args.get("seed"),
                "artifacts": artifact_payload,
                "has_results_csv": (run_dir / "results.csv").is_file(),
                "warnings": warnings,
            }
        )
    return runs


def discover_eval_reports(eval_root: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in sorted(eval_root.glob("*.json")):
        report = _read_json(path)
        if not report:
            continue
        counts = report.get("counts", {}) if isinstance(report.get("counts"), dict) else {}
        reports.append(
            {
                "path": _rel(path),
                "name": path.stem,
                "model": str(report.get("model", "")),
                "data": str(report.get("data", "")),
                "bbox_mAP50": _metric(report, "metrics_bbox", "mAP50"),
                "bbox_mAP50_95": _metric(report, "metrics_bbox", "mAP50_95"),
                "oks_mean": _metric(report, "oks", "mean"),
                "fn_rate": _metric(report, "rates", "false_negative_rate"),
                "fp_rate": _metric(report, "rates", "false_positive_rate"),
                "gt_wheels": counts.get("gt_wheels"),
                "pred_wheels_above_conf": counts.get("pred_wheels_above_conf"),
                "matched": counts.get("matched"),
            }
        )
    return reports


def discover_deployment_artifacts(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    artifacts = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in WEIGHT_EXTS
    ]
    return [
        {
            "path": _rel(path),
            "kind": path.suffix.lower().lstrip("."),
            "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
        }
        for path in artifacts
    ]


def attach_evals(runs: list[dict[str, Any]], eval_reports: list[dict[str, Any]]) -> None:
    model_to_reports: dict[str, list[dict[str, Any]]] = {}
    for report in eval_reports:
        if report["model"]:
            model_to_reports.setdefault(report["model"], []).append(report)

    for run in runs:
        artifact_paths = {artifact["path"] for artifact in run["artifacts"]}
        attached: list[dict[str, Any]] = []
        for artifact_path in sorted(artifact_paths):
            attached.extend(model_to_reports.get(artifact_path, []))
        run["eval_reports"] = attached
        run["best_eval"] = pick_best_eval(attached)


def pick_best_eval(reports: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not reports:
        return None
    return max(
        reports,
        key=lambda r: (
            r["bbox_mAP50"] if r["bbox_mAP50"] is not None else -1.0,
            r["oks_mean"] if r["oks_mean"] is not None else -1.0,
            -(r["fn_rate"] if r["fn_rate"] is not None else 1.0),
        ),
    )


def build_inventory(
    runs_root: Path,
    eval_root: Path,
    champion: Path,
    deployment_export_root: Path = DEFAULT_DEPLOYMENT_EXPORT_ROOT,
) -> dict[str, Any]:
    runs = discover_runs(runs_root)
    eval_reports = discover_eval_reports(eval_root)
    deployment_artifacts = discover_deployment_artifacts(deployment_export_root)
    attach_evals(runs, eval_reports)

    champion_path = _rel(champion)
    champion_runs = [
        run
        for run in runs
        if any(artifact["path"] == champion_path for artifact in run["artifacts"])
    ]
    run_artifacts = [artifact for run in runs for artifact in run["artifacts"]]
    all_artifacts = [*run_artifacts, *deployment_artifacts]
    return {
        "runs_root": _rel(runs_root),
        "eval_root": _rel(eval_root),
        "deployment_export_root": _rel(deployment_export_root),
        "champion": champion_path,
        "counts": {
            "train_runs": len(runs),
            "artifacts": len(all_artifacts),
            "run_artifacts": len(run_artifacts),
            "deployment_artifacts": len(deployment_artifacts),
            "pt_artifacts": sum(1 for a in all_artifacts if a["kind"] == "pt"),
            "onnx_artifacts": sum(1 for a in all_artifacts if a["kind"] == "onnx"),
            "tflite_artifacts": sum(1 for a in all_artifacts if a["kind"] == "tflite"),
            "coreml_artifacts": sum(1 for a in all_artifacts if a["kind"] == "mlmodel"),
            "eval_reports": len(eval_reports),
            "runs_with_eval": sum(1 for run in runs if run.get("eval_reports")),
            "runs_with_warnings": sum(1 for run in runs if run.get("warnings")),
        },
        "champion_run": champion_runs[0] if champion_runs else None,
        "runs": runs,
        "deployment_artifacts": deployment_artifacts,
        "eval_reports": eval_reports,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(inventory: dict[str, Any]) -> str:
    counts = inventory["counts"]
    champion_run = inventory.get("champion_run")
    lines = [
        "# Model Inventory",
        "",
        "Generated from local `runs/pose/**/args.yaml` and `outputs/eval/*.json`.",
        "",
        "## Summary",
        "",
        f"- Train runs: {counts['train_runs']}",
        (
            f"- Artifacts: {counts['artifacts']} (`.pt`={counts['pt_artifacts']}, "
            f"`.onnx`={counts['onnx_artifacts']}, `.tflite`={counts['tflite_artifacts']}, "
            f"`.mlmodel`={counts['coreml_artifacts']})"
        ),
        f"- Run artifacts: {counts['run_artifacts']}",
        f"- Deployment artifacts: {counts['deployment_artifacts']}",
        f"- Eval reports: {counts['eval_reports']}",
        f"- Runs with eval evidence: {counts['runs_with_eval']}",
        f"- Runs with lineage warnings: {counts['runs_with_warnings']}",
        f"- Champion artifact: `{inventory['champion']}`",
    ]
    if champion_run is not None:
        lines.extend(
            [
                f"- Champion run: `{champion_run['run_dir']}`",
                f"- Champion training data: `{champion_run['data']}`",
                f"- Champion source model: `{champion_run['source_model']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| Run | Data | Source model | Artifacts | Best eval mAP50 | Best eval OKS | FN | FP | Warnings |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for run in inventory["runs"]:
        best_eval = run.get("best_eval") or {}
        lines.append(
            "| "
            f"`{run['name']}` | "
            f"`{run['data']}` | "
            f"`{run['source_model']}` | "
            f"{len(run['artifacts'])} | "
            f"{_fmt(best_eval.get('bbox_mAP50'))} | "
            f"{_fmt(best_eval.get('oks_mean'))} | "
            f"{_fmt(best_eval.get('fn_rate'))} | "
            f"{_fmt(best_eval.get('fp_rate'))} | "
            f"{', '.join(run.get('warnings') or []) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Deployment Artifacts",
            "",
            "| Artifact | Kind | Size MB |",
            "|---|---:|---:|",
        ]
    )
    for artifact in inventory.get("deployment_artifacts", []):
        lines.append(
            f"| `{artifact['path']}` | {artifact['kind']} | {_fmt(artifact['size_mb'])} |"
        )
    if not inventory.get("deployment_artifacts"):
        lines.append("| none | n/a | n/a |")
    lines.extend(
        [
            "",
            "## Eval Reports",
            "",
            "| Report | Model | Data | mAP50 | OKS | FN | FP | GT / pred / matched |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for report in inventory["eval_reports"]:
        lines.append(
            "| "
            f"`{report['path']}` | "
            f"`{report['model']}` | "
            f"`{report['data']}` | "
            f"{_fmt(report['bbox_mAP50'])} | "
            f"{_fmt(report['oks_mean'])} | "
            f"{_fmt(report['fn_rate'])} | "
            f"{_fmt(report['fp_rate'])} | "
            f"{_fmt(report['gt_wheels'])} / {_fmt(report['pred_wheels_above_conf'])} / {_fmt(report['matched'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--deployment-export-root", type=Path, default=DEFAULT_DEPLOYMENT_EXPORT_ROOT)
    parser.add_argument(
        "--champion",
        type=Path,
        default=Path("runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt"),
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inventory = build_inventory(
        args.runs_root,
        args.eval_root,
        args.champion,
        args.deployment_export_root,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(inventory), encoding="utf-8")
    print(
        f"runs={inventory['counts']['train_runs']} artifacts={inventory['counts']['artifacts']} "
        f"eval_reports={inventory['counts']['eval_reports']} warnings={inventory['counts']['runs_with_warnings']}"
    )
    print(f"json={args.json_out}")
    print(f"markdown={args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
