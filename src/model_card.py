"""Generate the production model card for the wheel pose model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path("docs/MODEL_CARD.md")


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def metric(report: dict[str, Any], *keys: str, default: Any = "n/a") -> Any:
    cur: Any = report
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def eval_line(label: str, report: dict[str, Any]) -> str:
    counts = report.get("counts", {}) if isinstance(report.get("counts"), dict) else {}
    return (
        "| "
        f"{label} | "
        f"{metric(report, 'metrics_bbox', 'mAP50')} | "
        f"{metric(report, 'metrics_bbox', 'mAP50_95')} | "
        f"{metric(report, 'oks', 'mean')} | "
        f"{metric(report, 'rates', 'false_negative_rate')} | "
        f"{metric(report, 'rates', 'false_positive_rate')} | "
        f"{counts.get('gt_wheels', 'n/a')}/{counts.get('pred_wheels_above_conf', 'n/a')}/{counts.get('matched', 'n/a')} |"
    )


def build_model_card() -> str:
    inventory = read_json(Path("outputs/production_audit/model_inventory.json"))
    dataset_audit = read_json(Path("outputs/production_audit/dataset_audit.json"))
    senior = read_json(Path("outputs/production_audit/senior_ml_audit.json"))
    evidence = read_json(Path("outputs/production_audit/production_evidence_audit.json"))
    export_cert = read_json(Path("outputs/production_audit/export_certification.json"))
    tflite_cert = read_json(Path("outputs/production_audit/tflite_certification.json"))
    performance = read_json(Path("outputs/production_audit/performance_audit.json"))
    release = read_json(Path("outputs/production_audit/release_integrity.json"))
    pt_real = read_json(Path("outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_on_self_val.json"))
    pt_anchor = read_json(Path("outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s.json"))
    onnx_anchor = read_json(Path("outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_onnx_on_self_plus_ue_val.json"))
    tflite_anchor = read_json(Path("outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_tflite_on_self_plus_ue_val.json"))

    champion = inventory.get("champion_run") or {}
    counts = dataset_audit.get("counts", {}) if isinstance(dataset_audit.get("counts"), dict) else {}
    prod_blockers = senior.get("production_blockers", [])
    evidence_blockers = evidence.get("blockers", [])
    perf_pt = performance.get("benchmarks", {}).get("pytorch_cpu", {})
    perf_onnx = performance.get("benchmarks", {}).get("onnx_cpu", {})
    perf_litert = performance.get("benchmarks", {}).get("litert_cpu_smoke", {})

    lines = [
        "# Wheel Pose Model Card",
        "",
        "## Summary",
        "",
        "- Task: single-class wheel detection with three keypoints: `a`, `b`, `c_disc_bottom`.",
        "- Intended use: AR integration that raycasts wheel floor points and disc-bottom point into 3D.",
        "- Current status: integration-ready, not full production-ready until external Android/AR evidence is present.",
        f"- Production ready: {senior.get('production_ready', 'n/a')}",
        f"- Production blockers: {', '.join(prod_blockers) if isinstance(prod_blockers, list) else 'n/a'}",
        "",
        "## Champion",
        "",
        f"- PyTorch artifact: `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt`",
        f"- ONNX artifact: `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.onnx`",
        f"- TFLite artifact: `outputs/production_audit/tflite_export/best_float32.tflite`",
        f"- Training run: `{champion.get('run_dir', 'n/a')}`",
        f"- Training data config: `{champion.get('data', 'n/a')}`",
        f"- Source model: `{champion.get('source_model', 'n/a')}`",
        f"- Epochs / image size / batch: {champion.get('epochs', 'n/a')} / {champion.get('imgsz', 'n/a')} / {champion.get('batch', 'n/a')}",
        "",
        "## Data",
        "",
        f"- Dataset audit OK: {dataset_audit.get('ok', 'n/a')}",
        f"- Dataset configs checked: {counts.get('configs', 'n/a')}",
        f"- Total train images across configs: {counts.get('total_train_images', 'n/a')}",
        f"- Total val images across configs: {counts.get('total_val_images', 'n/a')}",
        f"- Total wheel labels across configs: {counts.get('total_wheel_labels', 'n/a')}",
        "- External 3D car pool: 300 clean GLBs from Sketchfab/Objaverse.",
        "- UE clean geometry labels: 132 frames / 548 wheels after QA filtering.",
        "",
        "## Metrics",
        "",
        "| Eval | bbox mAP50 | bbox mAP50-95 | OKS | FN rate | FP rate | GT/pred/matched |",
        "|---|---:|---:|---:|---:|---:|---:|",
        eval_line("PyTorch real-only validation", pt_real),
        eval_line("PyTorch mixed anchor validation", pt_anchor),
        eval_line("ONNX mixed anchor validation", onnx_anchor),
        eval_line("TFLite mixed anchor validation", tflite_anchor),
        "",
        "## Export And Runtime",
        "",
        f"- Export backend certification: {export_cert.get('certified', 'n/a')} (`{export_cert.get('scope', 'n/a')}`)",
        f"- TFLite package certification: {tflite_cert.get('certified', 'n/a')} (`{tflite_cert.get('scope', 'n/a')}`)",
        f"- PyTorch CPU mean latency: {metric(perf_pt, 'latency_ms', 'mean')} ms",
        f"- ONNX CPU mean latency: {metric(perf_onnx, 'latency_ms', 'mean')} ms",
        f"- LiteRT desktop smoke mean latency: {metric(perf_litert, 'latency_ms', 'mean')} ms",
        "- Android-device LiteRT validation is not yet present.",
        "",
        "## Production Evidence",
        "",
        f"- Evidence ready: {evidence.get('production_evidence_ready', 'n/a')}",
        f"- Evidence blockers: {', '.join(evidence_blockers) if isinstance(evidence_blockers, list) else 'n/a'}",
        "- Required evidence contract: `docs/PRODUCTION_EVIDENCE_CHECKLIST.md`.",
        "",
        "## Limitations",
        "",
        "- No human-labelled AR-device holdout has been evaluated yet.",
        "- No AR 3D replay/RANSAC validation report is present yet.",
        "- Android-device LiteRT latency/memory/output evidence is not present yet.",
        "- Synthetic/UE geometry labels are useful for coverage, but are not a replacement for real AR-device validation.",
        "",
        "## Release",
        "",
        f"- Release integrity OK: {release.get('ok', 'n/a')}",
        "- Deterministic package manifest: `docs/RELEASE_PACKAGE.md` / `outputs/production_audit/release_integrity.json`.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_model_card(), encoding="utf-8")
    print(f"model_card={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
