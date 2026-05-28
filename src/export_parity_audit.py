"""Analyze exported-model parity drift reports.

`check_export_drift.py` answers the gate question: did every sampled
frame pass strict PT-vs-exported parity? This script answers the senior
engineering question: why did it fail, how concentrated is the failure,
and what tolerance would be required before the current artifacts pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ONNX_DRIFT = Path("outputs/production_audit/onnx_drift_20.json")
DEFAULT_TFLITE_DRIFT = Path("outputs/production_audit/tflite_drift_20.json")
DEFAULT_JSON_OUT = Path("outputs/production_audit/export_parity_audit.json")
DEFAULT_MD_OUT = Path("docs/EXPORT_PARITY_AUDIT.md")

DEFAULT_SWEEP = (
    (2.0, 3.0, 0.05),
    (3.0, 4.0, 0.05),
    (5.0, 8.0, 0.10),
    (10.0, 15.0, 0.25),
)


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def classify_failures(sample: dict[str, Any]) -> list[str]:
    failures = sample.get("failures", [])
    if sample.get("n_pt") != sample.get("n_exported"):
        return ["count_mismatch"]
    if not failures and sample.get("matched"):
        return []
    categories: list[str] = []
    text = "\n".join(str(item) for item in failures).lower()
    if "detection count differs" in text:
        categories.append("count_mismatch")
    if "bbox drift" in text:
        categories.append("bbox_drift")
    if "keypoint drift" in text or "keypoint shape" in text:
        categories.append("keypoint_drift")
    if "conf drift" in text:
        categories.append("confidence_drift")
    if failures and not categories:
        categories.append("other")
    return categories


def _float(sample: dict[str, Any], key: str) -> float:
    try:
        return float(sample.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def passes_at(sample: dict[str, Any], *, bbox_atol: float, kp_atol: float, conf_atol: float) -> bool:
    if sample.get("n_pt") != sample.get("n_exported"):
        return False
    return (
        _float(sample, "max_bbox_drift_px") <= bbox_atol
        and _float(sample, "max_kp_drift_px") <= kp_atol
        and _float(sample, "max_conf_drift") <= conf_atol
    )


def summarize_report(path: Path, name: str) -> dict[str, Any]:
    report = read_json(path)
    samples = report.get("samples", []) if isinstance(report.get("samples"), list) else []
    category_counts: dict[str, int] = {
        "count_mismatch": 0,
        "bbox_drift": 0,
        "keypoint_drift": 0,
        "confidence_drift": 0,
        "other": 0,
    }
    failed_samples: list[dict[str, Any]] = []
    coordinate_scale_warnings = 0
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        pair_diagnostics = (
            sample.get("pair_diagnostics", [])
            if isinstance(sample.get("pair_diagnostics"), list)
            else []
        )
        sample_scale_warnings = sum(
            1
            for pair in pair_diagnostics
            if isinstance(pair, dict) and pair.get("coordinate_scale_warning")
        )
        coordinate_scale_warnings += sample_scale_warnings
        categories = classify_failures(sample)
        if categories:
            for category in categories:
                category_counts[category] = category_counts.get(category, 0) + 1
            failed_samples.append(
                {
                    "image": sample.get("image"),
                    "categories": categories,
                    "n_pt": sample.get("n_pt"),
                    "n_exported": sample.get("n_exported"),
                    "max_bbox_drift_px": sample.get("max_bbox_drift_px"),
                    "max_kp_drift_px": sample.get("max_kp_drift_px"),
                    "max_conf_drift": sample.get("max_conf_drift"),
                    "coordinate_scale_warnings": sample_scale_warnings,
                    "failures": sample.get("failures", []),
                }
            )

    sweep = []
    for bbox_atol, kp_atol, conf_atol in DEFAULT_SWEEP:
        passed = sum(
            1
            for sample in samples
            if isinstance(sample, dict)
            and passes_at(sample, bbox_atol=bbox_atol, kp_atol=kp_atol, conf_atol=conf_atol)
        )
        sweep.append(
            {
                "bbox_atol": bbox_atol,
                "kp_atol": kp_atol,
                "conf_atol": conf_atol,
                "samples_passed": passed,
                "samples_checked": len(samples),
                "all_passed": passed == len(samples) and bool(samples),
            }
        )

    return {
        "name": name,
        "path": str(path),
        "present": path.is_file(),
        "ok": bool(report.get("ok", False)),
        "samples_checked": report.get("samples_checked", len(samples)),
        "samples_matched": report.get("samples_matched", 0),
        "max_bbox_drift_px": report.get("max_bbox_drift_px"),
        "max_kp_drift_px": report.get("max_kp_drift_px"),
        "max_conf_drift": report.get("max_conf_drift"),
        "thresholds": report.get("thresholds", {}),
        "category_counts": category_counts,
        "coordinate_scale_warnings": coordinate_scale_warnings,
        "failed_samples": failed_samples,
        "tolerance_sweep": sweep,
    }


def build_audit(onnx_path: Path, tflite_path: Path) -> dict[str, Any]:
    reports = {
        "onnx": summarize_report(onnx_path, "onnx"),
        "tflite": summarize_report(tflite_path, "tflite"),
    }
    findings = build_findings(reports)
    return {
        "ok": all(report["present"] for report in reports.values()),
        "certified": all(report["ok"] for report in reports.values()),
        "schema_version": 1,
        "reports": reports,
        "findings": findings,
        "recommendation": (
            "Do not certify exported artifacts as drop-in replacements under the current "
            "strict parity policy. ONNX and TFLite aggregate eval remain usable evidence, "
            "but production needs either fixed export parity or an explicitly approved "
            "aggregate/AR-holdout acceptance policy."
        ),
        "summary": {
            name: {
                "ok": report["ok"],
                "samples_matched": report["samples_matched"],
                "samples_checked": report["samples_checked"],
                "category_counts": report["category_counts"],
            }
            for name, report in reports.items()
        },
    }


def build_findings(reports: dict[str, dict[str, Any]]) -> list[str]:
    findings: list[str] = []
    onnx = reports.get("onnx", {})
    tflite = reports.get("tflite", {})
    if onnx.get("category_counts") == tflite.get("category_counts"):
        findings.append(
            "ONNX and TFLite have identical failure category counts; the issue is likely "
            "shared exported-backend/postprocess parity rather than a TFLite-only runtime bug."
        )
    for name, report in reports.items():
        categories = report.get("category_counts", {})
        if categories.get("count_mismatch", 0) == 0 and not report.get("ok", False):
            findings.append(f"{name}: strict failures are drift-only; no detection-count mismatch.")
        if report.get("coordinate_scale_warnings", 0):
            findings.append(
                f"{name}: {report['coordinate_scale_warnings']} matched pairs look like "
                "exported normalized coordinates compared with PyTorch pixel coordinates."
            )
        else:
            findings.append(f"{name}: no coordinate-scale warnings in the official strict report.")
        sweep = report.get("tolerance_sweep", [])
        first_all_pass = next((row for row in sweep if row.get("all_passed")), None)
        if first_all_pass:
            findings.append(
                f"{name}: all sampled frames pass only at bbox={first_all_pass['bbox_atol']}px, "
                f"kp={first_all_pass['kp_atol']}px, conf={first_all_pass['conf_atol']}."
            )
    return findings


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Export Parity Audit",
        "",
        "Diagnostic summary for PT-vs-exported parity drift reports.",
        "",
        f"- Audit OK: {audit.get('ok')}",
        f"- Certified: {audit.get('certified')}",
        f"- Recommendation: {audit.get('recommendation', 'n/a')}",
        "",
        "| Export | Strict OK | Matched | Max bbox px | Max kp px | Max conf | Failure categories |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, report in audit.get("reports", {}).items():
        categories = ", ".join(
            f"{key}={value}" for key, value in report.get("category_counts", {}).items() if value
        ) or "none"
        lines.append(
            "| "
            f"{name} | "
            f"{report.get('ok')} | "
            f"{report.get('samples_matched')}/{report.get('samples_checked')} | "
            f"{_fmt(report.get('max_bbox_drift_px'))} | "
            f"{_fmt(report.get('max_kp_drift_px'))} | "
            f"{_fmt(report.get('max_conf_drift'))} | "
            f"{categories}; scale_warnings={report.get('coordinate_scale_warnings', 'n/a')} |"
        )
    lines.extend(["", "## Findings", ""])
    for finding in audit.get("findings", []):
        lines.append(f"- {finding}")
    lines.extend(["", "## Tolerance Sweep", ""])
    for name, report in audit.get("reports", {}).items():
        lines.extend(
            [
                f"### {name}",
                "",
                "| bbox px | kp px | conf | Passed |",
                "|---:|---:|---:|---:|",
            ]
        )
        for row in report.get("tolerance_sweep", []):
            lines.append(
                "| "
                f"{_fmt(row['bbox_atol'])} | "
                f"{_fmt(row['kp_atol'])} | "
                f"{_fmt(row['conf_atol'])} | "
                f"{row['samples_passed']}/{row['samples_checked']} |"
            )
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-drift", type=Path, default=DEFAULT_ONNX_DRIFT)
    parser.add_argument("--tflite-drift", type=Path, default=DEFAULT_TFLITE_DRIFT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = build_audit(args.onnx_drift, args.tflite_drift)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(audit), encoding="utf-8")
    print(f"ok={audit['ok']} certified={audit['certified']}")
    for name, report in audit["reports"].items():
        print(
            f"{name}: strict_ok={report['ok']} "
            f"matched={report['samples_matched']}/{report['samples_checked']} "
            f"categories={report['category_counts']}"
        )
    print(f"json={args.json_out}")
    print(f"markdown={args.md_out}")
    return 0 if audit["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
