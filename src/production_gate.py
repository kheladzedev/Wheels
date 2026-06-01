"""Machine-readable integration/production gate for the wheel model."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class GateItem:
    name: str
    ok: bool
    severity: str
    detail: str


def read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def strict_true(value: object) -> bool:
    return value is True


def metric(report: dict, *keys: str, default: float | None = None) -> float | None:
    cur: object = report
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    if isinstance(cur, bool):
        return default
    try:
        value = float(cur)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def file_gate(name: str, path: Path, severity: str = "fail") -> GateItem:
    return GateItem(name, path.is_file(), severity, str(path))


def certification_gate(
    name: str, path: Path, severity: str = "production_fail"
) -> GateItem:
    report = read_json(path)
    certified = strict_true(report.get("certified", False))
    if not path.is_file():
        detail = f"missing: {path}"
    else:
        status = report.get("status", "unknown")
        reason = report.get("reason", report.get("failure_reason", "n/a"))
        detail = f"certified={certified}, status={status}, reason={reason}"
    return GateItem(name, certified, severity, detail)


def report_ok_gate(
    name: str, path: Path, severity: str = "production_fail"
) -> GateItem:
    report = read_json(path)
    ok = strict_true(report.get("ok", False))
    if not path.is_file():
        detail = f"missing: {path}"
    else:
        failures = report.get("failures", [])
        detail = f"ok={ok}, failures={failures if failures else '[]'}"
    return GateItem(name, ok, severity, detail)


def dataset_audit_gate(
    name: str, path: Path, severity: str = "fail"
) -> GateItem:
    report = read_json(path)
    gate = report.get("gate") if isinstance(report.get("gate"), dict) else None
    if gate is not None:
        ok = strict_true(gate.get("ok", False))
    else:
        ok = strict_true(report.get("ok", False))
    if not path.is_file():
        detail = f"missing: {path}"
    elif gate is not None:
        detail = (
            f"gate_ok={ok}, overall_ok={report.get('ok', False)}, "
            f"scope={gate.get('scope', 'n/a')}, "
            f"failed_configs={gate.get('failed_configs', [])}, "
            f"missing_configs={gate.get('missing_configs', [])}"
        )
    else:
        failures = report.get("failures", [])
        detail = f"ok={ok}, failures={failures if failures else '[]'}"
    return GateItem(name, ok, severity, detail)


def evidence_ready_gate(
    name: str, path: Path, severity: str = "production_fail"
) -> GateItem:
    report = read_json(path)
    ready = strict_true(report.get("production_evidence_ready", False))
    if not path.is_file():
        detail = f"missing: {path}"
    else:
        blockers = report.get("blockers", [])
        detail = f"production_evidence_ready={ready}, blockers={blockers if blockers else '[]'}"
    return GateItem(name, ready, severity, detail)


def eval_quality_gate(
    name: str,
    path: Path,
    *,
    min_map50: float,
    min_oks: float,
    max_fn: float,
    max_fp: float = 1.0,
    severity: str = "production_fail",
) -> GateItem:
    report = read_json(path)
    map50 = metric(report, "metrics_bbox", "mAP50", default=0.0) or 0.0
    oks = metric(report, "oks", "mean", default=0.0) or 0.0
    fn = metric(report, "rates", "false_negative_rate", default=1.0) or 1.0
    fp = metric(report, "rates", "false_positive_rate", default=1.0) or 1.0
    ok = (
        path.is_file()
        and map50 >= min_map50
        and oks >= min_oks
        and fn <= max_fn
        and fp <= max_fp
    )
    if not path.is_file():
        detail = f"missing: {path}"
    else:
        detail = (
            f"bbox_mAP50={map50:.3f}>={min_map50:.3f}, "
            f"OKS={oks:.3f}>={min_oks:.3f}, FN={fn:.3f}<={max_fn:.3f}, "
            f"FP={fp:.3f}<={max_fp:.3f}"
        )
    return GateItem(name, ok, severity, detail)


def real_quality_metrics(
    champion_real_eval: Path,
    operating_point_audit: Path | None = None,
) -> dict[str, Any]:
    audit = read_json(operating_point_audit) if operating_point_audit is not None else {}
    selected = audit.get("selected") if isinstance(audit.get("selected"), dict) else None
    if strict_true(audit.get("ok", False)) and selected is not None:
        return {
            "source": "operating_point",
            "path": selected.get("path", str(operating_point_audit)),
            "conf": metric(selected, "conf", default=None),
            "map50": metric(selected, "bbox_mAP50", default=0.0) or 0.0,
            "oks": metric(selected, "oks_mean", default=0.0) or 0.0,
            "fn": metric(selected, "false_negative_rate", default=1.0) or 1.0,
            "fp": metric(selected, "false_positive_rate", default=1.0) or 1.0,
            "operating_point_ok": True,
        }
    report = read_json(champion_real_eval)
    return {
        "source": "default_eval",
        "path": str(champion_real_eval),
        "conf": metric(report, "thresholds", "conf", default=None),
        "map50": metric(report, "metrics_bbox", "mAP50", default=0.0) or 0.0,
        "oks": metric(report, "oks", "mean", default=0.0) or 0.0,
        "fn": metric(report, "rates", "false_negative_rate", default=1.0) or 1.0,
        "fp": metric(report, "rates", "false_positive_rate", default=1.0) or 1.0,
        "operating_point_ok": False,
    }


def build_gate_items(args: argparse.Namespace) -> list[GateItem]:
    anchor_eval = read_json(args.champion_anchor_eval)
    onnx_eval = read_json(args.onnx_eval)
    drift = read_json(args.onnx_drift)
    operating_point_audit = getattr(
        args,
        "operating_point_audit",
        Path("outputs/production_audit/operating_point_audit.json"),
    )
    real_quality = real_quality_metrics(args.champion_real_eval, operating_point_audit)

    real_map50 = float(real_quality["map50"])
    real_oks = float(real_quality["oks"])
    real_fn = float(real_quality["fn"])
    real_fp = float(real_quality["fp"])
    real_source = str(real_quality["source"])
    real_path = str(real_quality["path"])
    real_conf = real_quality.get("conf")
    real_conf_detail = (
        f", conf={float(real_conf):.3f}" if isinstance(real_conf, (int, float)) else ""
    )
    anchor_map50 = metric(anchor_eval, "metrics_bbox", "mAP50", default=0.0) or 0.0
    onnx_map50 = metric(onnx_eval, "metrics_bbox", "mAP50", default=0.0) or 0.0
    onnx_drift_ok = strict_true(drift.get("ok", False))

    items = [
        file_gate("champion_pt_exists", args.champion_pt),
        file_gate("champion_onnx_exists", args.champion_onnx, severity="warn"),
        file_gate("contract_doc_exists", Path("docs/AR_ML_CONTRACT.md")),
        file_gate(
            "production_audit_exists", Path("docs/PRODUCTION_READINESS_AUDIT.md")
        ),
        dataset_audit_gate("dataset_audit", args.dataset_audit, severity="fail"),
        GateItem(
            "real_only_operating_point",
            bool(real_quality["operating_point_ok"]),
            "fail",
            f"source={real_source}, report={real_path}{real_conf_detail}",
        ),
        report_ok_gate("performance_audit", args.performance_audit, severity="fail"),
        report_ok_gate("release_integrity", args.release_integrity, severity="fail"),
        report_ok_gate(
            "runtime_contract_audit", args.runtime_contract_audit, severity="fail"
        ),
        GateItem(
            "real_only_bbox_map50_target",
            real_map50 >= args.min_real_map50,
            "fail",
            f"{real_map50:.3f} >= {args.min_real_map50:.3f} ({real_source})",
        ),
        GateItem(
            "real_only_oks_floor",
            real_oks >= args.min_real_oks,
            "fail",
            f"{real_oks:.3f} >= {args.min_real_oks:.3f} ({real_source})",
        ),
        GateItem(
            "real_only_fn_ceiling",
            real_fn <= args.max_real_fn,
            "fail",
            f"{real_fn:.3f} <= {args.max_real_fn:.3f} ({real_source})",
        ),
        GateItem(
            "real_only_fp_ceiling",
            real_fp <= args.max_real_fp,
            "fail",
            f"{real_fp:.3f} <= {args.max_real_fp:.3f} ({real_source})",
        ),
        GateItem(
            "mixed_anchor_regression_signal",
            anchor_map50 >= args.min_anchor_map50,
            "warn",
            f"{anchor_map50:.3f} >= {args.min_anchor_map50:.3f}",
        ),
        GateItem(
            "onnx_aggregate_eval",
            onnx_map50 >= args.min_onnx_map50,
            "warn",
            f"{onnx_map50:.3f} >= {args.min_onnx_map50:.3f}",
        ),
        GateItem(
            "onnx_strict_parity_diagnostic",
            onnx_drift_ok,
            "warn",
            (
                f"samples={drift.get('samples_matched', 'n/a')}/"
                f"{drift.get('samples_checked', 'n/a')}, "
                f"max_kp={metric(drift, 'max_kp_drift_px', default=0.0):.3f}px"
            ),
        ),
        certification_gate("exported_backends_certified", args.export_certification),
        certification_gate("tflite_litert_certified", args.tflite_certified),
        evidence_ready_gate(
            "production_evidence_audit_ready", args.production_evidence_audit
        ),
        report_ok_gate("android_litert_device_eval", args.android_litert_eval),
        eval_quality_gate(
            "human_ar_holdout_eval",
            args.ar_holdout_eval,
            min_map50=args.min_ar_holdout_map50,
            min_oks=args.min_ar_holdout_oks,
            max_fn=args.max_ar_holdout_fn,
            max_fp=args.max_ar_holdout_fp,
        ),
        report_ok_gate("ar_3d_replay_eval", args.ar_3d_eval),
    ]
    return items


def evaluate(items: list[GateItem], mode: str) -> bool:
    severities = blocking_severities(mode)
    return all(item.ok for item in items if item.severity in severities)


def blocking_severities(mode: str) -> set[str]:
    if mode == "integration":
        return {"fail"}
    if mode == "production":
        return {"fail", "production_fail"}
    raise ValueError(f"unknown mode: {mode}")


def failed_items(items: list[GateItem], mode: str) -> list[str]:
    severities = blocking_severities(mode)
    return [item.name for item in items if not item.ok and item.severity in severities]


def warning_items(items: list[GateItem]) -> list[str]:
    return [item.name for item in items if not item.ok and item.severity == "warn"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("integration", "production"), default="production"
    )
    parser.add_argument(
        "--champion-pt",
        type=Path,
        default=Path(
            "runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt"
        ),
    )
    parser.add_argument(
        "--champion-onnx",
        type=Path,
        default=Path(
            "runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.onnx"
        ),
    )
    parser.add_argument(
        "--champion-real-eval",
        type=Path,
        default=Path(
            "outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_on_self_val.json"
        ),
    )
    parser.add_argument(
        "--operating-point-audit",
        type=Path,
        default=Path("outputs/production_audit/operating_point_audit.json"),
    )
    parser.add_argument(
        "--champion-anchor-eval",
        type=Path,
        default=Path("outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s.json"),
    )
    parser.add_argument(
        "--onnx-eval",
        type=Path,
        default=Path(
            "outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_onnx_on_self_plus_ue_val.json"
        ),
    )
    parser.add_argument(
        "--onnx-drift",
        type=Path,
        default=Path("outputs/production_audit/onnx_drift_20.json"),
    )
    parser.add_argument(
        "--tflite-certified",
        type=Path,
        default=Path("outputs/production_audit/tflite_certification.json"),
    )
    parser.add_argument(
        "--export-certification",
        type=Path,
        default=Path("outputs/production_audit/export_certification.json"),
    )
    parser.add_argument(
        "--ar-holdout-eval",
        type=Path,
        default=Path("outputs/production_audit/ar_device_holdout_eval.json"),
    )
    parser.add_argument(
        "--android-litert-eval",
        type=Path,
        default=Path("outputs/production_audit/android_litert_device_eval.json"),
    )
    parser.add_argument(
        "--ar-3d-eval",
        type=Path,
        default=Path("outputs/production_audit/ar_3d_replay_eval.json"),
    )
    parser.add_argument(
        "--dataset-audit",
        type=Path,
        default=Path("outputs/production_audit/dataset_audit.json"),
    )
    parser.add_argument(
        "--release-integrity",
        type=Path,
        default=Path("outputs/production_audit/release_integrity.json"),
    )
    parser.add_argument(
        "--performance-audit",
        type=Path,
        default=Path("outputs/production_audit/performance_audit.json"),
    )
    parser.add_argument(
        "--runtime-contract-audit",
        type=Path,
        default=Path("outputs/production_audit/runtime_contract_audit.json"),
    )
    parser.add_argument(
        "--production-evidence-audit",
        type=Path,
        default=Path("outputs/production_audit/production_evidence_audit.json"),
    )
    parser.add_argument("--min-real-map50", type=float, default=0.85)
    parser.add_argument("--min-real-oks", type=float, default=0.80)
    parser.add_argument("--max-real-fn", type=float, default=0.10)
    parser.add_argument("--max-real-fp", type=float, default=0.15)
    parser.add_argument("--min-anchor-map50", type=float, default=0.65)
    parser.add_argument("--min-onnx-map50", type=float, default=0.65)
    parser.add_argument("--min-ar-holdout-map50", type=float, default=0.85)
    parser.add_argument("--min-ar-holdout-oks", type=float, default=0.80)
    parser.add_argument("--max-ar-holdout-fn", type=float, default=0.10)
    parser.add_argument("--max-ar-holdout-fp", type=float, default=0.15)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    items = build_gate_items(args)
    ok = evaluate(items, args.mode)
    payload = {
        "mode": args.mode,
        "ok": ok,
        "items": [asdict(item) for item in items],
        "failed": failed_items(items, args.mode),
        "warnings": warning_items(items),
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    for item in items:
        status = "OK" if item.ok else item.severity.upper()
        print(f"{status:15} {item.name}: {item.detail}")
    print(f"Gate {args.mode}: {'PASS' if ok else 'FAIL'}")
    if args.json_out is not None:
        print(f"JSON: {args.json_out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
