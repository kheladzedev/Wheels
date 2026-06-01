"""Build a senior ML production-readiness evidence matrix.

This audit is intentionally stricter than a smoke test. It separates
"integration-ready evidence exists" from "production-ready evidence
exists", and keeps failed production blockers explicit instead of
turning missing Android/AR validation into a vague TODO.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_JSON_OUT = Path("outputs/production_audit/senior_ml_audit.json")
DEFAULT_MD_OUT = Path("docs/SENIOR_ML_AUDIT.md")
DEFAULT_ANDROID_LITERT_EVAL = Path("outputs/production_audit/android_litert_device_eval.json")
DEFAULT_AR_HOLDOUT_EVAL = Path("outputs/production_audit/ar_device_holdout_eval.json")
DEFAULT_AR_REPLAY_EVAL = Path("outputs/production_audit/ar_3d_replay_eval.json")
DEFAULT_PRODUCTION_EVIDENCE_AUDIT = Path("outputs/production_audit/production_evidence_audit.json")
DEFAULT_INTEGRATION_GATE = Path("outputs/production_audit/integration_gate.json")
DEFAULT_PRODUCTION_GATE = Path("outputs/production_audit/production_gate.json")
MIN_REAL_MAP50 = 0.85
MIN_REAL_OKS = 0.80
MAX_REAL_FN = 0.10
MAX_REAL_FP = 0.15


@dataclass
class Requirement:
    name: str
    category: str
    status: str
    integration_required: bool
    production_required: bool
    evidence: str
    detail: str

    @property
    def ok(self) -> bool:
        return self.status == "pass"


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def metric(report: dict[str, Any], *keys: str, default: float | None = None) -> float | None:
    cur: Any = report
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default


def _status(ok: bool, *, missing: bool = False) -> str:
    if missing:
        return "missing"
    return "pass" if ok else "fail"


def _req(
    name: str,
    category: str,
    status: str,
    evidence: Path | str,
    detail: str,
    *,
    integration_required: bool,
    production_required: bool,
) -> Requirement:
    return Requirement(
        name=name,
        category=category,
        status=status,
        integration_required=integration_required,
        production_required=production_required,
        evidence=str(evidence).replace("\\", "/"),
        detail=detail,
    )


def _report_ok_requirement(
    name: str,
    category: str,
    path: Path,
    *,
    integration_required: bool,
    production_required: bool,
) -> Requirement:
    report = read_json(path)
    missing = not path.is_file()
    ok = bool(report.get("ok", False))
    failures = report.get("failures", [])
    return _req(
        name,
        category,
        _status(ok, missing=missing),
        path,
        f"ok={ok}, failures={failures if failures else []}",
        integration_required=integration_required,
        production_required=production_required,
    )


def _dataset_audit_requirement(path: Path) -> Requirement:
    report = read_json(path)
    missing = not path.is_file()
    gate = report.get("gate") if isinstance(report.get("gate"), dict) else None
    if gate is not None:
        ok = gate.get("ok") is True
        detail = (
            f"gate_ok={ok}, overall_ok={report.get('ok', False)}, "
            f"scope={gate.get('scope', 'n/a')}, "
            f"failed_configs={gate.get('failed_configs', [])}, "
            f"missing_configs={gate.get('missing_configs', [])}"
        )
    else:
        ok = bool(report.get("ok", False))
        failures = report.get("failures", [])
        detail = f"ok={ok}, failures={failures if failures else []}"
    return _req(
        "dataset_format_and_leakage",
        "data",
        _status(ok, missing=missing),
        path,
        detail,
        integration_required=True,
        production_required=True,
    )


def _production_evidence_requirement(path: Path) -> Requirement:
    report = read_json(path)
    missing = not path.is_file()
    ready = bool(report.get("production_evidence_ready", False))
    blockers = report.get("blockers", [])
    return _req(
        "production_evidence_audit_ready",
        "production_validation",
        _status(ready, missing=missing),
        path,
        f"production_evidence_ready={ready}, blockers={blockers if blockers else []}",
        integration_required=False,
        production_required=True,
    )


def _gate_requirement(path: Path, mode: str, *, production_required: bool) -> Requirement:
    report = read_json(path)
    missing = not path.is_file()
    ok = bool(report.get("ok", False))
    failed = report.get("failed", [])
    return _req(
        f"{mode}_gate",
        "gating",
        _status(ok, missing=missing),
        path,
        f"ok={ok}, failed={failed if failed else []}",
        integration_required=mode == "integration",
        production_required=production_required,
    )


def _file_requirement(
    name: str,
    category: str,
    path: Path,
    *,
    integration_required: bool,
    production_required: bool,
) -> Requirement:
    return _req(
        name,
        category,
        _status(path.is_file(), missing=not path.is_file()),
        path,
        "present" if path.is_file() else "missing",
        integration_required=integration_required,
        production_required=production_required,
    )


def _champion_quality_requirement(
    path: Path,
    *,
    operating_point_path: Path | None = None,
) -> Requirement:
    operating_point = read_json(operating_point_path) if operating_point_path is not None else {}
    selected = (
        operating_point.get("selected")
        if isinstance(operating_point.get("selected"), dict)
        else None
    )
    if operating_point.get("ok") is True and selected is not None:
        missing = False
        map50 = metric(selected, "bbox_mAP50", default=0.0) or 0.0
        oks = metric(selected, "oks_mean", default=0.0) or 0.0
        fn = metric(selected, "false_negative_rate", default=1.0) or 1.0
        fp = metric(selected, "false_positive_rate", default=1.0) or 1.0
        conf = metric(selected, "conf", default=None)
        source = "operating_point"
        evidence = operating_point_path or path
        conf_text = f"{conf:.3f}" if conf is not None else "n/a"
        source_detail = (
            f"source={source}, report={selected.get('path', 'n/a')}, "
            f"conf={conf_text}, "
        )
    else:
        report = read_json(path)
        missing = not path.is_file()
        map50 = metric(report, "metrics_bbox", "mAP50", default=0.0) or 0.0
        oks = metric(report, "oks", "mean", default=0.0) or 0.0
        fn = metric(report, "rates", "false_negative_rate", default=1.0) or 1.0
        fp = metric(report, "rates", "false_positive_rate", default=1.0) or 1.0
        evidence = path
        source_detail = "source=default_eval, "
    ok = (
        map50 >= MIN_REAL_MAP50
        and oks >= MIN_REAL_OKS
        and fn <= MAX_REAL_FN
        and fp <= MAX_REAL_FP
    )
    return _req(
        "champion_real_validation_quality",
        "model_quality",
        _status(ok, missing=missing),
        evidence,
        (
            source_detail +
            f"bbox_mAP50={map50:.3f}>={MIN_REAL_MAP50:.3f}, "
            f"OKS={oks:.3f}>={MIN_REAL_OKS:.3f}, "
            f"FN={fn:.3f}<={MAX_REAL_FN:.3f}, "
            f"FP={fp:.3f}<={MAX_REAL_FP:.3f}"
        ),
        integration_required=True,
        production_required=True,
    )


def _model_inventory_requirement(path: Path) -> Requirement:
    report = read_json(path)
    missing = not path.is_file()
    counts = report.get("counts", {}) if isinstance(report.get("counts"), dict) else {}
    ok = bool(report.get("champion_run")) and int(counts.get("train_runs", 0)) > 0
    return _req(
        "model_inventory_lineage",
        "lineage",
        _status(ok, missing=missing),
        path,
        (
            f"train_runs={counts.get('train_runs', 'n/a')}, "
            f"artifacts={counts.get('artifacts', 'n/a')}, "
            f"eval_reports={counts.get('eval_reports', 'n/a')}, "
            f"champion_run={bool(report.get('champion_run'))}"
        ),
        integration_required=True,
        production_required=True,
    )


def _model_pool_requirement(path: Path) -> Requirement:
    glbs = len(list(path.glob("*.glb"))) if path.is_dir() else 0
    rejected = len(list((path / "rejected").glob("*.glb"))) if (path / "rejected").is_dir() else 0
    ok = glbs >= 300
    return _req(
        "external_3d_model_pool",
        "data",
        _status(ok, missing=not path.is_dir()),
        path,
        f"clean_glb={glbs}/300, rejected={rejected}",
        integration_required=True,
        production_required=False,
    )


def _ue_geometry_requirement(path: Path) -> Requirement:
    report = read_json(path)
    missing = not path.is_file()
    frames = int(report.get("frames_written", 0) or 0)
    wheels = int(report.get("wheels_written", 0) or 0)
    ok = frames >= 150 and wheels >= 500
    return _req(
        "ue_geometry_label_yield",
        "data",
        _status(ok, missing=missing),
        path,
        f"frames={frames}/150, wheels={wheels}/500",
        integration_required=True,
        production_required=False,
    )


def _export_drift_requirement(path: Path, name: str, *, production_required: bool) -> Requirement:
    report = read_json(path)
    missing = not path.is_file()
    ok = bool(report.get("ok", False))
    return _req(
        name,
        "export",
        _status(ok, missing=missing),
        path,
        (
            f"ok={ok}, matched={report.get('samples_matched', 'n/a')}/"
            f"{report.get('samples_checked', 'n/a')}, "
            f"max_bbox={float(report.get('max_bbox_drift_px', 0.0)):.3f}px, "
            f"max_kp={float(report.get('max_kp_drift_px', 0.0)):.3f}px"
        ),
        integration_required=False,
        production_required=production_required,
    )


def _tflite_cert_requirement(
    path: Path,
    *,
    name: str = "tflite_litert_certification",
) -> Requirement:
    report = read_json(path)
    missing = not path.is_file()
    certified = bool(report.get("certified", False))
    return _req(
        name,
        "export",
        _status(certified, missing=missing),
        path,
        (
            f"certified={certified}, status={report.get('status', 'n/a')}, "
            f"artifact={report.get('artifact', {}).get('path', 'n/a') if isinstance(report.get('artifact'), dict) else 'n/a'}"
        ),
        integration_required=False,
        production_required=True,
    )


def _ar_holdout_requirement(path: Path) -> Requirement:
    report = read_json(path)
    missing = not path.is_file()
    map50 = metric(report, "metrics_bbox", "mAP50", default=0.0) or 0.0
    oks = metric(report, "oks", "mean", default=0.0) or 0.0
    fn = metric(report, "rates", "false_negative_rate", default=1.0) or 1.0
    fp = metric(report, "rates", "false_positive_rate", default=1.0) or 1.0
    ok = (
        map50 >= MIN_REAL_MAP50
        and oks >= MIN_REAL_OKS
        and fn <= MAX_REAL_FN
        and fp <= MAX_REAL_FP
    )
    return _req(
        "human_labelled_ar_device_holdout",
        "production_validation",
        _status(ok, missing=missing),
        path,
        (
            f"bbox_mAP50={map50:.3f}>={MIN_REAL_MAP50:.3f}, "
            f"OKS={oks:.3f}>={MIN_REAL_OKS:.3f}, "
            f"FN={fn:.3f}<={MAX_REAL_FN:.3f}, "
            f"FP={fp:.3f}<={MAX_REAL_FP:.3f}"
        ),
        integration_required=False,
        production_required=True,
    )


def default_paths() -> argparse.Namespace:
    return argparse.Namespace(
        android_litert_eval=DEFAULT_ANDROID_LITERT_EVAL,
        ar_holdout_eval=DEFAULT_AR_HOLDOUT_EVAL,
        ar_replay_eval=DEFAULT_AR_REPLAY_EVAL,
        production_evidence_audit=DEFAULT_PRODUCTION_EVIDENCE_AUDIT,
        integration_gate=DEFAULT_INTEGRATION_GATE,
        production_gate=DEFAULT_PRODUCTION_GATE,
    )


def build_audit(args: argparse.Namespace | None = None) -> dict[str, Any]:
    args = args or default_paths()
    requirements = [
        _file_requirement(
            "champion_pytorch_artifact",
            "artifact",
            Path("runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt"),
            integration_required=True,
            production_required=True,
        ),
        _file_requirement(
            "champion_onnx_artifact",
            "artifact",
            Path("runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.onnx"),
            integration_required=True,
            production_required=False,
        ),
        _file_requirement(
            "champion_tflite_artifact",
            "artifact",
            Path("outputs/production_audit/tflite_export/best_float32.tflite"),
            integration_required=False,
            production_required=True,
        ),
        _file_requirement(
            "champion_coreml_artifact",
            "artifact",
            Path("outputs/production_audit/coreml_export/best.mlmodel"),
            integration_required=False,
            production_required=True,
        ),
        _model_inventory_requirement(Path("outputs/production_audit/model_inventory.json")),
        _report_ok_requirement(
            "model_selection_promotion_guard",
            "lineage",
            Path("outputs/production_audit/model_selection_audit.json"),
            integration_required=True,
            production_required=True,
        ),
        _model_pool_requirement(Path("data/sketchfab_cars")),
        _ue_geometry_requirement(Path("outputs/ue_tasks/render_sketchfab_geometry_labels_status.json")),
        _dataset_audit_requirement(Path("outputs/production_audit/dataset_audit.json")),
        _champion_quality_requirement(
            Path("outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_on_self_val.json"),
            operating_point_path=Path("outputs/production_audit/operating_point_audit.json"),
        ),
        _report_ok_requirement(
            "runtime_contract",
            "runtime_contract",
            Path("outputs/production_audit/runtime_contract_audit.json"),
            integration_required=True,
            production_required=True,
        ),
        _report_ok_requirement(
            "spec_compliance_contract",
            "runtime_contract",
            Path("outputs/production_audit/spec_compliance_audit.json"),
            integration_required=True,
            production_required=True,
        ),
        _report_ok_requirement(
            "performance_audit",
            "runtime_contract",
            Path("outputs/production_audit/performance_audit.json"),
            integration_required=True,
            production_required=True,
        ),
        _export_drift_requirement(
            Path("outputs/production_audit/onnx_drift_20.json"),
            "onnx_parity",
            production_required=False,
        ),
        _export_drift_requirement(
            Path("outputs/production_audit/tflite_drift_20.json"),
            "tflite_parity",
            production_required=False,
        ),
        _tflite_cert_requirement(
            Path("outputs/production_audit/export_certification.json"),
            name="export_backend_certification",
        ),
        _tflite_cert_requirement(Path("outputs/production_audit/tflite_certification.json")),
        _tflite_cert_requirement(
            Path("outputs/production_audit/coreml_certification.json"),
            name="coreml_certification",
        ),
        _report_ok_requirement(
            "android_litert_device_validation",
            "runtime_contract",
            args.android_litert_eval,
            integration_required=False,
            production_required=True,
        ),
        _ar_holdout_requirement(args.ar_holdout_eval),
        _report_ok_requirement(
            "ar_3d_replay_validation",
            "production_validation",
            args.ar_replay_eval,
            integration_required=False,
            production_required=True,
        ),
        _production_evidence_requirement(
            args.production_evidence_audit
        ),
        _file_requirement(
            "ar_holdout_evaluation_pipeline",
            "production_tooling",
            Path("src/evaluate_ar_holdout.py"),
            integration_required=True,
            production_required=False,
        ),
        _file_requirement(
            "ar_replay_validation_pipeline",
            "production_tooling",
            Path("src/validate_ar_replay.py"),
            integration_required=True,
            production_required=False,
        ),
        _gate_requirement(
            args.integration_gate,
            "integration",
            production_required=False,
        ),
        _gate_requirement(
            args.production_gate,
            "production",
            production_required=True,
        ),
    ]

    integration_blockers = [
        req.name for req in requirements if req.integration_required and not req.ok
    ]
    production_blockers = [
        req.name for req in requirements if req.production_required and not req.ok
    ]
    return {
        "schema_version": 1,
        "audit_ok": True,
        "integration_ready": not integration_blockers,
        "production_ready": not production_blockers,
        "counts": {
            "requirements": len(requirements),
            "passed": sum(1 for req in requirements if req.ok),
            "failed_or_missing": sum(1 for req in requirements if not req.ok),
            "integration_blockers": len(integration_blockers),
            "production_blockers": len(production_blockers),
        },
        "integration_blockers": integration_blockers,
        "production_blockers": production_blockers,
        "requirements": [asdict(req) | {"ok": req.ok} for req in requirements],
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Senior ML Audit",
        "",
        "Machine-readable production-readiness evidence matrix for the wheel-pose model.",
        "",
        f"- Audit OK: {audit.get('audit_ok')}",
        f"- Integration ready: {audit.get('integration_ready')}",
        f"- Production ready: {audit.get('production_ready')}",
        f"- Requirements: {audit.get('counts', {}).get('requirements', 'n/a')}",
        f"- Passed: {audit.get('counts', {}).get('passed', 'n/a')}",
        f"- Failed/missing: {audit.get('counts', {}).get('failed_or_missing', 'n/a')}",
        f"- Integration blockers: {', '.join(audit.get('integration_blockers', [])) or 'none'}",
        f"- Production blockers: {', '.join(audit.get('production_blockers', [])) or 'none'}",
        "",
        "| Requirement | Category | Status | Integration | Production | Evidence | Detail |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for req in audit.get("requirements", []):
        lines.append(
            "| "
            f"{req['name']} | "
            f"{req['category']} | "
            f"{req['status']} | "
            f"{req['integration_required']} | "
            f"{req['production_required']} | "
            f"`{req['evidence']}` | "
            f"{req['detail']} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    parser.add_argument("--android-litert-eval", type=Path, default=DEFAULT_ANDROID_LITERT_EVAL)
    parser.add_argument("--ar-holdout-eval", type=Path, default=DEFAULT_AR_HOLDOUT_EVAL)
    parser.add_argument("--ar-replay-eval", type=Path, default=DEFAULT_AR_REPLAY_EVAL)
    parser.add_argument(
        "--production-evidence-audit",
        type=Path,
        default=DEFAULT_PRODUCTION_EVIDENCE_AUDIT,
    )
    parser.add_argument("--integration-gate", type=Path, default=DEFAULT_INTEGRATION_GATE)
    parser.add_argument("--production-gate", type=Path, default=DEFAULT_PRODUCTION_GATE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = build_audit(args)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(audit), encoding="utf-8")
    print(
        f"audit_ok={audit['audit_ok']} integration_ready={audit['integration_ready']} "
        f"production_ready={audit['production_ready']} "
        f"production_blockers={audit['production_blockers']}"
    )
    print(f"json={args.json_out}")
    print(f"markdown={args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
