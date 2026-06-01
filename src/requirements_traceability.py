"""Generate a requirements-to-evidence traceability matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_JSON_OUT = Path("outputs/production_audit/requirements_traceability.json")
DEFAULT_MD_OUT = Path("docs/REQUIREMENTS_TRACEABILITY.md")


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def status_from_requirement(requirements: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for req in requirements:
        if req.get("name") == name:
            return req
    return {"name": name, "status": "missing", "detail": "requirement not found", "evidence": "n/a"}


def _row(
    *,
    requirement: str,
    status: str,
    evidence: str,
    detail: str,
    gap: str = "",
) -> dict[str, str]:
    return {
        "requirement": requirement,
        "status": status,
        "evidence": evidence,
        "detail": detail,
        "gap": gap,
    }


def build_traceability() -> dict[str, Any]:
    senior = read_json(Path("outputs/production_audit/senior_ml_audit.json"))
    evidence = read_json(Path("outputs/production_audit/production_evidence_audit.json"))
    inventory = read_json(Path("outputs/production_audit/model_inventory.json"))
    release = read_json(Path("outputs/production_audit/release_integrity.json"))
    reqs = senior.get("requirements", []) if isinstance(senior.get("requirements"), list) else []
    evidence_checks = {
        check.get("name"): check
        for check in evidence.get("checks", [])
        if isinstance(check, dict) and check.get("name")
    }

    def req_row(label: str, req_name: str, gap: str = "") -> dict[str, str]:
        req = status_from_requirement(reqs, req_name)
        return _row(
            requirement=label,
            status=str(req.get("status", "missing")),
            evidence=str(req.get("evidence", "n/a")),
            detail=str(req.get("detail", "")),
            gap=gap if req.get("status") != "pass" else "",
        )

    rows = [
        req_row("Champion model artifact exists", "champion_pytorch_artifact"),
        req_row("Exportable TFLite artifact exists", "champion_tflite_artifact"),
        req_row("Exportable CoreML artifact exists", "champion_coreml_artifact"),
        req_row("Model lineage and inventory are documented", "model_inventory_lineage"),
        req_row("Champion selection and promotion guard passes", "model_selection_promotion_guard"),
        req_row("Training/evaluation datasets pass format and leakage audit", "dataset_format_and_leakage"),
        req_row("Champion meets real-validation quality targets", "champion_real_validation_quality"),
        req_row("AR JSON runtime contract is implemented and smoke-tested", "runtime_contract"),
        req_row("ML deliverable matches the AR technical specification", "spec_compliance_contract"),
        req_row("Local performance audit passes", "performance_audit"),
        req_row("ONNX/TFLite export backends are certified", "export_backend_certification"),
        req_row("Desktop TFLite/LiteRT package is certified", "tflite_litert_certification"),
        req_row("Desktop CoreML package is certified", "coreml_certification"),
        req_row(
            "Android-device LiteRT evidence is validated",
            "android_litert_device_validation",
            gap=", ".join(evidence_checks.get("android_litert_device_validation", {}).get("failures", [])),
        ),
        req_row(
            "Human-labelled AR-device holdout passes production thresholds",
            "human_labelled_ar_device_holdout",
            gap=", ".join(evidence_checks.get("human_labelled_ar_device_holdout", {}).get("failures", [])),
        ),
        req_row(
            "AR-side 3D replay/RANSAC validation passes",
            "ar_3d_replay_validation",
            gap=", ".join(evidence_checks.get("ar_3d_replay_validation", {}).get("failures", [])),
        ),
        req_row(
            "Consolidated production evidence audit passes",
            "production_evidence_audit_ready",
            gap=", ".join(str(item) for item in evidence.get("blockers", [])),
        ),
        req_row("Production gate passes", "production_gate"),
    ]
    passed = sum(1 for row in rows if row["status"] == "pass")
    return {
        "schema_version": 1,
        "ok": True,
        "production_ready": bool(senior.get("production_ready", False)),
        "summary": {
            "requirements": len(rows),
            "passed": passed,
            "failed_or_missing": len(rows) - passed,
            "train_runs": inventory.get("counts", {}).get("train_runs"),
            "eval_reports": inventory.get("counts", {}).get("eval_reports"),
            "release_integrity_ok": release.get("ok"),
        },
        "rows": rows,
    }


def render_markdown(trace: dict[str, Any]) -> str:
    summary = trace.get("summary", {})
    lines = [
        "# Requirements Traceability",
        "",
        f"- Production ready: {trace.get('production_ready')}",
        f"- Requirements passed: {summary.get('passed')}/{summary.get('requirements')}",
        f"- Train runs inventoried: {summary.get('train_runs')}",
        f"- Eval reports inventoried: {summary.get('eval_reports')}",
        f"- Release integrity OK: {summary.get('release_integrity_ok')}",
        "",
        "| Requirement | Status | Evidence | Detail | Gap |",
        "|---|---|---|---|---|",
    ]
    for row in trace.get("rows", []):
        lines.append(
            "| "
            f"{row.get('requirement')} | "
            f"{row.get('status')} | "
            f"`{row.get('evidence')}` | "
            f"{row.get('detail')} | "
            f"{row.get('gap')} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trace = build_traceability()
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(trace), encoding="utf-8")
    print(
        f"ok={trace['ok']} production_ready={trace['production_ready']} "
        f"passed={trace['summary']['passed']}/{trace['summary']['requirements']}"
    )
    print(f"json={args.json_out}")
    print(f"markdown={args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
