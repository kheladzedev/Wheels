"""Audit completion of the original senior-ML wheel-pose objective.

This report is deliberately objective-level rather than tool-level: it
maps the requested work to concrete evidence and separates integration
readiness from full production certification.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_JSON_OUT = Path("outputs/production_audit/objective_completion_audit.json")
DEFAULT_MD_OUT = Path("docs/OBJECTIVE_COMPLETION_AUDIT.md")

MODEL_POOL_ROOT = Path("data/sketchfab_cars")
UE_CLEAN_ROOT = Path("data/incoming/ue_sketchfab_geometry_clean")
TARGET_MODELS = 300
MIN_UE_CLEAN_IMAGES = 120
MIN_UE_CLEAN_WHEELS = 500

PT_CHAMPION = Path("runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt")
ONNX_CHAMPION = Path("runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.onnx")
TFLITE_CHAMPION = Path("outputs/production_audit/tflite_export/best_float32.tflite")


@dataclass
class Requirement:
    id: str
    title: str
    status: str
    evidence: str
    detail: str
    production_required: bool = True


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def count_files(root: Path, pattern: str) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for _ in root.glob(pattern))


def count_wheels(root: Path) -> tuple[int, int]:
    images = count_files(root / "images", "*")
    wheels = 0
    annotation_root = root / "annotations"
    if annotation_root.is_dir():
        for path in annotation_root.glob("*.json"):
            payload = read_json(path)
            wheel_list = payload.get("wheels", [])
            if isinstance(wheel_list, list):
                wheels += len(wheel_list)
    return images, wheels


def status_for(condition: bool, *, present: bool = True) -> str:
    if not present:
        return "missing"
    return "pass" if condition else "fail"


def req(
    id: str,
    title: str,
    condition: bool,
    evidence: str,
    detail: str,
    *,
    present: bool = True,
    production_required: bool = True,
) -> Requirement:
    return Requirement(
        id=id,
        title=title,
        status=status_for(condition, present=present),
        evidence=evidence,
        detail=detail,
        production_required=production_required,
    )


def build_audit() -> dict[str, Any]:
    model_inventory = read_json(Path("outputs/production_audit/model_inventory.json"))
    dataset_audit = read_json(Path("outputs/production_audit/dataset_audit.json"))
    spec_compliance = read_json(Path("outputs/production_audit/spec_compliance_audit.json"))
    senior_audit = read_json(Path("outputs/production_audit/senior_ml_audit.json"))
    audit_suite = read_json(Path("outputs/production_audit/audit_suite_status.json"))
    release_integrity = read_json(Path("outputs/production_audit/release_integrity.json"))
    report_consistency = read_json(Path("outputs/production_audit/report_consistency_audit.json"))
    production_evidence = read_json(Path("outputs/production_audit/production_evidence_audit.json"))
    integration_gate = read_json(Path("outputs/production_audit/integration_gate.json"))
    production_gate = read_json(Path("outputs/production_audit/production_gate.json"))
    export_certification = read_json(Path("outputs/production_audit/export_certification.json"))
    tflite_certification = read_json(Path("outputs/production_audit/tflite_certification.json"))
    handoff_manifest = read_json(
        Path("outputs/production_audit/external_evidence_handoff_bundle_manifest.json")
    )
    handoff_verification = read_json(
        Path("outputs/production_audit/external_evidence_handoff_bundle_verification.json")
    )
    return_template = read_json(
        Path("outputs/production_audit/external_evidence_return_template_manifest.json")
    )

    inventory_counts = model_inventory.get("counts", {})
    dataset_counts = dataset_audit.get("counts", {})
    clean_glbs = count_files(MODEL_POOL_ROOT, "*.glb")
    rejected_glbs = count_files(MODEL_POOL_ROOT / "rejected", "*.glb")
    ue_images, ue_wheels = count_wheels(UE_CLEAN_ROOT)
    champion_files = [PT_CHAMPION, ONNX_CHAMPION, TFLITE_CHAMPION]
    champion_present = all(path.is_file() for path in champion_files)
    handoff_hash_match = (
        bool(handoff_manifest.get("bundle_sha256"))
        and handoff_manifest.get("bundle_sha256") == handoff_verification.get("bundle_sha256")
    )
    handoff_current_artifacts = handoff_verification.get("current_artifacts", [])
    handoff_current_artifacts_ok = (
        isinstance(handoff_current_artifacts, list)
        and len(handoff_current_artifacts) == handoff_manifest.get("artifact_count")
        and all(
            isinstance(item, dict) and item.get("ok") is True
            for item in handoff_current_artifacts
        )
    )
    integration_ready = bool(integration_gate.get("ok")) and bool(
        senior_audit.get("integration_ready", audit_suite.get("integration_ready"))
    )
    production_ready = bool(production_gate.get("ok")) and bool(
        senior_audit.get("production_ready", audit_suite.get("production_ready"))
    )

    requirements = [
        req(
            "model_inventory_reviewed",
            "Model inventory and lineage reviewed",
            bool(inventory_counts.get("train_runs")) and bool(inventory_counts.get("artifacts")),
            "outputs/production_audit/model_inventory.json; docs/MODEL_INVENTORY.md",
            (
                f"train_runs={inventory_counts.get('train_runs', 'n/a')}, "
                f"artifacts={inventory_counts.get('artifacts', 'n/a')}, "
                f"eval_reports={inventory_counts.get('eval_reports', 'n/a')}"
            ),
            present=bool(model_inventory),
        ),
        req(
            "training_data_reviewed",
            "Training data audit reviewed",
            bool(dataset_audit.get("ok")),
            "outputs/production_audit/dataset_audit.json; docs/DATASET_AUDIT.md",
            (
                f"configs={dataset_counts.get('configs', 'n/a')}, "
                f"failed={dataset_counts.get('failed', 'n/a')}, "
                f"wheel_labels={dataset_counts.get('total_wheel_labels', 'n/a')}"
            ),
            present=bool(dataset_audit),
        ),
        req(
            "technical_spec_compliance_reviewed",
            "AR technical specification compliance reviewed",
            bool(spec_compliance.get("ok")),
            "outputs/production_audit/spec_compliance_audit.json; docs/SPEC_COMPLIANCE_AUDIT.md",
            (
                f"ok={spec_compliance.get('ok', 'n/a')}, "
                f"failures={spec_compliance.get('failures', 'n/a')}"
            ),
            present=bool(spec_compliance),
        ),
        req(
            "three_d_model_collection_done",
            "300 external car-body GLBs collected",
            clean_glbs >= TARGET_MODELS,
            str(MODEL_POOL_ROOT),
            f"clean_glbs={clean_glbs}/{TARGET_MODELS}, rejected={rejected_glbs}",
            present=MODEL_POOL_ROOT.is_dir(),
        ),
        req(
            "ue_mcp_synthetic_data_done",
            "UE/MCP synthetic scan-style data generated and cleaned",
            ue_images >= MIN_UE_CLEAN_IMAGES and ue_wheels >= MIN_UE_CLEAN_WHEELS,
            str(UE_CLEAN_ROOT),
            (
                f"images={ue_images}/{MIN_UE_CLEAN_IMAGES}, "
                f"wheels={ue_wheels}/{MIN_UE_CLEAN_WHEELS}"
            ),
            present=UE_CLEAN_ROOT.is_dir(),
        ),
        req(
            "champion_artifacts_present",
            "Champion PT/ONNX/TFLite artifacts present",
            champion_present,
            "; ".join(str(path) for path in champion_files),
            ", ".join(f"{path.name}={path.is_file()}" for path in champion_files),
        ),
        req(
            "desktop_export_certified",
            "Desktop ONNX/TFLite export certification passed",
            bool(export_certification.get("certified")) and bool(tflite_certification.get("certified")),
            "outputs/production_audit/export_certification.json; outputs/production_audit/tflite_certification.json",
            (
                f"export_certified={export_certification.get('certified', 'n/a')}, "
                f"tflite_certified={tflite_certification.get('certified', 'n/a')}"
            ),
            present=bool(export_certification) and bool(tflite_certification),
        ),
        req(
            "integration_gate_passed",
            "Integration gate passed",
            integration_ready,
            "outputs/production_audit/integration_gate.json; outputs/production_audit/senior_ml_audit.json",
            (
                f"integration_gate_ok={integration_gate.get('ok', 'n/a')}, "
                f"senior_integration_ready={senior_audit.get('integration_ready', 'n/a')}"
            ),
            present=bool(integration_gate) and bool(senior_audit),
        ),
        req(
            "release_package_integrity",
            "Release package integrity passed",
            bool(release_integrity.get("ok")),
            "outputs/production_audit/release_integrity.json; docs/RELEASE_PACKAGE.md",
            f"release_integrity_ok={release_integrity.get('ok', 'n/a')}",
            present=bool(release_integrity),
        ),
        req(
            "report_consistency_passed",
            "Final report consistency audit passed",
            bool(report_consistency.get("ok")) and report_consistency.get("failures", []) == [],
            "outputs/production_audit/report_consistency_audit.json; docs/REPORT_CONSISTENCY_AUDIT.md",
            (
                f"report_consistency_ok={report_consistency.get('ok', 'n/a')}, "
                f"failures={report_consistency.get('failures', 'n/a')}"
            ),
            present=bool(report_consistency),
        ),
        req(
            "handoff_bundle_verified",
            "External evidence handoff bundle verified",
            bool(handoff_manifest.get("ok"))
            and bool(handoff_verification.get("ok"))
            and handoff_hash_match
            and handoff_current_artifacts_ok,
            "outputs/production_audit/external_evidence_handoff_bundle.zip",
            (
                f"manifest_ok={handoff_manifest.get('ok', 'n/a')}, "
                f"verification_ok={handoff_verification.get('ok', 'n/a')}, "
                f"artifacts={handoff_manifest.get('artifact_count', 'n/a')}, "
                f"sha_match={handoff_hash_match}, "
                f"current_artifacts_ok={handoff_current_artifacts_ok}"
            ),
            present=bool(handoff_manifest) and bool(handoff_verification),
        ),
        req(
            "return_drop_process_ready",
            "External evidence return/drop intake process ready",
            bool(return_template.get("ok"))
            and Path("src/import_external_evidence_drop.py").is_file()
            and Path("src/run_production_evidence_intake.py").is_file(),
            (
                "outputs/production_audit/external_evidence_return_template.zip; "
                "src/import_external_evidence_drop.py; src/run_production_evidence_intake.py"
            ),
            (
                f"template_ok={return_template.get('ok', 'n/a')}, "
                f"template_artifacts={return_template.get('artifact_count', 'n/a')}, "
                f"importer={Path('src/import_external_evidence_drop.py').is_file()}"
            ),
            present=bool(return_template),
        ),
        req(
            "production_evidence_present",
            "External Android/AR production evidence present",
            bool(production_evidence.get("production_evidence_ready")),
            "outputs/production_audit/production_evidence_audit.json",
            (
                f"production_evidence_ready={production_evidence.get('production_evidence_ready', 'n/a')}, "
                f"blockers={production_evidence.get('blockers', 'n/a')}"
            ),
            present=bool(production_evidence),
        ),
        req(
            "production_gate_passed",
            "Production gate passed",
            production_ready,
            "outputs/production_audit/production_gate.json; outputs/production_audit/senior_ml_audit.json",
            (
                f"production_gate_ok={production_gate.get('ok', 'n/a')}, "
                f"senior_production_ready={senior_audit.get('production_ready', 'n/a')}"
            ),
            present=bool(production_gate) and bool(senior_audit),
        ),
        req(
            "senior_report_generated",
            "Senior ML report generated",
            Path("docs/SENIOR_ML_AUDIT.md").is_file() and bool(senior_audit.get("audit_ok")),
            "docs/SENIOR_ML_AUDIT.md; outputs/production_audit/senior_ml_audit.json",
            (
                f"audit_ok={senior_audit.get('audit_ok', 'n/a')}, "
                f"production_blockers={senior_audit.get('production_blockers', 'n/a')}"
            ),
            present=bool(senior_audit),
        ),
        req(
            "executive_report_generated",
            "Executive report generated",
            Path("docs/EXECUTIVE_REPORT_RU.md").is_file()
            and Path("outputs/production_audit/requirements_traceability.json").is_file(),
            "docs/EXECUTIVE_REPORT_RU.md; outputs/production_audit/requirements_traceability.json",
            (
                f"executive_report={Path('docs/EXECUTIVE_REPORT_RU.md').is_file()}, "
                f"traceability_json={Path('outputs/production_audit/requirements_traceability.json').is_file()}"
            ),
        ),
    ]

    production_required = [row for row in requirements if row.production_required]
    objective_complete = production_ready and all(row.status == "pass" for row in production_required)
    blockers = sorted(
        {
            str(item)
            for item in (
                senior_audit.get("production_blockers", [])
                + production_evidence.get("blockers", [])
                + audit_suite.get("production_blockers", [])
            )
            if item
        }
    )
    failed_requirements = [
        row.id for row in production_required if row.status in {"fail", "missing"}
    ]

    return {
        "schema_version": 1,
        "ok": True,
        "objective_complete": objective_complete,
        "integration_ready": integration_ready,
        "production_ready": production_ready,
        "summary": {
            "requirements": len(requirements),
            "production_required": len(production_required),
            "passed": sum(1 for row in requirements if row.status == "pass"),
            "failed": sum(1 for row in requirements if row.status == "fail"),
            "missing": sum(1 for row in requirements if row.status == "missing"),
        },
        "failed_requirements": failed_requirements,
        "production_blockers": blockers,
        "requirements": [asdict(row) for row in requirements],
        "inputs": {
            "model_inventory": "outputs/production_audit/model_inventory.json",
            "dataset_audit": "outputs/production_audit/dataset_audit.json",
            "spec_compliance_audit": "outputs/production_audit/spec_compliance_audit.json",
            "senior_ml_audit": "outputs/production_audit/senior_ml_audit.json",
            "audit_suite_status": "outputs/production_audit/audit_suite_status.json",
            "release_integrity": "outputs/production_audit/release_integrity.json",
            "report_consistency_audit": "outputs/production_audit/report_consistency_audit.json",
            "production_evidence_audit": "outputs/production_audit/production_evidence_audit.json",
        },
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Objective Completion Audit",
        "",
        "This report maps the original senior-ML wheel-pose objective to concrete evidence.",
        "",
        f"- Objective complete: {audit['objective_complete']}",
        f"- Integration ready: {audit['integration_ready']}",
        f"- Production ready: {audit['production_ready']}",
        f"- Production blockers: {', '.join(audit['production_blockers']) if audit['production_blockers'] else 'none'}",
        "",
        "| Requirement | Status | Evidence | Detail |",
        "|---|---|---|---|",
    ]
    for row in audit["requirements"]:
        lines.append(
            f"| {row['title']} | {row['status'].upper()} | `{row['evidence']}` | {row['detail']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                "Integration work is ready for AR/app wiring, but the full objective is not complete "
                "until Android LiteRT device validation, human-labelled AR holdout evaluation, "
                "and AR 3D replay validation are returned and pass."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = build_audit()
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(audit), encoding="utf-8")
    print(
        f"ok={audit['ok']} objective_complete={audit['objective_complete']} "
        f"integration_ready={audit['integration_ready']} production_ready={audit['production_ready']}"
    )
    print(f"json={args.json_out}")
    print(f"markdown={args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
