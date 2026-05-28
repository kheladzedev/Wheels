"""Generate and validate the ML integration release artifact manifest.

The production audit has many reports, but AR needs a deterministic
package inventory: exact file paths, sizes, SHA256 hashes, and a hard
fail if any required artifact is missing or empty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_JSON_OUT = Path("outputs/production_audit/release_integrity.json")
DEFAULT_MD_OUT = Path("docs/RELEASE_PACKAGE.md")
SELF_REFERENTIAL_ARTIFACTS = {
    str(DEFAULT_JSON_OUT).replace("\\", "/"),
    str(DEFAULT_MD_OUT).replace("\\", "/"),
}

DEFAULT_REQUIRED_ARTIFACTS = [
    "runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt",
    "runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.onnx",
    "outputs/production_audit/tflite_export/best_float32.tflite",
    "configs/pose_dataset_real_v1_self.yaml",
    "configs/pose_dataset_real_v1_self_plus_ue_synthetic.yaml",
    "configs/pose_dataset_real_self_ue_plus_sketchfab_clean.yaml",
    "docs/AR_ML_CONTRACT.md",
    "docs/AR_MOCK_LOG_CONTRACT.md",
    "android_litert_harness/README.md",
    "android_litert_harness/AndroidLiteRtDeviceValidationTest.kt",
    "ar_holdout_harness/README.md",
    "ar_holdout_harness/ArHoldoutAnnotationWriter.kt",
    "ar_replay_harness/README.md",
    "ar_replay_harness/ArReplayLogger.kt",
    "src/evaluate_ar_holdout.py",
    "src/validate_ar_replay.py",
    "src/validate_android_litert_report.py",
    "src/tflite_certification.py",
    "src/production_evidence_audit.py",
    "src/ue_wheel_asset_filter.py",
    "src/model_selection_audit.py",
    "src/spec_compliance_audit.py",
    "src/import_external_evidence_drop.py",
    "src/run_production_evidence_intake.py",
    "src/verify_external_evidence_handoff_bundle.py",
    "src/requirements_traceability.py",
    "src/executive_report_ru.py",
    "src/objective_completion_audit.py",
    "src/report_consistency_audit.py",
    "src/production_audit_suite.py",
    "docs/MODEL_INVENTORY.md",
    "docs/MODEL_SELECTION_AUDIT.md",
    "docs/SPEC_COMPLIANCE_AUDIT.md",
    "docs/MODEL_CARD.md",
    "docs/DATASET_AUDIT.md",
    "docs/PERFORMANCE_AUDIT.md",
    "docs/SENIOR_ML_AUDIT.md",
    "docs/EXPORT_PARITY_AUDIT.md",
    "docs/EXPORT_CERTIFICATION.md",
    "docs/ANDROID_LITERT_DEVICE_REPORT.md",
    "docs/EXTERNAL_EVIDENCE_HANDOFF_BUNDLE.md",
    "docs/PRODUCTION_EVIDENCE_CHECKLIST.md",
    "docs/PRODUCTION_EVIDENCE_INTAKE.md",
    "docs/PRODUCTION_EVIDENCE_AUDIT.md",
    "docs/REQUIREMENTS_TRACEABILITY.md",
    "docs/EXECUTIVE_REPORT_RU.md",
    "docs/OBJECTIVE_COMPLETION_AUDIT.md",
    "scripts/create_android_litert_report_template.py",
    "scripts/create_ar_holdout_provenance_template.py",
    "scripts/create_ar_replay_log_template.py",
    "scripts/create_external_evidence_return_template.py",
    "scripts/build_external_evidence_handoff_bundle.py",
    "scripts/write_production_audit_report.py",
    "scripts/write_handoff_report.py",
    "outputs/production_audit/android_litert_device_report.template.json",
    "outputs/production_audit/ar_device_holdout_provenance.template.json",
    "outputs/production_audit/ar_3d_replay.template.jsonl",
    "outputs/production_audit/external_evidence_return_template.zip",
    "outputs/production_audit/external_evidence_return_template_manifest.json",
    "outputs/production_audit/external_evidence_handoff_bundle.zip",
    "outputs/production_audit/external_evidence_handoff_bundle_manifest.json",
    "outputs/production_audit/external_evidence_handoff_bundle_verification.json",
    "outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_on_self_val.json",
    "outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s.json",
    "outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_onnx_on_self_plus_ue_val.json",
    "outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_tflite_on_self_plus_ue_val.json",
    "outputs/production_audit/model_inventory.json",
    "outputs/production_audit/model_selection_audit.json",
    "outputs/production_audit/spec_compliance_audit.json",
    "outputs/production_audit/dataset_audit.json",
    "outputs/production_audit/performance_audit.json",
    "outputs/production_audit/senior_ml_audit.json",
    "outputs/production_audit/export_parity_audit.json",
    "outputs/production_audit/export_certification.json",
    "outputs/production_audit/production_evidence_audit.json",
    "outputs/production_audit/production_evidence_intake_preflight_status.json",
    "outputs/production_audit/requirements_traceability.json",
    "outputs/production_audit/objective_completion_audit.json",
    "outputs/production_audit/onnx_drift_20.json",
    "outputs/production_audit/tflite_drift_20.json",
    "outputs/production_audit/tflite_certification.json",
    "outputs/production_audit/litert_runtime_smoke.json",
    "outputs/production_audit/runtime_contract_audit.json",
    "outputs/production_audit/integration_gate.json",
    "outputs/production_audit/production_gate.json",
]

PRODUCTION_EVIDENCE_RELEASE_ARTIFACTS = [
    "outputs/production_audit/external_evidence_drop_import.json",
    "outputs/production_audit/production_evidence_intake_status.json",
    "outputs/production_audit/android_litert_device_eval.json",
    "outputs/production_audit/ar_device_holdout_eval.json",
    "outputs/production_audit/ar_device_holdout_pipeline.json",
    "outputs/production_audit/ar_3d_replay_eval.json",
]


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def required_artifacts_for_current_state(
    *,
    include_objective: bool = True,
    production_evidence_audit_path: Path = Path("outputs/production_audit/production_evidence_audit.json"),
) -> list[str]:
    artifacts = list(DEFAULT_REQUIRED_ARTIFACTS)
    if not include_objective:
        artifacts = [
            artifact
            for artifact in artifacts
            if artifact
            not in {
                "src/objective_completion_audit.py",
                "docs/OBJECTIVE_COMPLETION_AUDIT.md",
                "outputs/production_audit/objective_completion_audit.json",
            }
        ]
    evidence = read_json(production_evidence_audit_path)
    if evidence.get("production_evidence_ready") is True:
        for artifact in PRODUCTION_EVIDENCE_RELEASE_ARTIFACTS:
            if artifact not in artifacts:
                artifacts.append(artifact)
    return artifacts


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_artifact(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    size_bytes = path.stat().st_size if exists else 0
    return {
        "path": str(path).replace("\\", "/"),
        "exists": exists,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 3),
        "sha256": sha256_file(path) if exists and size_bytes > 0 else None,
    }


def build_manifest(paths: list[Path]) -> dict[str, Any]:
    artifacts = [inspect_artifact(path) for path in paths]
    failures: list[str] = []
    for artifact in artifacts:
        if artifact["path"] in SELF_REFERENTIAL_ARTIFACTS:
            failures.append(f"self_referential_artifact:{artifact['path']}")
        elif not artifact["exists"]:
            failures.append(f"missing:{artifact['path']}")
        elif artifact["size_bytes"] <= 0:
            failures.append(f"empty:{artifact['path']}")
        elif not artifact["sha256"]:
            failures.append(f"missing_sha256:{artifact['path']}")
    return {
        "ok": not failures,
        "schema_version": 1,
        "artifact_count": len(artifacts),
        "total_size_mb": round(sum(a["size_bytes"] for a in artifacts) / (1024 * 1024), 3),
        "failures": failures,
        "artifacts": artifacts,
    }


def render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Release Package",
        "",
        "This is the deterministic file inventory for the current ML integration package.",
        "",
        f"- OK: {manifest['ok']}",
        f"- Artifact count: {manifest['artifact_count']}",
        f"- Total size: {manifest['total_size_mb']} MB",
        f"- Failures: {', '.join(manifest['failures']) if manifest['failures'] else 'none'}",
        "",
        "| Path | Size MB | SHA256 |",
        "|---|---:|---|",
    ]
    for artifact in manifest["artifacts"]:
        sha = artifact["sha256"] or "n/a"
        lines.append(f"| `{artifact['path']}` | {artifact['size_mb']} | `{sha}` |")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        action="append",
        default=None,
        help="Required artifact path. May be passed multiple times. Defaults to the current release set.",
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts = args.artifact or required_artifacts_for_current_state()
    paths = [Path(p) for p in artifacts]
    manifest = build_manifest(paths)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(manifest), encoding="utf-8")
    print(
        f"ok={manifest['ok']} artifacts={manifest['artifact_count']} "
        f"total_size_mb={manifest['total_size_mb']}"
    )
    print(f"json={args.json_out}")
    print(f"markdown={args.md_out}")
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
