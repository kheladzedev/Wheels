from __future__ import annotations

import json
import hashlib
from pathlib import Path

from scripts.build_external_evidence_handoff_bundle import DEFAULT_BUNDLE_ARTIFACTS
from scripts.create_external_evidence_return_template import (
    build_manifest,
    build_template_files,
    write_template_zip,
)
from scripts.write_production_audit_report import build_package_artifacts
from src.release_integrity import PRODUCTION_EVIDENCE_RELEASE_ARTIFACTS
from src.report_consistency_audit import build_audit, render_markdown


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write(path: Path, text: str = "ok") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def package_digest(artifacts: list[dict]) -> str:
    canonical = json.dumps(
        artifacts,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_return_template(
    audit_root: Path,
    expected_artifact: dict,
    *,
    embedded_artifact: dict | None = None,
) -> None:
    zip_path = Path("outputs/production_audit/external_evidence_return_template.zip")
    artifact_path = Path(expected_artifact["path"])
    files = build_template_files(artifact_path)
    if embedded_artifact is not None:
        files["EXPECTED_ANDROID_ARTIFACT.json"] = (
            json.dumps(
                {
                    "schema_version": 1,
                    "expected_android_artifact": embedded_artifact,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
    artifacts = write_template_zip(zip_path, files, expected_android_artifact=artifact_path)
    manifest = build_manifest(zip_path, artifacts, expected_android_artifact=artifact_path)
    write_json(audit_root / "external_evidence_return_template_manifest.json", manifest)


def post_finalization_refresh(ok: bool = True) -> list[dict]:
    return [
        {
            "name": "write_production_audit_report",
            "command": ["python", "scripts/write_production_audit_report.py"],
            "returncode": 0 if ok else 1,
            "ok": ok,
        },
        {
            "name": "write_handoff_report",
            "command": ["python", "scripts/write_handoff_report.py"],
            "returncode": 0,
            "ok": True,
        },
        {
            "name": "release_integrity",
            "command": ["python", "src/release_integrity.py"],
            "returncode": 0,
            "ok": True,
        },
        {
            "name": "report_consistency_audit",
            "command": ["python", "src/report_consistency_audit.py"],
            "returncode": 0,
            "ok": True,
        },
    ]


def create_consistent_reports(root: Path) -> None:
    audit_root = root / "outputs" / "production_audit"
    write(root / "model.pt", "weights")
    tflite_path = audit_root / "tflite_export" / "best_float32.tflite"
    write(tflite_path, "tflite")
    model_sha = hashlib.sha256((root / "model.pt").read_bytes()).hexdigest()
    tflite_sha = sha256_file(tflite_path)
    expected_android_artifact = {
        "path": "outputs/production_audit/tflite_export/best_float32.tflite",
        "sha256": tflite_sha,
        "format": "tflite_float32",
    }
    package_artifacts = [
        {
            "role": "champion_pt",
            "path": "model.pt",
            "exists": True,
            "size_bytes": (root / "model.pt").stat().st_size,
            "sha256": model_sha,
        }
    ]
    write_json(
        audit_root / "release_integrity.json",
        {
            "ok": True,
            "artifact_count": 1,
            "artifacts": [
                {"path": "model.pt", "exists": True, "sha256": model_sha},
            ],
        },
    )
    write_json(
        audit_root / "audit_suite_status.json",
        {
            "ok": True,
            "integration_ready": True,
            "production_ready": False,
            "release_artifacts": 1,
        },
    )
    write_json(
        audit_root / "objective_completion_audit.json",
        {
            "objective_complete": False,
            "integration_ready": True,
            "production_ready": False,
            "failed_requirements": ["production_gate_passed"],
        },
    )
    write_json(audit_root / "integration_gate.json", {"ok": True})
    write_json(audit_root / "production_gate.json", {"ok": False})
    write_json(audit_root / "production_evidence_audit.json", {"production_evidence_ready": False})
    write_json(audit_root / "model_selection_audit.json", {"ok": True})
    write_json(audit_root / "spec_compliance_audit.json", {"ok": True})
    write_json(
        audit_root / "model_package_manifest.json",
        {
            "schema_version": 2,
            "package_artifacts": package_artifacts,
            "package_digest_sha256": package_digest(package_artifacts),
            "missing_artifacts": [],
        },
    )
    write_json(
        audit_root / "production_evidence_intake_preflight_status.json",
        {
            "finalization_required": True,
            "finalization_command": [
                "./.venv/bin/python",
                "src/production_audit_suite.py",
                "--with-pytest",
            ],
        },
    )
    write_json(
        audit_root / "external_evidence_handoff_bundle_verification.json",
        {
            "ok": True,
            "entry_count": len(DEFAULT_BUNDLE_ARTIFACTS),
            "expected_entry_count": len(DEFAULT_BUNDLE_ARTIFACTS),
            "required_artifact_count": len(DEFAULT_BUNDLE_ARTIFACTS),
            "bundle_sha256": "bundle-sha",
            "expected_bundle_sha256": "bundle-sha",
            "bundle_size_bytes": 123,
            "expected_bundle_size_bytes": 123,
            "missing_required_artifacts": [],
            "failures": [],
            "current_artifacts": [
                {"path": path, "ok": True} for path in DEFAULT_BUNDLE_ARTIFACTS
            ],
        },
    )
    write_return_template(audit_root, expected_android_artifact)
    for doc in [
        "PRODUCTION_READINESS_AUDIT.md",
        "HANDOFF_TODAY.md",
        "OBJECTIVE_COMPLETION_AUDIT.md",
        "MODEL_SELECTION_AUDIT.md",
        "SPEC_COMPLIANCE_AUDIT.md",
        "RELEASE_PACKAGE.md",
        "REQUIREMENTS_TRACEABILITY.md",
        "EXECUTIVE_REPORT_RU.md",
        "MODEL_CARD.md",
    ]:
        write(root / "docs" / doc)
    write(
        root / "docs" / "PRODUCTION_EVIDENCE_INTAKE.md",
        (
            "finalization_required=true\n"
            "--finalize\n"
            "./.venv/bin/python src/production_audit_suite.py --with-pytest\n"
        ),
    )


def test_report_consistency_audit_passes_consistent_reports(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    create_consistent_reports(tmp_path)

    audit = build_audit()

    assert audit["ok"] is True
    assert audit["failures"] == []


def test_report_consistency_audit_rejects_self_reference_and_stale_counts(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    create_consistent_reports(tmp_path)
    write_json(
        tmp_path / "outputs" / "production_audit" / "release_integrity.json",
        {
            "ok": True,
            "artifact_count": 1,
            "artifacts": [
                {
                    "path": "outputs/production_audit/release_integrity.json",
                    "exists": True,
                }
            ],
        },
    )
    write(tmp_path / "docs" / "EXECUTIVE_REPORT_RU.md", "Artifacts: 72\n")

    audit = build_audit()

    assert audit["ok"] is False
    assert "release_manifest_no_self_reference" in audit["failures"]
    assert "prefinal_reports_do_not_embed_release_counts" in audit["failures"]


def test_report_consistency_audit_rejects_stale_release_hashes(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    create_consistent_reports(tmp_path)
    write(tmp_path / "model.pt", "changed")

    audit = build_audit()

    assert audit["ok"] is False
    assert "release_manifest_hashes_current" in audit["failures"]
    assert "model_package_manifest_artifacts_current" in audit["failures"]


def test_report_consistency_audit_rejects_bad_model_package_digest(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    create_consistent_reports(tmp_path)
    package_path = tmp_path / "outputs" / "production_audit" / "model_package_manifest.json"
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["package_digest_sha256"] = "bad"
    package_path.write_text(json.dumps(package), encoding="utf-8")

    audit = build_audit()

    assert audit["ok"] is False
    assert "model_package_manifest_artifacts_current" in audit["failures"]


def test_report_consistency_audit_rejects_stale_return_template_artifact_sha(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    create_consistent_reports(tmp_path)
    manifest_path = (
        tmp_path
        / "outputs"
        / "production_audit"
        / "external_evidence_return_template_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["expected_android_artifact"]["sha256"] = "bad"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    audit = build_audit()

    assert audit["ok"] is False
    assert "external_evidence_return_template_artifact_current" in audit["failures"]


def test_report_consistency_audit_rejects_return_template_embedded_artifact_mismatch(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    create_consistent_reports(tmp_path)
    audit_root = tmp_path / "outputs" / "production_audit"
    manifest_path = audit_root / "external_evidence_return_template_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    write_return_template(
        audit_root,
        manifest["expected_android_artifact"],
        embedded_artifact={
            **manifest["expected_android_artifact"],
            "sha256": "embedded-bad",
        },
    )

    audit = build_audit()

    assert audit["ok"] is False
    assert "external_evidence_return_template_artifact_current" in audit["failures"]


def test_report_consistency_audit_rejects_return_template_missing_expected_entry(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    create_consistent_reports(tmp_path)
    audit_root = tmp_path / "outputs" / "production_audit"
    artifact = Path("outputs/production_audit/tflite_export/best_float32.tflite")
    files = build_template_files(artifact)
    files.pop("ar_3d_replay/ar_replay.jsonl.PLACEHOLDER")
    zip_path = Path("outputs/production_audit/external_evidence_return_template.zip")
    artifacts = write_template_zip(zip_path, files, expected_android_artifact=artifact)
    manifest = build_manifest(zip_path, artifacts, expected_android_artifact=artifact)
    write_json(audit_root / "external_evidence_return_template_manifest.json", manifest)

    audit = build_audit()

    assert audit["ok"] is False
    assert "external_evidence_return_template_artifact_current" in audit["failures"]


def test_report_consistency_audit_rejects_return_template_manifest_entry_hash_mismatch(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    create_consistent_reports(tmp_path)
    manifest_path = (
        tmp_path
        / "outputs"
        / "production_audit"
        / "external_evidence_return_template_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["sha256"] = "bad-entry-sha"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    audit = build_audit()

    assert audit["ok"] is False
    assert "external_evidence_return_template_artifact_current" in audit["failures"]


def test_package_artifacts_exclude_post_manifest_handoff(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "docs" / "HANDOFF_TODAY.md", "new handoff")

    artifacts = build_package_artifacts({"handoff_today": "docs/HANDOFF_TODAY.md"})

    assert "docs/HANDOFF_TODAY.md" not in {artifact["path"] for artifact in artifacts}


def test_report_consistency_markdown_lists_checks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    create_consistent_reports(tmp_path)

    markdown = render_markdown(build_audit())

    assert "Report Consistency Audit" in markdown
    assert "release_manifest_no_self_reference" in markdown


def test_report_consistency_audit_requires_intake_finalization_contract(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    create_consistent_reports(tmp_path)
    write_json(
        tmp_path / "outputs" / "production_audit" / "production_evidence_intake_preflight_status.json",
        {"finalization_required": False, "finalization_command": []},
    )

    audit = build_audit()

    assert audit["ok"] is False
    assert "intake_finalization_contract" in audit["failures"]


def test_report_consistency_audit_accepts_completed_intake_finalization(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    create_consistent_reports(tmp_path)
    write_json(
        tmp_path / "outputs" / "production_audit" / "production_evidence_audit.json",
        {"production_evidence_ready": True},
    )
    write_json(
        tmp_path / "outputs" / "production_audit" / "production_evidence_intake_status.json",
        {
            "finalization_required": False,
            "finalization_command": [
                "./.venv/bin/python",
                "src/production_audit_suite.py",
                "--with-pytest",
            ],
            "finalization": {"ok": True, "returncode": 0, "skipped": False},
            "post_finalization_refresh": post_finalization_refresh(),
        },
    )

    audit = build_audit()

    assert "intake_finalization_contract" not in audit["failures"]
    assert "intake_post_finalization_refresh_contract" not in audit["failures"]


def test_report_consistency_audit_requires_post_finalization_refresh_when_completed(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    create_consistent_reports(tmp_path)
    write_json(
        tmp_path / "outputs" / "production_audit" / "production_evidence_audit.json",
        {"production_evidence_ready": True},
    )
    write_json(
        tmp_path / "outputs" / "production_audit" / "production_evidence_intake_status.json",
        {
            "finalization_required": False,
            "finalization_command": [
                "./.venv/bin/python",
                "src/production_audit_suite.py",
                "--with-pytest",
            ],
            "finalization": {"ok": True, "returncode": 0, "skipped": False},
            "post_finalization_refresh": [],
        },
    )

    audit = build_audit()

    assert audit["ok"] is False
    assert "intake_post_finalization_refresh_contract" in audit["failures"]


def test_report_consistency_audit_rejects_failed_post_finalization_refresh(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    create_consistent_reports(tmp_path)
    write_json(
        tmp_path / "outputs" / "production_audit" / "production_evidence_audit.json",
        {"production_evidence_ready": True},
    )
    write_json(
        tmp_path / "outputs" / "production_audit" / "production_evidence_intake_status.json",
        {
            "finalization_required": False,
            "finalization_command": [
                "./.venv/bin/python",
                "src/production_audit_suite.py",
                "--with-pytest",
            ],
            "finalization": {"ok": True, "returncode": 0, "skipped": False},
            "post_finalization_refresh": post_finalization_refresh(ok=False),
        },
    )

    audit = build_audit()

    assert audit["ok"] is False
    assert "intake_post_finalization_refresh_contract" in audit["failures"]


def test_report_consistency_audit_requires_strict_handoff_bundle_verification(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    create_consistent_reports(tmp_path)
    write_json(
        tmp_path
        / "outputs"
        / "production_audit"
        / "external_evidence_handoff_bundle_verification.json",
        {
            "ok": True,
            "entry_count": len(DEFAULT_BUNDLE_ARTIFACTS),
            "expected_entry_count": len(DEFAULT_BUNDLE_ARTIFACTS),
            "required_artifact_count": 0,
            "missing_required_artifacts": [],
            "failures": [],
        },
    )

    audit = build_audit()

    assert audit["ok"] is False
    assert "handoff_bundle_verification_strict" in audit["failures"]


def test_report_consistency_audit_requires_handoff_bundle_hash_match(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    create_consistent_reports(tmp_path)
    report_path = (
        tmp_path
        / "outputs"
        / "production_audit"
        / "external_evidence_handoff_bundle_verification.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["expected_bundle_sha256"] = "different"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    audit = build_audit()

    assert audit["ok"] is False
    assert "handoff_bundle_verification_strict" in audit["failures"]


def test_report_consistency_audit_requires_current_handoff_artifacts(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    create_consistent_reports(tmp_path)
    report_path = (
        tmp_path
        / "outputs"
        / "production_audit"
        / "external_evidence_handoff_bundle_verification.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["current_artifacts"][0]["ok"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")

    audit = build_audit()

    assert audit["ok"] is False
    assert "handoff_bundle_verification_strict" in audit["failures"]


def test_report_consistency_audit_requires_production_evidence_artifacts_when_ready(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    create_consistent_reports(tmp_path)
    write_json(
        tmp_path / "outputs" / "production_audit" / "production_evidence_audit.json",
        {"production_evidence_ready": True},
    )

    audit = build_audit()

    assert audit["ok"] is False
    assert "production_evidence_release_artifacts" in audit["failures"]


def test_report_consistency_audit_accepts_production_evidence_artifacts_when_ready(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    create_consistent_reports(tmp_path)
    artifacts = [
        {"path": "model.pt", "exists": True},
        {"path": "docs/MODEL_CARD.md", "exists": True},
        *[
            {"path": artifact, "exists": True}
            for artifact in PRODUCTION_EVIDENCE_RELEASE_ARTIFACTS
        ],
    ]
    write_json(
        tmp_path / "outputs" / "production_audit" / "release_integrity.json",
        {
            "ok": True,
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
        },
    )
    write_json(
        tmp_path / "outputs" / "production_audit" / "audit_suite_status.json",
        {
            "ok": True,
            "integration_ready": True,
            "production_ready": False,
            "release_artifacts": len(artifacts),
        },
    )
    write_json(
        tmp_path / "outputs" / "production_audit" / "production_evidence_audit.json",
        {"production_evidence_ready": True},
    )

    audit = build_audit()

    assert "production_evidence_release_artifacts" not in audit["failures"]
