from __future__ import annotations

from src.release_integrity import (
    DEFAULT_REQUIRED_ARTIFACTS,
    PRODUCTION_EVIDENCE_RELEASE_ARTIFACTS,
    build_manifest,
    render_markdown,
    required_artifacts_for_current_state,
    sha256_file,
)


def test_release_integrity_hashes_existing_artifacts(tmp_path):
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"weights")

    manifest = build_manifest([artifact])

    assert manifest["ok"] is True
    assert manifest["artifact_count"] == 1
    assert manifest["artifacts"][0]["sha256"] == sha256_file(artifact)
    assert manifest["failures"] == []


def test_release_integrity_fails_missing_and_empty_artifacts(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_bytes(b"")
    missing = tmp_path / "missing.json"

    manifest = build_manifest([empty, missing])

    assert manifest["ok"] is False
    assert f"empty:{empty}" in manifest["failures"]
    assert f"missing:{missing}" in manifest["failures"]


def test_release_integrity_markdown_lists_hashes(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"abc")

    markdown = render_markdown(build_manifest([artifact]))

    assert "Release Package" in markdown
    assert str(artifact) in markdown
    assert sha256_file(artifact) in markdown


def test_release_integrity_includes_objective_completion_audit_artifacts():
    assert "src/objective_completion_audit.py" in DEFAULT_REQUIRED_ARTIFACTS
    assert "docs/OBJECTIVE_COMPLETION_AUDIT.md" in DEFAULT_REQUIRED_ARTIFACTS
    assert "outputs/production_audit/objective_completion_audit.json" in DEFAULT_REQUIRED_ARTIFACTS


def test_release_integrity_includes_final_report_source_runners_only():
    assert "src/model_selection_audit.py" in DEFAULT_REQUIRED_ARTIFACTS
    assert "src/spec_compliance_audit.py" in DEFAULT_REQUIRED_ARTIFACTS
    assert "src/report_consistency_audit.py" in DEFAULT_REQUIRED_ARTIFACTS
    assert "scripts/write_production_audit_report.py" in DEFAULT_REQUIRED_ARTIFACTS
    assert "scripts/write_handoff_report.py" in DEFAULT_REQUIRED_ARTIFACTS
    assert "docs/REPORT_CONSISTENCY_AUDIT.md" not in DEFAULT_REQUIRED_ARTIFACTS
    assert "outputs/production_audit/report_consistency_audit.json" not in DEFAULT_REQUIRED_ARTIFACTS
    assert "docs/HANDOFF_TODAY.md" not in DEFAULT_REQUIRED_ARTIFACTS
    assert "outputs/production_audit/model_package_manifest.json" not in DEFAULT_REQUIRED_ARTIFACTS


def test_release_integrity_tracks_intake_preflight_separately_from_final_status():
    assert "outputs/production_audit/production_evidence_intake_preflight_status.json" in DEFAULT_REQUIRED_ARTIFACTS
    assert "outputs/production_audit/production_evidence_intake_status.json" not in DEFAULT_REQUIRED_ARTIFACTS
    assert "outputs/production_audit/production_evidence_intake_status.json" in PRODUCTION_EVIDENCE_RELEASE_ARTIFACTS


def test_release_integrity_rejects_self_referential_artifacts(tmp_path):
    release_json = tmp_path / "outputs" / "production_audit" / "release_integrity.json"
    release_json.parent.mkdir(parents=True)
    release_json.write_text("{}", encoding="utf-8")

    manifest = build_manifest([release_json.relative_to(tmp_path)])

    assert manifest["ok"] is False
    assert "self_referential_artifact:outputs/production_audit/release_integrity.json" in manifest[
        "failures"
    ]


def test_default_release_artifacts_do_not_include_generated_manifest_outputs():
    assert "outputs/production_audit/release_integrity.json" not in DEFAULT_REQUIRED_ARTIFACTS
    assert "docs/RELEASE_PACKAGE.md" not in DEFAULT_REQUIRED_ARTIFACTS
    assert "outputs/production_audit/model_selection_audit.json" in DEFAULT_REQUIRED_ARTIFACTS
    assert "docs/MODEL_SELECTION_AUDIT.md" in DEFAULT_REQUIRED_ARTIFACTS
    assert "outputs/production_audit/spec_compliance_audit.json" in DEFAULT_REQUIRED_ARTIFACTS
    assert "docs/SPEC_COMPLIANCE_AUDIT.md" in DEFAULT_REQUIRED_ARTIFACTS


def test_release_integrity_adds_external_evidence_artifacts_when_ready(tmp_path):
    audit = tmp_path / "production_evidence_audit.json"
    audit.write_text('{"production_evidence_ready": true}', encoding="utf-8")

    artifacts = required_artifacts_for_current_state(production_evidence_audit_path=audit)

    for artifact in PRODUCTION_EVIDENCE_RELEASE_ARTIFACTS:
        assert artifact in artifacts


def test_release_integrity_skips_external_evidence_artifacts_until_ready(tmp_path):
    audit = tmp_path / "production_evidence_audit.json"
    audit.write_text('{"production_evidence_ready": false}', encoding="utf-8")

    artifacts = required_artifacts_for_current_state(production_evidence_audit_path=audit)

    for artifact in PRODUCTION_EVIDENCE_RELEASE_ARTIFACTS:
        assert artifact not in artifacts


def test_release_integrity_conditional_artifacts_respect_objective_filter(tmp_path):
    audit = tmp_path / "production_evidence_audit.json"
    audit.write_text('{"production_evidence_ready": true}', encoding="utf-8")

    artifacts = required_artifacts_for_current_state(
        include_objective=False,
        production_evidence_audit_path=audit,
    )

    assert "src/objective_completion_audit.py" not in artifacts
    assert "outputs/production_audit/objective_completion_audit.json" not in artifacts
    assert "outputs/production_audit/external_evidence_drop_import.json" in artifacts
