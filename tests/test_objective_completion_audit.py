from __future__ import annotations

from pathlib import Path

from src.objective_completion_audit import build_audit, render_markdown


def write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, text: str) -> None:
    write(path, text)


def create_common_evidence(root: Path, *, production_ready: bool) -> None:
    for i in range(300):
        write(root / "data" / "sketchfab_cars" / f"car_{i:03d}.glb")
    for i in range(120):
        write(root / "data" / "incoming" / "ue_sketchfab_geometry_clean" / "images" / f"{i:03d}.png")
        write_json(
            root
            / "data"
            / "incoming"
            / "ue_sketchfab_geometry_clean"
            / "annotations"
            / f"{i:03d}.json",
            '{"wheels": [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}, {"id": 5}]}',
        )
    write(root / "runs" / "pose" / "wheel_real_v1_self_plus_ue_synthetic_s" / "weights" / "best.pt")
    write(root / "runs" / "pose" / "wheel_real_v1_self_plus_ue_synthetic_s" / "weights" / "best.onnx")
    write(root / "outputs" / "production_audit" / "tflite_export" / "best_float32.tflite")
    write(root / "outputs" / "production_audit" / "coreml_export" / "best.mlmodel")
    write(root / "docs" / "SENIOR_ML_AUDIT.md")
    write(root / "docs" / "EXECUTIVE_REPORT_RU.md")
    write(root / "src" / "import_external_evidence_drop.py")
    write(root / "src" / "run_production_evidence_intake.py")
    write(root / "outputs" / "production_audit" / "requirements_traceability.json", "{}")

    write_json(
        root / "outputs" / "production_audit" / "model_inventory.json",
        '{"counts": {"train_runs": 2, "artifacts": 4, "eval_reports": 3}}',
    )
    write_json(
        root / "outputs" / "production_audit" / "dataset_audit.json",
        '{"ok": true, "counts": {"configs": 2, "failed": 0, "total_wheel_labels": 10}}',
    )
    write_json(
        root / "outputs" / "production_audit" / "spec_compliance_audit.json",
        '{"ok": true, "failures": []}',
    )
    write_json(
        root / "outputs" / "production_audit" / "release_integrity.json",
        '{"ok": true, "artifact_count": 12, "total_size_mb": 4.5, "failures": []}',
    )
    write_json(
        root / "outputs" / "production_audit" / "report_consistency_audit.json",
        '{"ok": true, "failures": []}',
    )
    write_json(
        root / "outputs" / "production_audit" / "export_certification.json",
        '{"certified": true}',
    )
    write_json(
        root / "outputs" / "production_audit" / "tflite_certification.json",
        '{"certified": true}',
    )
    write_json(
        root / "outputs" / "production_audit" / "coreml_certification.json",
        '{"certified": true}',
    )
    write_json(root / "outputs" / "production_audit" / "integration_gate.json", '{"ok": true}')
    write_json(
        root / "outputs" / "production_audit" / "external_evidence_handoff_bundle_manifest.json",
        '{"ok": true, "artifact_count": 2, "bundle_sha256": "abc"}',
    )
    write_json(
        root / "outputs" / "production_audit" / "external_evidence_handoff_bundle_verification.json",
        (
            '{"ok": true, "bundle_sha256": "abc", '
            '"current_artifacts": [{"path": "a", "ok": true}, {"path": "b", "ok": true}]}'
        ),
    )
    write_json(
        root / "outputs" / "production_audit" / "external_evidence_return_template_manifest.json",
        '{"ok": true, "artifact_count": 6}',
    )

    production_json = "true" if production_ready else "false"
    blockers = "[]" if production_ready else '["android_litert_device_validation"]'
    write_json(
        root / "outputs" / "production_audit" / "senior_ml_audit.json",
        (
            '{"audit_ok": true, "integration_ready": true, '
            f'"production_ready": {production_json}, "production_blockers": {blockers}}}'
        ),
    )
    write_json(
        root / "outputs" / "production_audit" / "production_gate.json",
        f'{{"ok": {production_json}}}',
    )
    write_json(
        root / "outputs" / "production_audit" / "audit_suite_status.json",
        (
            '{"integration_ready": true, '
            f'"production_ready": {production_json}, "production_blockers": {blockers}}}'
        ),
    )


def by_id(audit: dict, row_id: str) -> dict:
    return next(row for row in audit["requirements"] if row["id"] == row_id)


def test_objective_audit_keeps_objective_open_without_external_evidence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    create_common_evidence(tmp_path, production_ready=False)
    write_json(
        tmp_path / "outputs" / "production_audit" / "production_evidence_audit.json",
        (
            '{"ok": true, "production_evidence_ready": false, '
            '"blockers": ["android_litert_device_validation"]}'
        ),
    )

    audit = build_audit()

    assert audit["ok"] is True
    assert audit["integration_ready"] is True
    assert audit["production_ready"] is False
    assert audit["objective_complete"] is False
    assert by_id(audit, "production_evidence_present")["status"] == "fail"
    assert by_id(audit, "production_gate_passed")["status"] == "fail"
    assert "android_litert_device_validation" in audit["production_blockers"]


def test_objective_audit_can_mark_full_completion_when_production_evidence_passes(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    create_common_evidence(tmp_path, production_ready=True)
    write_json(
        tmp_path / "outputs" / "production_audit" / "production_evidence_audit.json",
        '{"ok": true, "production_evidence_ready": true, "blockers": []}',
    )

    audit = build_audit()

    assert audit["objective_complete"] is True
    assert audit["production_ready"] is True
    assert audit["failed_requirements"] == []
    assert by_id(audit, "ue_mcp_synthetic_data_done")["status"] == "pass"
    assert by_id(audit, "technical_spec_compliance_reviewed")["status"] == "pass"


def test_objective_audit_requires_spec_compliance_evidence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    create_common_evidence(tmp_path, production_ready=True)
    (tmp_path / "outputs" / "production_audit" / "spec_compliance_audit.json").unlink()
    write_json(
        tmp_path / "outputs" / "production_audit" / "production_evidence_audit.json",
        '{"ok": true, "production_evidence_ready": true, "blockers": []}',
    )

    audit = build_audit()

    assert audit["objective_complete"] is False
    assert by_id(audit, "technical_spec_compliance_reviewed")["status"] == "missing"
    assert "technical_spec_compliance_reviewed" in audit["failed_requirements"]


def test_objective_audit_requires_report_consistency_evidence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    create_common_evidence(tmp_path, production_ready=True)
    write_json(
        tmp_path / "outputs" / "production_audit" / "production_evidence_audit.json",
        '{"ok": true, "production_evidence_ready": true, "blockers": []}',
    )
    write_json(
        tmp_path / "outputs" / "production_audit" / "report_consistency_audit.json",
        '{"ok": false, "failures": ["release_manifest_hashes_current"]}',
    )

    audit = build_audit()

    assert audit["objective_complete"] is False
    assert by_id(audit, "report_consistency_passed")["status"] == "fail"


def test_objective_audit_requires_current_handoff_bundle_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    create_common_evidence(tmp_path, production_ready=True)
    write_json(
        tmp_path / "outputs" / "production_audit" / "production_evidence_audit.json",
        '{"ok": true, "production_evidence_ready": true, "blockers": []}',
    )
    write_json(
        tmp_path
        / "outputs"
        / "production_audit"
        / "external_evidence_handoff_bundle_verification.json",
        (
            '{"ok": true, "bundle_sha256": "abc", '
            '"current_artifacts": [{"path": "a", "ok": false}, {"path": "b", "ok": true}]}'
        ),
    )

    audit = build_audit()

    assert audit["objective_complete"] is False
    assert by_id(audit, "handoff_bundle_verified")["status"] == "fail"
    assert "handoff_bundle_verified" in audit["failed_requirements"]


def test_objective_completion_markdown_lists_decision(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    create_common_evidence(tmp_path, production_ready=False)
    write_json(
        tmp_path / "outputs" / "production_audit" / "production_evidence_audit.json",
        '{"ok": true, "production_evidence_ready": false, "blockers": []}',
    )

    markdown = render_markdown(build_audit())

    assert "Objective Completion Audit" in markdown
    assert "Objective complete: False" in markdown
    assert "Integration work is ready" in markdown
