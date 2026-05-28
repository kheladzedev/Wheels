from __future__ import annotations

from src.production_audit_suite import StepResult, build_steps, evaluate_suite_status


def test_build_steps_orders_release_around_gates():
    names = [step.name for step in build_steps(include_performance=True, include_pytest=True)]

    assert names.index("model_inventory") < names.index("model_selection_audit")
    assert names.index("model_selection_audit") < names.index("dataset_audit")
    assert names.index("runtime_contract_audit") < names.index("spec_compliance_audit")
    assert names.index("spec_compliance_audit") < names.index("release_integrity_pregate")
    assert names.index("release_integrity_pregate") < names.index("integration_gate")
    assert names.index("ar_replay_log_template") < names.index("production_evidence_audit")
    assert names.index("ar_holdout_provenance_template") < names.index("production_evidence_intake_preflight")
    assert names.index("production_evidence_intake_preflight") < names.index("production_evidence_audit")
    assert names.index("external_evidence_return_template") < names.index("external_evidence_handoff_bundle")
    assert names.index("external_evidence_handoff_bundle") < names.index("release_integrity_pregate")
    assert names.index("external_evidence_handoff_bundle") < names.index("production_evidence_audit")
    assert names.index("external_evidence_handoff_bundle") < names.index(
        "external_evidence_handoff_bundle_verify"
    )
    assert names.index("external_evidence_handoff_bundle_verify") < names.index(
        "release_integrity_pregate"
    )
    assert names.index("external_evidence_handoff_bundle_verify") < names.index(
        "production_evidence_audit"
    )
    assert names.index("ar_replay_log_template") < names.index("release_integrity_pregate")
    preflight = next(step for step in build_steps(include_performance=True, include_pytest=True) if step.name == "production_evidence_intake_preflight")
    assert preflight.allow_failure is True
    assert "--status-out" in preflight.cmd
    assert "outputs/production_audit/production_evidence_intake_preflight_status.json" in preflight.cmd
    assert names.index("production_gate_expected") < names.index("senior_ml_audit")
    assert names.index("senior_ml_audit") < names.index("release_integrity_final")
    assert names.index("senior_ml_audit") < names.index("requirements_traceability_final")
    assert names.index("requirements_traceability_final") < names.index("executive_report_ru_final")
    assert names.index("executive_report_ru_final") < names.index("objective_completion_audit")
    assert names.index("objective_completion_audit") < names.index("release_integrity_final")
    assert names.index("executive_report_ru_final") < names.index("release_integrity_final")
    assert names.index("release_integrity_final") < names.index("report_consistency_audit")
    assert names.index("handoff_report") < names.index("report_consistency_audit")
    assert names.index("production_readiness_report") < names.index("release_integrity_post_reports")
    assert names.index("release_integrity_post_reports") < names.index("report_consistency_audit")
    assert names.index("report_consistency_audit") < names.index("project_readiness")
    assert names[-1] == "pytest"
    assert "performance_audit" in names


def test_build_steps_can_skip_performance_and_pytest():
    names = [step.name for step in build_steps(include_performance=False, include_pytest=False)]

    assert "performance_audit" not in names
    assert "pytest" not in names


def test_pregate_release_integrity_excludes_late_objective_artifacts():
    steps = build_steps(include_performance=False, include_pytest=False)
    pregate_cmd = next(step.cmd for step in steps if step.name == "release_integrity_pregate")
    final_cmd = next(step.cmd for step in steps if step.name == "release_integrity_final")

    assert "outputs/production_audit/objective_completion_audit.json" not in pregate_cmd
    assert "outputs/production_audit/objective_completion_audit.json" in final_cmd


def test_evaluate_suite_status_allows_expected_production_blockers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    audit_root = tmp_path / "outputs" / "production_audit"
    audit_root.mkdir(parents=True)
    (audit_root / "integration_gate.json").write_text('{"ok": true}', encoding="utf-8")
    (audit_root / "production_gate.json").write_text(
        '{"ok": false, "failed": ["human_ar_holdout_eval"]}',
        encoding="utf-8",
    )
    (audit_root / "senior_ml_audit.json").write_text(
        '{"integration_ready": true, "production_ready": false, '
        '"production_blockers": ["human_labelled_ar_device_holdout"]}',
        encoding="utf-8",
    )
    (audit_root / "release_integrity.json").write_text(
        '{"ok": true, "artifact_count": 31}',
        encoding="utf-8",
    )
    (audit_root / "report_consistency_audit.json").write_text(
        '{"ok": true, "failures": []}',
        encoding="utf-8",
    )

    status = evaluate_suite_status(
        [StepResult("production_gate_expected", 1, True, True)],
        strict_production=False,
    )

    assert status["ok"] is True
    assert status["integration_ready"] is True
    assert status["production_ready"] is False
    assert status["report_consistency_ok"] is True


def test_evaluate_suite_status_strict_requires_production(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    audit_root = tmp_path / "outputs" / "production_audit"
    audit_root.mkdir(parents=True)
    (audit_root / "integration_gate.json").write_text('{"ok": true}', encoding="utf-8")
    (audit_root / "production_gate.json").write_text('{"ok": false}', encoding="utf-8")
    (audit_root / "senior_ml_audit.json").write_text(
        '{"integration_ready": true, "production_ready": false}',
        encoding="utf-8",
    )
    (audit_root / "release_integrity.json").write_text('{"ok": true}', encoding="utf-8")
    (audit_root / "report_consistency_audit.json").write_text('{"ok": true, "failures": []}', encoding="utf-8")

    status = evaluate_suite_status([], strict_production=True)

    assert status["ok"] is False


def test_evaluate_suite_status_production_ready_requires_report_consistency(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    audit_root = tmp_path / "outputs" / "production_audit"
    audit_root.mkdir(parents=True)
    (audit_root / "integration_gate.json").write_text('{"ok": true}', encoding="utf-8")
    (audit_root / "production_gate.json").write_text('{"ok": true}', encoding="utf-8")
    (audit_root / "senior_ml_audit.json").write_text(
        '{"integration_ready": true, "production_ready": true, "production_blockers": []}',
        encoding="utf-8",
    )
    (audit_root / "release_integrity.json").write_text('{"ok": true, "artifact_count": 31}', encoding="utf-8")
    (audit_root / "report_consistency_audit.json").write_text(
        '{"ok": false, "failures": ["release_manifest_hashes_current"]}',
        encoding="utf-8",
    )

    status = evaluate_suite_status([], strict_production=False)

    assert status["ok"] is False
    assert status["integration_ready"] is True
    assert status["production_ready"] is False
    assert status["report_consistency_ok"] is False
    assert status["report_consistency_failures"] == ["release_manifest_hashes_current"]
