from __future__ import annotations

from src.executive_report_ru import build_report, fmt, metric


def test_metric_reads_nested_value():
    assert metric({"a": {"b": 1}}, "a", "b") == 1
    assert metric({}, "missing") == "n/a"


def test_fmt_rounds_float():
    assert fmt(0.12345) == "0.123"
    assert fmt("x") == "x"


def test_executive_report_mentions_consolidated_evidence_gate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    audit_root = tmp_path / "outputs" / "production_audit"
    audit_root.mkdir(parents=True)
    (audit_root / "audit_suite_status.json").write_text(
        '{"integration_ready": true, "production_ready": false, "ok": true, '
        '"production_blockers": ["production_gate"]}',
        encoding="utf-8",
    )
    (audit_root / "production_evidence_audit.json").write_text(
        '{"production_evidence_ready": false, "checks": []}',
        encoding="utf-8",
    )
    (audit_root / "requirements_traceability.json").write_text(
        '{"summary": {"passed": 9, "requirements": 14}}',
        encoding="utf-8",
    )

    report = build_report()

    assert "Production evidence audit ready: False" in report
    assert "Consolidated production evidence gate: False" in report
    assert "Deterministic package manifest" in report
    assert "src/run_production_evidence_intake.py" in report


def test_executive_report_does_not_claim_integration_ready_when_gate_fails(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    audit_root = tmp_path / "outputs" / "production_audit"
    audit_root.mkdir(parents=True)
    (audit_root / "audit_suite_status.json").write_text(
        '{"integration_ready": false, "production_ready": false, "ok": false, '
        '"production_blockers": ["dataset_format_and_leakage", "champion_real_validation_quality"]}',
        encoding="utf-8",
    )
    (audit_root / "production_evidence_audit.json").write_text(
        '{"production_evidence_ready": false, "checks": []}',
        encoding="utf-8",
    )
    (audit_root / "requirements_traceability.json").write_text(
        '{"summary": {"passed": 9, "requirements": 16}}',
        encoding="utf-8",
    )

    report = build_report()

    assert "integration gate сейчас не закрыт" in report
    assert "модель является integration-ready" not in report
