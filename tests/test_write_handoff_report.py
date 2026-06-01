from __future__ import annotations

from scripts.write_handoff_report import _gate_status_line


def test_gate_status_line_reports_failed_integration_gate():
    line = _gate_status_line(
        {"ok": False, "failed": ["dataset_audit", "real_only_fp_ceiling"]},
        {"ok": False, "failed": ["production_evidence_audit_ready"]},
    )

    assert "Integration gate: FAIL" in line
    assert "dataset_audit" in line
    assert "real_only_fp_ceiling" in line
    assert "passes" not in line
