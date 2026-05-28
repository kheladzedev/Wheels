from __future__ import annotations

from src.requirements_traceability import build_traceability, render_markdown, status_from_requirement


def test_status_from_requirement_returns_named_requirement():
    reqs = [{"name": "a", "status": "pass", "detail": "ok", "evidence": "x"}]

    assert status_from_requirement(reqs, "a")["status"] == "pass"
    assert status_from_requirement(reqs, "missing")["status"] == "missing"


def test_render_markdown_contains_gap_and_summary():
    trace = {
        "production_ready": False,
        "summary": {
            "passed": 1,
            "requirements": 2,
            "train_runs": 10,
            "eval_reports": 19,
            "release_integrity_ok": True,
        },
        "rows": [
            {
                "requirement": "Production gate passes",
                "status": "fail",
                "evidence": "gate.json",
                "detail": "ok=False",
                "gap": "missing holdout",
            }
        ],
    }

    markdown = render_markdown(trace)

    assert "Requirements Traceability" in markdown
    assert "1/2" in markdown
    assert "Release integrity OK: True" in markdown
    assert "missing holdout" in markdown


def test_traceability_includes_consolidated_evidence_gate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    audit_root = tmp_path / "outputs" / "production_audit"
    audit_root.mkdir(parents=True)
    (audit_root / "senior_ml_audit.json").write_text(
        '{"production_ready": false, "requirements": ['
        '{"name": "production_evidence_audit_ready", "status": "fail", '
        '"detail": "production_evidence_ready=False", '
        '"evidence": "outputs/production_audit/production_evidence_audit.json"}'
        "]}",
        encoding="utf-8",
    )
    (audit_root / "production_evidence_audit.json").write_text(
        '{"blockers": ["ar_3d_replay_validation"], "checks": []}',
        encoding="utf-8",
    )

    trace = build_traceability()
    rows = {row["requirement"]: row for row in trace["rows"]}

    row = rows["Consolidated production evidence audit passes"]
    assert row["status"] == "fail"
    assert "ar_3d_replay_validation" in row["gap"]
