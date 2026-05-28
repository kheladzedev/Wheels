from __future__ import annotations

import spec_compliance_audit as sca


def test_spec_compliance_audit_passes_current_contract():
    audit = sca.build_audit()

    assert audit["ok"] is True
    assert audit["failures"] == []
    assert audit["confirmed_schema"]["frame_id"] == "spec-audit-frame"
    assert len(audit["confirmed_schema"]["wheels"]) == 1
    assert set(audit["confirmed_schema"]["wheels"][0]["points"]) == {
        "a",
        "b",
        "c_disc_bottom",
    }


def test_spec_compliance_audit_detects_missing_contract_anchor(monkeypatch):
    monkeypatch.setattr(
        sca,
        "_read_text",
        lambda path: "" if path.as_posix() == "docs/AR_ML_CONTRACT.md" else "ok",
    )

    audit = sca.build_audit()

    assert audit["ok"] is False
    assert "ar_ml_contract_document" in audit["failures"]


def test_spec_compliance_markdown_lists_checks():
    markdown = sca.render_markdown(
        {
            "ok": True,
            "failures": [],
            "policy": {"ml_scope": "2D", "ar_scope": "3D"},
            "checks": [
                {
                    "name": "confirmed_points_schema",
                    "ok": True,
                    "evidence": "src/postprocess_wheels.py",
                    "detail": "points=['a', 'b', 'c_disc_bottom']",
                }
            ],
        }
    )

    assert "Spec Compliance Audit" in markdown
    assert "confirmed_points_schema" in markdown
