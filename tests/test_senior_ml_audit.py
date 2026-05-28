from __future__ import annotations

import json
import argparse

from src.senior_ml_audit import (
    _production_evidence_requirement,
    build_audit,
    metric,
    read_json,
    render_markdown,
)


def test_metric_reads_nested_float():
    report = {"metrics_bbox": {"mAP50": "0.91"}}

    assert metric(report, "metrics_bbox", "mAP50") == 0.91
    assert metric(report, "missing", default=1.0) == 1.0


def test_read_json_returns_empty_for_bad_payload(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{bad", encoding="utf-8")

    assert read_json(path) == {}


def test_render_markdown_lists_blockers():
    audit = {
        "audit_ok": True,
        "integration_ready": True,
        "production_ready": False,
        "counts": {"requirements": 1, "passed": 0, "failed_or_missing": 1},
        "integration_blockers": [],
        "production_blockers": ["ar_3d_replay_validation"],
        "requirements": [
            {
                "name": "ar_3d_replay_validation",
                "category": "production_validation",
                "status": "missing",
                "integration_required": False,
                "production_required": True,
                "evidence": "outputs/production_audit/ar_3d_replay_eval.json",
                "detail": "missing",
            }
        ],
    }

    markdown = render_markdown(audit)

    assert "Senior ML Audit" in markdown
    assert "Production ready: False" in markdown
    assert "ar_3d_replay_validation" in markdown


def test_production_evidence_requirement_reports_consolidated_blockers(tmp_path):
    path = tmp_path / "production_evidence_audit.json"
    path.write_text(
        '{"production_evidence_ready": false, "blockers": ["android_litert_device_validation"]}',
        encoding="utf-8",
    )

    req = _production_evidence_requirement(path)

    assert req.name == "production_evidence_audit_ready"
    assert req.status == "fail"
    assert req.production_required is True
    assert "android_litert_device_validation" in req.detail


def test_build_audit_uses_external_evidence_path_overrides(tmp_path):
    android_eval = tmp_path / "android_eval.json"
    holdout_eval = tmp_path / "holdout_eval.json"
    replay_eval = tmp_path / "replay_eval.json"
    production_evidence = tmp_path / "production_evidence_audit.json"
    integration_gate = tmp_path / "integration_gate.json"
    production_gate = tmp_path / "production_gate.json"
    android_eval.write_text('{"ok": true, "failures": []}', encoding="utf-8")
    holdout_eval.write_text(
        '{"metrics_bbox":{"mAP50":0.9},"oks":{"mean":0.85},"rates":{"false_negative_rate":0.05}}',
        encoding="utf-8",
    )
    replay_eval.write_text('{"ok": true, "failures": []}', encoding="utf-8")
    production_evidence.write_text('{"production_evidence_ready": true, "blockers": []}', encoding="utf-8")
    integration_gate.write_text('{"ok": true, "failed": []}', encoding="utf-8")
    production_gate.write_text('{"ok": true, "failed": []}', encoding="utf-8")
    args = argparse.Namespace(
        android_litert_eval=android_eval,
        ar_holdout_eval=holdout_eval,
        ar_replay_eval=replay_eval,
        production_evidence_audit=production_evidence,
        integration_gate=integration_gate,
        production_gate=production_gate,
    )

    audit = build_audit(args)
    by_name = {req["name"]: req for req in audit["requirements"]}

    assert by_name["android_litert_device_validation"]["evidence"] == str(android_eval)
    assert by_name["human_labelled_ar_device_holdout"]["evidence"] == str(holdout_eval)
    assert by_name["ar_3d_replay_validation"]["evidence"] == str(replay_eval)
    assert by_name["production_evidence_audit_ready"]["evidence"] == str(production_evidence)
    assert by_name["integration_gate"]["evidence"] == str(integration_gate)
    assert by_name["production_gate"]["evidence"] == str(production_gate)
