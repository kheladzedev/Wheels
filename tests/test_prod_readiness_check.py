from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import scripts.prod_readiness_check as prc
from scripts.prod_readiness_check import (
    BatchAcceptance,
    DEFAULT_AUDIT_REPORT_PATH,
    DEFAULT_DATA_INCOMING_ROOT,
    DEFAULT_DEMO_SUMMARY_PATH,
    DEFAULT_RUNS_POSE_ROOT,
    OVERALL_AR_READY,
    OVERALL_BLOCKED,
    OVERALL_DEMO_READY,
    OVERALL_TRAINING_ALLOWED_AR_BLOCKED,
    ROOT,
    SemanticsRecord,
    compute_verdict,
    derive_batch_status,
    load_acceptance_files,
    load_audit_report,
    load_demo_summary,
    load_semantics_files,
    parse_semantics_md,
    render_markdown,
    run_check,
)


FIXED_NOW = datetime(2026, 5, 28, 12, 0, 0, tzinfo=timezone.utc)


def _accept_payload(**overrides) -> dict:
    payload = {
        "schema_version": 1,
        "source": "android_plugin",
        "batch_id": "android_plugin_real_v2",
        "export_date": "2026-06-10",
        "image_count": 100,
        "wheel_count": 350,
        "bbox_source": "PLUGIN_BUILD_v2.4.1",
        "keypoint_mapping": "floorray_v1",
        "requires_plugin_bbox": False,
        "validation_result": "PASS",
        "preview_result": "PASS",
        "bbox_audit_result": "PASS",
        "human_reviewer": "reviewer_b",
        "human_preview_accepted": True,
        "review_date": "2026-06-11",
        "review_notes": "ok",
        "status": "ACCEPT_FOR_TRAINING",
    }
    payload.update(overrides)
    return payload


def _make_batch_record(payload: dict, *, batch_id: str = "batch") -> BatchAcceptance:
    derived, failures = derive_batch_status(payload)
    return BatchAcceptance(
        batch_id=batch_id,
        path=f"/tmp/{batch_id}/metadata/acceptance_status.json",
        declared_status=payload.get("status"),
        derived_status=derived,
        failures=failures,
        payload=payload,
    )


def _semantics_record(**fields) -> SemanticsRecord:
    return SemanticsRecord(
        model_dir=fields.pop("model_dir", "wheel_model"),
        path="/tmp/runs/pose/wheel_model/SEMANTICS.md",
        fields=dict(fields),
    )


def _demo_summary(production: bool = False, ar: bool = False) -> dict:
    return {
        "demo_kind": "awe_demo_pack",
        "production_claim": production,
        "ar_ready_claim": ar,
        "items": [],
    }


def test_parse_semantics_md_normalises_booleans_and_skips_noise():
    text = """
---
# heading should be skipped
semantics_version: floorray_v1
trained_on_real_data: true
stale: false
trained_at: 2026-05-27
notes: post-exporter retrain
not a key value line
"""
    fields = parse_semantics_md(text)
    assert fields["semantics_version"] == "floorray_v1"
    assert fields["trained_on_real_data"] is True
    assert fields["stale"] is False
    assert fields["trained_at"] == "2026-05-27"
    assert fields["notes"] == "post-exporter retrain"


def test_derive_batch_status_accepts_clean_payload():
    derived, failures = derive_batch_status(_accept_payload())
    assert derived == "ACCEPT_FOR_TRAINING"
    assert failures == []


def test_derive_batch_status_forces_debug_on_plugin_bbox_blocker():
    payload = _accept_payload(requires_plugin_bbox=True)
    derived, failures = derive_batch_status(payload)
    assert derived == "ACCEPT_ONLY_AS_DEBUG"
    assert any("requires_plugin_bbox" in f for f in failures)


def test_derive_batch_status_rejects_placeholder_bbox_source():
    payload = _accept_payload(bbox_source="PLACEHOLDER")
    derived, failures = derive_batch_status(payload)
    assert derived == "REJECT_NEEDS_FIX"
    assert any("placeholder" in f.lower() for f in failures)


def test_derive_batch_status_rejects_missing_human_review():
    payload = _accept_payload(human_preview_accepted=False)
    derived, failures = derive_batch_status(payload)
    assert derived == "REJECT_NEEDS_FIX"
    assert any("human_preview_accepted" in f for f in failures)


def test_derive_batch_status_rejects_placeholder_reviewer():
    payload = _accept_payload(human_reviewer="FILL_ME")
    derived, _ = derive_batch_status(payload)
    assert derived == "REJECT_NEEDS_FIX"


def test_derive_batch_status_demotes_legacy_keypoint_mapping():
    payload = _accept_payload(keypoint_mapping="rim_v0")
    derived, _ = derive_batch_status(payload)
    assert derived == "ACCEPT_ONLY_AS_DEBUG"


def test_verdict_demo_pack_only_blocks_training_and_ar():
    verdict = compute_verdict(
        acceptances=[],
        audit=None,
        demo_summary=_demo_summary(),
        semantics=[],
        pose_runs_present=False,
        now=FIXED_NOW,
    )
    assert verdict["demo_ready"] is True
    assert verdict["training_allowed"] is False
    assert verdict["ar_ready_claim_allowed"] is False
    assert verdict["production_ready"] is False
    assert verdict["overall_status"] == OVERALL_DEMO_READY
    assert any("acceptance_status.json" in b for b in verdict["current_blockers"])


def test_verdict_no_inputs_at_all_is_blocked_on_real_data():
    verdict = compute_verdict(
        acceptances=[],
        audit=None,
        demo_summary=None,
        semantics=[],
        pose_runs_present=False,
        now=FIXED_NOW,
    )
    assert verdict["demo_ready"] is False
    assert verdict["overall_status"] == OVERALL_BLOCKED


def test_verdict_blocks_training_on_accept_only_as_debug():
    payload = _accept_payload(requires_plugin_bbox=True, status="ACCEPT_ONLY_AS_DEBUG")
    batch = _make_batch_record(payload, batch_id="android_plugin_real")
    assert batch.derived_status == "ACCEPT_ONLY_AS_DEBUG"

    verdict = compute_verdict(
        acceptances=[batch],
        audit=None,
        demo_summary=_demo_summary(),
        semantics=[],
        pose_runs_present=False,
        now=FIXED_NOW,
    )
    assert verdict["training_allowed"] is False
    assert verdict["ar_ready_claim_allowed"] is False
    assert verdict["overall_status"] == OVERALL_DEMO_READY
    assert any(
        "plugin" in b.lower() or "exporter" in b.lower()
        for b in verdict["current_blockers"]
    )


def test_verdict_blocks_training_on_requires_plugin_bbox_even_if_status_lies():
    payload = _accept_payload(requires_plugin_bbox=True, status="ACCEPT_FOR_TRAINING")
    batch = _make_batch_record(payload, batch_id="lying_batch")
    assert batch.derived_status == "ACCEPT_ONLY_AS_DEBUG"

    verdict = compute_verdict(
        acceptances=[batch],
        audit=None,
        demo_summary=None,
        semantics=[],
        pose_runs_present=False,
        now=FIXED_NOW,
    )
    assert verdict["training_allowed"] is False


def test_verdict_blocks_training_on_missing_human_preview_marker():
    payload = _accept_payload(human_preview_accepted=False)
    batch = _make_batch_record(payload)
    verdict = compute_verdict(
        acceptances=[batch],
        audit=None,
        demo_summary=None,
        semantics=[],
        pose_runs_present=False,
        now=FIXED_NOW,
    )
    assert verdict["training_allowed"] is False


def test_verdict_blocks_ar_ready_on_stale_model_even_if_other_gates_pass():
    payload = _accept_payload()
    batch = _make_batch_record(payload, batch_id="good_batch")
    stale = _semantics_record(
        model_dir="wheel_stale",
        semantics_version="floorray_v1",
        trained_on_real_data=True,
        stale=True,
        trained_at="2026-05-01",
    )
    audit = {
        "geometry_audit_pass": True,
        "bbox_audit_pass": True,
        "ar_replay_metric_pass": True,
        "export_parity_pass": True,
    }
    verdict = compute_verdict(
        acceptances=[batch],
        audit=audit,
        demo_summary=_demo_summary(),
        semantics=[stale],
        pose_runs_present=True,
        now=FIXED_NOW,
    )
    assert verdict["training_allowed"] is True
    assert verdict["ar_ready_claim_allowed"] is False
    assert any("stale" in b.lower() for b in verdict["current_blockers"])
    assert verdict["overall_status"] == OVERALL_TRAINING_ALLOWED_AR_BLOCKED


def test_verdict_grants_ar_ready_when_every_gate_passes():
    payload = _accept_payload()
    batch = _make_batch_record(payload, batch_id="prod_batch")
    fresh = _semantics_record(
        model_dir="wheel_floorray",
        semantics_version="floorray_v1",
        trained_on_real_data=True,
        stale=False,
        trained_at="2026-06-15",
    )
    audit = {
        "geometry_audit_pass": True,
        "bbox_audit_pass": True,
        "ar_replay_metric_pass": True,
        "export_parity_pass": True,
    }
    verdict = compute_verdict(
        acceptances=[batch],
        audit=audit,
        demo_summary=_demo_summary(),
        semantics=[fresh],
        pose_runs_present=True,
        now=FIXED_NOW,
    )
    assert verdict["training_allowed"] is True
    assert verdict["ar_ready_claim_allowed"] is True
    assert verdict["production_ready"] is False
    assert verdict["overall_status"] == OVERALL_AR_READY


def test_render_markdown_never_claims_production_ready_unless_gates_pass():
    verdict = compute_verdict(
        acceptances=[],
        audit=None,
        demo_summary=_demo_summary(),
        semantics=[],
        pose_runs_present=False,
        now=FIXED_NOW,
    )
    md = render_markdown(verdict)
    assert "production_ready" in md
    assert "`False`" in md or "False" in md
    forbidden = (
        "claim AR-ready",
        "production-ready",
        "AR-ready, production-ready",
    )
    for phrase in forbidden:
        assert phrase in md, f"missing the forbidden-claims warning: {phrase!r}"
    assert "Do **not** claim AR-ready" in md


def test_load_acceptance_files_reads_nested_batches(tmp_path):
    incoming = tmp_path / "data" / "incoming"
    batch_dir = incoming / "android_plugin_real" / "metadata"
    batch_dir.mkdir(parents=True)
    payload = _accept_payload(
        requires_plugin_bbox=True,
        bbox_source="PLACEHOLDER",
        status="REJECT_NEEDS_FIX",
        human_preview_accepted=False,
        human_reviewer=None,
    )
    (batch_dir / "acceptance_status.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    records = load_acceptance_files(incoming)
    assert len(records) == 1
    assert records[0].derived_status in {"ACCEPT_ONLY_AS_DEBUG", "REJECT_NEEDS_FIX"}


def test_run_check_end_to_end_writes_both_reports(tmp_path):
    out_dir = tmp_path / "prod_readiness"
    demo_path = tmp_path / "demo_summary.json"
    demo_path.write_text(json.dumps(_demo_summary()), encoding="utf-8")

    verdict = run_check(
        data_incoming_root=tmp_path / "incoming_does_not_exist",
        audit_report_path=tmp_path / "audit_missing.json",
        demo_summary_path=demo_path,
        runs_pose_root=tmp_path / "runs_pose_missing",
        out_dir=out_dir,
        now=FIXED_NOW,
    )
    assert (out_dir / "REPORT.json").is_file()
    assert (out_dir / "REPORT.md").is_file()
    on_disk = json.loads((out_dir / "REPORT.json").read_text(encoding="utf-8"))
    assert on_disk["overall_status"] == verdict["overall_status"]
    assert on_disk["demo_ready"] is True
    assert on_disk["training_allowed"] is False
    assert on_disk["ar_ready_claim_allowed"] is False


def test_module_root_anchors_to_repo_not_cwd():
    """ROOT is computed from ``__file__`` so it never depends on the caller's cwd."""
    assert ROOT.is_absolute()
    assert (ROOT / "pytest.ini").is_file()
    assert (ROOT / "scripts" / "prod_readiness_check.py").is_file()


def test_default_paths_anchored_under_root_module_constants():
    assert DEFAULT_DATA_INCOMING_ROOT == ROOT / "data" / "incoming"
    assert (
        DEFAULT_AUDIT_REPORT_PATH
        == ROOT / "outputs" / "full_pipeline_audit" / "REPORT.json"
    )
    assert (
        DEFAULT_DEMO_SUMMARY_PATH == ROOT / "outputs" / "awe_demo" / "demo_summary.json"
    )
    assert DEFAULT_RUNS_POSE_ROOT == ROOT / "runs" / "pose"


def test_default_paths_are_cwd_independent(monkeypatch, tmp_path):
    """Switching cwd must not change the constants — they are import-time absolutes."""
    snapshot = (
        ROOT,
        DEFAULT_DATA_INCOMING_ROOT,
        DEFAULT_AUDIT_REPORT_PATH,
        DEFAULT_DEMO_SUMMARY_PATH,
        DEFAULT_RUNS_POSE_ROOT,
    )

    deep_subdir = tmp_path / "deep" / "nested"
    deep_subdir.mkdir(parents=True)
    monkeypatch.chdir(deep_subdir)

    assert (
        prc.ROOT,
        prc.DEFAULT_DATA_INCOMING_ROOT,
        prc.DEFAULT_AUDIT_REPORT_PATH,
        prc.DEFAULT_DEMO_SUMMARY_PATH,
        prc.DEFAULT_RUNS_POSE_ROOT,
    ) == snapshot


def test_load_acceptance_files_detects_existing_status_at_default_layout(tmp_path):
    """Mirror the production layout (``data/incoming/<batch>/metadata/...``)."""
    incoming = tmp_path / "data" / "incoming"
    batch_meta = incoming / "android_plugin_real" / "metadata"
    batch_meta.mkdir(parents=True)
    payload = _accept_payload(
        requires_plugin_bbox=True,
        bbox_source="PLACEHOLDER",
        status="ACCEPT_ONLY_AS_DEBUG",
        human_preview_accepted=False,
        human_reviewer=None,
        validation_result="NOT_RUN",
        preview_result="NOT_RUN",
        bbox_audit_result="NOT_RUN",
    )
    (batch_meta / "acceptance_status.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    records = load_acceptance_files(incoming)
    assert len(records) == 1
    rec = records[0]
    assert rec.path.endswith(
        "data/incoming/android_plugin_real/metadata/acceptance_status.json"
    )
    assert rec.derived_status == "ACCEPT_ONLY_AS_DEBUG"


def test_load_audit_report_detects_existing_report_file(tmp_path):
    audit_dir = tmp_path / "outputs" / "full_pipeline_audit"
    audit_dir.mkdir(parents=True)
    audit_path = audit_dir / "REPORT.json"
    audit_path.write_text(
        json.dumps(
            {
                "geometry_audit_pass": True,
                "bbox_audit_pass": True,
                "ar_replay_metric_pass": False,
                "export_parity_pass": True,
            }
        ),
        encoding="utf-8",
    )

    audit = load_audit_report(audit_path)
    assert audit is not None
    assert audit["geometry_audit_pass"] is True
    assert audit["ar_replay_metric_pass"] is False


def test_load_demo_summary_detects_existing_file(tmp_path):
    demo_path = tmp_path / "outputs" / "awe_demo" / "demo_summary.json"
    demo_path.parent.mkdir(parents=True)
    demo_path.write_text(json.dumps(_demo_summary()), encoding="utf-8")
    summary = load_demo_summary(demo_path)
    assert summary is not None
    assert summary["production_claim"] is False


def test_run_check_picks_up_existing_files_in_fake_repo(tmp_path):
    """End-to-end: drop every input at its expected relative path, confirm detection."""
    incoming = tmp_path / "data" / "incoming"
    batch_meta = incoming / "android_plugin_real" / "metadata"
    batch_meta.mkdir(parents=True)
    debug_payload = _accept_payload(
        requires_plugin_bbox=True,
        bbox_source="PLACEHOLDER",
        status="ACCEPT_ONLY_AS_DEBUG",
        human_preview_accepted=False,
        human_reviewer=None,
        validation_result="NOT_RUN",
        preview_result="NOT_RUN",
        bbox_audit_result="NOT_RUN",
    )
    (batch_meta / "acceptance_status.json").write_text(
        json.dumps(debug_payload), encoding="utf-8"
    )

    audit_dir = tmp_path / "outputs" / "full_pipeline_audit"
    audit_dir.mkdir(parents=True)
    (audit_dir / "REPORT.json").write_text(
        json.dumps(
            {
                "geometry_audit_pass": False,
                "bbox_audit_pass": False,
                "ar_replay_metric_pass": False,
                "export_parity_pass": False,
            }
        ),
        encoding="utf-8",
    )

    demo_path = tmp_path / "outputs" / "awe_demo" / "demo_summary.json"
    demo_path.parent.mkdir(parents=True)
    demo_path.write_text(json.dumps(_demo_summary()), encoding="utf-8")

    runs_pose = tmp_path / "runs" / "pose"
    runs_pose.mkdir(parents=True)

    out_dir = tmp_path / "outputs" / "prod_readiness"
    verdict = run_check(
        data_incoming_root=incoming,
        audit_report_path=audit_dir / "REPORT.json",
        demo_summary_path=demo_path,
        runs_pose_root=runs_pose,
        out_dir=out_dir,
        now=FIXED_NOW,
    )

    assert verdict["inputs_seen"]["acceptance_files"], (
        "acceptance_status.json was placed at the canonical layout and must be detected"
    )
    assert verdict["inputs_seen"]["audit_report"] is True
    assert verdict["inputs_seen"]["demo_summary"] is True
    assert verdict["demo_ready"] is True
    assert verdict["training_allowed"] is False
    assert verdict["ar_ready_claim_allowed"] is False
    assert verdict["overall_status"] == OVERALL_DEMO_READY
    assert (out_dir / "REPORT.json").is_file()
    assert (out_dir / "REPORT.md").is_file()


def test_run_check_from_subdirectory_keeps_default_resolution(monkeypatch, tmp_path):
    """Calling ``run_check`` from a deep cwd with default paths must still hit
    the real repo's ``outputs/awe_demo`` / ``data/incoming`` / ``runs/pose``."""
    snapshot_root = ROOT
    deep_subdir = tmp_path / "deep" / "nested" / "cwd"
    deep_subdir.mkdir(parents=True)
    monkeypatch.chdir(deep_subdir)

    verdict = run_check(write=False, now=FIXED_NOW)

    assert prc.ROOT == snapshot_root
    for key in (
        "overall_status",
        "demo_ready",
        "training_allowed",
        "ar_ready_claim_allowed",
        "production_ready",
        "current_blockers",
        "next_safe_task",
    ):
        assert key in verdict
    assert verdict["production_ready"] is False


def test_load_semantics_files_detects_missing_keys(tmp_path):
    runs = tmp_path / "runs" / "pose"
    model_dir = runs / "wheel_x"
    model_dir.mkdir(parents=True)
    (model_dir / "SEMANTICS.md").write_text(
        "semantics_version: floorray_v1\ntrained_on_real_data: true\n",
        encoding="utf-8",
    )

    records = load_semantics_files(runs)
    assert len(records) == 1
    rec = records[0]
    assert rec.fields.get("trained_on_real_data") is True
    assert any("stale" in f for f in rec.failures)
    assert rec.is_real_floorray is False
