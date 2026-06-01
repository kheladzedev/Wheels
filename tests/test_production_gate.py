from __future__ import annotations

import argparse
import json
from pathlib import Path

import production_gate as pg


def _write_eval(
    path: Path, *, map50: float, oks: float, fn: float, fp: float = 0.0
) -> None:
    path.write_text(
        json.dumps(
            {
                "metrics_bbox": {"mAP50": map50},
                "oks": {"mean": oks},
                "rates": {"false_negative_rate": fn, "false_positive_rate": fp},
            }
        ),
        encoding="utf-8",
    )


def test_integration_gate_ignores_production_only_blockers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "AR_ML_CONTRACT.md").write_text("contract", encoding="utf-8")
    (tmp_path / "docs" / "PRODUCTION_READINESS_AUDIT.md").write_text(
        "audit", encoding="utf-8"
    )
    pt = tmp_path / "best.pt"
    onnx = tmp_path / "best.onnx"
    pt.write_bytes(b"pt")
    onnx.write_bytes(b"onnx")
    real_eval = tmp_path / "real.json"
    anchor_eval = tmp_path / "anchor.json"
    onnx_eval = tmp_path / "onnx.json"
    drift = tmp_path / "drift.json"
    operating_point = tmp_path / "operating_point_audit.json"
    dataset_audit = tmp_path / "dataset_audit.json"
    release_integrity = tmp_path / "release_integrity.json"
    performance_audit = tmp_path / "performance_audit.json"
    runtime_contract_audit = tmp_path / "runtime_contract_audit.json"
    export_certification = tmp_path / "export_certification.json"
    production_evidence_audit = tmp_path / "production_evidence_audit.json"
    _write_eval(real_eval, map50=0.9, oks=0.88, fn=0.05, fp=0.05)
    _write_eval(anchor_eval, map50=0.7, oks=0.88, fn=0.2)
    _write_eval(onnx_eval, map50=0.69, oks=0.88, fn=0.2)
    drift.write_text(
        '{"ok": false, "samples_matched": 14, "samples_checked": 20}', encoding="utf-8"
    )
    operating_point.write_text(
        json.dumps(
            {
                "ok": True,
                "selected": {
                    "path": "real.json",
                    "conf": 0.50,
                    "bbox_mAP50": 0.9,
                    "oks_mean": 0.88,
                    "false_negative_rate": 0.05,
                    "false_positive_rate": 0.05,
                },
            }
        ),
        encoding="utf-8",
    )
    dataset_audit.write_text('{"ok": true, "failures": []}', encoding="utf-8")
    release_integrity.write_text('{"ok": true, "failures": []}', encoding="utf-8")
    performance_audit.write_text('{"ok": true, "failures": []}', encoding="utf-8")
    runtime_contract_audit.write_text('{"ok": true, "failures": []}', encoding="utf-8")
    export_certification.write_text(
        '{"certified": true, "status": "certified"}', encoding="utf-8"
    )
    production_evidence_audit.write_text(
        '{"production_evidence_ready": false, "blockers": ["android_litert_device_validation"]}',
        encoding="utf-8",
    )

    args = argparse.Namespace(
        champion_pt=pt,
        champion_onnx=onnx,
        champion_real_eval=real_eval,
        operating_point_audit=operating_point,
        champion_anchor_eval=anchor_eval,
        onnx_eval=onnx_eval,
        onnx_drift=drift,
        export_certification=export_certification,
        tflite_certified=tmp_path / "missing_tflite.json",
        android_litert_eval=tmp_path / "missing_android_litert.json",
        ar_holdout_eval=tmp_path / "missing_holdout.json",
        ar_3d_eval=tmp_path / "missing_3d.json",
        dataset_audit=dataset_audit,
        release_integrity=release_integrity,
        performance_audit=performance_audit,
        runtime_contract_audit=runtime_contract_audit,
        production_evidence_audit=production_evidence_audit,
        min_real_map50=0.85,
        min_real_oks=0.8,
        max_real_fn=0.1,
        max_real_fp=0.15,
        min_anchor_map50=0.65,
        min_onnx_map50=0.65,
        min_ar_holdout_map50=0.85,
        min_ar_holdout_oks=0.8,
        max_ar_holdout_fn=0.1,
        max_ar_holdout_fp=0.15,
    )

    items = pg.build_gate_items(args)

    assert pg.evaluate(items, "integration") is True
    assert pg.evaluate(items, "production") is False
    assert any(
        item.name == "onnx_strict_parity_diagnostic" and not item.ok for item in items
    )
    assert "onnx_strict_parity_diagnostic" in pg.warning_items(items)
    assert "onnx_strict_parity_diagnostic" not in pg.failed_items(items, "production")
    assert "production_evidence_audit_ready" in pg.failed_items(items, "production")
    assert any(item.name == "exported_backends_certified" and item.ok for item in items)


def test_tflite_gate_requires_certified_true(tmp_path):
    missing = pg.certification_gate(
        "tflite_litert_certified", tmp_path / "missing.json"
    )
    assert missing.ok is False
    assert "missing" in missing.detail

    failed = tmp_path / "failed.json"
    failed.write_text(
        json.dumps({"certified": False, "status": "aggregate_pass_parity_failed"}),
        encoding="utf-8",
    )
    failed_item = pg.certification_gate("tflite_litert_certified", failed)
    assert failed_item.ok is False
    assert "certified=False" in failed_item.detail

    passed = tmp_path / "passed.json"
    passed.write_text(
        json.dumps({"certified": True, "status": "passed"}),
        encoding="utf-8",
    )
    assert pg.certification_gate("tflite_litert_certified", passed).ok is True


def test_tflite_gate_rejects_truthy_non_boolean_certification(tmp_path):
    report = tmp_path / "truthy_certified.json"
    report.write_text(
        json.dumps({"certified": 1, "status": "passed"}),
        encoding="utf-8",
    )

    assert pg.certification_gate("tflite_litert_certified", report).ok is False

    report.write_text(
        json.dumps({"certified": "true", "status": "passed"}),
        encoding="utf-8",
    )

    assert pg.certification_gate("tflite_litert_certified", report).ok is False


def test_ar_report_gate_requires_ok_true(tmp_path):
    failed = tmp_path / "ar_3d.json"
    failed.write_text(
        json.dumps({"ok": False, "failures": ["too_few_observations"]}),
        encoding="utf-8",
    )

    assert pg.report_ok_gate("ar_3d_replay_eval", failed).ok is False

    passed = tmp_path / "ar_3d_pass.json"
    passed.write_text(json.dumps({"ok": True, "failures": []}), encoding="utf-8")

    assert pg.report_ok_gate("ar_3d_replay_eval", passed).ok is True


def test_report_ok_gate_rejects_truthy_non_boolean_ok(tmp_path):
    report = tmp_path / "ar_3d_truthy.json"
    report.write_text(json.dumps({"ok": 1, "failures": []}), encoding="utf-8")

    assert pg.report_ok_gate("ar_3d_replay_eval", report).ok is False

    report.write_text(json.dumps({"ok": "true", "failures": []}), encoding="utf-8")

    assert pg.report_ok_gate("ar_3d_replay_eval", report).ok is False


def test_dataset_audit_gate_prefers_explicit_production_subset(tmp_path):
    report = tmp_path / "dataset_audit.json"
    report.write_text(
        json.dumps(
            {
                "ok": False,
                "failures": ["legacy_configs_failed"],
                "gate": {
                    "ok": True,
                    "scope": "configured_subset",
                    "configs": ["configs/pose_dataset_strict.yaml"],
                    "failed_configs": [],
                    "missing_configs": [],
                },
            }
        ),
        encoding="utf-8",
    )

    item = pg.dataset_audit_gate("dataset_audit", report)

    assert item.ok is True
    assert "gate_ok=True" in item.detail
    assert "overall_ok=False" in item.detail


def test_dataset_audit_gate_falls_back_to_overall_ok(tmp_path):
    report = tmp_path / "dataset_audit.json"
    report.write_text('{"ok": true, "failures": []}', encoding="utf-8")

    assert pg.dataset_audit_gate("dataset_audit", report).ok is True

    report.write_text('{"ok": false, "failures": ["bad"]}', encoding="utf-8")
    assert pg.dataset_audit_gate("dataset_audit", report).ok is False


def test_production_evidence_gate_requires_ready_true(tmp_path):
    failed = tmp_path / "evidence.json"
    failed.write_text(
        json.dumps(
            {
                "ok": True,
                "production_evidence_ready": False,
                "blockers": ["ar_3d_replay_validation"],
            }
        ),
        encoding="utf-8",
    )

    failed_item = pg.evidence_ready_gate("production_evidence_audit_ready", failed)

    assert failed_item.ok is False
    assert "ar_3d_replay_validation" in failed_item.detail

    passed = tmp_path / "evidence_pass.json"
    passed.write_text(
        json.dumps({"ok": True, "production_evidence_ready": True, "blockers": []}),
        encoding="utf-8",
    )

    assert pg.evidence_ready_gate("production_evidence_audit_ready", passed).ok is True


def test_production_evidence_gate_rejects_truthy_non_boolean_ready(tmp_path):
    report = tmp_path / "evidence_truthy.json"
    report.write_text(
        json.dumps({"ok": True, "production_evidence_ready": 1, "blockers": []}),
        encoding="utf-8",
    )

    assert pg.evidence_ready_gate("production_evidence_audit_ready", report).ok is False


def test_human_holdout_gate_requires_quality_thresholds(tmp_path):
    report = tmp_path / "holdout.json"
    _write_eval(report, map50=0.86, oks=0.82, fn=0.08)

    assert (
        pg.eval_quality_gate(
            "human_ar_holdout_eval",
            report,
            min_map50=0.85,
            min_oks=0.8,
            max_fn=0.1,
        ).ok
        is True
    )

    _write_eval(report, map50=0.7, oks=0.82, fn=0.08)
    assert (
        pg.eval_quality_gate(
            "human_ar_holdout_eval",
            report,
            min_map50=0.85,
            min_oks=0.8,
            max_fn=0.1,
        ).ok
        is False
    )

    _write_eval(report, map50=0.9, oks=0.85, fn=0.05, fp=0.10)
    assert (
        pg.eval_quality_gate(
            "human_ar_holdout_eval",
            report,
            min_map50=0.85,
            min_oks=0.8,
            max_fn=0.1,
            max_fp=0.15,
        ).ok
        is True
    )


def test_quality_gate_rejects_boolean_metric_values(tmp_path):
    report = tmp_path / "holdout_bool_metrics.json"
    report.write_text(
        json.dumps(
            {
                "metrics_bbox": {"mAP50": True},
                "oks": {"mean": True},
                "rates": {"false_negative_rate": False},
            }
        ),
        encoding="utf-8",
    )

    assert (
        pg.eval_quality_gate(
            "human_ar_holdout_eval",
            report,
            min_map50=0.85,
            min_oks=0.8,
            max_fn=0.1,
        ).ok
        is False
    )


def test_quality_gate_enforces_fp_ceiling(tmp_path):
    # A false-positive rate over the ceiling must fail the gate even when
    # mAP50/OKS/FN all pass — FP was previously ungated (silent pass).
    report = tmp_path / "holdout_fp.json"
    _write_eval(report, map50=0.9, oks=0.85, fn=0.05, fp=0.30)
    assert (
        pg.eval_quality_gate(
            "human_ar_holdout_eval",
            report,
            min_map50=0.85,
            min_oks=0.8,
            max_fn=0.1,
            max_fp=0.15,
        ).ok
        is False
    )


def test_real_quality_gates_use_selected_operating_point(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "AR_ML_CONTRACT.md").write_text("contract", encoding="utf-8")
    (tmp_path / "docs" / "PRODUCTION_READINESS_AUDIT.md").write_text(
        "audit", encoding="utf-8"
    )
    pt = tmp_path / "best.pt"
    onnx = tmp_path / "best.onnx"
    pt.write_bytes(b"pt")
    onnx.write_bytes(b"onnx")
    real_eval = tmp_path / "real_default.json"
    anchor_eval = tmp_path / "anchor.json"
    onnx_eval = tmp_path / "onnx.json"
    drift = tmp_path / "drift.json"
    operating_point = tmp_path / "operating_point_audit.json"
    _write_eval(real_eval, map50=0.91, oks=0.88, fn=0.06, fp=0.25)
    _write_eval(anchor_eval, map50=0.7, oks=0.88, fn=0.2)
    _write_eval(onnx_eval, map50=0.69, oks=0.88, fn=0.2)
    drift.write_text('{"ok": true, "samples_matched": 20, "samples_checked": 20}', encoding="utf-8")
    operating_point.write_text(
        json.dumps(
            {
                "ok": True,
                "selected": {
                    "path": "outputs/production_audit/threshold_conf080_real_val.json",
                    "conf": 0.80,
                    "bbox_mAP50": 0.903,
                    "oks_mean": 0.887,
                    "false_negative_rate": 0.094,
                    "false_positive_rate": 0.147,
                },
            }
        ),
        encoding="utf-8",
    )
    for name in (
        "dataset_audit",
        "release_integrity",
        "performance_audit",
        "runtime_contract_audit",
    ):
        (tmp_path / f"{name}.json").write_text('{"ok": true, "failures": []}', encoding="utf-8")
    export_certification = tmp_path / "export_certification.json"
    tflite_certification = tmp_path / "tflite_certification.json"
    export_certification.write_text('{"certified": true, "status": "certified"}', encoding="utf-8")
    tflite_certification.write_text('{"certified": true, "status": "certified"}', encoding="utf-8")
    production_evidence_audit = tmp_path / "production_evidence_audit.json"
    production_evidence_audit.write_text(
        '{"production_evidence_ready": false, "blockers": ["android_litert_device_validation"]}',
        encoding="utf-8",
    )
    args = argparse.Namespace(
        champion_pt=pt,
        champion_onnx=onnx,
        champion_real_eval=real_eval,
        champion_anchor_eval=anchor_eval,
        onnx_eval=onnx_eval,
        onnx_drift=drift,
        operating_point_audit=operating_point,
        export_certification=export_certification,
        tflite_certified=tflite_certification,
        android_litert_eval=tmp_path / "missing_android_litert.json",
        ar_holdout_eval=tmp_path / "missing_holdout.json",
        ar_3d_eval=tmp_path / "missing_3d.json",
        dataset_audit=tmp_path / "dataset_audit.json",
        release_integrity=tmp_path / "release_integrity.json",
        performance_audit=tmp_path / "performance_audit.json",
        runtime_contract_audit=tmp_path / "runtime_contract_audit.json",
        production_evidence_audit=production_evidence_audit,
        min_real_map50=0.85,
        min_real_oks=0.8,
        max_real_fn=0.1,
        max_real_fp=0.15,
        min_anchor_map50=0.65,
        min_onnx_map50=0.65,
        min_ar_holdout_map50=0.85,
        min_ar_holdout_oks=0.8,
        max_ar_holdout_fn=0.1,
        max_ar_holdout_fp=0.15,
    )

    items = pg.build_gate_items(args)

    fp_item = next(item for item in items if item.name == "real_only_fp_ceiling")
    op_item = next(item for item in items if item.name == "real_only_operating_point")
    assert fp_item.ok is True
    assert op_item.ok is True
    assert "conf=0.800" in op_item.detail
