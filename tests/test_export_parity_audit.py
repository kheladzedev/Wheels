from __future__ import annotations

import json

from src.export_parity_audit import (
    build_findings,
    build_audit,
    classify_failures,
    passes_at,
    render_markdown,
)


def test_classify_failures_detects_categories():
    sample = {
        "n_pt": 2,
        "n_exported": 2,
        "failures": [
            "bbox drift exceeds 2.0px",
            "keypoint drift exceeds 3.0px",
            "conf drift exceeds 0.05",
        ],
    }

    assert classify_failures(sample) == [
        "bbox_drift",
        "keypoint_drift",
        "confidence_drift",
    ]


def test_passes_at_respects_count_and_tolerances():
    sample = {
        "n_pt": 1,
        "n_exported": 1,
        "max_bbox_drift_px": 2.5,
        "max_kp_drift_px": 4.0,
        "max_conf_drift": 0.02,
    }

    assert passes_at(sample, bbox_atol=3.0, kp_atol=4.0, conf_atol=0.05) is True
    assert passes_at(sample, bbox_atol=2.0, kp_atol=4.0, conf_atol=0.05) is False
    sample["n_exported"] = 2
    assert passes_at(sample, bbox_atol=10.0, kp_atol=10.0, conf_atol=1.0) is False


def test_build_audit_summarizes_reports(tmp_path):
    drift = {
        "ok": False,
        "samples_checked": 2,
        "samples_matched": 1,
        "max_bbox_drift_px": 3.0,
        "max_kp_drift_px": 4.0,
        "max_conf_drift": 0.1,
        "thresholds": {"bbox_atol": 2.0},
        "samples": [
            {
                "matched": True,
                "n_pt": 1,
                "n_exported": 1,
                "max_bbox_drift_px": 0.1,
                "max_kp_drift_px": 0.1,
                "max_conf_drift": 0.01,
                "failures": [],
            },
            {
                "matched": False,
                "n_pt": 1,
                "n_exported": 1,
                "max_bbox_drift_px": 3.0,
                "max_kp_drift_px": 4.0,
                "max_conf_drift": 0.1,
                "pair_diagnostics": [{"coordinate_scale_warning": True}],
                "failures": ["bbox drift exceeds 2.0px", "conf drift exceeds 0.05"],
            },
        ],
    }
    onnx = tmp_path / "onnx.json"
    tflite = tmp_path / "tflite.json"
    onnx.write_text(json.dumps(drift), encoding="utf-8")
    tflite.write_text(json.dumps(drift), encoding="utf-8")

    audit = build_audit(onnx, tflite)

    assert audit["ok"] is True
    assert audit["certified"] is False
    assert audit["reports"]["onnx"]["category_counts"]["bbox_drift"] == 1
    assert audit["reports"]["onnx"]["category_counts"]["confidence_drift"] == 1
    assert audit["reports"]["onnx"]["coordinate_scale_warnings"] == 1


def test_render_markdown_includes_tolerance_sweep():
    audit = {
        "ok": True,
        "certified": False,
        "reports": {
            "onnx": {
                "ok": False,
                "samples_matched": 1,
                "samples_checked": 2,
                "max_bbox_drift_px": 3.0,
                "max_kp_drift_px": 4.0,
                "max_conf_drift": 0.1,
                "category_counts": {"bbox_drift": 1},
                "coordinate_scale_warnings": 0,
                "tolerance_sweep": [
                    {
                        "bbox_atol": 2.0,
                        "kp_atol": 3.0,
                        "conf_atol": 0.05,
                        "samples_passed": 1,
                        "samples_checked": 2,
                    }
                ],
            }
        },
        "findings": ["ONNX and TFLite have identical failure category counts."],
        "recommendation": "Do not certify.",
    }

    markdown = render_markdown(audit)

    assert "Export Parity Audit" in markdown
    assert "Tolerance Sweep" in markdown
    assert "bbox_drift=1" in markdown
    assert "Do not certify." in markdown


def test_build_findings_compares_export_failures():
    reports = {
        "onnx": {
            "ok": False,
            "category_counts": {"count_mismatch": 0, "bbox_drift": 1},
            "coordinate_scale_warnings": 0,
            "tolerance_sweep": [
                {
                    "bbox_atol": 10.0,
                    "kp_atol": 15.0,
                    "conf_atol": 0.25,
                    "all_passed": True,
                }
            ],
        },
        "tflite": {
            "ok": False,
            "category_counts": {"count_mismatch": 0, "bbox_drift": 1},
            "coordinate_scale_warnings": 2,
            "tolerance_sweep": [],
        },
    }

    findings = build_findings(reports)

    assert any("identical failure category" in finding for finding in findings)
    assert any("drift-only" in finding for finding in findings)
    assert any("coordinate" in finding for finding in findings)
