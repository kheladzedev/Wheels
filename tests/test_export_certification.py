from __future__ import annotations

import argparse

from src.export_certification import (
    certify_backend,
    count_scale_warnings,
    eval_summary,
    has_count_mismatch,
)


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        min_drift_samples=2,
        max_bbox_drift_px=10.0,
        max_kp_drift_px=15.0,
        max_conf_drift=0.25,
        min_export_map50=0.65,
        min_export_oks=0.80,
        max_map50_delta=0.02,
        max_map5095_delta=0.02,
        max_oks_delta=0.02,
        max_fn_delta=0.02,
        max_fp_delta=0.02,
    )


def _eval(map50: float = 0.69, oks: float = 0.88, fn: float = 0.28, fp: float = 0.26):
    return {
        "metrics_bbox": {"mAP50": map50, "mAP50_95": 0.61},
        "oks": {"mean": oks},
        "rates": {"false_negative_rate": fn, "false_positive_rate": fp},
    }


def _drift(**overrides):
    payload = {
        "samples_checked": 2,
        "samples_matched": 1,
        "ok": False,
        "max_bbox_drift_px": 8.0,
        "max_kp_drift_px": 13.0,
        "max_conf_drift": 0.22,
        "samples": [
            {
                "n_pt": 1,
                "n_exported": 1,
                "pair_diagnostics": [{"coordinate_scale_warning": False}],
                "failures": ["conf drift exceeds 0.05"],
            }
        ],
    }
    payload.update(overrides)
    return payload


def test_certify_backend_accepts_calibrated_drift_and_metric_parity():
    report = certify_backend(
        name="onnx",
        eval_report=_eval(map50=0.692, oks=0.888, fn=0.286, fp=0.268),
        drift_report=_drift(),
        pt_metrics=eval_summary(_eval(map50=0.697, oks=0.887, fn=0.286, fp=0.259)),
        args=_args(),
    )

    assert report["certified"] is True
    assert report["failures"] == []
    assert report["drift"]["strict_ok"] is False


def test_certify_backend_rejects_count_mismatch():
    drift = _drift(samples=[{"n_pt": 1, "n_exported": 2, "failures": []}])

    report = certify_backend(
        name="onnx",
        eval_report=_eval(),
        drift_report=drift,
        pt_metrics=eval_summary(_eval()),
        args=_args(),
    )

    assert report["certified"] is False
    assert "count_mismatch" in report["failures"]


def test_count_scale_warnings_and_mismatch_helpers():
    drift = _drift(
        samples=[
            {
                "n_pt": 1,
                "n_exported": 1,
                "pair_diagnostics": [
                    {"coordinate_scale_warning": True},
                    {"coordinate_scale_warning": False},
                ],
                "failures": [],
            }
        ]
    )

    assert count_scale_warnings(drift) == 1
    assert has_count_mismatch(drift) is False
