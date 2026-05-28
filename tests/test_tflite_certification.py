from __future__ import annotations

import argparse
import json

from src.tflite_certification import build_certification


def test_tflite_certification_passes_with_export_cert_and_litert_smoke(tmp_path):
    artifact = tmp_path / "best_float32.tflite"
    artifact.write_bytes(b"tflite")
    export_cert = tmp_path / "export_cert.json"
    export_cert.write_text(
        json.dumps(
            {
                "certified": True,
                "backends": {"tflite": {"certified": True, "failures": []}},
            }
        ),
        encoding="utf-8",
    )
    eval_report = tmp_path / "eval.json"
    eval_report.write_text(
        json.dumps(
            {
                "metrics_bbox": {"mAP50": 0.69, "mAP50_95": 0.61},
                "oks": {"mean": 0.88},
                "rates": {"false_negative_rate": 0.28, "false_positive_rate": 0.27},
                "counts": {
                    "images": 58,
                    "gt_wheels": 84,
                    "pred_wheels_above_conf": 82,
                    "matched": 60,
                },
            }
        ),
        encoding="utf-8",
    )
    drift = tmp_path / "drift.json"
    drift.write_text('{"ok": false, "samples_checked": 20, "samples_matched": 14}', encoding="utf-8")
    smoke = tmp_path / "smoke.json"
    smoke.write_text(
        json.dumps(
            {
                "ok": True,
                "runtime": "ai_edge_litert",
                "outputs": [{"shape": [1, 14, 8400]}],
                "latency_ms": {"mean": 10.0, "p95": 12.0},
            }
        ),
        encoding="utf-8",
    )

    report = build_certification(
        argparse.Namespace(
            artifact=artifact,
            export_certification=export_cert,
            tflite_eval=eval_report,
            tflite_drift=drift,
            litert_smoke=smoke,
            min_map50=0.65,
            min_oks=0.8,
        )
    )

    assert report["certified"] is True
    assert report["scope"] == "desktop_tflite_litert_package_not_android_device"
    assert report["multi_frame_drift"]["strict_ok"] is False


def test_tflite_certification_fails_without_litert_smoke(tmp_path):
    artifact = tmp_path / "best_float32.tflite"
    artifact.write_bytes(b"tflite")
    export_cert = tmp_path / "export_cert.json"
    export_cert.write_text(
        '{"backends": {"tflite": {"certified": true}}}',
        encoding="utf-8",
    )
    eval_report = tmp_path / "eval.json"
    eval_report.write_text('{"metrics_bbox":{"mAP50":0.69},"oks":{"mean":0.88}}', encoding="utf-8")
    drift = tmp_path / "drift.json"
    drift.write_text("{}", encoding="utf-8")
    smoke = tmp_path / "smoke.json"
    smoke.write_text('{"ok": false}', encoding="utf-8")

    report = build_certification(
        argparse.Namespace(
            artifact=artifact,
            export_certification=export_cert,
            tflite_eval=eval_report,
            tflite_drift=drift,
            litert_smoke=smoke,
            min_map50=0.65,
            min_oks=0.8,
        )
    )

    assert report["certified"] is False
    assert "litert_smoke_failed" in report["failures"]
