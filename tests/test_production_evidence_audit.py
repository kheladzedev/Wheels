from __future__ import annotations

import argparse
import hashlib
import json
import math

from src.production_evidence_audit import (
    ar_holdout_source_manifest_sha256,
    android_report_completeness_failures,
    build_audit,
    build_required_evidence,
    check_android_litert,
    check_ar_holdout,
    check_ar_replay,
    check_external_evidence_custody,
    render_markdown,
)
from src.validate_android_litert_report import build_report as build_android_litert_report


def _args(tmp_path):
    expected_android_artifact = tmp_path / "expected_android.tflite"
    expected_android_artifact.write_bytes(b"expected android artifact")
    return argparse.Namespace(
        android_litert_source=tmp_path / "android.json",
        android_litert_eval=tmp_path / "android_eval.json",
        ar_holdout_source=tmp_path / "ar_holdout",
        ar_holdout_eval=tmp_path / "ar_eval.json",
        ar_holdout_pipeline=tmp_path / "ar_pipeline.json",
        ar_replay_jsonl=tmp_path / "ar_replay.jsonl",
        ar_replay_eval=tmp_path / "replay_eval.json",
        external_evidence_import_report=tmp_path / "external_evidence_drop_import.json",
        expected_android_artifact=expected_android_artifact,
        min_ar_holdout_map50=0.85,
        min_ar_holdout_oks=0.8,
        max_ar_holdout_fn=0.1,
        min_ar_holdout_images=50,
        min_ar_holdout_gt_wheels=80,
    )


def _android_validator_args(tmp_path):
    return argparse.Namespace(
        expected_artifact=tmp_path / "expected_android.tflite",
        min_runs=20,
        max_mean_latency_ms=120.0,
        max_p95_latency_ms=180.0,
        max_peak_memory_mb=512.0,
    )


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_sha256(entries):
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_import_report(
    args,
    *,
    ok=True,
    dry_run=False,
    omit=None,
    bad_sha=None,
    dest_transform=None,
    dest_root=None,
):
    omit = set(omit or [])
    bad_sha = set(bad_sha or [])
    dest_root_value = dest_root or args.external_evidence_import_report.parent
    files = [
        args.android_litert_source,
        args.ar_holdout_source / "metadata" / "provenance.json",
        args.ar_replay_jsonl,
        *sorted((args.ar_holdout_source / "images").glob("*")),
        *sorted((args.ar_holdout_source / "annotations").glob("*")),
    ]
    planned = []
    copied = []
    for path in files:
        if not path.is_file() or path.name in omit:
            continue
        dest = str(dest_transform(path) if dest_transform else path)
        planned.append(
            {
                "source": path.name,
                "dest": dest,
                "size_bytes": path.stat().st_size,
                "sha256": _sha(path),
            }
        )
        copied.append(
            {
                "dest": dest,
                "size_bytes": path.stat().st_size,
                "sha256": "bad" if path.name in bad_sha else _sha(path),
            }
        )
    args.external_evidence_import_report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ok": ok,
                "dry_run": dry_run,
                "source": str(args.external_evidence_import_report.parent / "drop"),
                "source_kind": "directory",
                "source_sha256": None,
                "dest_root": str(dest_root_value),
                "expected_android_artifact": str(args.expected_android_artifact),
                "expected_android_artifact_sha256": _sha(args.expected_android_artifact),
                "failures": [] if ok else ["bad_drop"],
                "file_count": len(copied),
                "evidence_manifest_sha256": _manifest_sha256(planned),
                "planned": planned,
                "copied_artifacts": copied,
            }
        ),
        encoding="utf-8",
    )


def _write_canonical_input_files(args):
    args.android_litert_source.write_text("{}", encoding="utf-8")
    images = args.ar_holdout_source / "images"
    annotations = args.ar_holdout_source / "annotations"
    metadata = args.ar_holdout_source / "metadata"
    images.mkdir(parents=True)
    annotations.mkdir()
    metadata.mkdir()
    (images / "frame.jpg").write_bytes(b"jpg")
    (annotations / "frame.json").write_text("{}", encoding="utf-8")
    (metadata / "provenance.json").write_text("{}", encoding="utf-8")
    args.ar_replay_jsonl.write_text("{}\n", encoding="utf-8")


def _write_ready_android(args):
    expected_artifact_sha = _sha(args.expected_android_artifact)
    payload = {
        "schema_version": 1,
        "source_type": "android_litert_device_validation",
        "test_session_id": "android-litert-test-001",
        "test_app_version": "1.2.3",
        "test_date_utc": "2026-05-27",
        "device": {
            "model": "Pixel test",
            "manufacturer": "Google",
            "android_version": "15",
            "soc": "Tensor test",
            "is_emulator": False,
        },
        "runtime": "LiteRT",
        "artifact": {"sha256": expected_artifact_sha, "format": "tflite_float32"},
        "input": {"shape": [1, 640, 640, 3], "dtype": "float32", "profile": "zero_float32_smoke"},
        "latency_ms": {"runs": 30, "mean": 40.0, "p95": 70.0},
        "memory_mb": {"peak": 200.0},
        "output": {
            "shape": [1, 14, 8400],
            "finite": True,
            "min": 0.0,
            "max": 1.0,
            "mean": 0.5,
        },
    }
    args.android_litert_source.write_text(json.dumps(payload), encoding="utf-8")
    report = build_android_litert_report(
        args.android_litert_source,
        payload,
        argparse.Namespace(
            expected_artifact=args.expected_android_artifact,
            min_runs=20,
            max_mean_latency_ms=120.0,
            max_p95_latency_ms=180.0,
            max_peak_memory_mb=512.0,
        ),
    )
    args.android_litert_eval.write_text(json.dumps(report), encoding="utf-8")


def _write_ready_holdout(args):
    images = args.ar_holdout_source / "images"
    annotations = args.ar_holdout_source / "annotations"
    metadata = args.ar_holdout_source / "metadata"
    images.mkdir(parents=True)
    annotations.mkdir()
    metadata.mkdir()
    wheels_written = 0
    for index in range(50):
        stem = "frame" if index == 0 else f"frame_{index:03d}"
        wheel_count = 2 if index < 30 else 1
        wheels_written += wheel_count
        (images / f"{stem}.jpg").write_bytes(b"jpg")
        wheels = []
        for wheel_index in range(wheel_count):
            x1 = 10.0 + wheel_index * 30.0
            y1 = 20.0
            x2 = x1 + 20.0
            y2 = y1 + 20.0
            wheels.append(
                {
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "points": {
                        "a": [x1 + 2.0, y2 - 2.0],
                        "b": [x2 - 2.0, y2 - 2.0],
                        "c_disc_bottom": [(x1 + x2) / 2.0, y2 - 4.0],
                    },
                }
            )
        (annotations / f"{stem}.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "frame_id": stem,
                    "image": f"{stem}.jpg",
                    "wheels": wheels,
                }
            ),
            encoding="utf-8",
        )
    assert wheels_written == 80
    (metadata / "provenance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_type": "android_ar_device_human_labelled",
                "label_type": "human_reviewed",
                "capture_device": "Pixel test",
                "review_status": "accepted",
                "capture_app_version": "1.2.3",
                "capture_date_utc": "2026-05-27",
                "annotator": "labeler_a",
                "reviewer": "reviewer_b",
            }
        ),
        encoding="utf-8",
    )
    args.ar_holdout_eval.write_text(
        json.dumps(
            {
                "counts": {"images": 50, "gt_wheels": 80},
                "metrics_bbox": {"mAP50": 0.9},
                "oks": {"mean": 0.85},
                "rates": {"false_negative_rate": 0.05},
            }
        ),
        encoding="utf-8",
    )
    args.ar_holdout_pipeline.write_text(
        json.dumps(
            {
                "ok": True,
                "stage": "done",
                "source_root": str(args.ar_holdout_source),
                "source_manifest_sha256": ar_holdout_source_manifest_sha256(args.ar_holdout_source),
                "eval_returncode": 0,
                "eval_report": str(args.ar_holdout_eval),
                "eval_report_sha256": _sha(args.ar_holdout_eval),
                "evaluation": {
                    "ok": True,
                    "failures": [],
                    "thresholds": {
                        "min_map50": 0.85,
                        "min_oks": 0.8,
                        "max_fn": 0.1,
                        "min_images": 50,
                        "min_gt_wheels": 80,
                    },
                    "metrics": {
                        "bbox_mAP50": 0.9,
                        "oks_mean": 0.85,
                        "false_negative_rate": 0.05,
                        "images": 50,
                        "gt_wheels": 80,
                    },
                },
                "conversion": {"ok": True},
            }
        ),
        encoding="utf-8",
    )


def _write_ready_replay(args):
    args.ar_replay_jsonl.write_text(
        "\n".join(json.dumps(_replay_observation(index)) for index in range(30)) + "\n",
        encoding="utf-8",
    )
    args.ar_replay_eval.write_text(
        json.dumps(
            {
                "ok": True,
                "source": str(args.ar_replay_jsonl),
                "source_sha256": _sha(args.ar_replay_jsonl),
                "failures": [],
                "thresholds": {
                    "require_production_source": True,
                    "min_observations": 30,
                    "min_sessions": 1,
                    "min_floor_hit_rate": 0.9,
                    "require_ransac": True,
                    "min_inlier_rate": 0.7,
                    "max_median_residual": 0.02,
                    "max_p95_residual": 0.05,
                    "min_final_positions": 1,
                },
                "counts": {
                    "observations_total": 30,
                    "observations_valid": 30,
                    "schema_errors": 0,
                    "sessions": 1,
                    "floor_hits_complete": 30,
                    "production_source_observations": 30,
                    "ransac_labelled": 30,
                    "inliers": 30,
                    "outliers": 0,
                    "residuals": 30,
                    "recovered_planes": 30,
                    "c_plane_hits": 30,
                    "c_height_values": 30,
                    "final_disc_bottom_positions": 1,
                },
                "metrics": {
                    "floor_hit_rate": 1.0,
                    "inlier_rate": 1.0,
                    "median_residual": 0.004,
                    "p95_residual": 0.004,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_ready_core_reports(args):
    _write_ready_android(args)
    _write_ready_holdout(args)
    _write_ready_replay(args)


def _replay_observation(index):
    return {
        "schema_version": 1,
        "source_type": "android_ar_device_replay",
        "capture_device": "Pixel test",
        "capture_app_version": "1.2.3",
        "capture_date_utc": "2026-05-27",
        "session_id": "session-001",
        "frame_id": f"frame_{index:04d}",
        "capture_index": index,
        "camera_transform": None,
        "camera_pose_ref": f"pose_{index:04d}",
        "screen_points": {
            "a": [100.0, 200.0],
            "b": [150.0, 200.0],
            "c_disc_bottom": [125.0, 170.0],
        },
        "floor_raycast_hits": {
            "a": [1.0, 0.0, 2.0],
            "b": [1.5, 0.0, 2.0],
        },
        "inlier": True,
        "residual": 0.004,
        "recovered_plane": {
            "normal": [0.998, 0.0, 0.062],
            "point": [1.25, 0.0, 2.0],
            "support": 18,
        },
        "c_plane_hit": [1.25, 0.4, 2.0],
        "c_height_value": 0.4,
        "final_disc_bottom_position": [1.25, 0.4, 2.0] if index == 0 else None,
    }


def test_evidence_audit_reports_missing_blockers(tmp_path):
    audit = build_audit(_args(tmp_path))

    assert audit["ok"] is True
    assert audit["production_evidence_ready"] is False
    assert audit["blockers"] == [
        "android_litert_device_validation",
        "human_labelled_ar_device_holdout",
        "ar_3d_replay_validation",
    ]
    assert [item["name"] for item in audit["required_evidence"]] == audit["blockers"]
    assert [item["name"] for item in audit["next_actions"]] == audit["blockers"]
    assert str(tmp_path / "android.json") in audit["required_evidence"][0]["required_inputs"]
    assert "android_litert_harness/AndroidLiteRtDeviceValidationTest.kt" in audit["required_evidence"][0]["evidence_producer"]
    assert audit["required_evidence"][0]["thresholds"]["expected_output_shape"] == [1, 14, 8400]
    assert "ar_holdout_harness/ArHoldoutAnnotationWriter.kt" in audit["required_evidence"][1]["evidence_producer"]
    assert str(tmp_path / "ar_replay.jsonl") in audit["required_evidence"][2]["required_inputs"]
    assert "ar_replay_harness/ArReplayLogger.kt" in audit["required_evidence"][2]["evidence_producer"]


def test_evidence_audit_passes_ready_reports(tmp_path):
    args = _args(tmp_path)
    _write_ready_core_reports(args)
    _write_import_report(args)

    audit = build_audit(args)

    assert audit["production_evidence_ready"] is True
    assert audit["blockers"] == []


def test_evidence_audit_requires_import_custody_once_reports_are_ready(tmp_path):
    args = _args(tmp_path)
    _write_ready_core_reports(args)

    audit = build_audit(args)

    assert audit["production_evidence_ready"] is False
    assert "external_evidence_custody" in audit["blockers"]
    custody = next(check for check in audit["checks"] if check["name"] == "external_evidence_custody")
    assert f"missing_import_report:{args.external_evidence_import_report}" in custody["failures"]


def test_evidence_audit_rejects_import_custody_hash_mismatch(tmp_path):
    args = _args(tmp_path)
    _write_ready_core_reports(args)
    _write_import_report(args, bad_sha={"frame.jpg"})

    audit = build_audit(args)

    assert audit["production_evidence_ready"] is False
    custody = next(check for check in audit["checks"] if check["name"] == "external_evidence_custody")
    assert any(failure.startswith("input_sha256_mismatch:") for failure in custody["failures"])


def test_evidence_audit_rechecks_android_runtime_thresholds_even_if_report_says_ok(tmp_path):
    args = _args(tmp_path)
    _write_ready_android(args)
    report = json.loads(args.android_litert_eval.read_text(encoding="utf-8"))
    report["metrics"]["mean_latency_ms"] = 500.0
    report["metrics"]["p95_latency_ms"] = 700.0
    report["metrics"]["peak_memory_mb"] = 900.0
    args.android_litert_eval.write_text(json.dumps(report), encoding="utf-8")

    check = check_android_litert(args)

    assert check["ready"] is False
    assert "mean_latency:500.000>120.000" in check["failures"]
    assert "p95_latency:700.000>180.000" in check["failures"]
    assert "peak_memory:900.000>512.000" in check["failures"]


def test_evidence_audit_allows_android_p95_latency_below_mean(tmp_path):
    args = _args(tmp_path)
    _write_ready_android(args)
    source = json.loads(args.android_litert_source.read_text(encoding="utf-8"))
    source["latency_ms"]["mean"] = 70.0
    source["latency_ms"]["p95"] = 40.0
    args.android_litert_source.write_text(json.dumps(source), encoding="utf-8")
    report = build_android_litert_report(
        args.android_litert_source,
        source,
        argparse.Namespace(
            expected_artifact=args.expected_android_artifact,
            min_runs=20,
            max_mean_latency_ms=120.0,
            max_p95_latency_ms=180.0,
            max_peak_memory_mb=512.0,
        ),
    )
    args.android_litert_eval.write_text(json.dumps(report), encoding="utf-8")

    check = check_android_litert(args)

    assert check["ready"] is True
    assert "p95_latency_less_than_mean:40.000<70.000" not in check["failures"]


def test_evidence_audit_rejects_android_eval_non_finite_latency_and_memory(tmp_path):
    args = _args(tmp_path)
    _write_ready_android(args)
    report = json.loads(args.android_litert_eval.read_text(encoding="utf-8"))
    report["metrics"]["runs"] = math.nan
    report["metrics"]["mean_latency_ms"] = math.nan
    report["metrics"]["p95_latency_ms"] = math.inf
    report["metrics"]["peak_memory_mb"] = math.nan
    args.android_litert_eval.write_text(json.dumps(report), encoding="utf-8")

    check = check_android_litert(args)

    assert check["ready"] is False
    assert "missing_latency_runs" in check["failures"]
    assert "missing_mean_latency" in check["failures"]
    assert "missing_p95_latency" in check["failures"]
    assert "missing_peak_memory" in check["failures"]


def test_evidence_audit_rejects_android_eval_fractional_latency_runs(tmp_path):
    args = _args(tmp_path)
    _write_ready_android(args)
    report = json.loads(args.android_litert_eval.read_text(encoding="utf-8"))
    report["metrics"]["runs"] = 30.5
    args.android_litert_eval.write_text(json.dumps(report), encoding="utf-8")

    check = check_android_litert(args)

    assert check["ready"] is False
    assert "invalid_latency_runs" in check["failures"]


def test_evidence_audit_revalidates_android_source_even_if_eval_says_ok(tmp_path):
    args = _args(tmp_path)
    args.android_litert_source.write_text("{}", encoding="utf-8")
    expected_artifact_sha = _sha(args.expected_android_artifact)
    args.android_litert_eval.write_text(
        json.dumps(
            {
                "ok": True,
                "schema_version": 1,
                "source_schema_version": 1,
                "source_type": "android_litert_device_validation",
                "test_session_id": "android-litert-test-001",
                "test_app_version": "1.2.3",
                "test_date_utc": "2026-05-27",
                "source": str(args.android_litert_source),
                "source_sha256": _sha(args.android_litert_source),
                "failures": [],
                "thresholds": {
                    "expected_artifact": str(args.expected_android_artifact),
                    "expected_artifact_sha256": expected_artifact_sha,
                },
                "device": {
                    "model": "Pixel test",
                    "manufacturer": "Google",
                    "android_version": "15",
                    "soc": "Tensor test",
                    "is_emulator": False,
                },
                "runtime": "LiteRT",
                "artifact": {"sha256": expected_artifact_sha, "format": "tflite_float32"},
                "input": {"shape": [1, 640, 640, 3], "dtype": "float32", "profile": "zero_float32_smoke"},
                "metrics": {
                    "runs": 30,
                    "mean_latency_ms": 40.0,
                    "p95_latency_ms": 70.0,
                    "peak_memory_mb": 200.0,
                },
                "output": {
                    "shape": [1, 14, 8400],
                    "finite": True,
                    "min": 0.0,
                    "max": 1.0,
                    "mean": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )

    check = check_android_litert(args)

    assert check["ready"] is False
    all_failures = " ".join(check["failures"])
    assert "source_revalidation_failed" in all_failures
    assert "android_report_section_mismatch:device" in check["failures"]
    assert any(failure.startswith("android_report_metric_mismatch:runs:") for failure in check["failures"])


def test_evidence_audit_rejects_android_eval_threshold_and_failure_tampering(tmp_path):
    args = _args(tmp_path)
    _write_ready_android(args)
    report = json.loads(args.android_litert_eval.read_text(encoding="utf-8"))
    report["failures"] = ["manually-hidden"]
    report["thresholds"]["min_runs"] = 1
    report["thresholds"].pop("max_peak_memory_mb")
    args.android_litert_eval.write_text(json.dumps(report), encoding="utf-8")

    check = check_android_litert(args)

    assert check["ready"] is False
    assert "android_report_failures_mismatch" in check["failures"]
    assert "android_report_threshold_keys_mismatch" in check["failures"]
    assert "android_report_threshold_mismatch:min_runs:1!=20" in check["failures"]


def test_evidence_audit_rejects_android_threshold_shape_bool_tampering(tmp_path):
    args = _args(tmp_path)
    _write_ready_android(args)
    report = json.loads(args.android_litert_eval.read_text(encoding="utf-8"))
    report["thresholds"]["expected_input_shape"] = [True, 640, 640, 3]
    report["thresholds"]["expected_output_shape"] = [1.0, 14, 8400]
    args.android_litert_eval.write_text(json.dumps(report), encoding="utf-8")

    check = check_android_litert(args)

    assert check["ready"] is False
    assert (
        "android_report_threshold_mismatch:"
        "expected_input_shape:[True, 640, 640, 3]!=[1, 640, 640, 3]"
    ) in check["failures"]
    assert (
        "android_report_threshold_mismatch:"
        "expected_output_shape:[1.0, 14, 8400]!=[1, 14, 8400]"
    ) in check["failures"]


def test_evidence_audit_requires_android_source_schema_version(tmp_path):
    args = _args(tmp_path)
    _write_ready_android(args)
    report = json.loads(args.android_litert_eval.read_text(encoding="utf-8"))
    report.pop("source_schema_version")
    args.android_litert_eval.write_text(json.dumps(report), encoding="utf-8")

    check = check_android_litert(args)

    assert check["ready"] is False
    assert "unsupported_source_schema_version:missing" in check["failures"]


def test_evidence_audit_rejects_boolean_android_source_schema_version(tmp_path):
    args = _args(tmp_path)
    _write_ready_android(args)
    report = json.loads(args.android_litert_eval.read_text(encoding="utf-8"))
    report["source_schema_version"] = True
    args.android_litert_eval.write_text(json.dumps(report), encoding="utf-8")

    check = check_android_litert(args)

    assert check["ready"] is False
    assert "unsupported_source_schema_version:True" in check["failures"]


def test_evidence_audit_rejects_boolean_android_eval_schema_and_ok(tmp_path):
    args = _args(tmp_path)
    _write_ready_android(args)
    report = json.loads(args.android_litert_eval.read_text(encoding="utf-8"))
    report["schema_version"] = True
    report["ok"] = 1
    args.android_litert_eval.write_text(json.dumps(report), encoding="utf-8")

    check = check_android_litert(args)

    assert check["ready"] is False
    assert "android_report_field_mismatch:schema_version:True!=1" in check["failures"]
    assert "android_report_field_mismatch:ok:1!=True" in check["failures"]


def test_evidence_audit_rejects_android_eval_with_future_test_date(tmp_path):
    args = _args(tmp_path)
    _write_ready_android(args)
    report = json.loads(args.android_litert_eval.read_text(encoding="utf-8"))
    report["test_date_utc"] = "2999-01-01"
    args.android_litert_eval.write_text(json.dumps(report), encoding="utf-8")

    check = check_android_litert(args)

    assert check["ready"] is False
    assert "invalid_test_date_utc" in check["failures"]


def test_evidence_audit_rejects_holdout_annotations_without_schema_version(tmp_path):
    args = _args(tmp_path)
    _write_ready_holdout(args)
    annotation = args.ar_holdout_source / "annotations" / "frame.json"
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    payload.pop("schema_version")
    annotation.write_text(json.dumps(payload), encoding="utf-8")

    check = check_ar_holdout(args)

    assert check["ready"] is False
    assert any("invalid_annotations:" in failure for failure in check["failures"])
    assert any("schema_version must be 1" in failure for failure in check["failures"])


def test_evidence_audit_rejects_holdout_eval_with_spoofed_gt_wheel_count(tmp_path):
    args = _args(tmp_path)
    _write_ready_holdout(args)
    for annotation in (args.ar_holdout_source / "annotations").glob("*.json"):
        payload = json.loads(annotation.read_text(encoding="utf-8"))
        payload["wheels"] = []
        annotation.write_text(json.dumps(payload), encoding="utf-8")
    args.ar_holdout_pipeline.write_text(
        json.dumps(
            {
                "ok": True,
                "stage": "done",
                "source_root": str(args.ar_holdout_source),
                "source_manifest_sha256": ar_holdout_source_manifest_sha256(args.ar_holdout_source),
                "eval_returncode": 0,
                "eval_report": str(args.ar_holdout_eval),
                "eval_report_sha256": _sha(args.ar_holdout_eval),
                "evaluation": {"ok": True, "failures": []},
                "conversion": {"ok": True},
            }
        ),
        encoding="utf-8",
    )

    check = check_ar_holdout(args)

    assert check["ready"] is False
    assert check["source_gt_wheels"] == 0
    assert "source_gt_wheels:0<80" in check["failures"]
    assert "eval_gt_wheels_source_mismatch:80!=0" in check["failures"]


def test_evidence_audit_rejects_holdout_non_integer_eval_counts(tmp_path):
    args = _args(tmp_path)
    _write_ready_holdout(args)
    report = json.loads(args.ar_holdout_eval.read_text(encoding="utf-8"))
    report["counts"]["images"] = 50.5
    report["counts"]["gt_wheels"] = True
    args.ar_holdout_eval.write_text(json.dumps(report), encoding="utf-8")
    pipeline = json.loads(args.ar_holdout_pipeline.read_text(encoding="utf-8"))
    pipeline["eval_report_sha256"] = _sha(args.ar_holdout_eval)
    args.ar_holdout_pipeline.write_text(json.dumps(pipeline), encoding="utf-8")

    check = check_ar_holdout(args)

    assert check["ready"] is False
    assert "ar_holdout_count_not_integer:images:50.5" in check["failures"]
    assert "ar_holdout_count_not_integer:gt_wheels:True" in check["failures"]


def test_evidence_audit_rejects_holdout_non_integer_pipeline_counts_and_thresholds(tmp_path):
    args = _args(tmp_path)
    _write_ready_holdout(args)
    pipeline = json.loads(args.ar_holdout_pipeline.read_text(encoding="utf-8"))
    pipeline["eval_returncode"] = 0.0
    pipeline["evaluation"]["thresholds"]["min_images"] = 50.0
    pipeline["evaluation"]["metrics"]["gt_wheels"] = 80.0
    args.ar_holdout_pipeline.write_text(json.dumps(pipeline), encoding="utf-8")

    check = check_ar_holdout(args)

    assert check["ready"] is False
    assert "pipeline_eval_returncode_not_zero:0.0" in check["failures"]
    assert "pipeline_evaluation_threshold_not_integer:min_images:50.0" in check["failures"]
    assert "pipeline_evaluation_metric_count_not_integer:gt_wheels:80.0" in check["failures"]


def test_evidence_audit_counts_only_valid_holdout_wheels(tmp_path):
    args = _args(tmp_path)
    _write_ready_holdout(args)
    for annotation in (args.ar_holdout_source / "annotations").glob("*.json"):
        payload = json.loads(annotation.read_text(encoding="utf-8"))
        payload["wheels"] = [
            {
                "bbox_xyxy": [40, 10, 20, 30],
                "points": {"a": [42, 28], "b": [58, 28], "c_disc_bottom": ["bad", 24]},
            }
            for _ in payload["wheels"]
        ]
        annotation.write_text(json.dumps(payload), encoding="utf-8")
    args.ar_holdout_pipeline.write_text(
        json.dumps(
            {
                "ok": True,
                "stage": "done",
                "source_root": str(args.ar_holdout_source),
                "source_manifest_sha256": ar_holdout_source_manifest_sha256(args.ar_holdout_source),
                "eval_returncode": 0,
                "eval_report": str(args.ar_holdout_eval),
                "eval_report_sha256": _sha(args.ar_holdout_eval),
                "evaluation": {"ok": True, "failures": []},
                "conversion": {"ok": True},
            }
        ),
        encoding="utf-8",
    )

    check = check_ar_holdout(args)

    assert check["ready"] is False
    assert check["source_gt_wheels"] == 0
    assert "source_gt_wheels:0<80" in check["failures"]
    assert "eval_gt_wheels_source_mismatch:80!=0" in check["failures"]
    all_failures = " ".join(check["failures"])
    assert "wheel_invalid_bbox_order" in all_failures
    assert "wheel_invalid_point_c_disc_bottom" in all_failures


def test_evidence_audit_rejects_holdout_points_outside_bbox(tmp_path):
    args = _args(tmp_path)
    _write_ready_holdout(args)
    for annotation in (args.ar_holdout_source / "annotations").glob("*.json"):
        payload = json.loads(annotation.read_text(encoding="utf-8"))
        payload["wheels"] = [
            {
                "bbox_xyxy": [10, 10, 30, 30],
                "points": {"a": [9, 20], "b": [28, 28], "c_disc_bottom": [20, 31]},
            }
            for _ in payload["wheels"]
        ]
        annotation.write_text(json.dumps(payload), encoding="utf-8")
    args.ar_holdout_pipeline.write_text(
        json.dumps(
            {
                "ok": True,
                "stage": "done",
                "source_root": str(args.ar_holdout_source),
                "source_manifest_sha256": ar_holdout_source_manifest_sha256(args.ar_holdout_source),
                "eval_returncode": 0,
                "eval_report": str(args.ar_holdout_eval),
                "eval_report_sha256": _sha(args.ar_holdout_eval),
                "evaluation": {"ok": True, "failures": []},
                "conversion": {"ok": True},
            }
        ),
        encoding="utf-8",
    )

    check = check_ar_holdout(args)

    assert check["ready"] is False
    assert check["source_gt_wheels"] == 0
    assert "source_gt_wheels:0<80" in check["failures"]
    all_failures = " ".join(check["failures"])
    assert "wheel_point_a_outside_bbox" in all_failures
    assert "wheel_point_c_disc_bottom_outside_bbox" in all_failures


def test_evidence_audit_rejects_holdout_image_annotation_stem_mismatch(tmp_path):
    args = _args(tmp_path)
    _write_ready_holdout(args)
    (args.ar_holdout_source / "annotations" / "frame.json").rename(
        args.ar_holdout_source / "annotations" / "wrong_frame.json"
    )
    args.ar_holdout_pipeline.write_text(
        json.dumps(
            {
                "ok": True,
                "stage": "done",
                "source_root": str(args.ar_holdout_source),
                "source_manifest_sha256": ar_holdout_source_manifest_sha256(args.ar_holdout_source),
                "eval_returncode": 0,
                "eval_report": str(args.ar_holdout_eval),
                "eval_report_sha256": _sha(args.ar_holdout_eval),
                "evaluation": {"ok": True, "failures": []},
                "conversion": {"ok": True},
            }
        ),
        encoding="utf-8",
    )

    check = check_ar_holdout(args)

    assert check["ready"] is False
    assert any("missing_annotations_for_images" in failure for failure in check["failures"])
    assert any("missing_images_for_annotations" in failure for failure in check["failures"])


def test_evidence_audit_rejects_holdout_annotation_pointing_at_other_existing_image(tmp_path):
    args = _args(tmp_path)
    _write_ready_holdout(args)
    annotation = args.ar_holdout_source / "annotations" / "frame.json"
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    payload["frame_id"] = "frame_001"
    payload["image"] = "frame_001.jpg"
    annotation.write_text(json.dumps(payload), encoding="utf-8")
    args.ar_holdout_pipeline.write_text(
        json.dumps(
            {
                "ok": True,
                "stage": "done",
                "source_root": str(args.ar_holdout_source),
                "source_manifest_sha256": ar_holdout_source_manifest_sha256(args.ar_holdout_source),
                "eval_returncode": 0,
                "eval_report": str(args.ar_holdout_eval),
                "eval_report_sha256": _sha(args.ar_holdout_eval),
                "evaluation": {"ok": True, "failures": []},
                "conversion": {"ok": True},
            }
        ),
        encoding="utf-8",
    )

    check = check_ar_holdout(args)

    assert check["ready"] is False
    all_failures = " ".join(check["failures"])
    assert "annotation_frame_id_mismatch:frame.json:frame_001" in all_failures
    assert "annotation_image_field_mismatch" in all_failures


def test_evidence_audit_rejects_android_eval_for_stale_expected_artifact_sha(tmp_path):
    args = _args(tmp_path)
    _write_ready_android(args)
    report = json.loads(args.android_litert_eval.read_text(encoding="utf-8"))
    report["thresholds"]["expected_artifact_sha256"] = "old-artifact-sha"
    report["artifact"]["sha256"] = "old-artifact-sha"
    args.android_litert_eval.write_text(json.dumps(report), encoding="utf-8")

    check = check_android_litert(args)

    assert check["ready"] is False
    assert "expected_artifact_sha256_mismatch" in check["failures"]


def test_evidence_audit_rejects_android_eval_for_expected_artifact_path_mismatch(tmp_path):
    args = _args(tmp_path)
    _write_ready_android(args)
    report = json.loads(args.android_litert_eval.read_text(encoding="utf-8"))
    report["thresholds"]["expected_artifact"] = str(tmp_path / "old_expected.tflite")
    args.android_litert_eval.write_text(json.dumps(report), encoding="utf-8")

    check = check_android_litert(args)

    assert check["ready"] is False
    assert any(
        failure.startswith("expected_artifact_path_mismatch:")
        for failure in check["failures"]
    )


def test_evidence_audit_rejects_android_eval_for_wrong_input_contract(tmp_path):
    args = _args(tmp_path)
    _write_ready_android(args)
    report = json.loads(args.android_litert_eval.read_text(encoding="utf-8"))
    report["input"] = {"shape": [1, 3, 640, 640], "dtype": "uint8", "profile": "unknown"}
    args.android_litert_eval.write_text(json.dumps(report), encoding="utf-8")

    check = check_android_litert(args)

    assert check["ready"] is False
    assert "unexpected_input_shape:[1, 3, 640, 640]" in check["failures"]
    assert "unexpected_input_dtype:uint8" in check["failures"]
    assert "unexpected_input_profile:unknown" in check["failures"]


def test_evidence_audit_rejects_android_eval_non_integer_shapes(tmp_path):
    args = _args(tmp_path)
    _write_ready_android(args)
    report = json.loads(args.android_litert_eval.read_text(encoding="utf-8"))
    report["input"]["shape"] = [True, 640, 640, 3]
    report["output"]["shape"] = [1.0, 14, 8400]
    args.android_litert_eval.write_text(json.dumps(report), encoding="utf-8")

    check = check_android_litert(args)

    assert check["ready"] is False
    assert "unexpected_input_shape:[True, 640, 640, 3]" in check["failures"]
    assert "unexpected_output_shape:[1.0, 14, 8400]" in check["failures"]


def test_evidence_audit_rejects_android_section_bool_numeric_tampering(tmp_path):
    args = _args(tmp_path)
    _write_ready_android(args)
    report = json.loads(args.android_litert_eval.read_text(encoding="utf-8"))
    report["output"]["min"] = False
    args.android_litert_eval.write_text(json.dumps(report), encoding="utf-8")

    check = check_android_litert(args)

    assert check["ready"] is False
    assert "android_report_section_mismatch:output" in check["failures"]


def test_evidence_audit_rejects_android_eval_from_emulator(tmp_path):
    args = _args(tmp_path)
    _write_ready_android(args)
    report = json.loads(args.android_litert_eval.read_text(encoding="utf-8"))
    report["device"]["is_emulator"] = True
    args.android_litert_eval.write_text(json.dumps(report), encoding="utf-8")

    check = check_android_litert(args)

    assert check["ready"] is False
    assert "device_must_be_physical:is_emulator=True" in check["failures"]


def test_evidence_audit_rejects_android_degenerate_output_stats(tmp_path):
    args = _args(tmp_path)
    _write_ready_android(args)
    report = json.loads(args.android_litert_eval.read_text(encoding="utf-8"))
    report["output"] = {"shape": [1, 14, 8400], "finite": True, "min": 0.0, "max": 0.0, "mean": 0.0}
    args.android_litert_eval.write_text(json.dumps(report), encoding="utf-8")

    check = check_android_litert(args)

    assert check["ready"] is False
    assert "degenerate_output_range" in check["failures"]


def test_evidence_audit_rejects_android_output_mean_outside_range(tmp_path):
    args = _args(tmp_path)
    _write_ready_android(args)
    report = json.loads(args.android_litert_eval.read_text(encoding="utf-8"))
    report["output"] = {"shape": [1, 14, 8400], "finite": True, "min": -0.1, "max": 1.0, "mean": 1.5}
    args.android_litert_eval.write_text(json.dumps(report), encoding="utf-8")

    check = check_android_litert(args)

    assert check["ready"] is False
    assert "output_mean_outside_range" in check["failures"]


def test_android_validator_output_satisfies_evidence_audit_completeness(tmp_path):
    expected = tmp_path / "expected_android.tflite"
    expected.write_bytes(b"expected android artifact")
    source = tmp_path / "android.json"
    source.write_text("{}", encoding="utf-8")
    report = build_android_litert_report(
        source,
            {
                "schema_version": 1,
                "source_type": "android_litert_device_validation",
                "test_session_id": "android-litert-001",
                "test_app_version": "1.2.3",
                "test_date_utc": "2026-05-27",
                "device": {
                "model": "Pixel test",
                "manufacturer": "Google",
                "android_version": "15",
                "soc": "Tensor test",
                "is_emulator": False,
            },
            "runtime": "LiteRT",
            "artifact": {"sha256": _sha(expected), "format": "tflite_float32"},
            "input": {"shape": [1, 640, 640, 3], "dtype": "float32", "profile": "zero_float32_smoke"},
            "latency_ms": {"runs": 30, "mean": 40.0, "p95": 70.0},
            "memory_mb": {"peak": 200.0},
            "output": {"shape": [1, 14, 8400], "finite": True, "min": 0.0, "max": 1.0, "mean": 0.5},
        },
        _android_validator_args(tmp_path),
    )

    assert report["ok"] is True
    assert android_report_completeness_failures(report) == []


def test_required_evidence_commands_are_self_contained(tmp_path):
    args = _args(tmp_path)
    required = {item["name"]: item for item in build_required_evidence(args)}

    android_cmd = required["android_litert_device_validation"]["validation_command"]
    android_thresholds = required["android_litert_device_validation"]["thresholds"]
    assert android_cmd[android_cmd.index("--expected-artifact") + 1] == str(args.expected_android_artifact)
    assert android_cmd[android_cmd.index("--min-runs") + 1] == "20"
    assert android_cmd[android_cmd.index("--max-mean-latency-ms") + 1] == "120.0"
    assert android_cmd[android_cmd.index("--max-p95-latency-ms") + 1] == "180.0"
    assert android_cmd[android_cmd.index("--max-peak-memory-mb") + 1] == "512.0"
    assert android_thresholds["expected_input_shape"] == [1, 640, 640, 3]
    assert android_thresholds["expected_input_dtype"] == "float32"
    assert android_thresholds["expected_input_profile"] == "zero_float32_smoke"

    holdout_cmd = required["human_labelled_ar_device_holdout"]["validation_command"]
    assert holdout_cmd[holdout_cmd.index("--eval-out") + 1] == str(args.ar_holdout_eval)
    assert holdout_cmd[holdout_cmd.index("--status-out") + 1] == str(args.ar_holdout_pipeline)
    assert holdout_cmd[holdout_cmd.index("--min-map50") + 1] == "0.85"
    assert holdout_cmd[holdout_cmd.index("--min-oks") + 1] == "0.8"
    assert holdout_cmd[holdout_cmd.index("--max-fn") + 1] == "0.1"
    assert holdout_cmd[holdout_cmd.index("--min-images") + 1] == "50"
    assert holdout_cmd[holdout_cmd.index("--min-gt-wheels") + 1] == "80"

    replay_cmd = required["ar_3d_replay_validation"]["validation_command"]
    assert replay_cmd[replay_cmd.index("--min-observations") + 1] == "30"
    assert replay_cmd[replay_cmd.index("--min-sessions") + 1] == "1"
    assert replay_cmd[replay_cmd.index("--min-floor-hit-rate") + 1] == "0.9"
    assert replay_cmd[replay_cmd.index("--min-inlier-rate") + 1] == "0.7"
    assert replay_cmd[replay_cmd.index("--max-median-residual") + 1] == "0.02"
    assert replay_cmd[replay_cmd.index("--max-p95-residual") + 1] == "0.05"
    assert replay_cmd[replay_cmd.index("--min-final-positions") + 1] == "1"


def test_evidence_audit_rechecks_ar_replay_quality_even_if_report_says_ok(tmp_path):
    args = _args(tmp_path)
    _write_ready_replay(args)
    report = json.loads(args.ar_replay_eval.read_text(encoding="utf-8"))
    report["counts"]["floor_hits_complete"] = 20
    report["counts"]["ransac_labelled"] = 10
    report["counts"]["residuals"] = 10
    report["counts"]["recovered_planes"] = 10
    report["counts"]["c_plane_hits"] = 10
    report["counts"]["c_height_values"] = 10
    report["counts"]["final_disc_bottom_positions"] = 0
    report["metrics"]["floor_hit_rate"] = 20 / 30
    report["metrics"]["inlier_rate"] = 0.6
    report["metrics"]["median_residual"] = 0.04
    report["metrics"]["p95_residual"] = 0.08
    args.ar_replay_eval.write_text(json.dumps(report), encoding="utf-8")

    check = check_ar_replay(args)

    assert check["ready"] is False
    assert "incomplete_floor_hits:20!=30" in check["failures"]
    assert "incomplete_ransac_labels:10!=30" in check["failures"]
    assert "incomplete_residuals:10!=30" in check["failures"]
    assert "incomplete_recovered_planes:10!=30" in check["failures"]
    assert "incomplete_c_plane_hits:10!=30" in check["failures"]
    assert "incomplete_c_height_values:10!=30" in check["failures"]
    assert "final_positions:0<1" in check["failures"]
    assert "floor_hit_rate:0.667<0.900" in check["failures"]
    assert "inlier_rate:0.600<0.700" in check["failures"]
    assert "median_residual:0.040000>0.020000" in check["failures"]
    assert "p95_residual:0.080000>0.050000" in check["failures"]


def test_evidence_audit_rejects_ar_replay_fractional_counts_and_integer_thresholds(tmp_path):
    args = _args(tmp_path)
    _write_ready_replay(args)
    report = json.loads(args.ar_replay_eval.read_text(encoding="utf-8"))
    report["thresholds"]["min_observations"] = 30.5
    report["counts"]["observations_valid"] = 30.5
    args.ar_replay_eval.write_text(json.dumps(report), encoding="utf-8")

    check = check_ar_replay(args)

    assert check["ready"] is False
    assert "replay_threshold_not_integer:min_observations:30.5" in check["failures"]
    assert "replay_count_not_integer:observations_valid:30.5" in check["failures"]
    assert "replay_report_count_not_integer:observations_valid:30.5" in check["failures"]


def test_evidence_audit_rejects_ar_replay_boolean_float_thresholds_and_metrics(tmp_path):
    args = _args(tmp_path)
    _write_ready_replay(args)
    report = json.loads(args.ar_replay_eval.read_text(encoding="utf-8"))
    report["thresholds"]["min_floor_hit_rate"] = True
    report["thresholds"]["max_median_residual"] = False
    report["metrics"]["floor_hit_rate"] = True
    report["metrics"]["median_residual"] = False
    args.ar_replay_eval.write_text(json.dumps(report), encoding="utf-8")

    check = check_ar_replay(args)

    assert check["ready"] is False
    assert "replay_threshold_not_number:min_floor_hit_rate:True" in check["failures"]
    assert "replay_threshold_not_number:max_median_residual:False" in check["failures"]
    assert "replay_metric_not_number:floor_hit_rate:True" in check["failures"]
    assert "replay_metric_not_number:median_residual:False" in check["failures"]
    assert any(
        failure.startswith("replay_report_threshold_mismatch:min_floor_hit_rate:")
        for failure in check["failures"]
    )


def test_evidence_audit_rejects_ar_replay_p95_residual_below_median(tmp_path):
    args = _args(tmp_path)
    _write_ready_replay(args)
    report = json.loads(args.ar_replay_eval.read_text(encoding="utf-8"))
    report["metrics"]["median_residual"] = 0.006
    report["metrics"]["p95_residual"] = 0.004
    args.ar_replay_eval.write_text(json.dumps(report), encoding="utf-8")

    check = check_ar_replay(args)

    assert check["ready"] is False
    assert "p95_residual_less_than_median:0.004000<0.006000" in check["failures"]
    assert any(
        failure.startswith("replay_report_metric_mismatch:median_residual:")
        for failure in check["failures"]
    )


def test_evidence_audit_revalidates_ar_replay_source_even_if_report_says_ok(tmp_path):
    args = _args(tmp_path)
    args.ar_replay_jsonl.write_text("{}\n", encoding="utf-8")
    args.ar_replay_eval.write_text(
        json.dumps(
            {
                "ok": True,
                "source": str(args.ar_replay_jsonl),
                "source_sha256": _sha(args.ar_replay_jsonl),
                "failures": [],
                "thresholds": {
                    "require_production_source": True,
                    "min_observations": 30,
                    "min_sessions": 1,
                    "min_floor_hit_rate": 0.9,
                    "require_ransac": True,
                    "min_inlier_rate": 0.7,
                    "max_median_residual": 0.02,
                    "max_p95_residual": 0.05,
                    "min_final_positions": 1,
                },
                "counts": {
                    "observations_total": 30,
                    "observations_valid": 30,
                    "schema_errors": 0,
                    "sessions": 1,
                    "floor_hits_complete": 30,
                    "production_source_observations": 30,
                    "ransac_labelled": 30,
                    "inliers": 30,
                    "outliers": 0,
                    "residuals": 30,
                    "recovered_planes": 30,
                    "c_plane_hits": 30,
                    "c_height_values": 30,
                    "final_disc_bottom_positions": 1,
                },
                "metrics": {
                    "floor_hit_rate": 1.0,
                    "inlier_rate": 1.0,
                    "median_residual": 0.004,
                    "p95_residual": 0.006,
                },
            }
        ),
        encoding="utf-8",
    )

    check = check_ar_replay(args)

    assert check["ready"] is False
    all_failures = " ".join(check["failures"])
    assert "source_revalidation_failed" in all_failures
    assert "replay_report_count_mismatch:observations_total:30!=1" in check["failures"]
    assert "replay_report_count_mismatch:observations_valid:30!=0" in check["failures"]


def test_evidence_audit_revalidates_ar_replay_recovered_plane_normal(tmp_path):
    args = _args(tmp_path)
    _write_ready_replay(args)
    observations = [
        json.loads(line)
        for line in args.ar_replay_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    observations[0]["recovered_plane"]["normal"] = [2.0, 0.0, 0.0]
    args.ar_replay_jsonl.write_text(
        "\n".join(json.dumps(observation) for observation in observations) + "\n",
        encoding="utf-8",
    )
    report = json.loads(args.ar_replay_eval.read_text(encoding="utf-8"))
    report["source_sha256"] = _sha(args.ar_replay_jsonl)
    args.ar_replay_eval.write_text(json.dumps(report), encoding="utf-8")

    check = check_ar_replay(args)

    assert check["ready"] is False
    all_failures = " ".join(check["failures"])
    assert "source_revalidation_failed" in all_failures
    assert "replay_report_field_mismatch:ok:True!=False" in check["failures"]
    assert "replay_report_count_mismatch:observations_valid:30!=29" in check["failures"]


def test_evidence_audit_revalidates_ar_replay_capture_index_order(tmp_path):
    args = _args(tmp_path)
    _write_ready_replay(args)
    observations = [
        json.loads(line)
        for line in args.ar_replay_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    observations[2]["capture_index"] = 1
    args.ar_replay_jsonl.write_text(
        "\n".join(json.dumps(observation) for observation in observations) + "\n",
        encoding="utf-8",
    )
    report = json.loads(args.ar_replay_eval.read_text(encoding="utf-8"))
    report["source_sha256"] = _sha(args.ar_replay_jsonl)
    args.ar_replay_eval.write_text(json.dumps(report), encoding="utf-8")

    check = check_ar_replay(args)

    assert check["ready"] is False
    all_failures = " ".join(check["failures"])
    assert "source_revalidation_failed" in all_failures
    assert "replay_report_field_mismatch:ok:True!=False" in check["failures"]
    assert "replay_report_failures_mismatch" in check["failures"]
    assert "replay_report_count_mismatch:schema_errors:0!=1" in check["failures"]


def test_evidence_audit_rejects_ar_replay_threshold_and_failure_tampering(tmp_path):
    args = _args(tmp_path)
    _write_ready_replay(args)
    report = json.loads(args.ar_replay_eval.read_text(encoding="utf-8"))
    report["failures"] = ["manually-hidden"]
    report["thresholds"]["min_observations"] = 1
    report["thresholds"].pop("max_p95_residual")
    args.ar_replay_eval.write_text(json.dumps(report), encoding="utf-8")

    check = check_ar_replay(args)

    assert check["ready"] is False
    assert "replay_report_failures_mismatch" in check["failures"]
    assert "replay_report_threshold_keys_mismatch" in check["failures"]
    assert "replay_report_threshold_mismatch:min_observations:1!=30" in check["failures"]


def test_evidence_audit_rejects_ar_replay_bool_int_report_tampering(tmp_path):
    args = _args(tmp_path)
    _write_ready_replay(args)
    report = json.loads(args.ar_replay_eval.read_text(encoding="utf-8"))
    report["ok"] = 1
    report["thresholds"]["require_ransac"] = 1
    args.ar_replay_eval.write_text(json.dumps(report), encoding="utf-8")

    check = check_ar_replay(args)

    assert check["ready"] is False
    assert "replay_report_field_mismatch:ok:1!=True" in check["failures"]
    assert "replay_report_threshold_mismatch:require_ransac:1!=True" in check["failures"]


def test_evidence_audit_accepts_relative_report_source_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = _args(tmp_path)
    args.android_litert_source.write_text("{}", encoding="utf-8")
    args.android_litert_eval.write_text(
        json.dumps(
            {
                "ok": True,
                "source": args.android_litert_source.relative_to(tmp_path).as_posix(),
                "source_sha256": _sha(args.android_litert_source),
                "thresholds": {"expected_artifact_sha256": "abc"},
                "device": {
                    "model": "Pixel test",
                    "manufacturer": "Google",
                    "android_version": "15",
                },
                "runtime": "LiteRT",
                "artifact": {"sha256": "abc"},
                "metrics": {
                    "runs": 30,
                    "mean_latency_ms": 40.0,
                    "p95_latency_ms": 70.0,
                },
                "output": {"shape": [1, 14, 8400], "finite": True},
            }
        ),
        encoding="utf-8",
    )

    images = args.ar_holdout_source / "images"
    annotations = args.ar_holdout_source / "annotations"
    metadata = args.ar_holdout_source / "metadata"
    images.mkdir(parents=True)
    annotations.mkdir()
    metadata.mkdir()
    (images / "frame.jpg").write_bytes(b"jpg")
    (annotations / "frame.json").write_text("{}", encoding="utf-8")
    (metadata / "provenance.json").write_text(
        '{"source_type":"android_ar_device_human_labelled","label_type":"human_reviewed","capture_device":"Pixel test","review_status":"accepted"}',
        encoding="utf-8",
    )
    args.ar_holdout_eval.write_text(
        '{"metrics_bbox":{"mAP50":0.9},"oks":{"mean":0.85},"rates":{"false_negative_rate":0.05}}',
        encoding="utf-8",
    )
    args.ar_holdout_pipeline.write_text(
        json.dumps(
            {
                "ok": True,
                "stage": "done",
                "source_root": args.ar_holdout_source.relative_to(tmp_path).as_posix(),
                "source_manifest_sha256": ar_holdout_source_manifest_sha256(args.ar_holdout_source),
                "eval_returncode": 0,
                "eval_report": args.ar_holdout_eval.relative_to(tmp_path).as_posix(),
                "eval_report_sha256": _sha(args.ar_holdout_eval),
                "evaluation": {"ok": True, "failures": []},
                "conversion": {"ok": True},
            }
        ),
        encoding="utf-8",
    )

    args.ar_replay_jsonl.write_text("{}\n", encoding="utf-8")
    args.ar_replay_eval.write_text(
        json.dumps(
            {
                "ok": True,
                "source": args.ar_replay_jsonl.relative_to(tmp_path).as_posix(),
                "source_sha256": _sha(args.ar_replay_jsonl),
                "thresholds": {
                    "require_production_source": True,
                    "min_observations": 30,
                    "min_sessions": 1,
                },
                "counts": {
                    "observations_valid": 30,
                    "sessions": 1,
                    "production_source_observations": 30,
                },
            }
        ),
        encoding="utf-8",
    )

    assert "report_source_mismatch" not in " ".join(check_android_litert(args)["failures"])
    assert "pipeline_source_mismatch" not in " ".join(check_ar_holdout(args)["failures"])
    assert "report_source_mismatch" not in " ".join(check_ar_replay(args)["failures"])


def test_evidence_audit_rejects_ar_replay_report_for_different_source(tmp_path):
    args = _args(tmp_path)
    args.ar_replay_jsonl.write_text('{"expected":true}\n', encoding="utf-8")
    old_replay = tmp_path / "old_ar_replay.jsonl"
    old_replay.write_text('{"old":true}\n', encoding="utf-8")
    args.ar_replay_eval.write_text(
        json.dumps(
            {
                "ok": True,
                "source": str(old_replay),
                "source_sha256": _sha(old_replay),
                "thresholds": {
                    "require_production_source": True,
                    "min_observations": 30,
                    "min_sessions": 1,
                },
                "counts": {
                    "observations_valid": 30,
                    "sessions": 1,
                    "production_source_observations": 30,
                },
            }
        ),
        encoding="utf-8",
    )

    check = check_ar_replay(args)

    assert check["ready"] is False
    assert f"report_source_mismatch:{old_replay}" in check["failures"]


def test_evidence_audit_rejects_android_report_for_changed_source_content(tmp_path):
    args = _args(tmp_path)
    args.android_litert_source.write_text("before", encoding="utf-8")
    old_sha = _sha(args.android_litert_source)
    args.android_litert_source.write_text("after", encoding="utf-8")
    args.android_litert_eval.write_text(
        json.dumps(
            {
                "ok": True,
                "source": str(args.android_litert_source),
                "source_sha256": old_sha,
                "thresholds": {"expected_artifact_sha256": "abc"},
                "device": {
                    "model": "Pixel test",
                    "manufacturer": "Google",
                    "android_version": "15",
                },
                "runtime": "LiteRT",
                "artifact": {"sha256": "abc"},
                "metrics": {
                    "runs": 30,
                    "mean_latency_ms": 40.0,
                    "p95_latency_ms": 70.0,
                },
                "output": {"shape": [1, 14, 8400], "finite": True},
            }
        ),
        encoding="utf-8",
    )

    check = check_android_litert(args)

    assert check["ready"] is False
    assert "report_source_sha256_mismatch" in check["failures"]


def test_evidence_audit_rejects_ar_replay_report_for_changed_source_content(tmp_path):
    args = _args(tmp_path)
    args.ar_replay_jsonl.write_text('{"before":true}\n', encoding="utf-8")
    old_sha = _sha(args.ar_replay_jsonl)
    args.ar_replay_jsonl.write_text('{"after":true}\n', encoding="utf-8")
    args.ar_replay_eval.write_text(
        json.dumps(
            {
                "ok": True,
                "source": str(args.ar_replay_jsonl),
                "source_sha256": old_sha,
                "thresholds": {
                    "require_production_source": True,
                    "min_observations": 30,
                    "min_sessions": 1,
                },
                "counts": {
                    "observations_valid": 30,
                    "sessions": 1,
                    "production_source_observations": 30,
                },
            }
        ),
        encoding="utf-8",
    )

    check = check_ar_replay(args)

    assert check["ready"] is False
    assert "report_source_sha256_mismatch" in check["failures"]


def test_evidence_audit_rejects_ar_holdout_pipeline_for_changed_source_content(tmp_path):
    args = _args(tmp_path)
    images = args.ar_holdout_source / "images"
    annotations = args.ar_holdout_source / "annotations"
    metadata = args.ar_holdout_source / "metadata"
    images.mkdir(parents=True)
    annotations.mkdir()
    metadata.mkdir()
    (images / "frame.jpg").write_bytes(b"jpg")
    annotation = annotations / "frame.json"
    annotation.write_text("{}", encoding="utf-8")
    (metadata / "provenance.json").write_text(
        '{"source_type":"android_ar_device_human_labelled","label_type":"human_reviewed","capture_device":"Pixel test","review_status":"accepted"}',
        encoding="utf-8",
    )
    old_manifest = ar_holdout_source_manifest_sha256(args.ar_holdout_source)
    annotation.write_text('{"changed":true}', encoding="utf-8")
    args.ar_holdout_eval.write_text(
        '{"metrics_bbox":{"mAP50":0.9},"oks":{"mean":0.85},"rates":{"false_negative_rate":0.05}}',
        encoding="utf-8",
    )
    args.ar_holdout_pipeline.write_text(
        json.dumps(
            {
                "ok": True,
                "stage": "done",
                "source_root": str(args.ar_holdout_source),
                "source_manifest_sha256": old_manifest,
                "eval_returncode": 0,
                "eval_report": str(args.ar_holdout_eval),
                "eval_report_sha256": _sha(args.ar_holdout_eval),
                "evaluation": {"ok": True, "failures": []},
                "conversion": {"ok": True},
            }
        ),
        encoding="utf-8",
    )

    check = check_ar_holdout(args)

    assert check["ready"] is False
    assert "pipeline_source_manifest_sha256_mismatch" in check["failures"]


def test_evidence_audit_rejects_ar_holdout_eval_report_changed_after_pipeline(tmp_path):
    args = _args(tmp_path)
    _write_ready_holdout(args)
    args.ar_holdout_eval.write_text(
        json.dumps(
            {
                "counts": {"images": 50, "gt_wheels": 80},
                "metrics_bbox": {"mAP50": 0.99},
                "oks": {"mean": 0.99},
                "rates": {"false_negative_rate": 0.0},
            }
        ),
        encoding="utf-8",
    )

    check = check_ar_holdout(args)

    assert check["ready"] is False
    assert "pipeline_eval_report_sha256_mismatch" in check["failures"]


def test_evidence_audit_rejects_ar_holdout_pipeline_evaluation_mismatch(tmp_path):
    args = _args(tmp_path)
    _write_ready_holdout(args)
    pipeline = json.loads(args.ar_holdout_pipeline.read_text(encoding="utf-8"))
    pipeline["evaluation"]["metrics"]["bbox_mAP50"] = 0.99
    pipeline["evaluation"]["thresholds"]["min_images"] = 1
    args.ar_holdout_pipeline.write_text(json.dumps(pipeline), encoding="utf-8")

    check = check_ar_holdout(args)

    assert check["ready"] is False
    assert "pipeline_evaluation_metric_mismatch:bbox_mAP50:0.99!=0.9" in check["failures"]
    assert "pipeline_evaluation_threshold_mismatch:min_images:1!=50" in check["failures"]


def test_external_evidence_custody_accepts_dest_paths_relative_to_dest_root(tmp_path):
    args = _args(tmp_path)
    _write_canonical_input_files(args)
    _write_import_report(
        args,
        dest_root=tmp_path,
        dest_transform=lambda path: path.relative_to(tmp_path),
    )

    custody = check_external_evidence_custody(args, required=True)

    assert custody["ready"] is True
    assert custody["failures"] == []


def test_external_evidence_custody_accepts_dest_paths_relative_to_report_dir(tmp_path):
    args = _args(tmp_path)
    _write_canonical_input_files(args)
    _write_import_report(args, dest_transform=lambda path: path.relative_to(tmp_path))

    custody = check_external_evidence_custody(args, required=True)

    assert custody["ready"] is True
    assert custody["failures"] == []


def test_external_evidence_custody_rejects_minimal_handwritten_import_report(tmp_path):
    args = _args(tmp_path)
    _write_canonical_input_files(args)
    _write_import_report(args)
    report = json.loads(args.external_evidence_import_report.read_text(encoding="utf-8"))
    for key in ("schema_version", "source", "source_kind", "dest_root", "file_count", "planned"):
        report.pop(key, None)
    args.external_evidence_import_report.write_text(json.dumps(report), encoding="utf-8")

    custody = check_external_evidence_custody(args, required=True)

    assert custody["ready"] is False
    assert "unsupported_import_report_schema:missing" in custody["failures"]
    assert "missing_import_source" in custody["failures"]
    assert "unsupported_import_source_kind:missing" in custody["failures"]
    assert "missing_import_dest_root" in custody["failures"]
    assert "invalid_import_file_count:missing" in custody["failures"]
    assert "missing_planned_artifacts" in custody["failures"]


def test_external_evidence_custody_rejects_boolean_import_report_schema(tmp_path):
    args = _args(tmp_path)
    _write_canonical_input_files(args)
    _write_import_report(args)
    report = json.loads(args.external_evidence_import_report.read_text(encoding="utf-8"))
    report["schema_version"] = True
    args.external_evidence_import_report.write_text(json.dumps(report), encoding="utf-8")

    custody = check_external_evidence_custody(args, required=True)

    assert custody["ready"] is False
    assert "unsupported_import_report_schema:True" in custody["failures"]


def test_external_evidence_custody_rejects_manifest_mismatch(tmp_path):
    args = _args(tmp_path)
    _write_canonical_input_files(args)
    _write_import_report(args)
    report = json.loads(args.external_evidence_import_report.read_text(encoding="utf-8"))
    report["evidence_manifest_sha256"] = "bad-manifest"
    args.external_evidence_import_report.write_text(json.dumps(report), encoding="utf-8")

    custody = check_external_evidence_custody(args, required=True)

    assert custody["ready"] is False
    assert "evidence_manifest_sha256_mismatch" in custody["failures"]


def test_external_evidence_custody_rejects_planned_copied_mismatch(tmp_path):
    args = _args(tmp_path)
    _write_canonical_input_files(args)
    _write_import_report(args)
    report = json.loads(args.external_evidence_import_report.read_text(encoding="utf-8"))
    report["planned"][0]["sha256"] = "planned-bad-sha"
    report["evidence_manifest_sha256"] = _manifest_sha256(report["planned"])
    args.external_evidence_import_report.write_text(json.dumps(report), encoding="utf-8")

    custody = check_external_evidence_custody(args, required=True)

    assert custody["ready"] is False
    assert any(
        failure.startswith("copied_artifact_sha256_mismatch:")
        for failure in custody["failures"]
    )


def test_external_evidence_custody_rejects_non_integer_artifact_sizes(tmp_path):
    args = _args(tmp_path)
    _write_canonical_input_files(args)
    _write_import_report(args)
    report = json.loads(args.external_evidence_import_report.read_text(encoding="utf-8"))
    report["planned"][0]["size_bytes"] = True
    report["copied_artifacts"][0]["size_bytes"] = True
    report["evidence_manifest_sha256"] = _manifest_sha256(report["planned"])
    args.external_evidence_import_report.write_text(json.dumps(report), encoding="utf-8")

    custody = check_external_evidence_custody(args, required=True)

    assert custody["ready"] is False
    planned_dest = report["planned"][0]["dest"]
    copied_dest = report["copied_artifacts"][0]["dest"]
    assert f"planned_artifact_invalid_size_bytes:{planned_dest}:True" in custody["failures"]
    assert f"copied_artifact_invalid_size_bytes:{copied_dest}:True" in custody["failures"]
    assert any(
        failure.startswith("input_invalid_import_size_bytes:")
        for failure in custody["failures"]
    )


def test_external_evidence_custody_rejects_stale_artifact_size(tmp_path):
    args = _args(tmp_path)
    _write_canonical_input_files(args)
    _write_import_report(args)
    report = json.loads(args.external_evidence_import_report.read_text(encoding="utf-8"))
    report["planned"][0]["size_bytes"] += 1
    report["copied_artifacts"][0]["size_bytes"] += 1
    report["evidence_manifest_sha256"] = _manifest_sha256(report["planned"])
    args.external_evidence_import_report.write_text(json.dumps(report), encoding="utf-8")

    custody = check_external_evidence_custody(args, required=True)

    assert custody["ready"] is False
    assert any(
        failure.startswith("input_size_mismatch:")
        for failure in custody["failures"]
    )


def test_external_evidence_custody_rejects_stale_expected_android_artifact_sha(
    tmp_path,
):
    args = _args(tmp_path)
    _write_canonical_input_files(args)
    _write_import_report(args)
    report = json.loads(args.external_evidence_import_report.read_text(encoding="utf-8"))
    report["expected_android_artifact_sha256"] = "old-artifact-sha"
    args.external_evidence_import_report.write_text(json.dumps(report), encoding="utf-8")

    custody = check_external_evidence_custody(args, required=True)

    assert custody["ready"] is False
    assert "import_expected_android_artifact_sha256_mismatch" in custody["failures"]


def test_external_evidence_custody_rejects_expected_android_artifact_path_mismatch(
    tmp_path,
):
    args = _args(tmp_path)
    _write_canonical_input_files(args)
    _write_import_report(args)
    report = json.loads(args.external_evidence_import_report.read_text(encoding="utf-8"))
    report["expected_android_artifact"] = str(tmp_path / "old_expected_android.tflite")
    args.external_evidence_import_report.write_text(json.dumps(report), encoding="utf-8")

    custody = check_external_evidence_custody(args, required=True)

    assert custody["ready"] is False
    assert any(
        failure.startswith("import_expected_android_artifact_path_mismatch:")
        for failure in custody["failures"]
    )


def test_external_evidence_custody_rejects_retained_zip_source_sha_mismatch(tmp_path):
    args = _args(tmp_path)
    _write_canonical_input_files(args)
    source_zip = tmp_path / "drop.zip"
    source_zip.write_bytes(b"zip-v1")
    _write_import_report(args)
    report = json.loads(args.external_evidence_import_report.read_text(encoding="utf-8"))
    report["source"] = str(source_zip)
    report["source_kind"] = "zip"
    report["source_sha256"] = _sha(source_zip)
    source_zip.write_bytes(b"zip-v2")
    args.external_evidence_import_report.write_text(json.dumps(report), encoding="utf-8")

    custody = check_external_evidence_custody(args, required=True)

    assert custody["ready"] is False
    assert "zip_source_sha256_mismatch" in custody["failures"]


def test_evidence_audit_rejects_spoofed_reports_without_source_proof(tmp_path):
    args = _args(tmp_path)
    args.android_litert_source.write_text("{}", encoding="utf-8")
    args.android_litert_eval.write_text(
        json.dumps(
            {
                "ok": True,
                "source": str(args.android_litert_source),
                "source_sha256": _sha(args.android_litert_source),
                "thresholds": {"expected_artifact_sha256": "abc"},
                "artifact": {"sha256": "wrong"},
            }
        ),
        encoding="utf-8",
    )

    images = args.ar_holdout_source / "images"
    annotations = args.ar_holdout_source / "annotations"
    metadata = args.ar_holdout_source / "metadata"
    images.mkdir(parents=True)
    annotations.mkdir()
    metadata.mkdir()
    (images / "frame.jpg").write_bytes(b"jpg")
    (annotations / "frame.json").write_text("{}", encoding="utf-8")
    (metadata / "provenance.json").write_text(
        '{"source_type":"android_ar_device_human_labelled","label_type":"human_reviewed","capture_device":"FILL_ME"}',
        encoding="utf-8",
    )
    args.ar_holdout_eval.write_text(
        json.dumps(
            {
                "metrics_bbox": {"mAP50": 0.99},
                "oks": {"mean": 0.99},
                "rates": {"false_negative_rate": 0.0},
            }
        ),
        encoding="utf-8",
    )

    replay_source = tmp_path / "ar_3d_replay.template.jsonl"
    replay_source.write_text("{}\n", encoding="utf-8")
    args.ar_replay_eval.write_text(
        json.dumps(
            {
                "ok": True,
                "source": str(replay_source),
                "source_sha256": _sha(replay_source),
                "thresholds": {"require_production_source": False},
                "counts": {
                    "observations_valid": 30,
                    "production_source_observations": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    audit = build_audit(args)

    assert audit["production_evidence_ready"] is False
    all_failures = " ".join(
        failure
        for check in audit["checks"]
        for failure in check["failures"]
    )
    assert "artifact_sha256_mismatch" in all_failures
    assert "missing_device_model" in all_failures
    assert "missing_valid_observation_count" not in all_failures
    assert "invalid_provenance" in all_failures
    assert "template_source_not_allowed" in all_failures
    assert "production_source_not_required_in_report" in all_failures
    assert "min_observations_too_low" in all_failures


def test_render_markdown_lists_failures(tmp_path):
    audit = build_audit(_args(tmp_path))
    markdown = render_markdown(audit)

    assert "Production Evidence Audit" in markdown
    assert "Required Evidence" in markdown
    assert "Current Checks" in markdown
    assert "android_litert_device_validation" in markdown
    assert "android_litert_harness/AndroidLiteRtDeviceValidationTest.kt" in markdown
    assert "ar_holdout_harness/ArHoldoutAnnotationWriter.kt" in markdown
    assert "ar_replay_harness/ArReplayLogger.kt" in markdown
    assert "validate_android_litert_report.py" in markdown
    assert "missing_source" in markdown
