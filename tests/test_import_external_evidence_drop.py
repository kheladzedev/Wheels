from __future__ import annotations

import json
import zipfile
import hashlib
import math
from pathlib import Path

from src.import_external_evidence_drop import (
    MAX_DROP_FILE_BYTES,
    MAX_DROP_TOTAL_BYTES,
    build_import_report,
    destination_for,
    destination_for_drop_path,
    safe_posix_path,
    zip_entry_guard_failures,
)

EXPECTED_ARTIFACT_BYTES = b"test tflite"
EXPECTED_ARTIFACT_SHA = hashlib.sha256(EXPECTED_ARTIFACT_BYTES).hexdigest()


def _write_drop(root: Path, *, artifact_sha: str = "abc") -> None:
    (root / "ar_device_holdout" / "images").mkdir(parents=True)
    (root / "ar_device_holdout" / "annotations").mkdir()
    (root / "ar_device_holdout" / "metadata").mkdir()
    (root / "ar_3d_replay").mkdir()
    (root / "android_litert_device_report.json").write_text(
        json.dumps(
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
                "artifact": {"sha256": artifact_sha, "format": "tflite_float32"},
                "input": {"shape": [1, 640, 640, 3], "dtype": "float32", "profile": "zero_float32_smoke"},
                "latency_ms": {"runs": 30, "mean": 40.0, "p95": 70.0},
                "memory_mb": {"peak": 128.0},
                "output": {"shape": [1, 14, 8400], "finite": True, "min": 0.0, "max": 1.0, "mean": 0.5},
            }
        ),
        encoding="utf-8",
    )
    for i in range(50):
        frame_id = f"frame_{i:04d}"
        image_name = f"{frame_id}.jpg"
        (root / "ar_device_holdout" / "images" / image_name).write_bytes(b"jpg")
        (root / "ar_device_holdout" / "annotations" / f"{frame_id}.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frame_id": frame_id,
                "image": image_name,
                    "wheels": [
                        {
                            "bbox_xyxy": [10, 10, 30, 30],
                            "points": {"a": [12, 28], "b": [28, 28], "c_disc_bottom": [20, 24]},
                        },
                        {
                            "bbox_xyxy": [40, 10, 60, 30],
                            "points": {"a": [42, 28], "b": [58, 28], "c_disc_bottom": [50, 24]},
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
    (root / "ar_device_holdout" / "metadata" / "provenance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_type": "android_ar_device_human_labelled",
                "label_type": "human_reviewed",
                "capture_device": "Pixel test",
                "capture_app_version": "1.2.3",
                "capture_date_utc": "2026-05-27",
                "annotator": "annotator_a",
                "reviewer": "reviewer_b",
                "review_status": "accepted",
            }
        ),
        encoding="utf-8",
    )
    (root / "ar_3d_replay" / "ar_replay.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_type": "android_ar_device_replay",
                    "capture_device": "Pixel test",
                    "capture_app_version": "1.2.3",
                    "capture_date_utc": "2026-05-27",
                    "session_id": "s1",
                    "frame_id": f"frame_{i:04d}",
                    "capture_index": i,
                    "camera_transform": None,
                    "camera_pose_ref": f"pose_{i:04d}",
                    "screen_points": {
                        "a": [100.0, 200.0],
                        "b": [150.0, 200.0],
                        "c_disc_bottom": [125.0, 170.0],
                    },
                    "floor_raycast_hits": {"a": [1.0, 0.0, 2.0], "b": [1.5, 0.0, 2.0]},
                    "inlier": True,
                    "residual": 0.004,
                    "recovered_plane": {
                        "normal": [0.998, 0.0, 0.062],
                        "point": [1.25, 0.0, 2.0],
                        "support": 18,
                    },
                    "c_plane_hit": [1.25, 0.4, 2.0],
                    "c_height_value": 0.4,
                    "final_disc_bottom_position": [1.25, 0.0, 2.0] if i == 0 else None,
                }
            )
            for i in range(30)
        )
        + "\n",
        encoding="utf-8",
    )


def test_safe_posix_path_rejects_traversal_and_absolute_paths():
    assert safe_posix_path("../evil.json") is None
    assert safe_posix_path("/tmp/evil.json") is None
    assert safe_posix_path("nested/../../evil.json") is None
    assert safe_posix_path("./android_litert_device_report.json") is not None


def test_destination_for_accepts_supported_prefixes():
    assert destination_for(safe_posix_path("android_litert_device_report.json")) == Path(
        "android_litert_device_report.json"
    )
    assert destination_for(safe_posix_path("data/incoming/ar_3d_replay/ar_replay.jsonl")) == Path(
        "ar_3d_replay/ar_replay.jsonl"
    )
    assert destination_for(safe_posix_path("incoming/ar_device_holdout/images/a.jpg")) == Path(
        "ar_device_holdout/images/a.jpg"
    )
    assert destination_for(safe_posix_path("outputs/production_audit/ar_3d_replay.template.jsonl")) is None


def test_destination_for_drop_path_accepts_single_root_folder_prefix():
    assert destination_for_drop_path(safe_posix_path("evidence_drop/android_litert_device_report.json")) == Path(
        "android_litert_device_report.json"
    )
    assert destination_for_drop_path(
        safe_posix_path("evidence_drop/data/incoming/ar_3d_replay/ar_replay.jsonl")
    ) == Path("ar_3d_replay/ar_replay.jsonl")
    assert destination_for_drop_path(
        safe_posix_path("evidence_drop/ar_device_holdout/annotations/frame_0001.json")
    ) == Path("ar_device_holdout/annotations/frame_0001.json")


def test_import_external_evidence_drop_dry_run_from_directory(tmp_path):
    drop = tmp_path / "drop"
    dest = tmp_path / "incoming"
    _write_drop(drop)

    report = build_import_report(drop, dest_root=dest, dry_run=True, overwrite=False)

    assert report["ok"] is True
    assert report["dry_run"] is True
    assert report["file_count"] == 103
    assert report["source_kind"] == "directory"
    assert report["source_sha256"] is None
    assert report["evidence_manifest_sha256"]
    assert all(item["sha256"] for item in report["planned"])
    assert report["copied"] == []
    assert not (dest / "android_litert_device_report.json").exists()


def test_import_external_evidence_drop_copies_expected_files(tmp_path):
    drop = tmp_path / "drop"
    dest = tmp_path / "incoming"
    _write_drop(drop)

    report = build_import_report(drop, dest_root=dest, dry_run=False, overwrite=False)

    assert report["ok"] is True
    assert (dest / "android_litert_device_report.json").is_file()
    assert (dest / "ar_device_holdout" / "images" / "frame_0001.jpg").is_file()
    assert (dest / "ar_device_holdout" / "annotations" / "frame_0001.json").is_file()
    assert (dest / "ar_device_holdout" / "metadata" / "provenance.json").is_file()
    assert (dest / "ar_3d_replay" / "ar_replay.jsonl").is_file()
    assert len(report["copied_artifacts"]) == 103
    assert {
        item["dest"]: item["sha256"] for item in report["copied_artifacts"]
    }[str(dest / "android_litert_device_report.json")]
    assert "./.venv/bin/python src/run_production_evidence_intake.py --finalize" in report["next_commands"]


def test_import_external_evidence_drop_rejects_unsafe_zip_entry(tmp_path):
    archive = tmp_path / "drop.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../evil.json", "{}")

    report = build_import_report(archive, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "unsafe_zip_entry:../evil.json" in report["failures"]


def test_import_external_evidence_drop_rejects_zip_symlink_entry(tmp_path):
    archive = tmp_path / "drop.zip"
    info = zipfile.ZipInfo("android_litert_device_report.json")
    info.external_attr = 0o120777 << 16
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(info, "target")

    report = build_import_report(archive, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "zip_symlink_not_allowed:android_litert_device_report.json" in report["failures"]


def test_zip_entry_guard_rejects_oversized_uncompressed_payload():
    info = zipfile.ZipInfo("ar_device_holdout/images/frame_0001.jpg")
    info.file_size = MAX_DROP_FILE_BYTES + 1

    failures = zip_entry_guard_failures(info, MAX_DROP_TOTAL_BYTES - 1)

    assert (
        f"zip_entry_too_large:ar_device_holdout/images/frame_0001.jpg:"
        f"{MAX_DROP_FILE_BYTES + 1}>{MAX_DROP_FILE_BYTES}"
    ) in failures
    assert f"zip_total_uncompressed_too_large:{MAX_DROP_TOTAL_BYTES + MAX_DROP_FILE_BYTES}>{MAX_DROP_TOTAL_BYTES}" in failures


def test_import_external_evidence_drop_accepts_root_prefixed_zip(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    archive = tmp_path / "drop.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in sorted(p for p in drop.rglob("*") if p.is_file()):
            zf.write(path, "evidence_drop/" + path.relative_to(drop).as_posix())

    report = build_import_report(archive, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is True
    assert report["source_kind"] == "zip"
    assert report["source_sha256"]
    assert report["file_count"] == 103
    assert {
        item["dest"] for item in report["planned"]
    } >= {
        str(tmp_path / "incoming" / "android_litert_device_report.json"),
        str(tmp_path / "incoming" / "ar_3d_replay" / "ar_replay.jsonl"),
    }


def test_import_external_evidence_drop_reports_missing_required_inputs(tmp_path):
    drop = tmp_path / "drop"
    drop.mkdir()
    (drop / "android_litert_device_report.json").write_text("{}", encoding="utf-8")

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "missing_required:ar_holdout_images" in report["failures"]
    assert "missing_required:ar_replay_jsonl" in report["failures"]


def test_import_external_evidence_drop_rejects_too_small_production_drop(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    for i in range(1, 50):
        (drop / "ar_device_holdout" / "images" / f"frame_{i:04d}.jpg").unlink()
        (drop / "ar_device_holdout" / "annotations" / f"frame_{i:04d}.json").unlink()
    (drop / "ar_3d_replay" / "ar_replay.jsonl").write_text(
        (drop / "ar_3d_replay" / "ar_replay.jsonl").read_text(encoding="utf-8").splitlines()[0] + "\n",
        encoding="utf-8",
    )

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "ar_holdout_too_few_images:1<50" in report["failures"]
    assert "ar_holdout_too_few_gt_wheels:2<80" in report["failures"]
    assert "ar_replay_too_few_observations:1<30" in report["failures"]


def test_import_external_evidence_drop_reports_holdout_stem_mismatch(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    (drop / "ar_device_holdout" / "annotations" / "frame_0001.json").unlink()
    (drop / "ar_device_holdout" / "annotations" / "frame_9999.json").write_text(
        '{"frame_id":"frame_9999","image":"frame_9999.jpg","wheels":[]}',
        encoding="utf-8",
    )

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "ar_holdout_missing_annotations:frame_0001" in report["failures"]
    assert "ar_holdout_missing_images:frame_9999" in report["failures"]


def test_import_external_evidence_drop_rejects_bad_holdout_extensions(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    (drop / "ar_device_holdout" / "images" / "frame_0002.txt").write_text("not an image", encoding="utf-8")
    (drop / "ar_device_holdout" / "annotations" / "frame_0002.txt").write_text("{}", encoding="utf-8")

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "ar_holdout_bad_image_extensions:ar_device_holdout/images/frame_0002.txt" in report["failures"]
    assert "ar_holdout_bad_annotation_extensions:ar_device_holdout/annotations/frame_0002.txt" in report["failures"]


def test_import_external_evidence_drop_rejects_invalid_holdout_annotation_json(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    (drop / "ar_device_holdout" / "annotations" / "frame_0001.json").write_text("{", encoding="utf-8")

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "ar_holdout_invalid_annotation_json:frame_0001.json" in report["failures"]


def test_import_external_evidence_drop_rejects_holdout_annotation_contract_errors(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    (drop / "ar_device_holdout" / "annotations" / "frame_0001.json").write_text(
        '{"frame_id":"wrong","image":"nested/frame_0002.jpg","wheels":{}}',
        encoding="utf-8",
    )

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "ar_holdout_annotation_frame_id_mismatch:frame_0001.json:wrong" in report["failures"]
    assert "ar_holdout_annotation_image_not_filename:frame_0001.json:nested/frame_0002.jpg" in report["failures"]
    assert "ar_holdout_annotation_wheels_not_array:frame_0001.json" in report["failures"]


def test_import_external_evidence_drop_rejects_holdout_annotation_missing_schema_version(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    annotation_path = drop / "ar_device_holdout" / "annotations" / "frame_0001.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    payload.pop("schema_version")
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "ar_holdout_annotation_unsupported_schema_version:frame_0001.json:missing" in report["failures"]


def test_import_external_evidence_drop_rejects_boolean_schema_versions(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    android_path = drop / "android_litert_device_report.json"
    android = json.loads(android_path.read_text(encoding="utf-8"))
    android["schema_version"] = True
    android_path.write_text(json.dumps(android), encoding="utf-8")
    replay_path = drop / "ar_3d_replay" / "ar_replay.jsonl"
    observations = [json.loads(line) for line in replay_path.read_text(encoding="utf-8").splitlines()]
    observations[0]["schema_version"] = True
    replay_path.write_text(
        "\n".join(json.dumps(observation) for observation in observations) + "\n",
        encoding="utf-8",
    )

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "android_report_unsupported_schema_version:True" in report["failures"]
    assert "ar_replay_line_unsupported_schema_version:1:True" in report["failures"]


def test_import_external_evidence_drop_rejects_holdout_annotation_missing_image_reference(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    (drop / "ar_device_holdout" / "annotations" / "frame_0001.json").write_text(
        '{"frame_id":"frame_0001","image":"frame_9999.jpg","wheels":[]}',
        encoding="utf-8",
    )

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "ar_holdout_annotation_image_missing:frame_0001.json:frame_9999.jpg" in report["failures"]
    assert "ar_holdout_annotation_image_stem_mismatch:frame_0001.json:frame_9999.jpg" in report["failures"]


def test_import_external_evidence_drop_rejects_invalid_holdout_wheel_schema(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    annotation_path = drop / "ar_device_holdout" / "annotations" / "frame_0001.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    payload["wheels"] = [
        {"bbox": [10, 10, 30, 30], "keypoints": []},
        {
            "bbox_xyxy": [40, 10, 20, 30],
            "points": {"a": [42, 28], "b": [58, 28]},
        },
    ]
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "ar_holdout_wheel_invalid_bbox_xyxy:frame_0001.json:wheel[0]" in report["failures"]
    assert "ar_holdout_wheel_missing_points:frame_0001.json:wheel[0]" in report["failures"]
    assert "ar_holdout_wheel_invalid_bbox_order:frame_0001.json:wheel[1]" in report["failures"]
    assert "ar_holdout_wheel_invalid_point_c_disc_bottom:frame_0001.json:wheel[1]" in report["failures"]


def test_import_external_evidence_drop_rejects_holdout_points_outside_bbox(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    annotation_path = drop / "ar_device_holdout" / "annotations" / "frame_0001.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    payload["wheels"] = [
        {
            "bbox_xyxy": [10, 10, 30, 30],
            "points": {"a": [9, 20], "b": [28, 28], "c_disc_bottom": [20, 31]},
        }
    ]
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "ar_holdout_wheel_point_a_outside_bbox:frame_0001.json:wheel[0]" in report["failures"]
    assert (
        "ar_holdout_wheel_point_c_disc_bottom_outside_bbox:frame_0001.json:wheel[0]"
        in report["failures"]
    )


def test_import_external_evidence_drop_requires_overwrite_for_existing_files(tmp_path):
    drop = tmp_path / "drop"
    dest = tmp_path / "incoming"
    _write_drop(drop)
    (dest).mkdir()
    (dest / "android_litert_device_report.json").write_text("old", encoding="utf-8")

    report = build_import_report(drop, dest_root=dest, dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert f"would_overwrite:{dest / 'android_litert_device_report.json'}" in report["failures"]


def test_import_external_evidence_drop_rejects_directory_symlinks(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    outside = tmp_path / "outside.json"
    outside.write_text('{"leak":true}', encoding="utf-8")
    (drop / "ar_device_holdout" / "annotations" / "leaked.json").symlink_to(outside)

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "symlink_not_allowed:ar_device_holdout/annotations/leaked.json" in report["failures"]


def test_import_external_evidence_drop_rejects_leftover_placeholders(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    (drop / "android_litert_device_report.json.PLACEHOLDER").write_text(
        "replace me",
        encoding="utf-8",
    )

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "placeholder_file_not_allowed:android_litert_device_report.json.PLACEHOLDER" in report["failures"]


def test_import_external_evidence_drop_rejects_empty_required_files(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    (drop / "ar_3d_replay" / "ar_replay.jsonl").write_text("", encoding="utf-8")

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "empty_file:ar_3d_replay/ar_replay.jsonl" in report["failures"]


def test_import_external_evidence_drop_rejects_invalid_android_report_json(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    (drop / "android_litert_device_report.json").write_text("{", encoding="utf-8")

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "android_report_invalid_json_object" in report["failures"]


def test_import_external_evidence_drop_rejects_incomplete_android_report_contract(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    (drop / "android_litert_device_report.json").write_text(
        '{"device":{"model":"Pixel test"},"runtime":"LiteRT"}',
        encoding="utf-8",
    )

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "android_report_missing_device_manufacturer" in report["failures"]
    assert "android_report_missing_artifact_object" in report["failures"]
    assert "android_report_missing_latency_ms_object" in report["failures"]
    assert "android_report_missing_output_object" in report["failures"]


def test_import_external_evidence_drop_rejects_wrong_android_artifact_hash(tmp_path):
    drop = tmp_path / "drop"
    expected_artifact = tmp_path / "expected.tflite"
    expected_artifact.write_bytes(EXPECTED_ARTIFACT_BYTES)
    _write_drop(drop, artifact_sha="wrong-sha")

    report = build_import_report(
        drop,
        dest_root=tmp_path / "incoming",
        dry_run=True,
        overwrite=False,
        expected_android_artifact=expected_artifact,
    )

    assert report["ok"] is False
    assert report["expected_android_artifact_sha256"] == EXPECTED_ARTIFACT_SHA
    assert "android_report_artifact_sha256_mismatch" in report["failures"]


def test_import_external_evidence_drop_accepts_matching_expected_artifact_metadata(tmp_path):
    drop = tmp_path / "drop"
    expected_artifact = tmp_path / "expected.tflite"
    expected_artifact.write_bytes(EXPECTED_ARTIFACT_BYTES)
    _write_drop(drop, artifact_sha=EXPECTED_ARTIFACT_SHA)
    (drop / "EXPECTED_ANDROID_ARTIFACT.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "expected_android_artifact": {
                    "path": str(expected_artifact),
                    "sha256": EXPECTED_ARTIFACT_SHA,
                    "format": "tflite_float32",
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_import_report(
        drop,
        dest_root=tmp_path / "incoming",
        dry_run=True,
        overwrite=False,
        expected_android_artifact=expected_artifact,
    )

    assert report["ok"] is True
    assert report["expected_android_artifact_metadata_count"] == 1


def test_import_external_evidence_drop_rejects_stale_expected_artifact_metadata(tmp_path):
    drop = tmp_path / "drop"
    expected_artifact = tmp_path / "expected.tflite"
    expected_artifact.write_bytes(EXPECTED_ARTIFACT_BYTES)
    _write_drop(drop, artifact_sha=EXPECTED_ARTIFACT_SHA)
    (drop / "EXPECTED_ANDROID_ARTIFACT.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "expected_android_artifact": {
                    "path": str(expected_artifact),
                    "sha256": "old-sha",
                    "format": "tflite_float32",
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_import_report(
        drop,
        dest_root=tmp_path / "incoming",
        dry_run=True,
        overwrite=False,
        expected_android_artifact=expected_artifact,
    )

    assert report["ok"] is False
    assert report["expected_android_artifact_metadata_count"] == 1
    assert (
        "expected_android_artifact_metadata_sha256_mismatch:"
        "EXPECTED_ANDROID_ARTIFACT.json:old-sha"
    ) in report["failures"]


def test_import_external_evidence_drop_rejects_root_prefixed_stale_expected_artifact_metadata_zip(tmp_path):
    drop = tmp_path / "drop"
    expected_artifact = tmp_path / "expected.tflite"
    expected_artifact.write_bytes(EXPECTED_ARTIFACT_BYTES)
    _write_drop(drop, artifact_sha=EXPECTED_ARTIFACT_SHA)
    archive = tmp_path / "drop.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in sorted(p for p in drop.rglob("*") if p.is_file()):
            zf.write(path, "evidence_drop/" + path.relative_to(drop).as_posix())
        zf.writestr(
            "evidence_drop/EXPECTED_ANDROID_ARTIFACT.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "expected_android_artifact": {
                        "path": str(expected_artifact),
                        "sha256": "old-sha",
                        "format": "tflite_float32",
                    },
                }
            ),
        )

    report = build_import_report(
        archive,
        dest_root=tmp_path / "incoming",
        dry_run=True,
        overwrite=False,
        expected_android_artifact=expected_artifact,
    )

    assert report["ok"] is False
    assert report["expected_android_artifact_metadata_count"] == 1
    assert (
        "expected_android_artifact_metadata_sha256_mismatch:"
        "evidence_drop/EXPECTED_ANDROID_ARTIFACT.json:old-sha"
    ) in report["failures"]


def test_import_external_evidence_drop_rejects_invalid_ar_replay_jsonl(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    (drop / "ar_3d_replay" / "ar_replay.jsonl").write_text("not-json\n[]\n", encoding="utf-8")

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "ar_replay_invalid_json_line:1" in report["failures"]
    assert "ar_replay_line_not_object:2" in report["failures"]


def test_import_external_evidence_drop_rejects_incomplete_ar_replay_observation(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    (drop / "ar_3d_replay" / "ar_replay.jsonl").write_text(
        '{"source_type":"android_ar_device_replay","capture_device":"Pixel test","session_id":"s1"}\n',
        encoding="utf-8",
    )

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "ar_replay_line_missing_frame_id:1" in report["failures"]
    assert "ar_replay_line_missing_capture_index:1" in report["failures"]
    assert "ar_replay_line_missing_screen_points:1" in report["failures"]
    assert "ar_replay_line_missing_floor_raycast_hits:1" in report["failures"]


def test_import_external_evidence_drop_rejects_replay_without_camera_pose_evidence(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    observations = [
        json.loads(line)
        for line in (drop / "ar_3d_replay" / "ar_replay.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for observation in observations:
        observation["camera_pose_ref"] = None
        observation["camera_transform"] = None
    (drop / "ar_3d_replay" / "ar_replay.jsonl").write_text(
        "\n".join(json.dumps(observation) for observation in observations) + "\n",
        encoding="utf-8",
    )

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "ar_replay_line_missing_camera_pose_evidence:1" in report["failures"]


def test_import_external_evidence_drop_rejects_malformed_inline_camera_transform(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    observations = [
        json.loads(line)
        for line in (drop / "ar_3d_replay" / "ar_replay.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for observation in observations:
        observation["camera_pose_ref"] = None
        observation["camera_transform"] = {"R": [[1.0, 0.0], [0.0, 1.0]], "t": [0.0, 1.0, 2.0]}
    (drop / "ar_3d_replay" / "ar_replay.jsonl").write_text(
        "\n".join(json.dumps(observation) for observation in observations) + "\n",
        encoding="utf-8",
    )

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "ar_replay_line_invalid_camera_transform:1" in report["failures"]


def test_import_external_evidence_drop_rejects_replay_without_recovered_plane(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    observations = [
        json.loads(line)
        for line in (drop / "ar_3d_replay" / "ar_replay.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for observation in observations:
        observation.pop("recovered_plane")
    (drop / "ar_3d_replay" / "ar_replay.jsonl").write_text(
        "\n".join(json.dumps(observation) for observation in observations) + "\n",
        encoding="utf-8",
    )

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "ar_replay_line_missing_recovered_plane:1" in report["failures"]
    assert "ar_replay_missing_recovered_planes:0!=30" in report["failures"]


def test_import_external_evidence_drop_rejects_non_unit_recovered_plane_normal(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    observations = [
        json.loads(line)
        for line in (drop / "ar_3d_replay" / "ar_replay.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for observation in observations:
        observation["recovered_plane"]["normal"] = [2.0, 0.0, 0.0]
    (drop / "ar_3d_replay" / "ar_replay.jsonl").write_text(
        "\n".join(json.dumps(observation) for observation in observations) + "\n",
        encoding="utf-8",
    )

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "ar_replay_line_invalid_recovered_plane:1" in report["failures"]
    assert "ar_replay_missing_recovered_planes:0!=30" in report["failures"]


def test_import_external_evidence_drop_rejects_invalid_ar_replay_capture_index(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    observations = [
        json.loads(line)
        for line in (drop / "ar_3d_replay" / "ar_replay.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    observations[0]["capture_index"] = -1
    observations[1]["capture_index"] = True
    (drop / "ar_3d_replay" / "ar_replay.jsonl").write_text(
        "\n".join(json.dumps(observation) for observation in observations) + "\n",
        encoding="utf-8",
    )

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "ar_replay_line_negative_capture_index:1:-1" in report["failures"]
    assert "ar_replay_line_missing_capture_index:2" in report["failures"]


def test_import_external_evidence_drop_rejects_decreasing_ar_replay_capture_index(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    observations = [
        json.loads(line)
        for line in (drop / "ar_3d_replay" / "ar_replay.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    observations[2]["capture_index"] = 0
    (drop / "ar_3d_replay" / "ar_replay.jsonl").write_text(
        "\n".join(json.dumps(observation) for observation in observations) + "\n",
        encoding="utf-8",
    )

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "ar_replay_line_decreasing_capture_index:3:0<1:previous_line=2" in report["failures"]


def test_import_external_evidence_drop_rejects_repeated_replay_frame_without_wheel_identity(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    observations = [
        json.loads(line)
        for line in (drop / "ar_3d_replay" / "ar_replay.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    observations[2]["capture_index"] = 1
    observations[2]["frame_id"] = "frame_0001"
    (drop / "ar_3d_replay" / "ar_replay.jsonl").write_text(
        "\n".join(json.dumps(observation) for observation in observations) + "\n",
        encoding="utf-8",
    )

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "ar_replay_repeated_frame_missing_wheel_identity:s1:frame_0001:1:lines=2,3" in report["failures"]


def test_import_external_evidence_drop_accepts_repeated_replay_frame_with_unique_wheel_identity(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    observations = [
        json.loads(line)
        for line in (drop / "ar_3d_replay" / "ar_replay.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    observations[1]["wheel_index"] = 0
    observations[2]["capture_index"] = 1
    observations[2]["frame_id"] = "frame_0001"
    observations[2]["camera_pose_ref"] = "pose_0001"
    observations[2]["wheel_index"] = 1
    (drop / "ar_3d_replay" / "ar_replay.jsonl").write_text(
        "\n".join(json.dumps(observation) for observation in observations) + "\n",
        encoding="utf-8",
    )

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert "ar_replay_repeated_frame_missing_wheel_identity:s1:frame_0001:1:lines=2,3" not in report["failures"]
    assert "ar_replay_line_decreasing_capture_index:3:0<1:previous_line=2" not in report["failures"]


def test_import_external_evidence_drop_rejects_replay_without_c_plane_reconstruction(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    observations = [
        json.loads(line)
        for line in (drop / "ar_3d_replay" / "ar_replay.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for observation in observations:
        observation["c_plane_hit"] = None
        observation["c_height_value"] = None
    (drop / "ar_3d_replay" / "ar_replay.jsonl").write_text(
        "\n".join(json.dumps(observation) for observation in observations) + "\n",
        encoding="utf-8",
    )

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "ar_replay_line_missing_c_plane_hit:1" in report["failures"]
    assert "ar_replay_line_missing_c_height_value:1" in report["failures"]
    assert "ar_replay_missing_c_plane_hits:0!=30" in report["failures"]
    assert "ar_replay_missing_c_height_values:0!=30" in report["failures"]


def test_import_external_evidence_drop_rejects_negative_ar_replay_residual(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    observations = [
        json.loads(line)
        for line in (drop / "ar_3d_replay" / "ar_replay.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    observations[0]["residual"] = -0.001
    (drop / "ar_3d_replay" / "ar_replay.jsonl").write_text(
        "\n".join(json.dumps(observation) for observation in observations) + "\n",
        encoding="utf-8",
    )

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "ar_replay_line_negative_residual:1" in report["failures"]
    assert "ar_replay_missing_residuals:29!=30" in report["failures"]


def test_import_external_evidence_drop_rejects_negative_ar_replay_c_height(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    observations = [
        json.loads(line)
        for line in (drop / "ar_3d_replay" / "ar_replay.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    observations[0]["c_height_value"] = -0.1
    (drop / "ar_3d_replay" / "ar_replay.jsonl").write_text(
        "\n".join(json.dumps(observation) for observation in observations) + "\n",
        encoding="utf-8",
    )

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "ar_replay_line_negative_c_height_value:1" in report["failures"]
    assert "ar_replay_missing_c_height_values:29!=30" in report["failures"]


def test_import_external_evidence_drop_rejects_replay_missing_app_version_and_real_date(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    observations = [
        json.loads(line)
        for line in (drop / "ar_3d_replay" / "ar_replay.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for observation in observations:
        observation["capture_app_version"] = "FILL_ME"
        observation["capture_date_utc"] = "2026-99-99"
    (drop / "ar_3d_replay" / "ar_replay.jsonl").write_text(
        "\n".join(json.dumps(observation) for observation in observations) + "\n",
        encoding="utf-8",
    )

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "ar_replay_line_missing_capture_app_version:1" in report["failures"]
    assert "ar_replay_line_invalid_capture_date_utc:1" in report["failures"]


def test_import_external_evidence_drop_rejects_wrong_android_input_contract(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    report_path = drop / "android_litert_device_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["input"] = {"shape": [1, 3, 640, 640], "dtype": "uint8", "profile": "unknown"}
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "android_report_unexpected_input_shape:[1, 3, 640, 640]" in report["failures"]
    assert "android_report_unexpected_input_dtype:uint8" in report["failures"]
    assert "android_report_unexpected_input_profile:unknown" in report["failures"]


def test_import_external_evidence_drop_rejects_non_integer_android_shapes(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    report_path = drop / "android_litert_device_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["input"]["shape"] = [True, 640, 640, 3]
    payload["output"]["shape"] = [1.0, 14, 8400]
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "android_report_unexpected_input_shape:[True, 640, 640, 3]" in report["failures"]
    assert "android_report_unexpected_output_shape:[1.0, 14, 8400]" in report["failures"]


def test_import_external_evidence_drop_rejects_android_missing_app_version_and_real_date(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    report_path = drop / "android_litert_device_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["test_app_version"] = "FILL_ME"
    payload["test_date_utc"] = "2026-99-99"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "android_report_missing_test_app_version" in report["failures"]
    assert "android_report_invalid_test_date_utc" in report["failures"]


def test_import_external_evidence_drop_rejects_android_latency_outside_runtime_gate(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    report_path = drop / "android_litert_device_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["latency_ms"]["mean"] = 250.0
    payload["latency_ms"]["p95"] = -1.0
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "android_report_mean_latency_high:250.000>120.000" in report["failures"]
    assert "android_report_invalid_p95_latency:-1.000" in report["failures"]


def test_import_external_evidence_drop_allows_android_p95_below_mean(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    report_path = drop / "android_litert_device_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["latency_ms"]["mean"] = 70.0
    payload["latency_ms"]["p95"] = 40.0
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is True
    assert "android_report_p95_latency_less_than_mean:40.000<70.000" not in report["failures"]


def test_import_external_evidence_drop_rejects_android_non_finite_latency_and_memory(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    report_path = drop / "android_litert_device_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["latency_ms"] = {"runs": math.nan, "mean": math.nan, "p95": math.inf}
    payload["memory_mb"] = {"peak": math.nan}
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "android_report_missing_latency_runs" in report["failures"]
    assert "android_report_missing_latency_mean" in report["failures"]
    assert "android_report_missing_latency_p95" in report["failures"]
    assert "android_report_missing_peak_memory" in report["failures"]


def test_import_external_evidence_drop_rejects_fractional_android_latency_runs(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    report_path = drop / "android_litert_device_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["latency_ms"]["runs"] = 30.5
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "android_report_invalid_latency_runs" in report["failures"]


def test_import_external_evidence_drop_rejects_android_degenerate_output_stats(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    report_path = drop / "android_litert_device_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["output"] = {"shape": [1, 14, 8400], "finite": True, "min": 0.0, "max": 0.0, "mean": 0.0}
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "android_report_degenerate_output_range" in report["failures"]


def test_import_external_evidence_drop_rejects_android_output_mean_outside_range(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    report_path = drop / "android_litert_device_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["output"] = {"shape": [1, 14, 8400], "finite": True, "min": -0.1, "max": 1.0, "mean": 1.5}
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "android_report_output_mean_outside_range" in report["failures"]


def test_import_external_evidence_drop_rejects_future_dates(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    android_report_path = drop / "android_litert_device_report.json"
    android_report = json.loads(android_report_path.read_text(encoding="utf-8"))
    android_report["test_date_utc"] = "2999-01-01"
    android_report_path.write_text(json.dumps(android_report), encoding="utf-8")

    provenance_path = drop / "ar_device_holdout" / "metadata" / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["capture_date_utc"] = "2999-01-01"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    replay_path = drop / "ar_3d_replay" / "ar_replay.jsonl"
    observations = [json.loads(line) for line in replay_path.read_text(encoding="utf-8").splitlines()]
    for observation in observations:
        observation["capture_date_utc"] = "2999-01-01"
    replay_path.write_text("\n".join(json.dumps(obs) for obs in observations) + "\n", encoding="utf-8")

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "android_report_invalid_test_date_utc" in report["failures"]
    assert "ar_holdout_invalid_capture_date_utc" in report["failures"]
    assert "ar_replay_line_invalid_capture_date_utc:1" in report["failures"]


def test_import_external_evidence_drop_rejects_emulator_android_report(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    report_path = drop / "android_litert_device_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["device"]["is_emulator"] = True
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "android_report_device_must_be_physical:True" in report["failures"]


def test_import_external_evidence_drop_rejects_missing_or_nonaccepted_holdout_review_status(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    provenance_path = drop / "ar_device_holdout" / "metadata" / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance.pop("review_status")
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "ar_holdout_invalid_review_status:missing" in report["failures"]

    provenance["review_status"] = "approved"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "ar_holdout_invalid_review_status:approved" in report["failures"]


def test_import_external_evidence_drop_rejects_nonreviewed_holdout_label_type(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    provenance_path = drop / "ar_device_holdout" / "metadata" / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["label_type"] = "human_labelled"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "ar_holdout_invalid_label_type:human_labelled" in report["failures"]


def test_import_external_evidence_drop_rejects_impossible_holdout_capture_date(tmp_path):
    drop = tmp_path / "drop"
    _write_drop(drop)
    provenance_path = drop / "ar_device_holdout" / "metadata" / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["capture_date_utc"] = "2026-99-99"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    report = build_import_report(drop, dest_root=tmp_path / "incoming", dry_run=True, overwrite=False)

    assert report["ok"] is False
    assert "ar_holdout_invalid_capture_date_utc" in report["failures"]
