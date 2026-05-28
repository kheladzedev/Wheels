from __future__ import annotations

import json
import hashlib

from src.validate_ar_replay import ReplayThresholds, build_report, load_jsonl


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _obs(i: int, *, inlier: bool = True, residual: float = 0.004) -> dict:
    return {
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
        "floor_raycast_hits": {
            "a": [1.0, 0.0, 2.0],
            "b": [1.5, 0.0, 2.0],
        },
        "inlier": inlier,
        "residual": residual,
        "recovered_plane": {
            "normal": [0.998, 0.0, 0.062],
            "point": [1.25, 0.0, 2.0],
            "support": 18,
        },
        "c_plane_hit": [1.25, 0.4, 2.0],
        "c_height_value": 0.4,
        "final_disc_bottom_position": [1.25, 0.4, 2.0],
    }


def test_ar_replay_report_passes_complete_ransac_batch(tmp_path):
    source = tmp_path / "replay.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    observations = [_obs(i) for i in range(4)]

    report = build_report(
        observations,
        ReplayThresholds(min_observations=4, min_final_positions=4),
        source=source,
    )

    assert report["ok"] is True
    assert report["source_sha256"] == _sha(source)
    assert report["counts"]["observations_valid"] == 4
    assert report["metrics"]["floor_hit_rate"] == 1.0
    assert report["metrics"]["inlier_rate"] == 1.0


def test_ar_replay_report_fails_schema_and_quality_gates(tmp_path):
    observations = [_obs(0, inlier=False, residual=0.2), {"session_id": "s1"}]

    report = build_report(
        observations,
        ReplayThresholds(
            min_observations=2,
            min_inlier_rate=0.7,
            max_median_residual=0.02,
            max_p95_residual=0.05,
        ),
        source=tmp_path / "bad.jsonl",
    )

    assert report["ok"] is False
    assert "schema_errors" in report["failures"]
    assert any(f.startswith("inlier_rate_low") for f in report["failures"])
    assert any(f.startswith("median_residual_high") for f in report["failures"])


def test_ar_replay_report_rejects_negative_residual(tmp_path):
    observations = [_obs(i) for i in range(4)]
    observations[0]["residual"] = -0.001

    report = build_report(
        observations,
        ReplayThresholds(min_observations=4, min_final_positions=4),
        source=tmp_path / "negative_residual.jsonl",
    )

    assert report["ok"] is False
    assert "schema_errors" in report["failures"]
    assert any("residual must be non-negative" in error for error in report["errors"])


def test_ar_replay_report_rejects_template_or_synthetic_source(tmp_path):
    observations = [_obs(i) for i in range(4)]
    for obs in observations:
        obs["source_type"] = "FILL_ME_android_ar_device_replay"

    report = build_report(
        observations,
        ReplayThresholds(min_observations=4, min_final_positions=4),
        source=tmp_path / "template.jsonl",
    )

    assert report["ok"] is False
    assert any(f.startswith("missing_production_source") for f in report["failures"])


def test_ar_replay_report_rejects_missing_schema_version(tmp_path):
    observations = [_obs(i) for i in range(4)]
    for obs in observations:
        obs.pop("schema_version")

    report = build_report(
        observations,
        ReplayThresholds(min_observations=4, min_final_positions=4),
        source=tmp_path / "missing_schema.jsonl",
    )

    assert report["ok"] is False
    assert "schema_errors" in report["failures"]
    assert any("unsupported schema_version missing" in error for error in report["errors"])


def test_ar_replay_report_rejects_boolean_schema_version(tmp_path):
    observations = [_obs(i) for i in range(4)]
    observations[0]["schema_version"] = True

    report = build_report(
        observations,
        ReplayThresholds(min_observations=4, min_final_positions=4),
        source=tmp_path / "boolean_schema.jsonl",
    )

    assert report["ok"] is False
    assert "schema_errors" in report["failures"]
    assert any("unsupported schema_version True" in error for error in report["errors"])


def test_ar_replay_report_rejects_placeholder_capture_device(tmp_path):
    observations = [_obs(i) for i in range(4)]
    for obs in observations:
        obs["capture_device"] = "FILL_ME"

    report = build_report(
        observations,
        ReplayThresholds(min_observations=4, min_final_positions=4),
        source=tmp_path / "placeholder_device.jsonl",
    )

    assert report["ok"] is False
    assert "schema_errors" in report["failures"]
    assert any("capture_device" in error for error in report["errors"])


def test_ar_replay_report_requires_capture_app_version_and_real_date(tmp_path):
    observations = [_obs(i) for i in range(4)]
    for obs in observations:
        obs["capture_app_version"] = "FILL_ME"
        obs["capture_date_utc"] = "2026-99-99"

    report = build_report(
        observations,
        ReplayThresholds(min_observations=4, min_final_positions=4),
        source=tmp_path / "bad_provenance.jsonl",
    )

    assert report["ok"] is False
    assert "schema_errors" in report["failures"]
    assert any("capture_app_version" in error for error in report["errors"])
    assert any("capture_date_utc" in error for error in report["errors"])


def test_ar_replay_report_rejects_future_capture_date(tmp_path):
    observations = [_obs(i) for i in range(4)]
    for obs in observations:
        obs["capture_date_utc"] = "2999-01-01"

    report = build_report(
        observations,
        ReplayThresholds(min_observations=4, min_final_positions=4),
        source=tmp_path / "future_capture_date.jsonl",
    )

    assert report["ok"] is False
    assert "schema_errors" in report["failures"]
    assert any("capture_date_utc" in error for error in report["errors"])


def test_ar_replay_report_requires_camera_pose_evidence(tmp_path):
    observations = [_obs(i) for i in range(4)]
    for obs in observations:
        obs["camera_pose_ref"] = "FILL_ME_pose"

    report = build_report(
        observations,
        ReplayThresholds(min_observations=4, min_final_positions=4),
        source=tmp_path / "placeholder_pose.jsonl",
    )

    assert report["ok"] is False
    assert "schema_errors" in report["failures"]
    assert any("camera_pose_ref" in error for error in report["errors"])


def test_ar_replay_report_accepts_inline_camera_transform(tmp_path):
    observations = [_obs(i) for i in range(4)]
    for obs in observations:
        obs["camera_pose_ref"] = None
        obs["camera_transform"] = {"R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [0, 1, 2]}

    report = build_report(
        observations,
        ReplayThresholds(min_observations=4, min_final_positions=4),
        source=tmp_path / "inline_pose.jsonl",
    )

    assert report["ok"] is True


def test_ar_replay_report_rejects_malformed_inline_camera_transform(tmp_path):
    observations = [_obs(i) for i in range(4)]
    for obs in observations:
        obs["camera_pose_ref"] = None
        obs["camera_transform"] = {"foo": "bar"}

    report = build_report(
        observations,
        ReplayThresholds(min_observations=4, min_final_positions=4),
        source=tmp_path / "bad_inline_pose.jsonl",
    )

    assert report["ok"] is False
    assert "schema_errors" in report["failures"]
    assert any("camera_transform" in error for error in report["errors"])


def test_ar_replay_report_requires_recovered_plane_when_ransac_required(tmp_path):
    observations = [_obs(i) for i in range(4)]
    for obs in observations:
        obs.pop("recovered_plane")

    report = build_report(
        observations,
        ReplayThresholds(min_observations=4, min_final_positions=4),
        source=tmp_path / "missing_plane.jsonl",
    )

    assert report["ok"] is False
    assert any(failure.startswith("missing_recovered_planes") for failure in report["failures"])
    assert report["counts"]["recovered_planes"] == 0


def test_ar_replay_report_rejects_malformed_recovered_plane(tmp_path):
    observations = [_obs(i) for i in range(4)]
    for obs in observations:
        obs["recovered_plane"] = {"normal": [1.0, 0.0, 0.0], "point": [1.0, 2.0], "support": 0}

    report = build_report(
        observations,
        ReplayThresholds(min_observations=4, min_final_positions=4),
        source=tmp_path / "bad_plane.jsonl",
    )

    assert report["ok"] is False
    assert "schema_errors" in report["failures"]
    assert any("recovered_plane" in error for error in report["errors"])


def test_ar_replay_report_rejects_non_unit_recovered_plane_normal(tmp_path):
    observations = [_obs(i) for i in range(4)]
    for obs in observations:
        obs["recovered_plane"]["normal"] = [2.0, 0.0, 0.0]

    report = build_report(
        observations,
        ReplayThresholds(min_observations=4, min_final_positions=4),
        source=tmp_path / "non_unit_plane.jsonl",
    )

    assert report["ok"] is False
    assert "schema_errors" in report["failures"]
    assert any("unit normal vec3" in error for error in report["errors"])
    assert report["counts"]["recovered_planes"] == 0


def test_ar_replay_report_requires_c_plane_reconstruction_when_ransac_required(tmp_path):
    observations = [_obs(i) for i in range(4)]
    for obs in observations:
        obs["c_plane_hit"] = None
        obs["c_height_value"] = None

    report = build_report(
        observations,
        ReplayThresholds(min_observations=4, min_final_positions=4),
        source=tmp_path / "missing_c_plane.jsonl",
    )

    assert report["ok"] is False
    assert any(failure.startswith("missing_c_plane_hits") for failure in report["failures"])
    assert any(failure.startswith("missing_c_height_values") for failure in report["failures"])
    assert report["counts"]["c_plane_hits"] == 0
    assert report["counts"]["c_height_values"] == 0


def test_ar_replay_report_rejects_malformed_c_height_value(tmp_path):
    observations = [_obs(i) for i in range(4)]
    for obs in observations:
        obs["c_height_value"] = "bad-height"

    report = build_report(
        observations,
        ReplayThresholds(min_observations=4, min_final_positions=4),
        source=tmp_path / "bad_c_height.jsonl",
    )

    assert report["ok"] is False
    assert "schema_errors" in report["failures"]
    assert any("c_height_value" in error for error in report["errors"])


def test_ar_replay_report_rejects_negative_c_height_value(tmp_path):
    observations = [_obs(i) for i in range(4)]
    observations[0]["c_height_value"] = -0.1

    report = build_report(
        observations,
        ReplayThresholds(min_observations=4, min_final_positions=4),
        source=tmp_path / "negative_c_height.jsonl",
    )

    assert report["ok"] is False
    assert "schema_errors" in report["failures"]
    assert any("c_height_value must be non-negative" in error for error in report["errors"])


def test_ar_replay_report_rejects_invalid_capture_index_values(tmp_path):
    observations = [_obs(i) for i in range(4)]
    observations[0]["capture_index"] = -1
    observations[1]["capture_index"] = True

    report = build_report(
        observations,
        ReplayThresholds(min_observations=4, min_final_positions=4),
        source=tmp_path / "bad_capture_index.jsonl",
    )

    assert report["ok"] is False
    assert "schema_errors" in report["failures"]
    assert any("capture_index must be a non-negative integer" in error for error in report["errors"])
    assert report["counts"]["observations_valid"] == 2


def test_ar_replay_report_rejects_decreasing_capture_index_within_session(tmp_path):
    observations = [_obs(i) for i in range(4)]
    observations[0]["_line_no"] = 1
    observations[1]["_line_no"] = 2
    observations[2]["_line_no"] = 3
    observations[3]["_line_no"] = 4
    observations[2]["capture_index"] = 0

    report = build_report(
        observations,
        ReplayThresholds(min_observations=4, min_final_positions=4),
        source=tmp_path / "decreasing_capture_index.jsonl",
    )

    assert report["ok"] is False
    assert "schema_errors" in report["failures"]
    assert any("capture_index must not decrease within session s1" in error for error in report["errors"])
    assert report["counts"]["observations_valid"] == 4


def test_ar_replay_report_rejects_repeated_frame_without_wheel_identity(tmp_path):
    observations = [_obs(i) for i in range(4)]
    observations[0]["_line_no"] = 1
    observations[1]["_line_no"] = 2
    observations[2]["_line_no"] = 3
    observations[3]["_line_no"] = 4
    observations[2]["capture_index"] = 1
    observations[2]["frame_id"] = "frame_0001"

    report = build_report(
        observations,
        ReplayThresholds(min_observations=4, min_final_positions=4),
        source=tmp_path / "duplicate_frame_without_wheel_identity.jsonl",
    )

    assert report["ok"] is False
    assert "schema_errors" in report["failures"]
    assert any("requires wheel_index or wheel_track_id" in error for error in report["errors"])


def test_ar_replay_report_accepts_repeated_frame_with_unique_wheel_identity(tmp_path):
    observations = [_obs(i) for i in range(4)]
    observations[1]["wheel_index"] = 0
    observations[2]["capture_index"] = 1
    observations[2]["frame_id"] = "frame_0001"
    observations[2]["camera_pose_ref"] = "pose_0001"
    observations[2]["wheel_index"] = 1

    report = build_report(
        observations,
        ReplayThresholds(min_observations=4, min_final_positions=4),
        source=tmp_path / "duplicate_frame_with_wheel_identity.jsonl",
    )

    assert report["ok"] is True


def test_load_jsonl_records_bad_lines(tmp_path):
    path = tmp_path / "replay.jsonl"
    path.write_text(json.dumps(_obs(0)) + "\nnot-json\n[]\n", encoding="utf-8")

    observations = load_jsonl(path)

    assert len(observations) == 3
    assert observations[0]["frame_id"] == "frame_0000"
    assert "_load_error" in observations[1]
    assert "_load_error" in observations[2]
