from __future__ import annotations

from scripts.create_ar_replay_log_template import build_template
from src.validate_ar_replay import ReplayThresholds, build_report


def test_ar_replay_template_has_expected_shape():
    observations = build_template(
        observations=3,
        session_id="s_template",
        capture_device="FILL_ME",
        source_type="FILL_ME_android_ar_device_replay",
    )

    assert len(observations) == 3
    assert observations[0]["schema_version"] == 1
    assert observations[0]["session_id"] == "s_template"
    assert observations[0]["source_type"].startswith("FILL_ME")
    assert observations[0]["capture_app_version"].startswith("FILL_ME")
    assert observations[0]["capture_date_utc"].startswith("FILL_ME")
    assert observations[0]["camera_pose_ref"].startswith("FILL_ME")
    assert set(observations[0]["screen_points"]) == {"a", "b", "c_disc_bottom"}
    assert set(observations[0]["floor_raycast_hits"]) == {"a", "b"}


def test_ar_replay_template_is_not_production_evidence(tmp_path):
    observations = build_template(
        observations=4,
        session_id="s_template",
        capture_device="FILL_ME",
        source_type="FILL_ME_android_ar_device_replay",
    )

    report = build_report(
        observations,
        ReplayThresholds(min_observations=4, min_final_positions=4),
        source=tmp_path / "ar_3d_replay.template.jsonl",
    )

    assert report["ok"] is False
    assert "schema_errors" in report["failures"]
    assert any("camera_pose_ref" in error for error in report["errors"])


def test_filled_ar_replay_template_can_satisfy_validator(tmp_path):
    observations = build_template(
        observations=4,
        session_id="s_device",
        capture_device="Pixel test",
        capture_app_version="1.2.3",
        capture_date_utc="2026-05-27",
        source_type="android_ar_device_replay",
        camera_pose_ref_prefix="pose_s_device",
    )

    report = build_report(
        observations,
        ReplayThresholds(min_observations=4, min_final_positions=4),
        source=tmp_path / "ar_replay.jsonl",
    )

    assert report["ok"] is True
    assert report["counts"]["production_source_observations"] == 4
