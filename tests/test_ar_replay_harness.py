from __future__ import annotations

from pathlib import Path

from scripts.build_external_evidence_handoff_bundle import DEFAULT_BUNDLE_ARTIFACTS
from src.release_integrity import DEFAULT_REQUIRED_ARTIFACTS


HARNESS_README = Path("ar_replay_harness/README.md")
HARNESS_LOGGER = Path("ar_replay_harness/ArReplayLogger.kt")


def test_ar_replay_harness_documents_production_jsonl_flow():
    text = HARNESS_README.read_text(encoding="utf-8")

    assert "data/incoming/ar_3d_replay/ar_replay.jsonl" in text
    assert "src/validate_ar_replay.py" in text
    assert "raycast hits" in text
    assert "RANSAC labels" in text
    assert "unique `wheelIndex` or" in text
    assert "decreasing" in text
    assert "disc-bottom 3D position" in text


def test_ar_replay_harness_writes_validator_compatible_keys():
    text = HARNESS_LOGGER.read_text(encoding="utf-8")

    assert "class ArReplayLogger" in text
    assert 'SOURCE_TYPE_ANDROID_AR_DEVICE_REPLAY = "android_ar_device_replay"' in text
    assert 'DEFAULT_FILE_NAME = "ar_replay.jsonl"' in text
    assert '"schema_version"' in text
    assert '"capture_app_version"' in text
    assert '"capture_date_utc"' in text
    assert "captureDateUtc must be a real UTC date" in text
    assert "must not be in the future" in text
    assert "LocalDate.now(ZoneOffset.UTC)" in text
    assert '"screen_points"' in text
    assert '"floor_raycast_hits"' in text
    assert "exactly one of cameraTransform or non-blank cameraPoseRef is required" in text
    assert "captureIndex must be non-decreasing within one session" in text
    assert "repeated frame/captureIndex rows require wheelIndex or wheelTrackId" in text
    assert "repeated frame/captureIndex rows require unique wheel identity" in text
    assert '"wheel_index"' in text
    assert '"wheel_track_id"' in text
    assert "wheelIndex must be non-negative when present" in text
    assert "wheelTrackId must be non-blank when present" in text
    assert "floorHits.a and floorHits.b are required" in text
    assert "ransac result is required" in text
    assert "support must be positive for production replay evidence" in text
    assert "recoveredPlane.normal must be a unit vector" in text
    assert "cHeightValue must be finite for production replay evidence" in text
    assert "cHeightValue must be non-negative for production replay evidence" in text
    assert '"c_disc_bottom"' in text
    assert '"inlier"' in text
    assert '"residual"' in text
    assert '"c_plane_hit"' in text
    assert '"c_height_value"' in text
    assert '"final_disc_bottom_position"' in text
    assert "appendText(observation.toString() + \"\\n\")" in text


def test_ar_replay_harness_is_in_handoff_and_release_sets():
    assert str(HARNESS_README) in DEFAULT_BUNDLE_ARTIFACTS
    assert str(HARNESS_LOGGER) in DEFAULT_BUNDLE_ARTIFACTS
    assert str(HARNESS_README) in DEFAULT_REQUIRED_ARTIFACTS
    assert str(HARNESS_LOGGER) in DEFAULT_REQUIRED_ARTIFACTS
