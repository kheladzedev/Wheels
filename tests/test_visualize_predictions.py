"""Tests for confirmed-schema prediction visualization."""

from __future__ import annotations

import pytest

from visualize_predictions import validate_confirmed_payload


def _confirmed_payload(frame_id: str | None = "frame_001") -> dict:
    payload = {
        "wheels": [
            {
                "bbox_xyxy": [10.0, 20.0, 110.0, 120.0],
                "confidence": 0.91,
                "points": {
                    "a": [20.0, 110.0],
                    "b": [100.0, 110.0],
                    "c_disc_bottom": [60.0, 90.0],
                },
            }
        ]
    }
    if frame_id is not None:
        payload["frame_id"] = frame_id
    return payload


def test_visualizer_accepts_confirmed_schema_payload() -> None:
    payload = _confirmed_payload()

    wheels = validate_confirmed_payload(payload, source_label="test.json")

    assert wheels == payload["wheels"]


def test_visualizer_rejects_legacy_wheel_bbox_keypoints_shape() -> None:
    payload = {
        "frame_id": "frame_001",
        "wheels": [
            {
                "wheel_bbox": [10.0, 20.0, 110.0, 120.0],
                "confidence": 0.91,
                "keypoints": [],
            }
        ],
    }

    with pytest.raises(ValueError, match="confirmed schema"):
        validate_confirmed_payload(payload, source_label="legacy.json")


def test_visualizer_strict_mode_requires_frame_id() -> None:
    with pytest.raises(ValueError, match="frame_id"):
        validate_confirmed_payload(
            _confirmed_payload(frame_id=None),
            source_label="no-frame.json",
            require_frame_id=True,
        )
