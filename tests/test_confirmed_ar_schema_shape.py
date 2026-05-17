"""Regression tests for the AR-team confirmed JSON shape (2026-05-13).

These tests pin the shape that `to_confirmed_schema` emits — the
primary `<stem>.json` output the AR client consumes. If a refactor
adds or renames a field, one of these tests must be updated
*explicitly* (with AR-team sign-off in
`docs/OPEN_QUESTIONS_AR_SPEC.md`), not silently by relaxing the
assertion.

Pinned invariants:
  * Top-level keys are exactly {frame_id, wheels}.
  * Each wheel is exactly {bbox_xyxy, confidence, points}.
  * points keys are exactly {a, b, c_disc_bottom}.
  * No 3D / world / RANSAC / plane fields leak into the response.
  * No track_id, no per-keypoint visibility, no per-keypoint confidence.
  * No timestamp (AR matches via frame_id only).
"""

from __future__ import annotations

from postprocess_wheels import build_ar_payload, to_confirmed_schema

ALLOWED_TOP_LEVEL = {"frame_id", "wheels"}
ALLOWED_WHEEL_KEYS = {"bbox_xyxy", "confidence", "points"}
ALLOWED_POINT_KEYS = {"a", "b", "c_disc_bottom"}

# Strings that must NEVER appear as a field name (or substring of one)
# in the confirmed AR JSON. Splits the responsibilities — 3D / RANSAC /
# tracking / plane recovery all live on the AR side.
FORBIDDEN_KEY_SUBSTRINGS = (
    "track_id",
    "track",
    "world",
    "plane",
    "ransac",
    "raycast",
    "intrinsic",
    "extrinsic",
    "imu",
    "depth",
    "z_world",
    "z_axis",
    "3d",
    "visibility",
    "keypoints_confidence",
    "point_confidence",
    "kp_confidence",
    "timestamp",
)


def _two_wheel_legacy_payload() -> dict:
    """Two-wheel fixture with all keypoints visible — mirror of _demo()
    but stripped of occluded second-wheel noise so to_confirmed_schema
    emits both."""
    detections = [
        {
            "class_name": "wheel",
            "bbox": [100, 200, 200, 300],
            "confidence": 0.93,
            "keypoints": [
                {"xy": [110, 288], "visibility": 2, "confidence": 0.95},
                {"xy": [190, 289], "visibility": 2, "confidence": 0.92},
                {"xy": [150, 270], "visibility": 2, "confidence": 0.88},
            ],
        },
        {
            "class_name": "wheel",
            "bbox": [300, 200, 380, 280],
            "confidence": 0.88,
            "keypoints": [
                {"xy": [308, 268], "visibility": 2, "confidence": 0.90},
                {"xy": [372, 269], "visibility": 2, "confidence": 0.85},
                {"xy": [340, 252], "visibility": 2, "confidence": 0.80},
            ],
        },
    ]
    return build_ar_payload(
        detections,
        conf_threshold=0.25,
        frame_id="schema-regression-frame-0001",
    )


def _collect_all_keys(payload: object, into: set[str]) -> None:
    """Walk a JSON-like structure, collecting every dict key into `into`."""
    if isinstance(payload, dict):
        for k, v in payload.items():
            into.add(k)
            _collect_all_keys(v, into)
    elif isinstance(payload, list):
        for item in payload:
            _collect_all_keys(item, into)


def test_confirmed_top_level_keys_are_exact() -> None:
    confirmed = to_confirmed_schema(_two_wheel_legacy_payload())
    assert set(confirmed.keys()) == ALLOWED_TOP_LEVEL


def test_confirmed_top_level_frame_id_is_string() -> None:
    confirmed = to_confirmed_schema(_two_wheel_legacy_payload())
    assert isinstance(confirmed["frame_id"], str)
    assert confirmed["frame_id"] == "schema-regression-frame-0001"


def test_confirmed_wheels_is_list() -> None:
    confirmed = to_confirmed_schema(_two_wheel_legacy_payload())
    assert isinstance(confirmed["wheels"], list)
    # Both demo wheels are fully visible — confirmed schema emits both.
    assert len(confirmed["wheels"]) == 2


def test_confirmed_each_wheel_keys_are_exact() -> None:
    confirmed = to_confirmed_schema(_two_wheel_legacy_payload())
    for i, w in enumerate(confirmed["wheels"]):
        assert set(w.keys()) == ALLOWED_WHEEL_KEYS, (
            f"wheel[{i}] keys {sorted(w.keys())} != {sorted(ALLOWED_WHEEL_KEYS)}"
        )


def test_confirmed_each_wheel_points_keys_are_exact() -> None:
    confirmed = to_confirmed_schema(_two_wheel_legacy_payload())
    for i, w in enumerate(confirmed["wheels"]):
        assert set(w["points"].keys()) == ALLOWED_POINT_KEYS, (
            f"wheel[{i}].points keys {sorted(w['points'].keys())} != "
            f"{sorted(ALLOWED_POINT_KEYS)}"
        )


def test_confirmed_bbox_is_xyxy_four_floats() -> None:
    confirmed = to_confirmed_schema(_two_wheel_legacy_payload())
    for w in confirmed["wheels"]:
        bbox = w["bbox_xyxy"]
        assert isinstance(bbox, list) and len(bbox) == 4
        x1, y1, x2, y2 = bbox
        assert x1 < x2 and y1 < y2
        assert all(isinstance(v, float) for v in bbox)


def test_confirmed_confidence_is_float_in_unit_range() -> None:
    confirmed = to_confirmed_schema(_two_wheel_legacy_payload())
    for w in confirmed["wheels"]:
        c = w["confidence"]
        assert isinstance(c, float)
        assert 0.0 <= c <= 1.0


def test_confirmed_points_are_2d_pixel_pairs() -> None:
    confirmed = to_confirmed_schema(_two_wheel_legacy_payload())
    for w in confirmed["wheels"]:
        for key in ALLOWED_POINT_KEYS:
            xy = w["points"][key]
            assert isinstance(xy, list) and len(xy) == 2
            assert all(isinstance(v, float) for v in xy)


def test_confirmed_schema_has_no_forbidden_keys_anywhere() -> None:
    confirmed = to_confirmed_schema(_two_wheel_legacy_payload())
    all_keys: set[str] = set()
    _collect_all_keys(confirmed, all_keys)
    lowered = {k.lower() for k in all_keys}
    leaks: list[tuple[str, str]] = []
    for key in lowered:
        for needle in FORBIDDEN_KEY_SUBSTRINGS:
            if needle in key:
                leaks.append((key, needle))
    assert not leaks, (
        "Forbidden field name(s) leaked into the confirmed AR schema "
        f"(ML must not emit 3D / tracking / RANSAC / plane / timestamp): {leaks}"
    )


def test_confirmed_schema_empty_wheels_is_valid() -> None:
    """Zero detections → empty wheels list, not a missing field."""
    payload = build_ar_payload([], conf_threshold=0.25, frame_id="empty-frame")
    confirmed = to_confirmed_schema(payload)
    assert set(confirmed.keys()) == ALLOWED_TOP_LEVEL
    assert confirmed["wheels"] == []
    assert confirmed["frame_id"] == "empty-frame"


def test_confirmed_schema_drops_partially_occluded_wheels() -> None:
    """Wheels with any visibility<2 keypoint must NOT appear in the
    confirmed response (the schema represents only fully-visible
    wheels — `visibility` is not part of the response itself)."""
    detections = [
        {
            "class_name": "wheel",
            "bbox": [100, 200, 200, 300],
            "confidence": 0.93,
            "keypoints": [
                {"xy": [110, 288], "visibility": 2, "confidence": 0.95},
                {"xy": [190, 289], "visibility": 2, "confidence": 0.92},
                {"xy": [150, 270], "visibility": 2, "confidence": 0.88},
            ],
        },
        {
            "class_name": "wheel",
            "bbox": [300, 200, 380, 280],
            "confidence": 0.88,
            "keypoints": [
                {"xy": [308, 268], "visibility": 2, "confidence": 0.90},
                {"xy": [372, 269], "visibility": 1, "confidence": 0.55},
                {"xy": [340, 252], "visibility": 2, "confidence": 0.80},
            ],
        },
    ]
    payload = build_ar_payload(detections, conf_threshold=0.25, frame_id="occl-frame")
    confirmed = to_confirmed_schema(payload)
    assert len(confirmed["wheels"]) == 1
