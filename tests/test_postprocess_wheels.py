"""Tests for the AR JSON payload builder."""

from __future__ import annotations

import postprocess_wheels
from postprocess_wheels import (
    KEYPOINT_NAMES,
    N_KEYPOINTS,
    build_ar_payload,
    to_confirmed_schema,
    visibility_from_keypoint_confidence,
)


def _wheel(bbox, conf, kp_xys, kp_visibilities=None, kp_confs=None):
    kp_visibilities = kp_visibilities or [2] * N_KEYPOINTS
    kp_confs = kp_confs or [0.9] * N_KEYPOINTS
    return {
        "class_name": "wheel",
        "bbox": list(bbox),
        "confidence": conf,
        "keypoints": [
            {"xy": list(xy), "visibility": v, "confidence": c}
            for xy, v, c in zip(kp_xys, kp_visibilities, kp_confs)
        ],
    }


def test_passthrough_preserves_three_keypoints_in_order():
    det = _wheel((0, 0, 100, 100), 0.9, [(10, 10), (50, 90), (50, 95)])
    payload = build_ar_payload([det])

    assert len(payload["wheels"]) == 1
    kps = payload["wheels"][0]["keypoints"]
    assert [kp["name"] for kp in kps] == list(KEYPOINT_NAMES)
    assert kps[0]["xy"] == [10.0, 10.0]
    assert kps[2]["xy"] == [50.0, 95.0]


def test_frame_id_and_timestamp_echoed():
    det = _wheel((0, 0, 10, 10), 0.9, [(1, 1), (2, 2), (3, 3)])
    payload = build_ar_payload([det], frame_id="frame_42", timestamp=1234.5)

    assert payload["frame_id"] == "frame_42"
    assert payload["timestamp"] == 1234.5


def test_frame_metadata_omitted_when_none():
    det = _wheel((0, 0, 10, 10), 0.9, [(1, 1), (2, 2), (3, 3)])
    payload = build_ar_payload([det])

    assert "frame_id" not in payload
    assert "timestamp" not in payload


def test_conf_threshold_drops_low_confidence_wheels():
    detections = [
        _wheel((0, 0, 10, 10), 0.9, [(1, 1), (2, 2), (3, 3)]),
        _wheel((20, 20, 30, 30), 0.1, [(21, 21), (22, 22), (23, 23)]),
    ]
    payload = build_ar_payload(detections, conf_threshold=0.5)
    assert payload["stats"]["n_wheels"] == 1
    assert payload["wheels"][0]["confidence"] == 0.9


def test_wheel_with_wrong_keypoint_count_is_dropped():
    bad = {
        "class_name": "wheel",
        "bbox": [0, 0, 10, 10],
        "confidence": 0.9,
        "keypoints": [{"xy": [1, 1], "visibility": 2, "confidence": 0.9}],  # only 1
    }
    payload = build_ar_payload([bad])
    assert payload["stats"]["n_wheels"] == 0


def test_non_wheel_classes_are_ignored():
    junk = {
        "class_name": "car",
        "bbox": [0, 0, 10, 10],
        "confidence": 0.99,
        "keypoints": [],
    }
    payload = build_ar_payload([junk])
    assert payload["stats"]["n_wheels"] == 0


def test_sort_order_largest_first():
    small = _wheel((0, 0, 10, 10), 0.9, [(1, 1), (2, 2), (3, 3)])
    big = _wheel((0, 0, 100, 100), 0.9, [(10, 10), (50, 90), (50, 95)])
    payload = build_ar_payload([small, big])
    bboxes = [w["wheel_bbox"] for w in payload["wheels"]]
    assert bboxes[0] == [0.0, 0.0, 100.0, 100.0]
    assert bboxes[1] == [0.0, 0.0, 10.0, 10.0]


def test_visibility_passthrough_keeps_occlusion_signal():
    det = _wheel(
        (0, 0, 10, 10), 0.9, kp_xys=[(1, 1), (2, 2), (3, 3)], kp_visibilities=[2, 1, 0]
    )
    payload = build_ar_payload([det])
    vis = [kp["visibility"] for kp in payload["wheels"][0]["keypoints"]]
    assert vis == [2, 1, 0]


def test_deprecated_target_schema_converter_is_removed():
    assert not hasattr(postprocess_wheels, "to_target_schema")
    assert not hasattr(postprocess_wheels, "INTERNAL_TO_TARGET_KP")


def test_visibility_from_keypoint_confidence_preserves_threshold_boundaries():
    assert visibility_from_keypoint_confidence(0.5) == 2
    assert visibility_from_keypoint_confidence(0.499999) == 1
    assert visibility_from_keypoint_confidence(0.15) == 1
    assert visibility_from_keypoint_confidence(0.149999) == 0


# ---------------------------------------------------------------------------
# Confirmed schema (AR-team response 2026-05-13)
# Used by infer_image.py --confirmed-schema.
# ---------------------------------------------------------------------------


def test_to_confirmed_schema_renames_keypoints_to_ab_c_disc_bottom():
    det = _wheel(
        (10, 20, 60, 80),
        0.93,
        kp_xys=[(15, 70), (55, 72), (35, 60)],
        kp_visibilities=[2, 2, 2],
        kp_confs=[0.91, 0.90, 0.88],
    )
    payload = build_ar_payload([det], frame_id="frame_0001", timestamp=123.456)
    confirmed = to_confirmed_schema(payload)

    w = confirmed["wheels"][0]
    assert set(w["points"].keys()) == {"a", "b", "c_disc_bottom"}
    assert w["points"]["a"] == [15.0, 70.0]
    assert w["points"]["b"] == [55.0, 72.0]
    assert w["points"]["c_disc_bottom"] == [35.0, 60.0]


def test_to_confirmed_schema_uses_bbox_xyxy_not_xywh():
    det = _wheel((10, 20, 60, 80), 0.93, kp_xys=[(15, 70), (55, 72), (35, 60)])
    payload = build_ar_payload([det])
    confirmed = to_confirmed_schema(payload)
    w = confirmed["wheels"][0]
    # bbox_xyxy carries the unmodified four corners — not the xywh form.
    assert w["bbox_xyxy"] == [10.0, 20.0, 60.0, 80.0]
    assert "bbox_xywh" not in w


def test_to_confirmed_schema_drops_per_keypoint_metadata():
    det = _wheel(
        (0, 0, 100, 100),
        0.9,
        kp_xys=[(10, 85), (90, 86), (50, 68)],
        kp_visibilities=[2, 2, 2],
        kp_confs=[0.95, 0.92, 0.88],
    )
    payload = build_ar_payload([det])
    confirmed = to_confirmed_schema(payload)
    w = confirmed["wheels"][0]

    # No parallel metadata dicts.
    assert "keypoints_confidence" not in w
    assert "visibility" not in w

    # Each `points[name]` value is bare [x, y] — no nested confidence.
    for name in ("a", "b", "c_disc_bottom"):
        v = w["points"][name]
        assert isinstance(v, list)
        assert len(v) == 2
        # No key called "confidence" inside the xy entries.
        # (`v` is a list, not a dict; confirms structure as well.)
        assert not isinstance(v, dict)


def test_to_confirmed_schema_skips_occluded_wheels():
    fully_visible = _wheel(
        (0, 0, 100, 100),
        0.95,
        kp_xys=[(10, 85), (90, 86), (50, 68)],
        kp_visibilities=[2, 2, 2],
    )
    has_occluded = _wheel(
        (200, 200, 300, 300),
        0.92,
        kp_xys=[(210, 285), (290, 286), (250, 268)],
        kp_visibilities=[2, 0, 2],
    )
    payload = build_ar_payload([fully_visible, has_occluded])
    confirmed = to_confirmed_schema(payload)
    # Only the fully-visible wheel survives.
    assert len(confirmed["wheels"]) == 1
    assert confirmed["wheels"][0]["bbox_xyxy"] == [0.0, 0.0, 100.0, 100.0]


def test_to_confirmed_schema_skips_partially_visible_wheels():
    fully_visible = _wheel(
        (0, 0, 100, 100),
        0.95,
        kp_xys=[(10, 85), (90, 86), (50, 68)],
        kp_visibilities=[2, 2, 2],
    )
    partially_visible = _wheel(
        (200, 200, 300, 300),
        0.92,
        kp_xys=[(210, 285), (290, 286), (250, 268)],
        kp_visibilities=[2, 1, 2],
    )
    payload = build_ar_payload([fully_visible, partially_visible])

    confirmed = to_confirmed_schema(payload)

    assert len(confirmed["wheels"]) == 1
    assert confirmed["wheels"][0]["bbox_xyxy"] == [0.0, 0.0, 100.0, 100.0]


def test_to_confirmed_schema_skips_bad_floor_ray_geometry():
    good = _wheel(
        (0, 0, 100, 100),
        0.95,
        kp_xys=[(10, 85), (90, 86), (50, 68)],
        kp_visibilities=[2, 2, 2],
    )
    bad = _wheel(
        (200, 200, 300, 300),
        0.92,
        # A/B are high in the bbox and C is below the floor-ray line.
        kp_xys=[(210, 230), (290, 235), (250, 295)],
        kp_visibilities=[2, 2, 2],
    )
    payload = build_ar_payload([good, bad])

    confirmed = to_confirmed_schema(payload)

    assert len(confirmed["wheels"]) == 1
    assert confirmed["wheels"][0]["bbox_xyxy"] == [0.0, 0.0, 100.0, 100.0]


def test_to_confirmed_schema_omits_timestamp():
    det = _wheel((0, 0, 100, 100), 0.9, kp_xys=[(10, 85), (90, 86), (50, 68)])
    payload = build_ar_payload([det], frame_id="f", timestamp=123.456)
    confirmed = to_confirmed_schema(payload)
    assert "timestamp" not in confirmed


def test_to_confirmed_schema_empty_wheels():
    payload = build_ar_payload([], frame_id="frame_42", timestamp=0.0)
    confirmed = to_confirmed_schema(payload)
    assert confirmed == {"frame_id": "frame_42", "wheels": []}


def test_to_confirmed_schema_no_frame_id_omits_top_level_key():
    det = _wheel((0, 0, 100, 100), 0.9, kp_xys=[(10, 85), (90, 86), (50, 68)])
    payload = build_ar_payload([det])
    confirmed = to_confirmed_schema(payload)
    assert set(confirmed.keys()) == {"wheels"}


def test_to_confirmed_schema_per_wheel_keys_exactly():
    det = _wheel((0, 0, 100, 100), 0.9, kp_xys=[(10, 85), (90, 86), (50, 68)])
    payload = build_ar_payload([det])
    confirmed = to_confirmed_schema(payload)
    w = confirmed["wheels"][0]
    assert set(w.keys()) == {"bbox_xyxy", "confidence", "points"}


def test_to_confirmed_schema_all_values_are_python_floats():
    det = _wheel((0, 0, 100, 100), 0.9, kp_xys=[(10, 85), (90, 86), (50, 68)])
    payload = build_ar_payload([det])
    confirmed = to_confirmed_schema(payload)
    w = confirmed["wheels"][0]

    for v in w["bbox_xyxy"]:
        assert type(v) is float

    for name in ("a", "b", "c_disc_bottom"):
        xy = w["points"][name]
        for v in xy:
            assert type(v) is float
