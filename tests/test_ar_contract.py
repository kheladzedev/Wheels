"""Contract tests pinning the AR JSON schemas.

These tests are *shape* guards on the load-bearing ML -> AR contract, not
behavior tests. The behavior of `build_ar_payload` / `to_target_schema`
is covered by `tests/test_postprocess_wheels.py`. Anything here that
fails means a payload field, key, type, ordering, or omission has
drifted from what the AR client agreed to consume.

Two halves:

  - The **current (legacy) AR payload** produced by `build_ar_payload`.
    This is what `infer_image.py` writes today.
  - The **target AR payload** produced by `to_target_schema(...)`, per
    `docs/AR_ML_CONTRACT.md` and `docs/OPEN_QUESTIONS_AR_SPEC.md`.

Any breakage here requires explicit AR-team sign-off before merge.
"""

from __future__ import annotations

from typing import Sequence

from postprocess_wheels import (
    INTERNAL_TO_TARGET_KP,
    KEYPOINT_NAMES,
    N_KEYPOINTS,
    build_ar_payload,
    to_confirmed_schema,
    to_target_schema,
)


# ---------------------------------------------------------------------------
# Helpers — mirrors the pattern in tests/test_postprocess_wheels.py.
# ---------------------------------------------------------------------------


def _wheel(
    bbox: Sequence[float],
    conf: float,
    kp_xys: Sequence[Sequence[float]],
    kp_visibilities: Sequence[int] | None = None,
    kp_confs: Sequence[float | None] | None = None,
) -> dict:
    kp_visibilities = (
        list(kp_visibilities) if kp_visibilities is not None else [2] * N_KEYPOINTS
    )
    kp_confs = list(kp_confs) if kp_confs is not None else [0.9] * N_KEYPOINTS
    kps: list[dict] = []
    for xy, v, c in zip(kp_xys, kp_visibilities, kp_confs):
        kp: dict = {"xy": list(xy), "visibility": v}
        # Per `build_ar_payload`, the per-kp confidence key is optional —
        # absence is the signal for "None" downstream. Allow callers to
        # pass `None` to exercise that path.
        if c is not None:
            kp["confidence"] = c
        kps.append(kp)
    return {
        "class_name": "wheel",
        "bbox": list(bbox),
        "confidence": conf,
        "keypoints": kps,
    }


def _fully_loaded_payload() -> dict:
    """One wheel, all visibility=2, all confidences set, frame metadata set.

    The shape-pinning baseline — every optional path is exercised.
    """
    det = _wheel(
        bbox=(10, 20, 60, 80),
        conf=0.93,
        kp_xys=[(15, 30), (55, 75), (35, 79)],
        kp_visibilities=[2, 2, 2],
        kp_confs=[0.91, 0.90, 0.88],
    )
    return build_ar_payload([det], frame_id="frame_0001", timestamp=123.456)


# ---------------------------------------------------------------------------
# Current AR payload — top-level shape
# ---------------------------------------------------------------------------


def test_current_top_level_keys_with_frame_metadata():
    payload = _fully_loaded_payload()
    # No `image`, `image_size`, or `thresholds` — those are infer_image.py
    # add-ons, not part of `build_ar_payload`'s output.
    assert set(payload.keys()) == {"wheels", "stats", "frame_id", "timestamp"}


def test_current_top_level_keys_without_frame_metadata():
    det = _wheel((0, 0, 10, 10), 0.9, [(1, 1), (2, 2), (3, 3)])
    payload = build_ar_payload([det])
    assert set(payload.keys()) == {"wheels", "stats"}
    assert "frame_id" not in payload
    assert "timestamp" not in payload


def test_current_wheels_is_list_stats_is_dict_with_n_wheels_only():
    payload = _fully_loaded_payload()
    assert isinstance(payload["wheels"], list)
    assert isinstance(payload["stats"], dict)
    assert set(payload["stats"].keys()) == {"n_wheels"}
    assert isinstance(payload["stats"]["n_wheels"], int)
    assert payload["stats"]["n_wheels"] == len(payload["wheels"])


# ---------------------------------------------------------------------------
# Current AR payload — per-wheel shape
# ---------------------------------------------------------------------------


def test_current_wheel_has_exact_keyset():
    payload = _fully_loaded_payload()
    w = payload["wheels"][0]
    assert set(w.keys()) == {"wheel_bbox", "keypoints", "confidence", "warnings"}


def test_current_wheel_bbox_is_list_of_four_floats():
    payload = _fully_loaded_payload()
    bbox = payload["wheels"][0]["wheel_bbox"]
    assert isinstance(bbox, list)
    assert len(bbox) == 4
    for v in bbox:
        # Pin "float, not int" — `build_ar_payload` runs `float(...)` on
        # each coord so AR's deserializer can assume a uniform type.
        assert isinstance(v, float)
        assert not isinstance(v, bool)


def test_current_wheel_confidence_is_float_in_unit_interval():
    payload = _fully_loaded_payload()
    conf = payload["wheels"][0]["confidence"]
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0


def test_current_wheel_warnings_is_list():
    payload = _fully_loaded_payload()
    warnings = payload["wheels"][0]["warnings"]
    assert isinstance(warnings, list)
    # Empty by default — pinned so a refactor that starts emitting
    # default warnings has to update this test.
    assert warnings == []


# ---------------------------------------------------------------------------
# Current AR payload — keypoints shape
# ---------------------------------------------------------------------------


def test_current_keypoints_length_is_exactly_three():
    payload = _fully_loaded_payload()
    kps = payload["wheels"][0]["keypoints"]
    assert isinstance(kps, list)
    assert len(kps) == N_KEYPOINTS == 3


def test_current_keypoint_names_in_canonical_order():
    payload = _fully_loaded_payload()
    names = [kp["name"] for kp in payload["wheels"][0]["keypoints"]]
    # Internal naming, the load-bearing 0/1/2 index identity.
    assert names == ["rim_left", "rim_right", "disc_bottom"]
    assert tuple(names) == KEYPOINT_NAMES


def test_current_keypoint_has_exact_keyset():
    payload = _fully_loaded_payload()
    for kp in payload["wheels"][0]["keypoints"]:
        assert set(kp.keys()) == {"name", "xy", "visibility", "confidence"}


def test_current_keypoint_xy_is_list_of_two_floats():
    payload = _fully_loaded_payload()
    for kp in payload["wheels"][0]["keypoints"]:
        xy = kp["xy"]
        assert isinstance(xy, list)
        assert len(xy) == 2
        for v in xy:
            assert isinstance(v, float)
            assert not isinstance(v, bool)


def test_current_keypoint_visibility_is_int_in_coco_set():
    payload = _fully_loaded_payload()
    for kp in payload["wheels"][0]["keypoints"]:
        vis = kp["visibility"]
        assert isinstance(vis, int)
        assert not isinstance(vis, bool)
        assert vis in {0, 1, 2}


def test_current_keypoint_confidence_is_float_in_unit_interval_or_none():
    payload = _fully_loaded_payload()
    for kp in payload["wheels"][0]["keypoints"]:
        c = kp["confidence"]
        # Allowed to be None (model didn't emit per-kp confidence) — pin
        # that as a legal value.
        assert c is None or isinstance(c, float)
        if isinstance(c, float):
            assert 0.0 <= c <= 1.0


def test_current_keypoint_confidence_none_is_legal():
    """If a detection omits per-kp `confidence`, the slot survives as None.

    Pins the "no per-kp confidence" code path that `build_ar_payload`
    handles in the `if "confidence" in kp` branch.
    """
    det = _wheel(
        (0, 0, 10, 10),
        0.9,
        kp_xys=[(1, 1), (2, 2), (3, 3)],
        kp_confs=[None, None, None],
    )
    payload = build_ar_payload([det])
    for kp in payload["wheels"][0]["keypoints"]:
        assert kp["confidence"] is None


# ---------------------------------------------------------------------------
# Target AR payload — top-level shape
# ---------------------------------------------------------------------------


def test_target_top_level_keys_with_frame_metadata():
    target = to_target_schema(_fully_loaded_payload())
    assert set(target.keys()) == {"wheels", "frame_id", "timestamp"}
    # Explicit drop list — these were carried over for debug only.
    for dropped in ("stats", "image", "image_size", "thresholds"):
        assert dropped not in target


def test_target_top_level_keys_without_frame_metadata():
    det = _wheel((0, 0, 10, 10), 0.9, [(1, 1), (2, 2), (3, 3)])
    payload = build_ar_payload([det])
    target = to_target_schema(payload)
    assert set(target.keys()) == {"wheels"}
    assert "frame_id" not in target
    assert "timestamp" not in target


# ---------------------------------------------------------------------------
# Target AR payload — per-wheel shape
# ---------------------------------------------------------------------------


def test_target_wheel_has_exact_keyset():
    target = to_target_schema(_fully_loaded_payload())
    w = target["wheels"][0]
    assert set(w.keys()) == {
        "bbox_xywh",
        "confidence",
        "keypoints",
        "keypoints_confidence",
        "visibility",
    }


def test_target_wheel_bbox_xywh_is_list_of_four_floats():
    target = to_target_schema(_fully_loaded_payload())
    bbox = target["wheels"][0]["bbox_xywh"]
    assert isinstance(bbox, list)
    assert len(bbox) == 4
    for v in bbox:
        assert isinstance(v, float)
        assert not isinstance(v, bool)


def test_target_wheel_confidence_is_float_in_unit_interval():
    target = to_target_schema(_fully_loaded_payload())
    conf = target["wheels"][0]["confidence"]
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0


def test_target_keypoints_groups_are_dicts():
    target = to_target_schema(_fully_loaded_payload())
    w = target["wheels"][0]
    # Per §9 — three parallel dicts (NOT an array of objects).
    assert isinstance(w["keypoints"], dict)
    assert isinstance(w["keypoints_confidence"], dict)
    assert isinstance(w["visibility"], dict)


def test_target_keypoints_groups_share_exact_key_order():
    target = to_target_schema(_fully_loaded_payload())
    w = target["wheels"][0]
    expected = ("point_a", "point_b", "point_c_disc_bottom")
    # Order is fixed — AR's deserializer reads positionally in some paths.
    assert tuple(w["keypoints"].keys()) == expected
    assert tuple(w["keypoints_confidence"].keys()) == expected
    assert tuple(w["visibility"].keys()) == expected


def test_target_keypoints_values_per_name():
    target = to_target_schema(_fully_loaded_payload())
    w = target["wheels"][0]
    for name in ("point_a", "point_b", "point_c_disc_bottom"):
        xy = w["keypoints"][name]
        assert isinstance(xy, list)
        assert len(xy) == 2
        for v in xy:
            assert isinstance(v, float)
            assert not isinstance(v, bool)

        c = w["keypoints_confidence"][name]
        assert c is None or isinstance(c, float)
        if isinstance(c, float):
            assert 0.0 <= c <= 1.0

        vis = w["visibility"][name]
        assert isinstance(vis, int)
        assert not isinstance(vis, bool)
        assert vis in {0, 1, 2}


def test_target_keypoint_confidence_none_is_legal():
    """`None` per-kp confidence survives the schema conversion."""
    det = _wheel(
        (0, 0, 10, 10),
        0.9,
        kp_xys=[(1, 1), (2, 2), (3, 3)],
        kp_confs=[None, None, None],
    )
    payload = build_ar_payload([det])
    target = to_target_schema(payload)
    w = target["wheels"][0]
    for name in ("point_a", "point_b", "point_c_disc_bottom"):
        assert w["keypoints_confidence"][name] is None


# ---------------------------------------------------------------------------
# Numerical-fidelity round-trip
# ---------------------------------------------------------------------------


def test_round_trip_frame_metadata_is_bit_for_bit_equal():
    payload = _fully_loaded_payload()
    target = to_target_schema(payload)
    assert target["frame_id"] == payload["frame_id"]
    assert target["timestamp"] == payload["timestamp"]


def test_round_trip_bbox_xyxy_to_xywh_is_bit_for_bit():
    payload = _fully_loaded_payload()
    target = to_target_schema(payload)
    for w_cur, w_tgt in zip(payload["wheels"], target["wheels"]):
        x1, y1, x2, y2 = w_cur["wheel_bbox"]
        assert w_tgt["bbox_xywh"] == [x1, y1, x2 - x1, y2 - y1]


def test_round_trip_wheel_confidence_preserved():
    payload = _fully_loaded_payload()
    target = to_target_schema(payload)
    for w_cur, w_tgt in zip(payload["wheels"], target["wheels"]):
        assert w_tgt["confidence"] == w_cur["confidence"]


def test_round_trip_keypoint_fields_map_by_name():
    """For each (internal_name -> target_name), xy/conf/vis match positionally."""
    payload = _fully_loaded_payload()
    target = to_target_schema(payload)
    for w_cur, w_tgt in zip(payload["wheels"], target["wheels"]):
        # Build an index by internal name so we can compare positionally
        # without depending on KEYPOINT_NAMES ordering twice.
        cur_by_name = {kp["name"]: kp for kp in w_cur["keypoints"]}
        for internal_name, target_name in INTERNAL_TO_TARGET_KP.items():
            cur = cur_by_name[internal_name]
            assert w_tgt["keypoints"][target_name] == cur["xy"]
            assert w_tgt["keypoints_confidence"][target_name] == cur["confidence"]
            assert w_tgt["visibility"][target_name] == cur["visibility"]


# ---------------------------------------------------------------------------
# Negative invariants — fields AR will NEVER see
# ---------------------------------------------------------------------------

_FORBIDDEN_WHEEL_FIELDS = (
    "track_id",  # §5 — AR owns cross-frame association.
    "world_xyz",  # 3D leak.
    "plane",  # 3D leak.
    "plane_normal",  # 3D leak.
)

_FORBIDDEN_KP_FIELDS = (
    "world_xyz",
    "depth",
)


def test_current_wheel_does_not_leak_3d_or_tracking_fields():
    payload = _fully_loaded_payload()
    w = payload["wheels"][0]
    for forbidden in _FORBIDDEN_WHEEL_FIELDS:
        assert forbidden not in w, (
            f"Current schema leaked '{forbidden}' — AR owns tracking and 3D. "
            f"Coordinate with AR before adding."
        )


def test_current_keypoint_does_not_leak_3d_fields():
    payload = _fully_loaded_payload()
    for kp in payload["wheels"][0]["keypoints"]:
        for forbidden in _FORBIDDEN_KP_FIELDS:
            assert forbidden not in kp, (
                f"Current keypoint leaked '{forbidden}' — 3D is AR's job."
            )


def test_target_wheel_does_not_leak_3d_or_tracking_fields():
    target = to_target_schema(_fully_loaded_payload())
    w = target["wheels"][0]
    for forbidden in _FORBIDDEN_WHEEL_FIELDS:
        assert forbidden not in w
    # Renamed-away field — the target uses `bbox_xywh`, not `wheel_bbox`.
    assert "wheel_bbox" not in w
    # `warnings` was an explicit drop in §10.
    assert "warnings" not in w


def test_target_keypoint_groups_do_not_leak_3d_fields():
    target = to_target_schema(_fully_loaded_payload())
    w = target["wheels"][0]
    for name in ("point_a", "point_b", "point_c_disc_bottom"):
        # The dict values are lists/floats/ints, not nested objects —
        # but pin the absence of nested 3D info via the parent dict's
        # value type to make the intent obvious.
        assert isinstance(w["keypoints"][name], list)
        assert all(isinstance(v, float) for v in w["keypoints"][name])


# ---------------------------------------------------------------------------
# Visibility = 0 contract — slot is preserved, never dropped
# ---------------------------------------------------------------------------


def test_current_keeps_invisible_keypoint_slot():
    det = _wheel(
        (0, 0, 10, 10),
        0.9,
        kp_xys=[(1, 1), (2, 2), (3, 3)],
        kp_visibilities=[2, 1, 0],
        kp_confs=[0.9, 0.5, 0.1],
    )
    payload = build_ar_payload([det])
    kps = payload["wheels"][0]["keypoints"]
    # All three slots are present in the list — AR decides what to do
    # with visibility=0, but ML must always emit the slot.
    assert len(kps) == N_KEYPOINTS
    assert kps[2]["name"] == "disc_bottom"
    assert kps[2]["visibility"] == 0
    # `xy` is preserved as-is — `build_ar_payload` does not zero it out
    # for invisible keypoints. Pin that so a "helpful" refactor doesn't
    # start silently overwriting coordinates.
    assert "xy" in kps[2]


def test_target_keeps_invisible_keypoint_slot():
    det = _wheel(
        (0, 0, 10, 10),
        0.9,
        kp_xys=[(1, 1), (2, 2), (3, 3)],
        kp_visibilities=[2, 1, 0],
        kp_confs=[0.9, 0.5, 0.1],
    )
    payload = build_ar_payload([det])
    target = to_target_schema(payload)
    w = target["wheels"][0]
    # All three target-named slots present in all three parallel dicts.
    for group in ("keypoints", "keypoints_confidence", "visibility"):
        assert set(w[group].keys()) == {
            "point_a",
            "point_b",
            "point_c_disc_bottom",
        }
    assert w["visibility"]["point_c_disc_bottom"] == 0


# ---------------------------------------------------------------------------
# Confirmed schema (AR-team response 2026-05-13)
# Tight shape pins for `to_confirmed_schema` — the post-confirmation
# target documented in `docs/AR_ML_CONTRACT.md` "JSON shape".
# ---------------------------------------------------------------------------


def test_confirmed_top_level_keys_with_frame_id():
    confirmed = to_confirmed_schema(_fully_loaded_payload())
    assert set(confirmed.keys()) == {"frame_id", "wheels"}


def test_confirmed_top_level_keys_without_frame_id():
    det = _wheel((0, 0, 10, 10), 0.9, [(1, 1), (2, 2), (3, 3)])
    payload = build_ar_payload([det])
    confirmed = to_confirmed_schema(payload)
    assert set(confirmed.keys()) == {"wheels"}


def test_confirmed_no_timestamp_anywhere():
    confirmed = to_confirmed_schema(_fully_loaded_payload())
    assert "timestamp" not in confirmed


def test_confirmed_per_wheel_keys_exactly():
    confirmed = to_confirmed_schema(_fully_loaded_payload())
    w = confirmed["wheels"][0]
    assert set(w.keys()) == {"bbox_xyxy", "confidence", "points"}


def test_confirmed_points_dict_has_three_specific_keys_in_order():
    confirmed = to_confirmed_schema(_fully_loaded_payload())
    w = confirmed["wheels"][0]
    # Order is fixed: a, b, c_disc_bottom. Mirrors the canonical kp index
    # order from KEYPOINT_NAMES.
    assert tuple(w["points"].keys()) == ("a", "b", "c_disc_bottom")


def test_confirmed_points_values_are_float_pairs():
    confirmed = to_confirmed_schema(_fully_loaded_payload())
    w = confirmed["wheels"][0]
    for name in ("a", "b", "c_disc_bottom"):
        xy = w["points"][name]
        assert isinstance(xy, list)
        assert len(xy) == 2
        for v in xy:
            assert isinstance(v, float)
            assert not isinstance(v, bool)


def test_confirmed_bbox_xyxy_is_four_floats():
    confirmed = to_confirmed_schema(_fully_loaded_payload())
    bbox = confirmed["wheels"][0]["bbox_xyxy"]
    assert isinstance(bbox, list)
    assert len(bbox) == 4
    for v in bbox:
        assert isinstance(v, float)
        assert not isinstance(v, bool)


def test_confirmed_confidence_in_zero_one_range():
    confirmed = to_confirmed_schema(_fully_loaded_payload())
    conf = confirmed["wheels"][0]["confidence"]
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0


def test_confirmed_dropped_fields_absent():
    confirmed = to_confirmed_schema(_fully_loaded_payload())
    w = confirmed["wheels"][0]
    # Per-wheel fields explicitly dropped in the confirmed schema.
    for dropped in (
        "wheel_bbox",
        "bbox_xywh",
        "keypoints",
        "keypoints_confidence",
        "visibility",
        "warnings",
    ):
        assert dropped not in w, f"Confirmed wheel leaked '{dropped}'."
    # Top-level fields the converter must drop.
    for dropped in ("stats", "image", "image_size", "thresholds"):
        assert dropped not in confirmed, f"Confirmed top level leaked '{dropped}'."


def test_confirmed_forbidden_3d_and_tracking_fields_absent():
    confirmed = to_confirmed_schema(_fully_loaded_payload())
    w = confirmed["wheels"][0]
    for forbidden in _FORBIDDEN_WHEEL_FIELDS:
        assert forbidden not in w, (
            f"Confirmed schema leaked '{forbidden}' — AR owns tracking and 3D."
        )
    # Same invariant applied to the per-point xy entries.
    for name in ("a", "b", "c_disc_bottom"):
        xy = w["points"][name]
        # Pure list — no nested dict with 3D fields.
        assert isinstance(xy, list)
        for forbidden in _FORBIDDEN_KP_FIELDS:
            # Defensive: ensure xy isn't a dict accidentally carrying 3D.
            assert not isinstance(xy, dict) or forbidden not in xy


def test_confirmed_skips_occluded_wheels():
    fully_visible = _wheel(
        bbox=(0, 0, 100, 100),
        conf=0.95,
        kp_xys=[(10, 10), (90, 10), (50, 95)],
        kp_visibilities=[2, 2, 2],
        kp_confs=[0.9, 0.9, 0.9],
    )
    partially_occluded = _wheel(
        bbox=(200, 200, 300, 300),
        conf=0.92,
        kp_xys=[(210, 210), (290, 210), (250, 295)],
        kp_visibilities=[2, 0, 2],
        kp_confs=[0.9, 0.1, 0.9],
    )
    payload = build_ar_payload(
        [fully_visible, partially_occluded], frame_id="f", timestamp=1.0
    )
    confirmed = to_confirmed_schema(payload)
    # Only the fully-visible wheel survives in the confirmed schema.
    assert len(confirmed["wheels"]) == 1
    assert confirmed["wheels"][0]["bbox_xyxy"] == [0.0, 0.0, 100.0, 100.0]


# ---------------------------------------------------------------------------
# Confirmed schema — explicit negative invariants per the AR-team contract
# (goal §2: track_id / timestamp / 3D / per-kp confidence / visibility /
# thresholds / stats / warnings / image_size / image path all forbidden).
# ---------------------------------------------------------------------------


_CONFIRMED_FORBIDDEN_TOP_LEVEL = (
    "timestamp",
    "stats",
    "thresholds",
    "image",
    "image_size",
    "warnings",
    "track_id",
    "world_xyz",
    "depth",
    "plane",
    "plane_normal",
)

_CONFIRMED_FORBIDDEN_PER_WHEEL = (
    "track_id",
    "timestamp",
    "visibility",
    "keypoints",
    "keypoints_confidence",
    "points_confidence",
    "warnings",
    "stats",
    "image",
    "image_size",
    "thresholds",
    "world_xyz",
    "depth",
    "plane",
    "plane_normal",
    "wheel_bbox",
    "bbox_xywh",
)


def test_confirmed_no_forbidden_top_level_field():
    confirmed = to_confirmed_schema(_fully_loaded_payload())
    for forbidden in _CONFIRMED_FORBIDDEN_TOP_LEVEL:
        assert forbidden not in confirmed, (
            f"Confirmed schema leaked top-level '{forbidden}'."
        )


def test_confirmed_no_forbidden_per_wheel_field():
    confirmed = to_confirmed_schema(_fully_loaded_payload())
    w = confirmed["wheels"][0]
    for forbidden in _CONFIRMED_FORBIDDEN_PER_WHEEL:
        assert forbidden not in w, f"Confirmed wheel leaked '{forbidden}'."


def test_confirmed_uses_bbox_xyxy_not_bbox_xywh():
    confirmed = to_confirmed_schema(_fully_loaded_payload())
    w = confirmed["wheels"][0]
    assert "bbox_xyxy" in w
    assert "bbox_xywh" not in w
    x1, y1, x2, y2 = w["bbox_xyxy"]
    # xyxy: x2 > x1, y2 > y1. xywh would have x2 < x1 + width.
    assert x2 > x1 and y2 > y1


def test_confirmed_points_contains_only_a_b_c_disc_bottom():
    confirmed = to_confirmed_schema(_fully_loaded_payload())
    w = confirmed["wheels"][0]
    assert set(w["points"].keys()) == {"a", "b", "c_disc_bottom"}
    # No legacy / target names leak through.
    for legacy_name in ("rim_left", "rim_right", "disc_bottom"):
        assert legacy_name not in w["points"]
    for target_name in ("point_a", "point_b", "point_c_disc_bottom"):
        assert target_name not in w["points"]


def test_confirmed_point_value_is_two_floats_no_confidence_no_visibility():
    """Per-point payload must be just [x, y]. No dict, no extra fields."""
    confirmed = to_confirmed_schema(_fully_loaded_payload())
    w = confirmed["wheels"][0]
    for name in ("a", "b", "c_disc_bottom"):
        v = w["points"][name]
        # Pure 2-float list — never a dict with confidence/visibility/depth.
        assert isinstance(v, list)
        assert len(v) == 2
        assert all(isinstance(coord, float) for coord in v)
        assert not any(isinstance(coord, bool) for coord in v)


def test_confirmed_frame_id_value_is_preserved_verbatim():
    payload = _fully_loaded_payload()
    confirmed = to_confirmed_schema(payload)
    assert confirmed["frame_id"] == "frame_0001"


def test_confirmed_frame_id_must_be_present_when_inference_supplies_it():
    """Inference always supplies a frame_id (explicit or derived from the
    image stem — see infer_image.determine_frame_id). The confirmed
    converter must propagate it unchanged.
    """
    det = _wheel((0, 0, 10, 10), 0.9, [(1, 1), (2, 2), (3, 3)])
    payload = build_ar_payload([det], frame_id="img_001", timestamp=None)
    confirmed = to_confirmed_schema(payload)
    assert "frame_id" in confirmed
    assert confirmed["frame_id"] == "img_001"


# ---------------------------------------------------------------------------
# infer_image.determine_frame_id — stem-fallback for the confirmed schema.
# ---------------------------------------------------------------------------


def test_determine_frame_id_uses_explicit_when_provided():
    from pathlib import Path

    from infer_image import determine_frame_id

    assert determine_frame_id("frame_0042", Path("foo/bar/sample.jpg")) == "frame_0042"


def test_determine_frame_id_falls_back_to_image_stem():
    from pathlib import Path

    from infer_image import determine_frame_id

    assert determine_frame_id(None, Path("foo/bar/img_001.jpg")) == "img_001"


def test_determine_frame_id_treats_empty_string_as_unset():
    from pathlib import Path

    from infer_image import determine_frame_id

    assert determine_frame_id("", Path("baz/image.png")) == "image"


# ---------------------------------------------------------------------------
# Confirmed schema — full negative sweep against any unknown key
# ---------------------------------------------------------------------------


def test_confirmed_schema_top_level_keyset_is_a_closed_set():
    """The confirmed schema's top-level keys are exactly {frame_id, wheels}
    when frame_id is present, or {wheels} when not. Any *other* key is a
    contract violation.
    """
    confirmed = to_confirmed_schema(_fully_loaded_payload())
    assert confirmed.keys() <= {"frame_id", "wheels"}, (
        f"Unknown top-level key(s): {set(confirmed.keys()) - {'frame_id', 'wheels'}}"
    )


def test_confirmed_schema_per_wheel_keyset_is_a_closed_set():
    """Per-wheel keys are exactly {bbox_xyxy, confidence, points}."""
    confirmed = to_confirmed_schema(_fully_loaded_payload())
    for w in confirmed["wheels"]:
        assert set(w.keys()) == {"bbox_xyxy", "confidence", "points"}, (
            f"Unknown per-wheel keys: {set(w.keys())}"
        )
