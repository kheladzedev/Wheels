"""Headless tests for the prefill / drag-edit helpers added to
``manual_keypoint_annotator``.

The OpenCV mouse loop itself is not unit-tested, but the pure helpers
that it consumes (``load_draft_wheels``, ``strip_draft_wheel``,
``find_hit_keypoint``, ``find_hit_bbox``, ``apply_keypoint_drag``) are
each exercised end-to-end so a regression cannot silently corrupt the
prefill flow.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from manual_keypoint_annotator import (
    DRAFT_TOPLEVEL_DROP_KEYS,
    DRAFT_WHEEL_DROP_KEYS,
    DRAG_HIT_RADIUS_PX,
    apply_keypoint_drag,
    find_hit_bbox,
    find_hit_keypoint,
    load_draft_wheels,
    strip_draft_wheel,
)


# ---------------------------------------------------------------------------
# strip_draft_wheel
# ---------------------------------------------------------------------------


def test_strip_draft_wheel_removes_known_draft_keys():
    raw = {
        "bbox_xyxy": [10, 20, 50, 60],
        "points": {"a": [12, 58], "b": [48, 58], "c_disc_bottom": [30, 50]},
        "_detector_conf": 0.81,
        "_vehicle_conf": 0.9,
        "_mask_area_px": 2000,
        "_needs_review": True,
        "_review_reasons": ["mask_small"],
    }
    cleaned = strip_draft_wheel(raw)
    assert set(cleaned.keys()) == {"bbox_xyxy", "points"}
    assert cleaned["bbox_xyxy"] == [10.0, 20.0, 50.0, 60.0]
    assert cleaned["points"] == {
        "a": [12.0, 58.0],
        "b": [48.0, 58.0],
        "c_disc_bottom": [30.0, 50.0],
    }


def test_strip_draft_wheel_drops_unknown_point_keys():
    raw = {
        "bbox_xyxy": [0, 0, 10, 10],
        "points": {
            "a": [1, 9],
            "b": [9, 9],
            "c_disc_bottom": [5, 5],
            "rim_left": [1, 1],  # forbidden by the plugin contract
        },
    }
    cleaned = strip_draft_wheel(raw)
    assert set(cleaned["points"].keys()) == {"a", "b", "c_disc_bottom"}


def test_strip_draft_wheel_known_droplist_matches_constant():
    raw = {key: 1 for key in DRAFT_WHEEL_DROP_KEYS}
    raw.update(
        {
            "bbox_xyxy": [0, 0, 10, 10],
            "points": {"a": [1, 9], "b": [9, 9], "c_disc_bottom": [5, 5]},
        }
    )
    cleaned = strip_draft_wheel(raw)
    for key in DRAFT_WHEEL_DROP_KEYS:
        assert key not in cleaned


# ---------------------------------------------------------------------------
# load_draft_wheels
# ---------------------------------------------------------------------------


def _write_draft(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_draft_wheels_round_trip(tmp_path: Path):
    p = tmp_path / "frame_0001.json"
    _write_draft(
        p,
        {
            "frame_id": "frame_0001",
            "image": "frame_0001.jpg",
            "wheels": [
                {
                    "bbox_xyxy": [100, 200, 160, 260],
                    "points": {
                        "a": [105, 258],
                        "b": [155, 258],
                        "c_disc_bottom": [130, 240],
                    },
                    "_detector_conf": 0.8,
                    "_needs_review": True,
                }
            ],
            "_draft": True,
            "_warning": "NOT_GROUND_TRUTH_REQUIRES_HUMAN_REVIEW",
        },
    )
    wheels = load_draft_wheels(p)
    assert len(wheels) == 1
    w = wheels[0]
    assert set(w.keys()) == {"bbox_xyxy", "points"}
    assert w["bbox_xyxy"] == [100.0, 200.0, 160.0, 260.0]
    assert w["points"]["a"] == [105.0, 258.0]
    # Sanity check: top-level draft markers are validated only conceptually
    # via the wheels list; the helper does not need to surface them.
    assert "_draft" in DRAFT_TOPLEVEL_DROP_KEYS


def test_load_draft_wheels_missing_file_returns_empty(tmp_path: Path):
    assert load_draft_wheels(tmp_path / "nonexistent.json") == []


def test_load_draft_wheels_invalid_json_returns_empty(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    assert load_draft_wheels(p) == []


def test_load_draft_wheels_skips_wheels_missing_required_points(tmp_path: Path):
    p = tmp_path / "partial.json"
    _write_draft(
        p,
        {
            "wheels": [
                {  # ok
                    "bbox_xyxy": [0, 0, 10, 10],
                    "points": {"a": [1, 9], "b": [9, 9], "c_disc_bottom": [5, 5]},
                },
                {  # missing c_disc_bottom
                    "bbox_xyxy": [20, 20, 30, 30],
                    "points": {"a": [21, 29], "b": [29, 29]},
                },
                {  # missing bbox
                    "points": {"a": [1, 1], "b": [2, 2], "c_disc_bottom": [3, 3]},
                },
            ],
        },
    )
    wheels = load_draft_wheels(p)
    assert len(wheels) == 1
    assert wheels[0]["bbox_xyxy"] == [0.0, 0.0, 10.0, 10.0]


def test_load_draft_wheels_real_auto_annotate_shape(tmp_path: Path):
    """Round-trips the exact shape ``auto_annotate_wheels.py`` emits."""
    p = tmp_path / "real_000.json"
    _write_draft(
        p,
        {
            "frame_id": "real_000",
            "image": "real_000.jpg",
            "wheels": [
                {
                    "bbox_xyxy": [434, 601, 492, 649],
                    "points": {
                        "a": [442.0, 644.0],
                        "b": [478.0, 644.0],
                        "c_disc_bottom": [463.0, 639.6],
                    },
                    "_detector_conf": 0.8251,
                    "_vehicle_conf": 0.8927,
                    "_vehicle_class": 2,
                    "_mask_area_px": 2218,
                    "_needs_review": True,
                    "_review_reasons": ["mask_small", "small_bbox"],
                }
            ],
            "_draft": True,
            "_warning": "NOT_GROUND_TRUTH_REQUIRES_HUMAN_REVIEW",
            "_annotation_method": "coco_vehicle+sam2_grid_prompts",
        },
    )
    wheels = load_draft_wheels(p)
    assert len(wheels) == 1
    assert set(wheels[0].keys()) == {"bbox_xyxy", "points"}
    assert "_detector_conf" not in wheels[0]


# ---------------------------------------------------------------------------
# find_hit_keypoint
# ---------------------------------------------------------------------------


@pytest.fixture
def two_wheels() -> list[dict]:
    return [
        {
            "bbox_xyxy": [100.0, 100.0, 160.0, 160.0],
            "points": {
                "a": [105.0, 158.0],
                "b": [155.0, 158.0],
                "c_disc_bottom": [130.0, 140.0],
            },
        },
        {
            "bbox_xyxy": [300.0, 200.0, 380.0, 280.0],
            "points": {
                "a": [305.0, 278.0],
                "b": [375.0, 278.0],
                "c_disc_bottom": [340.0, 260.0],
            },
        },
    ]


def test_find_hit_keypoint_picks_closest_within_radius(two_wheels):
    # Display scale = 1, click 2px from wheel 0 a-point.
    hit = find_hit_keypoint(two_wheels, (107.0, 158.0), scale=1.0)
    assert hit == (0, "a")


def test_find_hit_keypoint_returns_none_outside_radius(two_wheels):
    # Click halfway between wheels — out of every hit radius.
    hit = find_hit_keypoint(two_wheels, (200.0, 200.0), scale=1.0)
    assert hit is None


def test_find_hit_keypoint_respects_display_scale(two_wheels):
    # Image is rendered at 0.5x — wheel-0 a-point is at display (52.5, 79).
    hit = find_hit_keypoint(two_wheels, (52.5, 79.0), scale=0.5)
    assert hit == (0, "a")


def test_find_hit_keypoint_hit_radius_constant_is_sane():
    # Defensive — if someone shrinks this below 6, drag becomes near-unusable
    # at 1.0 scale on a 4-pixel marker.
    assert DRAG_HIT_RADIUS_PX >= 8


# ---------------------------------------------------------------------------
# find_hit_bbox
# ---------------------------------------------------------------------------


def test_find_hit_bbox_finds_containing_wheel(two_wheels):
    assert find_hit_bbox(two_wheels, (130.0, 130.0), scale=1.0) == 0
    assert find_hit_bbox(two_wheels, (340.0, 240.0), scale=1.0) == 1


def test_find_hit_bbox_returns_none_outside_any(two_wheels):
    assert find_hit_bbox(two_wheels, (10.0, 10.0), scale=1.0) is None


def test_find_hit_bbox_prefers_smaller_bbox_on_overlap():
    wheels = [
        {
            "bbox_xyxy": [0.0, 0.0, 200.0, 200.0],
            "points": {"a": [1, 1], "b": [2, 2], "c_disc_bottom": [3, 3]},
        },
        {
            "bbox_xyxy": [80.0, 80.0, 120.0, 120.0],
            "points": {"a": [81, 81], "b": [82, 82], "c_disc_bottom": [83, 83]},
        },
    ]
    # Point (100, 100) is inside both — small wheel should win.
    assert find_hit_bbox(wheels, (100.0, 100.0), scale=1.0) == 1


# ---------------------------------------------------------------------------
# apply_keypoint_drag
# ---------------------------------------------------------------------------


def test_apply_keypoint_drag_moves_one_point(two_wheels):
    out = apply_keypoint_drag(two_wheels, 0, "c_disc_bottom", (135.0, 142.0))
    assert out[0]["points"]["c_disc_bottom"] == [135.0, 142.0]
    # Other points / wheels untouched.
    assert out[0]["points"]["a"] == [105.0, 158.0]
    assert out[1]["points"]["c_disc_bottom"] == [340.0, 260.0]


def test_apply_keypoint_drag_does_not_mutate_input(two_wheels):
    snapshot = json.loads(json.dumps(two_wheels))
    apply_keypoint_drag(two_wheels, 0, "a", (999.0, 999.0))
    assert two_wheels == snapshot


def test_apply_keypoint_drag_rejects_bad_kp_name(two_wheels):
    with pytest.raises(ValueError):
        apply_keypoint_drag(two_wheels, 0, "rim_left", (0.0, 0.0))


def test_apply_keypoint_drag_rejects_out_of_range(two_wheels):
    with pytest.raises(IndexError):
        apply_keypoint_drag(two_wheels, 5, "a", (0.0, 0.0))
