"""Tests for the pure helpers in src/auto_annotate_wheels.py.

The model-loading and prediction paths require ultralytics weights on
disk and pull MPS/CUDA at first use, so they are deliberately not
exercised here. The geometric helpers and the drop/flag decisions are
all pure numpy and fully testable.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.auto_annotate_wheels import (
    AB_BAND_FRACTION,
    DEFAULT_DROP_CONF,
    DEFAULT_REVIEW_CONF,
    HARD_MIN_MASK_PX,
    SOFT_MIN_MASK_PX,
    aspect_outside,
    bbox_from_mask,
    bbox_touches_edge,
    keypoints_from_mask,
    mask_circularity,
    mask_touches_edge,
    review_reasons,
    should_drop,
    tire_darkness,
)


# ---------------------------------------------------------------------------
# bbox_from_mask
# ---------------------------------------------------------------------------


def test_bbox_from_mask_empty_returns_none() -> None:
    m = np.zeros((50, 50), dtype=np.uint8)
    assert bbox_from_mask(m) is None


def test_bbox_from_mask_rectangle() -> None:
    m = np.zeros((100, 100), dtype=np.uint8)
    m[20:60, 30:70] = 1
    assert bbox_from_mask(m) == [30, 20, 70, 60]


def test_bbox_from_mask_single_pixel() -> None:
    m = np.zeros((20, 20), dtype=np.uint8)
    m[5, 7] = 1
    assert bbox_from_mask(m) == [7, 5, 8, 6]


def test_bbox_from_mask_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        bbox_from_mask(np.zeros((3, 4, 5), dtype=np.uint8))


# ---------------------------------------------------------------------------
# keypoints_from_mask
# ---------------------------------------------------------------------------


def _circle_mask(h: int, w: int, cx: int, cy: int, r: int) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r).astype(np.uint8)


def test_keypoints_from_mask_circle_geometry() -> None:
    # A side-on wheel: filled disc. Bbox is tight around the disc.
    cx, cy, r = 100, 100, 40
    h, w = 200, 200
    m = _circle_mask(h, w, cx, cy, r)
    bbox = [cx - r, cy - r, cx + r + 1, cy + r + 1]
    kp = keypoints_from_mask(m, bbox)
    assert kp is not None

    a = kp["a"]
    b = kp["b"]
    c = kp["c_disc_bottom"]

    # A is to the left of B.
    assert a[0] < b[0]
    # A and B sit near the lower edge of the bbox (bottom band).
    bottom_band_top = bbox[3] - (bbox[3] - bbox[1]) * AB_BAND_FRACTION
    assert a[1] >= bottom_band_top - 1
    assert b[1] >= bottom_band_top - 1
    # C sits on the vertical centreline (within 1 px slack for int rounding).
    assert abs(c[0] - cx) <= 1
    # C is above the bottom of the bbox (rim above floor contact).
    assert c[1] < bbox[3] - 1


def test_keypoints_from_mask_degenerate_returns_none() -> None:
    h, w = 100, 100
    m = np.zeros((h, w), dtype=np.uint8)
    # Bbox covers a region with no mask.
    assert keypoints_from_mask(m, [10, 10, 50, 50]) is None


def test_keypoints_from_mask_inverted_bbox_returns_none() -> None:
    m = np.ones((40, 40), dtype=np.uint8)
    assert keypoints_from_mask(m, [20, 20, 10, 10]) is None


def test_keypoints_from_mask_C_above_tyre_bottom() -> None:
    """C (disc bottom) sits visibly above A/B (tyre footprint) — spec rule.

    Per ``docs/KEYPOINT_SPEC.md``, C is the lowest visible point of the
    metal rim, not of the rubber tyre. With C_OFFSET_FRACTION = 0.175,
    that means C must be at least 10 % of the bbox height above A and B
    on a clean side-on disc mask.
    """
    cx, cy, r = 100, 100, 50
    m = _circle_mask(200, 200, cx, cy, r)
    bbox = [cx - r, cy - r, cx + r + 1, cy + r + 1]
    kp = keypoints_from_mask(m, bbox)
    assert kp is not None
    c_y = kp["c_disc_bottom"][1]
    a_y = kp["a"][1]
    # Bbox height ≈ 101; A sits ~10 % up from the bottom (AB band) and C
    # sits 17.5 % up from the mask bottom. Net separation is ~8 px on a
    # 101 px bbox, scales linearly with bbox height — pick 5 px as a
    # conservative floor that's also enough to be obvious in a preview.
    assert c_y < a_y - 5, (
        f"C must sit visibly above A/B (tyre footprint), "
        f"got C.y={c_y}, A.y={a_y} (need C < A - 5)"
    )


def test_keypoints_from_mask_points_inside_bbox() -> None:
    # Rectangular mask, exact-fit bbox.
    h, w = 60, 80
    m = np.zeros((h, w), dtype=np.uint8)
    m[10:50, 15:65] = 1
    bbox = [15, 10, 65, 50]
    kp = keypoints_from_mask(m, bbox)
    assert kp is not None
    for name, (px, py) in kp.items():
        assert bbox[0] - 1 <= px <= bbox[2] + 1, f"{name} x outside bbox"
        assert bbox[1] - 1 <= py <= bbox[3] + 1, f"{name} y outside bbox"


# ---------------------------------------------------------------------------
# edge / aspect helpers
# ---------------------------------------------------------------------------


def test_bbox_touches_edge_detects_each_side() -> None:
    shape = (100, 100)
    assert bbox_touches_edge([0, 10, 30, 30], shape)
    assert bbox_touches_edge([10, 0, 30, 30], shape)
    assert bbox_touches_edge([10, 10, 100, 30], shape)
    assert bbox_touches_edge([10, 10, 30, 100], shape)
    assert not bbox_touches_edge([10, 10, 30, 30], shape)


def test_mask_touches_edge_detects_each_side() -> None:
    m = np.zeros((50, 50), dtype=np.uint8)
    m[0, 25] = 1
    assert mask_touches_edge(m)

    m = np.zeros((50, 50), dtype=np.uint8)
    m[20:30, 20:30] = 1
    assert not mask_touches_edge(m)


def test_aspect_outside_thresholds() -> None:
    # Square — well within.
    assert not aspect_outside([0, 0, 100, 100])
    # Very wide.
    assert aspect_outside([0, 0, 300, 50])
    # Very tall.
    assert aspect_outside([0, 0, 50, 300])
    # Degenerate.
    assert aspect_outside([0, 0, 0, 0])


# ---------------------------------------------------------------------------
# review_reasons / should_drop
# ---------------------------------------------------------------------------


def _decent_mask() -> np.ndarray:
    m = np.zeros((200, 200), dtype=np.uint8)
    m[60:140, 60:140] = 1
    return m


def test_review_reasons_high_conf_centered_clean() -> None:
    m = _decent_mask()
    reasons = review_reasons(
        detector_conf=0.9,
        mask=m,
        bbox=[60, 60, 140, 140],
        image_shape=(200, 200),
    )
    assert reasons == []


def test_review_reasons_low_conf_flagged() -> None:
    m = _decent_mask()
    reasons = review_reasons(
        detector_conf=0.35,
        mask=m,
        bbox=[60, 60, 140, 140],
        image_shape=(200, 200),
    )
    assert "low_detector_conf" in reasons


def test_review_reasons_small_mask_flagged() -> None:
    m = np.zeros((200, 200), dtype=np.uint8)
    m[100:105, 100:105] = 1  # 25 px, well below SOFT_MIN_MASK_PX
    reasons = review_reasons(
        detector_conf=0.9,
        mask=m,
        bbox=[100, 100, 105, 105],
        image_shape=(200, 200),
    )
    assert "mask_small" in reasons


def test_review_reasons_edge_touch_flagged() -> None:
    m = np.zeros((200, 200), dtype=np.uint8)
    m[0:80, 60:140] = 1  # touches top
    reasons = review_reasons(
        detector_conf=0.9,
        mask=m,
        bbox=[60, 0, 140, 80],
        image_shape=(200, 200),
    )
    assert "mask_touches_edge" in reasons
    assert "bbox_touches_edge" in reasons


def test_review_reasons_extreme_aspect_flagged() -> None:
    m = np.zeros((200, 200), dtype=np.uint8)
    m[50:60, 20:180] = 1
    reasons = review_reasons(
        detector_conf=0.9,
        mask=m,
        bbox=[20, 50, 180, 60],
        image_shape=(200, 200),
    )
    assert "extreme_aspect" in reasons


def test_should_drop_low_conf() -> None:
    m = _decent_mask()
    assert should_drop(
        detector_conf=DEFAULT_DROP_CONF - 0.01,
        mask=m,
        bbox=[60, 60, 140, 140],
    )


def test_should_drop_tiny_mask() -> None:
    m = np.zeros((50, 50), dtype=np.uint8)
    m[10:12, 10:12] = 1  # 4 px
    assert should_drop(
        detector_conf=0.9,
        mask=m,
        bbox=[10, 10, 12, 12],
    )


def test_should_drop_none_bbox() -> None:
    assert should_drop(detector_conf=0.9, mask=_decent_mask(), bbox=None)


def test_should_drop_pass_through_when_healthy() -> None:
    m = _decent_mask()
    assert not should_drop(
        detector_conf=DEFAULT_REVIEW_CONF + 0.05,
        mask=m,
        bbox=[60, 60, 140, 140],
    )


# ---------------------------------------------------------------------------
# mask_circularity
# ---------------------------------------------------------------------------


def test_mask_circularity_empty_returns_zero() -> None:
    assert mask_circularity(np.zeros((30, 30), dtype=np.uint8)) == 0.0


def test_mask_circularity_circle_close_to_one() -> None:
    m = _circle_mask(200, 200, 100, 100, 60)
    # Discretisation drops a perfect 1.0 slightly; anything above 0.85 is
    # plenty to distinguish from a square (0.785) or oval (0.5-0.6).
    assert mask_circularity(m) > 0.85


def test_mask_circularity_square_around_pi_over_four() -> None:
    m = np.zeros((100, 100), dtype=np.uint8)
    m[20:80, 20:80] = 1
    c = mask_circularity(m)
    assert 0.70 < c < 0.85, c


def test_mask_circularity_thin_line_near_zero() -> None:
    m = np.zeros((100, 100), dtype=np.uint8)
    m[50, 10:90] = 1
    assert mask_circularity(m) < 0.20


def test_mask_circularity_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        mask_circularity(np.zeros((3, 3, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# tire_darkness
# ---------------------------------------------------------------------------


def test_tire_darkness_empty_mask_is_max_bright() -> None:
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    assert tire_darkness(img, np.zeros((20, 20), dtype=np.uint8)) == 255.0


def test_tire_darkness_black_image_is_zero() -> None:
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1
    assert tire_darkness(img, mask) == 0.0


def test_tire_darkness_white_image_is_max() -> None:
    img = np.full((20, 20, 3), 255, dtype=np.uint8)
    mask = np.ones((20, 20), dtype=np.uint8)
    assert tire_darkness(img, mask) == 255.0


def test_tire_darkness_shape_mismatch_raises() -> None:
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        tire_darkness(img, np.zeros((10, 10), dtype=np.uint8))


def test_tire_darkness_only_mask_pixels_count() -> None:
    # Half-white, half-black image; mask only over the black half.
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[:, 10:] = 255
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[:, :10] = 1
    assert tire_darkness(img, mask) == 0.0


# ---------------------------------------------------------------------------
# review_reasons: new soft flags
# ---------------------------------------------------------------------------


def test_review_reasons_small_bbox_flagged() -> None:
    # 25×25 mask — under the 28×1.5=42 px soft threshold.
    m = np.zeros((100, 100), dtype=np.uint8)
    m[40:65, 40:65] = 1
    reasons = review_reasons(
        detector_conf=0.9,
        mask=m,
        bbox=[40, 40, 65, 65],
        image_shape=(100, 100),
    )
    assert "small_bbox" in reasons


def test_review_reasons_low_circularity_flagged() -> None:
    m = np.zeros((100, 100), dtype=np.uint8)
    m[40:60, 10:90] = 1  # very wide rectangle
    reasons = review_reasons(
        detector_conf=0.9,
        mask=m,
        bbox=[10, 40, 90, 60],
        image_shape=(100, 100),
    )
    assert "low_circularity" in reasons


def test_review_reasons_light_mask_flagged_when_image_given() -> None:
    m = _circle_mask(100, 100, 50, 50, 25)
    img = np.full((100, 100, 3), 220, dtype=np.uint8)  # bright
    reasons = review_reasons(
        detector_conf=0.9,
        mask=m,
        bbox=[25, 25, 76, 76],
        image_shape=(100, 100),
        image_bgr=img,
    )
    assert "light_mask" in reasons


def test_review_reasons_skips_light_check_without_image() -> None:
    m = _circle_mask(100, 100, 50, 50, 25)
    reasons = review_reasons(
        detector_conf=0.9,
        mask=m,
        bbox=[25, 25, 76, 76],
        image_shape=(100, 100),
    )
    assert "light_mask" not in reasons
