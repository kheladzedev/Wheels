"""Tests for the keypoint-evaluation primitives.

Covers:
  - box_iou geometry edge cases.
  - greedy prediction → GT matching.
  - per-keypoint pixel error with visibility filtering.
  - OKS at the perfect-match and far-off extremes.
  - Stage 5 additions: bbox-area bucketing, occlusion classifier,
    per-pair slicing, worst-N failure ranking.
"""

from __future__ import annotations

from pathlib import Path

import eval_keypoints as eval_mod
from eval_keypoints import (
    DEFAULT_SIGMAS,
    GTWheel,
    KEYPOINT_NAMES,
    MatchedPair,
    PredWheel,
    box_iou,
    classify_bbox_area,
    has_occlusion,
    match_predictions_to_gt,
    oks_for_match,
    per_keypoint_pixel_errors,
    slice_records,
    top_n_failures,
)


def _gt(bbox, kps, vis=(2, 2, 2)) -> GTWheel:
    return GTWheel(
        bbox_xyxy=tuple(bbox),
        keypoints_xy=[tuple(p) for p in kps],
        visibilities=list(vis),
    )


def _pred(bbox, kps, conf=0.9) -> PredWheel:
    return PredWheel(
        bbox_xyxy=tuple(bbox),
        confidence=conf,
        keypoints_xy=[tuple(p) for p in kps],
    )


# ---- box_iou -----------------------------------------------------------


def test_box_iou_disjoint():
    a = (0, 0, 10, 10)
    b = (20, 20, 30, 30)
    assert box_iou(a, b) == 0.0


def test_box_iou_identical():
    a = (0, 0, 10, 10)
    assert box_iou(a, a) == 1.0


def test_box_iou_half_overlap():
    # Two 10x10 boxes, one shifted by 5px in x → intersection = 5x10 = 50,
    # union = 100 + 100 - 50 = 150 → IoU = 50/150 = 1/3.
    a = (0, 0, 10, 10)
    b = (5, 0, 15, 10)
    iou = box_iou(a, b)
    assert abs(iou - (1.0 / 3.0)) < 1e-9


# ---- match_predictions_to_gt ------------------------------------------


def test_match_predictions_to_gt():
    # Three GT wheels at distinct locations.
    gts = [
        _gt((0, 0, 100, 100), [(10, 10), (90, 50), (50, 90)]),
        _gt((200, 0, 300, 100), [(210, 10), (290, 50), (250, 90)]),
        _gt((400, 0, 500, 100), [(410, 10), (490, 50), (450, 90)]),
    ]
    preds = [
        # High-conf match to GT 0 (heavy overlap).
        _pred((5, 5, 105, 105), [(11, 11), (89, 49), (51, 89)], conf=0.95),
        # Decent match to GT 1.
        _pred((202, 2, 302, 102), [(211, 11), (291, 51), (249, 89)], conf=0.80),
        # False positive: nowhere near any GT.
        _pred((600, 600, 700, 700), [(610, 610), (690, 650), (650, 690)], conf=0.70),
    ]
    # GT 2 (400..500) has no prediction nearby → FN.

    result = match_predictions_to_gt(preds, gts, iou_threshold=0.5)

    # Two matches expected, in confidence-desc order.
    assert len(result.matches) == 2
    match_dict = dict(result.matches)
    assert match_dict[0] == 0  # highest-conf pred → GT 0
    assert match_dict[1] == 1  # second pred → GT 1

    assert result.unmatched_preds == [2]  # the conf=0.70 outlier
    assert result.unmatched_gt == [2]  # GT 2 was never matched


# ---- per_keypoint_pixel_errors ----------------------------------------


def test_per_keypoint_error_ignores_invisible_gt():
    # Pred is offset by (3, 4) → distance 5 from each GT kp.
    gt = _gt(
        (0, 0, 100, 100),
        kps=[(10, 10), (50, 50), (80, 80)],
        vis=(2, 0, 2),  # middle keypoint is invisible
    )
    pred = _pred(
        (0, 0, 100, 100),
        kps=[(13, 14), (53, 54), (83, 84)],
    )
    errs = per_keypoint_pixel_errors(pred, gt)

    assert errs[0] is not None
    assert errs[1] is None  # visibility=0 → no error reported
    assert errs[2] is not None
    assert abs(errs[0] - 5.0) < 1e-9
    assert abs(errs[2] - 5.0) < 1e-9


def test_eval_report_uses_confirmed_ar_point_names():
    assert KEYPOINT_NAMES == ("a", "b", "c_disc_bottom")

    gt = _gt(
        (0, 0, 100, 100),
        kps=[(10, 10), (50, 50), (80, 80)],
        vis=(2, 2, 2),
    )
    pred = _pred(
        (0, 0, 100, 100),
        kps=[(13, 14), (53, 54), (83, 84)],
    )

    metrics = eval_mod.compute_metrics([([pred], [gt])], conf_threshold=0.25)

    assert set(metrics["per_keypoint_pixel_error"].keys()) == {
        "a",
        "b",
        "c_disc_bottom",
    }


# ---- OKS ---------------------------------------------------------------


def test_oks_perfect_match():
    gt = _gt(
        (0, 0, 100, 100),
        kps=[(10, 10), (50, 50), (80, 80)],
        vis=(2, 2, 2),
    )
    pred = _pred(
        (0, 0, 100, 100),
        kps=[(10, 10), (50, 50), (80, 80)],
    )
    oks = oks_for_match(pred, gt, sigmas=DEFAULT_SIGMAS)
    assert oks is not None
    assert abs(oks - 1.0) < 1e-9


def test_oks_far_off():
    # GT bbox area is small (10x10) → s = 10. Predictions are wildly off
    # (1000+ pixels away) → exp(-huge) ≈ 0.
    gt = _gt(
        (0, 0, 10, 10),
        kps=[(5, 5), (6, 6), (7, 7)],
        vis=(2, 2, 2),
    )
    pred = _pred(
        (0, 0, 10, 10),
        kps=[(5000, 5000), (6000, 6000), (7000, 7000)],
    )
    oks = oks_for_match(pred, gt, sigmas=DEFAULT_SIGMAS)
    assert oks is not None
    assert oks < 1e-6


# ---- classify_bbox_area -----------------------------------------------


def test_classify_bbox_area_small_medium_large():
    # Three GT bboxes with areas 30**2, 60**2, 200**2 → three buckets.
    small = (0.0, 0.0, 30.0, 30.0)
    medium = (0.0, 0.0, 60.0, 60.0)
    large = (0.0, 0.0, 200.0, 200.0)
    assert classify_bbox_area(small) == "small"
    assert classify_bbox_area(medium) == "medium"
    assert classify_bbox_area(large) == "large"

    # Boundary conditions: COCO convention places area == 32**2 in
    # ``medium`` (lower bound inclusive) and area == 96**2 in ``large``.
    on_small_medium = (0.0, 0.0, 32.0, 32.0)
    on_medium_large = (0.0, 0.0, 96.0, 96.0)
    assert classify_bbox_area(on_small_medium) == "medium"
    assert classify_bbox_area(on_medium_large) == "large"


# ---- has_occlusion ----------------------------------------------------


def test_has_occlusion_true_when_any_visibility_one():
    # Visibility == 1 anywhere in the GT means an annotator marked at
    # least one keypoint as occluded-but-localised.
    occluded = _gt((0, 0, 10, 10), kps=[(1, 1), (2, 2), (3, 3)], vis=(2, 1, 2))
    visible = _gt((0, 0, 10, 10), kps=[(1, 1), (2, 2), (3, 3)], vis=(2, 2, 2))
    missing = _gt((0, 0, 10, 10), kps=[(1, 1), (2, 2), (3, 3)], vis=(0, 2, 2))
    assert has_occlusion(occluded) is True
    assert has_occlusion(visible) is False
    # vis=0 is "missing label", not "occluded" — must not flip the flag.
    assert has_occlusion(missing) is False


# ---- slice_records ----------------------------------------------------


def test_slice_records_by_bbox_area_partitions_pairs_not_images():
    # Two images, each with a small AND a large wheel. Slicing must
    # route every matched pair to its own bucket regardless of which
    # image it came from — i.e. it operates per pair, not per image.
    def _make_image() -> tuple[list[PredWheel], list[GTWheel]]:
        # Small wheel (30x30 = area 900 < 32**2 = 1024).
        small_gt = _gt(
            (0, 0, 30, 30), kps=[(10, 10), (20, 10), (15, 25)], vis=(2, 2, 2)
        )
        small_pred = _pred((1, 1, 31, 31), kps=[(10, 10), (20, 10), (15, 25)], conf=0.9)
        # Large wheel (200x200 = area 40000 >= 96**2 = 9216).
        large_gt = _gt(
            (300, 0, 500, 200),
            kps=[(320, 20), (480, 20), (400, 180)],
            vis=(2, 2, 2),
        )
        large_pred = _pred(
            (301, 1, 501, 201),
            kps=[(320, 20), (480, 20), (400, 180)],
            conf=0.85,
        )
        return [small_pred, large_pred], [small_gt, large_gt]

    image_records = [_make_image(), _make_image()]

    buckets = slice_records(image_records, axis="bbox_area")
    # Both images contribute one pair each to small and to large.
    assert len(buckets["small"]) == 2
    assert len(buckets["medium"]) == 0
    assert len(buckets["large"]) == 2

    # n_matched per bucket counts only the wheels in that bucket — not
    # the total number of wheels across images.
    from eval_keypoints import _aggregate_pairs  # noqa: PLC0415

    small_pairs = [(p[0], g[0]) for p, g in buckets["small"]]
    large_pairs = [(p[0], g[0]) for p, g in buckets["large"]]
    assert _aggregate_pairs(small_pairs)["n_matched"] == 2
    assert _aggregate_pairs(large_pairs)["n_matched"] == 2


# ---- top_n_failures ---------------------------------------------------


def _mp(stem: str, e: float, vis=(2, 2, 2)) -> MatchedPair:
    """MatchedPair with uniform (e, 0) translation, so mean error == e.

    Pred bbox is identical to GT bbox, so OKS is computed on a well-
    defined scale and only the keypoint offset drives the score.
    """
    gt = _gt(
        (0, 0, 100, 100),
        kps=[(10, 10), (50, 50), (80, 80)],
        vis=vis,
    )
    pred = _pred(
        (0, 0, 100, 100),
        kps=[(10 + e, 10), (50 + e, 50), (80 + e, 80)],
        conf=0.9,
    )
    return MatchedPair(image_path=Path(f"/tmp/{stem}.jpg"), pred=pred, gt=gt)


def test_top_n_failures_returns_highest_error_first():
    # Build 5 matched pairs whose mean per-keypoint errors are exactly
    # [1, 50, 2, 30, 10].
    pairs = [
        _mp("a", 1.0),
        _mp("b", 50.0),
        _mp("c", 2.0),
        _mp("d", 30.0),
        _mp("e", 10.0),
    ]
    out = top_n_failures(pairs, n=3)
    assert len(out) == 3
    # Worst-first order: 50, 30, 10.
    assert out[0]["score_px"] == 50.0
    assert out[1]["score_px"] == 30.0
    assert out[2]["score_px"] == 10.0
    # Image stems thread through.
    assert Path(out[0]["image"]).stem == "b"
    assert Path(out[1]["image"]).stem == "d"
    assert Path(out[2]["image"]).stem == "e"


def test_top_n_failures_returns_all_when_n_exceeds_len():
    # 3 pairs, asked for 10 → expect 3 entries, no padding.
    pairs = [_mp("a", 1.0), _mp("b", 5.0), _mp("c", 2.0)]
    out = top_n_failures(pairs, n=10)
    assert len(out) == 3
    # Still worst-first ordered.
    assert out[0]["score_px"] == 5.0
    assert out[1]["score_px"] == 2.0
    assert out[2]["score_px"] == 1.0


def test_top_n_failures_empty_input_returns_empty_list():
    # Empty input must not raise and must return [].
    out = top_n_failures([], n=5)
    assert out == []


def test_top_n_failures_skips_invisible_keypoints_in_score():
    # GT visibility (2, 0, 2): only keypoints 0 and 2 contribute to the
    # score. Pred offsets are (3, 4), (1000, 1000), (3, 4) → distances
    # 5, ignored, 5 → mean == 5.0, not (5 + None_as_zero + 5) / 3.
    gt = _gt(
        (0, 0, 100, 100),
        kps=[(10, 10), (50, 50), (80, 80)],
        vis=(2, 0, 2),
    )
    pred = _pred(
        (0, 0, 100, 100),
        kps=[(13, 14), (1050, 1050), (83, 84)],
        conf=0.9,
    )
    pair = MatchedPair(image_path=Path("/tmp/half_visible.jpg"), pred=pred, gt=gt)

    out = top_n_failures([pair], n=1)
    assert len(out) == 1
    row = out[0]
    assert row["score_px"] is not None
    # Mean over the two visible keypoints only.
    assert abs(row["score_px"] - 5.0) < 1e-9
    # And the per-keypoint list still carries the None for the invisible
    # keypoint — None is preserved (not coerced to 0).
    assert row["per_keypoint_px"][1] is None
    assert row["per_keypoint_px"][0] is not None
    assert row["per_keypoint_px"][2] is not None


def test_top_n_failures_preserves_input_order_for_tied_scores():
    # heapq.nlargest is used for top-k ranking; equal scores must remain
    # deterministic so failure catalogues do not churn between runs.
    pairs = [_mp("a", 5.0), _mp("b", 5.0), _mp("c", 5.0)]

    out = top_n_failures(pairs, n=2)

    assert [Path(row["image"]).stem for row in out] == ["a", "b"]


def test_match_records_reuse_one_matching_pass(monkeypatch):
    def _make_image(offset: float) -> tuple[list[PredWheel], list[GTWheel]]:
        gt = _gt(
            (offset, 0, offset + 100, 100),
            kps=[(offset + 10, 10), (offset + 90, 50), (offset + 50, 90)],
        )
        pred = _pred(
            (offset + 1, 1, offset + 101, 101),
            kps=[(offset + 10, 10), (offset + 90, 50), (offset + 50, 90)],
        )
        return [pred], [gt]

    image_records = [
        (Path("/tmp/a.jpg"), *_make_image(0.0)),
        (Path("/tmp/b.jpg"), *_make_image(200.0)),
    ]
    calls = 0
    original = eval_mod.match_predictions_to_gt

    def counting_match(preds, gts, iou_threshold=eval_mod.DEFAULT_IOU_MATCH):
        nonlocal calls
        calls += 1
        return original(preds, gts, iou_threshold=iou_threshold)

    monkeypatch.setattr(eval_mod, "match_predictions_to_gt", counting_match)

    matched_records = eval_mod.match_image_records(image_records)
    assert calls == len(image_records)

    metrics = eval_mod.compute_metrics_from_match_records(
        matched_records,
    )
    sliced = eval_mod.compute_sliced_metrics_from_match_records(matched_records)
    pairs = eval_mod.collect_matched_pairs_from_match_records(matched_records)

    assert calls == len(image_records)
    assert metrics["counts"]["matched"] == 2
    assert sliced["by_bbox_area"]["large"]["n_matched"] == 2
    assert [p.image_path.name for p in pairs] == ["a.jpg", "b.jpg"]
