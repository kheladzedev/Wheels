"""Tests for the center-point matcher."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from src.models.matcher import assign_targets_batched, assign_targets_single
from src.models.mobilenetv2_skipless_pose import FEATURE_STRIDE, N_KEYPOINTS

GRID = (20, 20)  # 640 / 32


def test_empty_image_produces_all_negative_assignment() -> None:
    """Image with zero wheels → no positives, no NaNs."""
    bboxes = torch.zeros(0, 4)
    kpts = torch.zeros(0, N_KEYPOINTS, 2)
    vis = torch.zeros(0, N_KEYPOINTS)
    r = assign_targets_single(bboxes, kpts, vis, GRID)
    assert r.cls_target.sum() == 0
    assert not r.pos_mask.any()
    assert r.bbox_target.shape == (400, 4)


def test_single_wheel_assigns_one_positive() -> None:
    bboxes = torch.tensor([[100.0, 200.0, 200.0, 300.0]])
    kpts = torch.tensor([[[110.0, 220.0], [190.0, 220.0], [150.0, 290.0]]])
    vis = torch.ones(1, N_KEYPOINTS)
    r = assign_targets_single(bboxes, kpts, vis, GRID)
    assert int(r.pos_mask.sum()) == 1

    # GT center is (150, 250). Cell stride 32 → cell-center (4*32+16, 7*32+16) = (144, 240).
    # That maps to row 7, col 4 → cell index 7*20 + 4 = 144.
    pos_idx = int(torch.where(r.pos_mask)[0].item())
    assert pos_idx == 144

    # Visibility is propagated.
    assert torch.equal(r.vis_target[pos_idx], torch.ones(N_KEYPOINTS))


def test_bbox_target_is_in_stride_units_and_nonnegative() -> None:
    bboxes = torch.tensor([[64.0, 64.0, 192.0, 192.0]])  # 4×4 stride box
    kpts = torch.zeros(1, N_KEYPOINTS, 2)
    vis = torch.ones(1, N_KEYPOINTS)
    r = assign_targets_single(bboxes, kpts, vis, GRID, stride=FEATURE_STRIDE)
    pos_idx = int(torch.where(r.pos_mask)[0].item())
    l, t, rr, b = r.bbox_target[pos_idx].tolist()
    # GT center is (128, 128). Stride 32 cell-center is (128, 128) too
    # (row 3, col 3 → (3.5*32, 3.5*32) = (112, 112)). Distances in stride
    # units: l = (112 - 64)/32 = 1.5, etc. — we only assert >= 0 and sane.
    assert all(v >= 0 for v in (l, t, rr, b))
    assert max(l, t, rr, b) < 10  # well-bounded


def test_larger_wheel_wins_on_cell_collision() -> None:
    """If two wheels land in the same cell, the larger one keeps it.

    The matcher sorts wheels by area ascending and writes positives in
    that order, so the largest writes last and wins. Verify by giving
    two wheels with the same center, different sizes.
    """
    bboxes = torch.tensor(
        [
            [140.0, 140.0, 160.0, 160.0],  # small, 400 px² area
            [80.0, 80.0, 220.0, 220.0],  # large, 19_600 px² area
        ]
    )
    kpts = torch.tensor(
        [
            [[145.0, 145.0], [155.0, 145.0], [150.0, 158.0]],  # small wheel kpts
            [[90.0, 90.0], [210.0, 90.0], [150.0, 215.0]],  # large wheel kpts
        ]
    )
    vis = torch.ones(2, N_KEYPOINTS)
    r = assign_targets_single(bboxes, kpts, vis, GRID)

    # Exactly one positive cell (both share the same center).
    assert int(r.pos_mask.sum()) == 1
    pos_idx = int(torch.where(r.pos_mask)[0].item())
    # Bbox target should be the large wheel's (l+r ~ 4.4 stride units),
    # not the small wheel's (l+r < 1 stride unit).
    bw = r.bbox_target[pos_idx, 0] + r.bbox_target[pos_idx, 2]
    assert bw > 3.0, f"expected large wheel's width to win, got {float(bw)}"


def test_batched_assignment_stacks_correctly() -> None:
    img0_b = torch.tensor([[100.0, 200.0, 200.0, 300.0]])
    img0_k = torch.zeros(1, N_KEYPOINTS, 2)
    img0_v = torch.ones(1, N_KEYPOINTS)

    img1_b = torch.tensor([[100.0, 100.0, 200.0, 200.0], [400.0, 400.0, 500.0, 500.0]])
    img1_k = torch.zeros(2, N_KEYPOINTS, 2)
    img1_v = torch.ones(2, N_KEYPOINTS)

    img2_b = torch.zeros(0, 4)
    img2_k = torch.zeros(0, N_KEYPOINTS, 2)
    img2_v = torch.zeros(0, N_KEYPOINTS)

    r = assign_targets_batched(
        [img0_b, img1_b, img2_b],
        [img0_k, img1_k, img2_k],
        [img0_v, img1_v, img2_v],
        GRID,
    )
    assert r.pos_mask.shape == (3, 400)
    assert int(r.pos_mask[0].sum()) == 1
    assert int(r.pos_mask[1].sum()) == 2
    assert int(r.pos_mask[2].sum()) == 0


def test_mismatched_batch_lengths_raises() -> None:
    with pytest.raises(ValueError):
        assign_targets_batched(
            [torch.zeros(0, 4)],
            [torch.zeros(0, N_KEYPOINTS, 2), torch.zeros(0, N_KEYPOINTS, 2)],
            [torch.zeros(0, N_KEYPOINTS)],
            GRID,
        )
