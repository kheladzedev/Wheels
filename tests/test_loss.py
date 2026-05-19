"""Tests for the production-model loss functions."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
from src.models.loss import (
    PoseLoss,
    focal_loss,
    giou_loss,
    keypoint_smooth_l1_loss,
    oks_keypoint_loss,
)
from src.models.matcher import assign_targets_batched
from src.models.mobilenetv2_skipless_pose import (
    FEATURE_STRIDE,
    N_KEYPOINTS,
    MobileNetV2SkiplessPose,
)

GRID = (20, 20)


def test_focal_loss_zero_on_perfect_predictions() -> None:
    """Perfect predictions (logit → ±inf in the right direction) → loss ~ 0."""
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
    logits = torch.tensor([-50.0, 50.0, -50.0, 50.0])
    val = float(focal_loss(logits, targets))
    assert val < 1e-6


def test_focal_loss_positive_on_wrong_predictions() -> None:
    """Hard mistakes → strictly positive loss, well above zero."""
    targets = torch.tensor([0.0, 1.0])
    logits = torch.tensor([8.0, -8.0])  # backwards
    assert float(focal_loss(logits, targets)) > 0.5


def test_giou_loss_zero_on_perfect_box_match() -> None:
    """Same box → IoU = 1, GIoU = 1, loss = 0."""
    pred = torch.tensor([[1.0, 1.0, 2.0, 2.0]])
    gt = torch.tensor([[1.0, 1.0, 2.0, 2.0]])
    val = float(giou_loss(pred, gt))
    assert abs(val) < 1e-5


def test_giou_loss_increases_with_offset() -> None:
    """A slightly misaligned box → small positive loss; far-off → larger."""
    gt = torch.tensor([[1.0, 1.0, 2.0, 2.0]])
    near = torch.tensor([[1.1, 1.1, 2.1, 2.1]])
    far = torch.tensor([[3.0, 3.0, 4.0, 4.0]])
    near_loss = float(giou_loss(near, gt))
    far_loss = float(giou_loss(far, gt))
    assert 0 < near_loss < far_loss


def test_oks_keypoint_loss_zero_on_perfect_match_visible() -> None:
    """Visible keypoints with zero offset error → loss ~ 0."""
    pred = torch.zeros(2, N_KEYPOINTS, 2)
    gt = torch.zeros(2, N_KEYPOINTS, 2)
    vis = torch.ones(2, N_KEYPOINTS)
    diag = torch.tensor([5.0, 5.0])
    val = float(oks_keypoint_loss(pred, gt, vis, diag))
    assert abs(val) < 1e-5


def test_oks_keypoint_loss_ignores_invisible_keypoints() -> None:
    """Wrong predictions on invisible kpts must not contribute."""
    pred = torch.tensor([[[10.0, 10.0], [0.0, 0.0], [0.0, 0.0]]])
    gt = torch.zeros(1, N_KEYPOINTS, 2)
    vis_all = torch.ones(1, N_KEYPOINTS)
    vis_masked = torch.tensor([[0.0, 1.0, 1.0]])  # kpt 0 invisible
    diag = torch.tensor([5.0])
    loss_all = float(oks_keypoint_loss(pred, gt, vis_all, diag))
    loss_masked = float(oks_keypoint_loss(pred, gt, vis_masked, diag))
    # Masking out the only wrong keypoint must drop loss substantially.
    assert loss_all > 0.5
    assert loss_masked < 1e-5


def test_oks_keypoint_loss_penalises_small_wheels_more() -> None:
    """Same pixel error on a smaller wheel → larger OKS loss."""
    pred = torch.tensor([[[1.0, 0.0]]] * 2)  # 1 unit dx error
    pred = torch.cat([pred, torch.zeros(2, 2, 2)], dim=1)  # pad to K=3
    gt = torch.zeros(2, N_KEYPOINTS, 2)
    vis = torch.cat([torch.ones(2, 1), torch.zeros(2, 2)], dim=1)  # only kp 0
    small_diag = torch.tensor([1.0])
    big_diag = torch.tensor([20.0])
    small_loss = float(oks_keypoint_loss(pred[:1], gt[:1], vis[:1], small_diag))
    big_loss = float(oks_keypoint_loss(pred[:1], gt[:1], vis[:1], big_diag))
    assert small_loss > big_loss


def test_smooth_l1_keypoint_loss_has_gradient_for_far_floor_ray_points() -> None:
    """A/B floor-ray offsets start far from zero and must still get gradient."""
    pred = torch.zeros(1, N_KEYPOINTS, 2, requires_grad=True)
    gt = torch.tensor([[[-1.25, 1.25], [1.25, 1.25], [0.0, 0.75]]])
    vis = torch.ones(1, N_KEYPOINTS)

    loss = keypoint_smooth_l1_loss(pred, gt, vis, beta=0.5)
    loss.backward()

    assert float(loss.detach()) > 0
    assert pred.grad is not None
    assert float(pred.grad[0, 0].abs().sum()) > 0.1
    assert float(pred.grad[0, 1].abs().sum()) > 0.1


def test_combined_pose_loss_runs_with_random_input() -> None:
    """End-to-end forward + loss + backward on synthetic batch."""
    m = MobileNetV2SkiplessPose(pretrained=False)
    m.train()  # train mode so loss has full path
    x = torch.randn(2, 3, 640, 640)
    preds = m(x)

    gt_bboxes = [
        torch.tensor([[100.0, 200.0, 200.0, 300.0], [400.0, 250.0, 500.0, 350.0]]),
        torch.tensor([[300.0, 280.0, 400.0, 380.0]]),
    ]
    gt_kpt = [
        torch.tensor(
            [
                [[120.0, 230.0], [180.0, 230.0], [150.0, 290.0]],
                [[420.0, 280.0], [480.0, 280.0], [450.0, 340.0]],
            ]
        ),
        torch.tensor([[[320.0, 300.0], [380.0, 300.0], [350.0, 370.0]]]),
    ]
    gt_vis = [
        torch.ones(2, N_KEYPOINTS),
        torch.ones(1, N_KEYPOINTS),
    ]
    match = assign_targets_batched(gt_bboxes, gt_kpt, gt_vis, GRID, FEATURE_STRIDE)

    crit = PoseLoss()
    ld = crit(preds, match)

    assert ld["total"].requires_grad
    assert math.isfinite(float(ld["total"].detach()))
    for comp in ("cls", "bbox", "kpt", "vis"):
        assert float(ld[comp]) >= 0
    # Smoke gradient: the loss has at least one trainable connection.
    ld["total"].backward()


def test_pose_loss_empty_gt_only_contributes_cls() -> None:
    """Image without wheels → bbox/kpt/vis components stay at zero,
    but the cls component still trains the model to predict negatives.
    """
    m = MobileNetV2SkiplessPose(pretrained=False)
    m.train()
    x = torch.randn(1, 3, 640, 640)
    preds = m(x)

    match = assign_targets_batched(
        [torch.zeros(0, 4)],
        [torch.zeros(0, N_KEYPOINTS, 2)],
        [torch.zeros(0, N_KEYPOINTS)],
        GRID,
        FEATURE_STRIDE,
    )
    crit = PoseLoss()
    ld = crit(preds, match)

    assert float(ld["cls"]) > 0
    assert float(ld["bbox"]) == 0
    assert float(ld["kpt"]) == 0
    assert float(ld["vis"]) == 0
