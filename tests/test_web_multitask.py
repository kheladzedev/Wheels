"""Tests for the web multi-task model (``src/models/web_multitask.py``).

The web deployment target (separate from the frozen Android/iOS 2D
contract) runs a single network that emits detection + keypoints AND a
floor estimate ``{pitch, roll, delta_z}`` from global features, at a
512x512 input. Floor estimation lets the web client recover 3D without
an AR framework; it is a *different* product surface, so the 2D AR
contract in ``docs/AR_ML_CONTRACT.md`` is untouched.

These tests pin the module's structure and training mechanics:
forward shapes at 512, a finite 3-DoF floor output, a multi-task loss
with learnable uncertainty weighting (Kendall et al.) that backprops,
the ``detach_floor`` switch used during the 2D-only training stage, and
the staged freezing that the training schedule relies on.
"""

from __future__ import annotations

import torch

from models import web_multitask as wm
from models.mobilenetv2_skipless_pose import N_KEYPOINTS


def _model():
    return wm.WebMultiTaskModel(pretrained=False)


def test_forward_shapes_at_512():
    model = _model()
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    # stride-32 backbone => 16x16 grid at 512
    assert out["cls"].shape == (2, 1, 16, 16)
    assert out["bbox"].shape == (2, 4, 16, 16)
    assert out["kpt"].shape == (2, N_KEYPOINTS * 2, 16, 16)
    assert out["vis"].shape == (2, N_KEYPOINTS, 16, 16)
    # floor head: pitch, roll, delta_z
    assert out["floor"].shape == (2, 3)


def test_floor_output_is_finite():
    model = _model()
    out = model(torch.randn(3, 3, 512, 512))
    assert torch.isfinite(out["floor"]).all()


def test_floor_loss_zero_for_perfect_prediction():
    gt = torch.tensor([[0.1, -0.05, 1.7], [0.2, 0.0, 1.5]])
    loss = wm.floor_loss(gt.clone(), gt)
    assert float(loss) < 1e-7


def test_floor_loss_known_offset_value():
    # A residual equal to the per-DoF scale -> normalized diff 1.0 ->
    # smooth_l1(1.0, beta=1.0) = 0.5, averaged over the 3 DoFs.
    gt = torch.zeros(1, 3)
    pred = torch.tensor([[wm.FLOOR_SCALE[0], 0.0, 0.0]])
    assert abs(float(wm.floor_loss(pred, gt)) - 0.5 / 3.0) < 1e-6


def test_floor_loss_balances_dofs_by_scale():
    # An angular error of one angular-scale must cost the same as a
    # delta_z error of one delta_z-scale: no single DoF dominates.
    gt = torch.zeros(1, 3)
    l_angle = wm.floor_loss(torch.tensor([[wm.FLOOR_SCALE[0], 0.0, 0.0]]), gt)
    l_dz = wm.floor_loss(torch.tensor([[0.0, 0.0, wm.FLOOR_SCALE[2]]]), gt)
    assert abs(float(l_angle) - float(l_dz)) < 1e-6


def test_multitask_loss_combines_and_backprops():
    model = _model()
    crit = wm.MultiTaskLoss()
    x = torch.randn(2, 3, 512, 512, requires_grad=False)
    out = model(x)
    # cheap surrogate pose loss so the test stays backbone-only
    pose_loss = out["cls"].abs().mean() + out["kpt"].abs().mean()
    gt_floor = torch.zeros(2, 3)
    res = crit(pose_loss=pose_loss, floor_pred=out["floor"], gt_floor=gt_floor)
    assert torch.isfinite(res["total"])
    res["total"].backward()
    # gradient must reach the backbone specifically, and the uncertainty
    # weights must themselves learn.
    enc_grad = sum(
        p.grad.abs().sum() for p in model.encoder.parameters() if p.grad is not None
    )
    assert float(enc_grad) > 0
    assert crit.log_var.grad is not None


def test_floor_grad_reaches_backbone_when_not_detached():
    # Positive control for the detach test: floor-only, NOT detached.
    model = _model()
    crit = wm.MultiTaskLoss()
    out = model(torch.randn(2, 3, 512, 512))
    res = crit(
        pose_loss=torch.zeros((), requires_grad=True),
        floor_pred=out["floor"],
        gt_floor=torch.ones(2, 3),
        detach_floor=False,
    )
    res["total"].backward()
    enc_grad = sum(
        p.grad.abs().sum() for p in model.encoder.parameters() if p.grad is not None
    )
    assert float(enc_grad) > 0


def test_uncertainty_weights_are_learnable_parameters():
    crit = wm.MultiTaskLoss()
    names = {n for n, _ in crit.named_parameters()}
    assert any("log_var" in n for n in names)
    # two tasks => two log-variance scalars
    n_params = sum(p.numel() for p in crit.parameters())
    assert n_params == 2


def test_detach_floor_blocks_floor_grad_into_backbone():
    model = _model()
    crit = wm.MultiTaskLoss()
    out = model(torch.randn(2, 3, 512, 512))
    # floor-only objective, floor detached => no grad into backbone
    res = crit(
        pose_loss=torch.zeros((), requires_grad=True),
        floor_pred=out["floor"],
        gt_floor=torch.ones(2, 3),
        detach_floor=True,
    )
    res["total"].backward()
    backbone_grad = sum(
        p.grad.abs().sum() for p in model.encoder.parameters() if p.grad is not None
    )
    assert float(backbone_grad) == 0.0


def test_stage_2d_freezes_floor_head():
    model = _model()
    model.set_stage("2d")
    assert all(not p.requires_grad for p in model.floor_head.parameters())
    model.set_stage("joint")
    assert all(p.requires_grad for p in model.floor_head.parameters())
