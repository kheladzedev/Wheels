"""Tests for the differentiable 3D reconstruction loss
(``src/models/reconstruction_loss.py``).

Stage-2/4 auxiliary loss (``docs/AR_REPLAY_METRIC_PLAN.md`` §6): a torch
module that reprojects predicted screen-space A/B onto the floor and C
onto the recovered vertical wheel plane, then penalises the 3D residual
vs ground truth with a Huber loss. It does NOT change the 2D output
contract; it is an optional training signal, gated behind a ramp and
detach, off by default in ``PoseLoss``.

Correctness is pinned by round-trip: feed the pixels obtained by
projecting the GT 3D scene and the loss must be ~0; perturbing pixels
must increase it; gradients must flow to the predicted pixels; grazing
rays must stay finite; the ramp must scale the term linearly.

GT pixels are produced by the trusted numpy forward model
(``eval3d_floorray``) so the torch module's only job under test is the
differentiable reconstruction.
"""

from __future__ import annotations

import numpy as np
import torch

import eval3d_floorray as g
from models import reconstruction_loss as rl


IMG_W, IMG_H = 640, 480


def _scene_tensors(disc_height=0.30, ab_height=0.0, fov=60.0):
    """Build a batch of frames (one wheel) and return torch tensors:
    K, R, C, gt floor A/B, gt disc 3D, and the GT-projected pixels."""
    a_floor = np.array([-0.15, 0.0, 0.0])
    b_floor = np.array([0.15, 0.0, 0.0])
    a_src = np.array([-0.15, 0.0, ab_height])
    b_src = np.array([0.15, 0.0, ab_height])
    disc = np.array([0.0, 0.0, disc_height])
    eyes = [(0.0, -2.0, 1.4), (0.7, -1.8, 1.5), (-0.6, -2.1, 1.3), (0.3, -2.3, 1.55)]
    Ks, Rs, Cs, a_px, b_px, c_px = [], [], [], [], [], []
    for eye in eyes:
        K = g.intrinsics_from_fov(fov, IMG_W, IMG_H)
        R, C = g.look_at(np.array(eye, float), np.array([0.0, 0.0, 0.25]))
        cam = g.Camera(K=K, R=R, C=C)
        Ks.append(K)
        Rs.append(R)
        Cs.append(C)
        a_px.append(g.project(cam, a_src[None])[0])
        b_px.append(g.project(cam, b_src[None])[0])
        c_px.append(g.project(cam, disc[None])[0])

    def t(x):
        return torch.tensor(np.array(x), dtype=torch.float64)

    return {
        "K": t(Ks),
        "R": t(Rs),
        "C": t(Cs),
        "gt_a": t([a_floor] * len(eyes)),
        "gt_b": t([b_floor] * len(eyes)),
        "gt_disc": t([disc] * len(eyes)),
        "a_px": t(a_px),
        "b_px": t(b_px),
        "c_px": t(c_px),
    }


def _loss_for(scene, **kw):
    return rl.reconstruction_loss(
        a_px=scene["a_px"],
        b_px=scene["b_px"],
        c_px=scene["c_px"],
        K=scene["K"],
        R=scene["R"],
        C=scene["C"],
        gt_a=scene["gt_a"],
        gt_b=scene["gt_b"],
        gt_disc=scene["gt_disc"],
        **kw,
    )


def test_perfect_prediction_gives_near_zero_loss():
    scene = _scene_tensors()
    out = _loss_for(scene)
    assert float(out["total"]) < 1e-6


def test_loss_increases_with_pixel_error():
    scene = _scene_tensors()
    base = float(_loss_for(scene)["total"])
    scene["c_px"] = scene["c_px"] + 5.0  # shift C by 5 px
    worse = float(_loss_for(scene)["total"])
    assert worse > base + 1e-6


def test_gradient_flows_to_predicted_pixels():
    scene = _scene_tensors()
    scene["c_px"] = (scene["c_px"] + 4.0).requires_grad_(True)
    out = _loss_for(scene)
    out["total"].backward()
    assert scene["c_px"].grad is not None
    assert torch.isfinite(scene["c_px"].grad).all()
    assert scene["c_px"].grad.abs().sum() > 0


def test_huber_bounds_large_residual():
    # Doubling an already-large pixel error should grow the loss roughly
    # linearly (Huber), not quadratically.
    scene = _scene_tensors()
    s1 = dict(scene)
    s1["c_px"] = scene["c_px"] + 40.0
    s2 = dict(scene)
    s2["c_px"] = scene["c_px"] + 80.0
    l1 = float(
        rl.reconstruction_loss(
            a_px=s1["a_px"],
            b_px=s1["b_px"],
            c_px=s1["c_px"],
            K=s1["K"],
            R=s1["R"],
            C=s1["C"],
            gt_a=s1["gt_a"],
            gt_b=s1["gt_b"],
            gt_disc=s1["gt_disc"],
        )["total"]
    )
    l2 = float(
        rl.reconstruction_loss(
            a_px=s2["a_px"],
            b_px=s2["b_px"],
            c_px=s2["c_px"],
            K=s2["K"],
            R=s2["R"],
            C=s2["C"],
            gt_a=s2["gt_a"],
            gt_b=s2["gt_b"],
            gt_disc=s2["gt_disc"],
        )["total"]
    )
    # quadratic would be ~4x; Huber linear is ~2x. Allow slack.
    assert l2 < 3.0 * l1


def test_grazing_ray_stays_finite():
    # Camera nearly level with the floor -> A/B rays graze the floor plane.
    a_floor = np.array([-0.15, 3.0, 0.0])
    b_floor = np.array([0.15, 3.0, 0.0])
    disc = np.array([0.0, 3.0, 0.30])
    eye = np.array([0.0, 0.0, 0.05])  # almost on the floor
    K = g.intrinsics_from_fov(60.0, IMG_W, IMG_H)
    R, C = g.look_at(eye, np.array([0.0, 3.0, 0.10]))
    cam = g.Camera(K=K, R=R, C=C)

    def t(x):
        return torch.tensor(np.array(x), dtype=torch.float64)[None]

    # Perturb the pixels so the residual is nonzero — otherwise the loss
    # is ~0 regardless of the clamp and the grazing path is never tested.
    out = rl.reconstruction_loss(
        a_px=t(g.project(cam, a_floor[None])[0]) + 8.0,
        b_px=t(g.project(cam, b_floor[None])[0]) - 8.0,
        c_px=t(g.project(cam, disc[None])[0]) + 8.0,
        K=t(K),
        R=t(R),
        C=t(C),
        gt_a=t(a_floor),
        gt_b=t(b_floor),
        gt_disc=t(disc),
    )
    # finite despite a near-parallel ray: the denom clamp + grazing weight
    # must keep t bounded and the gradient sane.
    assert torch.isfinite(out["total"])
    assert float(out["total"]) >= 0.0


def test_ramp_scales_loss_linearly():
    scene = _scene_tensors()
    scene["c_px"] = scene["c_px"] + 6.0
    full = float(_loss_for(scene, ramp=1.0)["total"])
    half = float(_loss_for(scene, ramp=0.5)["total"])
    zero = float(_loss_for(scene, ramp=0.0)["total"])
    assert zero == 0.0
    assert abs(half - 0.5 * full) < 1e-6


def test_ramp_zero_suppresses_nan_residual():
    # ramp=0 must yield an *exact* zero loss even if a pixel is NaN
    # (0.0 * NaN == NaN in IEEE float; the term must short-circuit).
    scene = _scene_tensors()
    bad = scene["c_px"].clone()
    bad[0, 0] = float("nan")
    out = _loss_for({**scene, "c_px": bad}, ramp=0.0)
    assert torch.isfinite(out["total"])
    assert float(out["total"]) == 0.0


def test_mixed_dtype_inputs_do_not_crash():
    # Calibration tensors can disagree in dtype with the model's float32
    # pixels (K float32 from one path, R float64 from a numpy loader).
    # einsum would otherwise raise "expected Double but found Float".
    scene = _scene_tensors()
    out = rl.reconstruction_loss(
        a_px=scene["a_px"].float(),
        b_px=scene["b_px"].float(),
        c_px=scene["c_px"].float(),
        K=scene["K"].float(),  # float32
        R=scene["R"],  # float64 (mismatch)
        C=scene["C"],  # float64
        gt_a=scene["gt_a"].float(),
        gt_b=scene["gt_b"].float(),
        gt_disc=scene["gt_disc"].float(),
    )
    assert torch.isfinite(out["total"])
    assert float(out["total"]) < 1e-4


def test_detach_plane_false_propagates_grad_to_ab_pixels():
    # Positive control for the detach test: with detach_plane=False the C
    # term *must* push gradient back into A/B (plane depends on them).
    scene = _scene_tensors(ab_height=0.0)
    scene["a_px"] = scene["a_px"].clone().requires_grad_(True)
    out = rl.reconstruction_loss(
        a_px=scene["a_px"],
        b_px=scene["b_px"],
        c_px=scene["c_px"] + 5.0,
        K=scene["K"],
        R=scene["R"],
        C=scene["C"],
        gt_a=scene["gt_a"],
        gt_b=scene["gt_b"],
        gt_disc=scene["gt_disc"],
        w_ab=0.0,
        w_c=1.0,
        detach_plane=False,
    )
    out["total"].backward()
    assert scene["a_px"].grad is not None
    assert scene["a_px"].grad.abs().sum() > 0


def test_detach_plane_isolates_c_gradient_from_ab_pixels():
    # With the vertical plane detached, the C term must not push gradient
    # back into the A/B pixel predictions.
    scene = _scene_tensors(ab_height=0.0)
    scene["a_px"] = scene["a_px"].clone().requires_grad_(True)
    scene["b_px"] = scene["b_px"].clone().requires_grad_(True)
    out = rl.reconstruction_loss(
        a_px=scene["a_px"],
        b_px=scene["b_px"],
        c_px=scene["c_px"] + 5.0,
        K=scene["K"],
        R=scene["R"],
        C=scene["C"],
        gt_a=scene["gt_a"],
        gt_b=scene["gt_b"],
        gt_disc=scene["gt_disc"],
        w_ab=0.0,
        w_c=1.0,
        detach_plane=True,
    )
    out["total"].backward()
    # only the C term is active (w_ab=0); detached plane => no A/B grad
    assert scene["a_px"].grad is None or scene["a_px"].grad.abs().sum() == 0
