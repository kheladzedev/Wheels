"""Differentiable 3D reconstruction loss for the floor-ray wheel contract.

The torch counterpart of ``src/eval3d_floorray.py``: reprojects the
model's predicted screen-space ``a`` / ``b`` onto the floor plane and
``c_disc_bottom`` onto the recovered vertical wheel plane, then
penalises the 3D residual against ground truth with a Huber loss. The
whole chain is differentiable, so gradients flow back to the predicted
pixel coordinates.

This is the Stage-2/4 auxiliary signal of
``docs/AR_REPLAY_METRIC_PLAN.md`` (§6 "offline eval first, training
loss later"). It is **off by default** and must stay so until the
replay metric stabilises on real-device data and a 3D error budget is
agreed (``docs/OPEN_QUESTIONS_AR_SPEC.md`` §9). It does **not** change
the frozen 2D ML output contract (``docs/AR_ML_CONTRACT.md``) — the
model still emits 2D pixels; only the training-time supervision gains a
3D term, behind a ``ramp`` (warm-up schedule) and ``detach_plane``
(isolate C's gradient from the A/B plane fit).

Conventions match ``src/eval3d_floorray.py``: world floor = z = 0,
camera ``X_cam = R @ (X_world - C)`` (OpenCV axes), image top-left
origin. Inputs are per-frame batched tensors.

Numerical care for grazing angles (rays nearly parallel to a plane):

  - the ray/plane denominator is clamped away from zero (keeps ``t``
    finite), and
  - each reconstructed point is weighted by a smooth grazing factor in
    ``[0, 1]`` that decays as the incidence angle flattens, so an
    ill-conditioned reconstruction cannot dominate the gradient, and
  - the residual itself is Huber (smooth-L1), bounding the gradient of
    any single large outlier.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Smallest |cos| we allow in the ray/plane denominator before clamping.
_MIN_DENOM = 1e-2
# Below this |cos| the reconstruction is downweighted toward zero.
_GRAZING_FULL = 0.05


def _expand_plane(
    normal: torch.Tensor, offset: torch.Tensor, batch: int, like: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if normal.dim() == 1:
        normal = normal.to(like).unsqueeze(0).expand(batch, -1)
    if offset.dim() == 0:
        offset = offset.to(like).expand(batch)
    return normal, offset


def pixel_to_ray(
    K: torch.Tensor, R: torch.Tensor, C: torch.Tensor, uv: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Back-project pixels to world rays. All inputs batched (B, ...).

    ``K`` (B,3,3), ``R`` (B,3,3) world->camera, ``C`` (B,3), ``uv`` (B,2).
    Returns ``(origin (B,3), direction (B,3))`` with unit directions.

    Calibration tensors are cast to ``uv``'s dtype so a float32 model
    output can meet float64 numpy-derived ``K`` / ``R`` / ``C`` without
    the einsum raising a dtype mismatch.
    """
    K = K.to(uv.dtype)
    R = R.to(uv.dtype)
    C = C.to(uv.dtype)
    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]
    u = uv[:, 0]
    v = uv[:, 1]
    dir_cam = torch.stack(
        [(u - cx) / fx, (v - cy) / fy, torch.ones_like(u)], dim=-1
    )  # (B,3)
    # camera->world is R^T; X_world_dir = R^T @ dir_cam
    dir_world = torch.einsum("bij,bj->bi", R.transpose(1, 2), dir_cam)
    dir_world = F.normalize(dir_world, dim=-1)
    return C, dir_world


def ray_plane_intersect(
    origin: torch.Tensor,
    direction: torch.Tensor,
    normal: torch.Tensor,
    offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Intersect batched rays with batched planes ``n.X = offset``.

    Returns ``(points (B,3), grazing_weight (B,))``. The denominator is
    clamped (sign-preserving) so ``t`` stays finite at grazing angles;
    ``grazing_weight`` smoothly decays to 0 as the ray flattens against
    the plane. ``t`` is clamped to be non-negative (no intersections
    behind the camera).
    """
    denom = (normal * direction).sum(-1)  # (B,) = cos(angle to normal)
    abs_d = denom.abs()
    sign = torch.where(denom >= 0, 1.0, -1.0).to(denom.dtype)
    denom_safe = sign * abs_d.clamp(min=_MIN_DENOM)
    t = (offset - (normal * origin).sum(-1)) / denom_safe
    t = t.clamp(min=0.0)
    points = origin + t.unsqueeze(-1) * direction
    grazing_weight = (abs_d / _GRAZING_FULL).clamp(max=1.0)
    return points, grazing_weight


def _vertical_plane(
    a_floor: torch.Tensor, b_floor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vertical plane through two floor anchors. ``a_floor``/``b_floor``
    are (B,3) on the floor. Returns horizontal unit normals (B,3) and
    offsets (B,)."""
    up = torch.tensor([0.0, 0.0, 1.0], dtype=a_floor.dtype, device=a_floor.device)
    base = b_floor - a_floor
    n = torch.cross(up.expand_as(base), base, dim=-1)
    n = F.normalize(n, dim=-1)
    offset = (n * a_floor).sum(-1)
    return n, offset


def _huber_point(
    pred: torch.Tensor, gt: torch.Tensor, beta: float, weight: torch.Tensor
) -> torch.Tensor:
    """Per-sample Huber over 3D coords, weighted, mean over batch."""
    per = F.smooth_l1_loss(pred, gt, beta=beta, reduction="none").sum(-1)  # (B,)
    return (per * weight).mean()


def reconstruction_loss(
    a_px: torch.Tensor,
    b_px: torch.Tensor,
    c_px: torch.Tensor,
    K: torch.Tensor,
    R: torch.Tensor,
    C: torch.Tensor,
    gt_a: torch.Tensor,
    gt_b: torch.Tensor,
    gt_disc: torch.Tensor,
    w_ab: float = 1.0,
    w_c: float = 1.0,
    beta: float = 0.05,
    ramp: float = 1.0,
    detach_plane: bool = True,
) -> dict[str, torch.Tensor]:
    """3D reconstruction loss over a batch of frames.

    A/B pixels reproject onto the floor; C reprojects onto the vertical
    plane recovered from the A/B floor reconstructions. ``detach_plane``
    stops C's gradient from flowing back into the A/B fit (recommended
    when first switching the term on). ``ramp`` is the warm-up multiplier
    on the whole term; ``ramp=0`` yields an exact zero loss.

    ``beta`` is the Huber knee in the GT's units. The default 0.05 suits
    metre-scale GT (the tests); for centimetre GT (the UE export) pass a
    cm-scale ``beta`` (~5.0), otherwise the knee sits at 0.5 mm and the
    term is pure L1 for every realistic residual. All tensors are (B, ...).
    """
    # ramp==0 must yield an exact zero (the term is off): short-circuit
    # so a NaN/Inf pixel during warm-up cannot poison the graph via
    # IEEE 0.0 * NaN == NaN.
    if ramp == 0.0:
        z = a_px.sum() * 0.0  # keeps the grad graph attached, value 0
        return {
            "total": z,
            "l_ab": z.detach(),
            "l_c": z.detach(),
            "l_a": z.detach(),
            "l_b": z.detach(),
        }

    batch = a_px.shape[0]
    floor_n = torch.tensor([0.0, 0.0, 1.0], dtype=a_px.dtype, device=a_px.device)
    floor_off = torch.zeros((), dtype=a_px.dtype, device=a_px.device)
    fn, fo = _expand_plane(floor_n, floor_off, batch, a_px)

    oa, da = pixel_to_ray(K, R, C, a_px)
    ob, db = pixel_to_ray(K, R, C, b_px)
    a3d, wa = ray_plane_intersect(oa, da, fn, fo)
    b3d, wb = ray_plane_intersect(ob, db, fn, fo)

    plane_a = a3d.detach() if detach_plane else a3d
    plane_b = b3d.detach() if detach_plane else b3d
    pn, po = _vertical_plane(plane_a, plane_b)

    oc, dc = pixel_to_ray(K, R, C, c_px)
    c3d, wc = ray_plane_intersect(oc, dc, pn, po)

    l_a = _huber_point(a3d, gt_a, beta, wa)
    l_b = _huber_point(b3d, gt_b, beta, wb)
    l_c = _huber_point(c3d, gt_disc, beta, wc)

    l_ab = l_a + l_b
    total = ramp * (w_ab * l_ab + w_c * l_c)
    return {
        "total": total,
        "l_ab": l_ab.detach(),
        "l_c": l_c.detach(),
        "l_a": l_a.detach(),
        "l_b": l_b.detach(),
    }


class ReconstructionLoss(nn.Module):
    """nn.Module wrapper around :func:`reconstruction_loss`.

    Off by default in the main objective: instantiate and add its
    ``total`` to ``PoseLoss`` output only inside a gated Stage-4
    experiment, with ``ramp`` scheduled from 0. Keeps the 2D contract
    unchanged.
    """

    def __init__(
        self,
        w_ab: float = 1.0,
        w_c: float = 1.0,
        beta: float = 0.05,
        detach_plane: bool = True,
    ) -> None:
        super().__init__()
        self.w_ab = w_ab
        self.w_c = w_c
        self.beta = beta
        self.detach_plane = detach_plane

    def forward(
        self,
        a_px: torch.Tensor,
        b_px: torch.Tensor,
        c_px: torch.Tensor,
        K: torch.Tensor,
        R: torch.Tensor,
        C: torch.Tensor,
        gt_a: torch.Tensor,
        gt_b: torch.Tensor,
        gt_disc: torch.Tensor,
        ramp: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        return reconstruction_loss(
            a_px,
            b_px,
            c_px,
            K,
            R,
            C,
            gt_a,
            gt_b,
            gt_disc,
            w_ab=self.w_ab,
            w_c=self.w_c,
            beta=self.beta,
            ramp=ramp,
            detach_plane=self.detach_plane,
        )
