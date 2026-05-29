"""Web multi-task wheel model: detection + keypoints + floor estimate.

A *separate* deployment surface from the frozen Android/iOS 2D AR
contract (``docs/AR_ML_CONTRACT.md``). The web client has no AR
framework to own the 3D pipeline, so this network additionally
regresses the camera-vs-floor pose ``{pitch, roll, delta_z}`` from
global features. With that floor estimate the web runtime can recover
3D itself, while the mobile contract path keeps emitting only 2D.

Architecture (``docs/MODEL_ARCHITECTURE_PROPOSAL.md`` lineage):

  - MobileNetV2 encoder at 512x512 input -> stride-32, 16x16, 1280-ch
    feature map (shared trunk).
  - Pose head: the same FCOS-style cls / bbox / kpt / vis head used by
    the skipless mobile model (reused, not reimplemented).
  - Floor head: global-average-pooled trunk features -> small MLP ->
    3 scalars ``[pitch, roll, delta_z]``.

Training is **staged** (``set_stage``): 2D-only first (floor head
frozen), then floor, then the 3D reconstruction loss, then joint — with
``detach_floor`` available so the floor task does not perturb the shared
trunk before the 2D head is stable. Task weighting uses learnable
homoscedastic uncertainty (Kendall, Gal & Cipolla 2018) instead of hand
tuning. The ``delta_z`` output is scale-relative; a metric anchor /
prior for ``delta_z`` and the FOV-at-inference decision are open
(tracked in the goal), so the head is built to make swapping that in a
one-line change.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mobilenetv2_skipless_pose import WheelPoseHead
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

FLOOR_DOF = 3  # pitch, roll, delta_z
_STAGES = ("2d", "floor", "recon", "joint")


class FloorHead(nn.Module):
    """Global-feature regressor for ``{pitch, roll, delta_z}``.

    Global-average-pools the trunk feature map, then a 2-layer MLP. The
    three outputs are returned raw (no activation): pitch/roll are
    angles in radians, ``delta_z`` is scale-relative until a metric
    anchor is wired in.
    """

    def __init__(self, in_channels: int = 1280, mid: int = 128) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, FLOOR_DOF),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)  # (B, C)
        return self.mlp(pooled)  # (B, 3)


class WebMultiTaskModel(nn.Module):
    """Shared MobileNetV2 trunk -> pose head + floor head, 512x512 input."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        self.encoder = mobilenet_v2(weights=weights).features
        self.head = WheelPoseHead(in_channels=1280)
        self.floor_head = FloorHead(in_channels=1280)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(x)  # (B, 1280, H/32, W/32)
        out = self.head(feat)
        out["floor"] = self.floor_head(feat)
        return out

    def set_stage(self, stage: str) -> None:
        """Freeze / unfreeze heads for the staged training schedule.

        ``2d``    floor head frozen (train detection + keypoints only)
        ``floor`` floor head trains; pose head still trains
        ``recon`` everything trains (3D reconstruction loss switched on
                  by the trainer with a ramp)
        ``joint`` everything trains
        """
        if stage not in _STAGES:
            raise ValueError(f"stage must be one of {_STAGES}, got {stage!r}")
        floor_on = stage in ("floor", "recon", "joint")
        for p in self.floor_head.parameters():
            p.requires_grad_(floor_on)


# Per-DoF reference magnitudes for [pitch(rad), roll(rad), delta_z].
# pitch/roll live on a ~0.1 rad scale while delta_z is metres-scale; a
# single shared Huber would let delta_z dominate the angular residuals
# ~15x. Residuals are normalised by these before the Huber so each DoF
# contributes comparably. Tune per dataset once real GT lands.
FLOOR_SCALE = (0.1, 0.1, 1.5)


def floor_loss(
    floor_pred: torch.Tensor,
    gt_floor: torch.Tensor,
    scale: tuple[float, float, float] = FLOOR_SCALE,
    beta: float = 1.0,
) -> torch.Tensor:
    """Scale-normalised Huber on ``[pitch, roll, delta_z]`` (B,3).

    Each DoF residual is divided by its reference ``scale`` so the loss
    is balanced across the mixed (radian / metric) units; ``beta`` is the
    Huber knee on the *normalised* residual.
    """
    s = torch.tensor(scale, dtype=floor_pred.dtype, device=floor_pred.device)
    diff = (floor_pred - gt_floor) / s
    return F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=beta)


class MultiTaskLoss(nn.Module):
    """Combine pose + floor losses with learnable uncertainty weights.

    Kendall et al. homoscedastic weighting: for task losses ``L_i`` with
    learnable log-variances ``s_i``,

        L = sum_i exp(-s_i) * L_i + s_i

    Two tasks (pose, floor) => two scalar parameters. ``detach_floor``
    drops the floor task's gradient (used in the 2D-only stage); the
    floor term is still reported for logging.
    """

    def __init__(self) -> None:
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(2))

    def forward(
        self,
        pose_loss: torch.Tensor,
        floor_pred: torch.Tensor,
        gt_floor: torch.Tensor,
        floor_beta: float = 1.0,
        detach_floor: bool = False,
    ) -> dict[str, torch.Tensor]:
        l_floor = floor_loss(floor_pred, gt_floor, beta=floor_beta)
        floor_term_input = l_floor.detach() if detach_floor else l_floor

        pose_term = torch.exp(-self.log_var[0]) * pose_loss + self.log_var[0]
        floor_term = torch.exp(-self.log_var[1]) * floor_term_input + self.log_var[1]
        total = pose_term + floor_term
        return {
            "total": total,
            "pose": pose_loss.detach(),
            "floor": l_floor.detach(),
            "log_var": self.log_var.detach().clone(),
        }
