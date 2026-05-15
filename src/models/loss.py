"""Loss functions for the MobileNetV2-skipless wheel pose detector.

Components (per docs/MODEL_ARCHITECTURE_PROPOSAL.md §2.4):

  L_total = lambda_cls  * L_focal(cls)
          + lambda_bbox * L_giou(bbox)   # positive cells only
          + lambda_kpt  * L_oks(kpt)     # positive cells only
          + lambda_vis  * L_bce(vis)     # positive cells only

All loss components operate on the flat (B, H*W, ...) layout produced
by the matcher. Positives are gated by `pos_mask`. If an image has no
ground-truth wheels, only the focal cls loss contributes (which still
trains the model to predict P(wheel)=0 everywhere).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.matcher import MatchResult
from src.models.mobilenetv2_skipless_pose import FEATURE_STRIDE, N_KEYPOINTS


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Standard Lin et al. focal loss, returned as a sum reduction.

    Caller divides by the positive count for FCOS-style normalisation.
    """
    p = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t).pow(gamma) * bce).sum()


def _ltrb_to_xyxy(ltrb: torch.Tensor) -> torch.Tensor:
    """Convert (l, t, r, b) distances → (x1, y1, x2, y2) in stride units.

    Cell-center is implicit at (0, 0); we only need relative geometry
    for GIoU, so we don't add cell centers here.
    """
    return torch.stack(
        [-ltrb[..., 0], -ltrb[..., 1], ltrb[..., 2], ltrb[..., 3]],
        dim=-1,
    )


def giou_loss(pred_ltrb: torch.Tensor, gt_ltrb: torch.Tensor) -> torch.Tensor:
    """1 - GIoU on positive cells, summed.

    Inputs are (N_pos, 4) — caller has already gathered positive cells.
    """
    pred_xyxy = _ltrb_to_xyxy(pred_ltrb)
    gt_xyxy = _ltrb_to_xyxy(gt_ltrb)

    pred_area = (pred_xyxy[..., 2] - pred_xyxy[..., 0]).clamp(min=0) * (
        pred_xyxy[..., 3] - pred_xyxy[..., 1]
    ).clamp(min=0)
    gt_area = (gt_xyxy[..., 2] - gt_xyxy[..., 0]) * (gt_xyxy[..., 3] - gt_xyxy[..., 1])

    inter_x1 = torch.maximum(pred_xyxy[..., 0], gt_xyxy[..., 0])
    inter_y1 = torch.maximum(pred_xyxy[..., 1], gt_xyxy[..., 1])
    inter_x2 = torch.minimum(pred_xyxy[..., 2], gt_xyxy[..., 2])
    inter_y2 = torch.minimum(pred_xyxy[..., 3], gt_xyxy[..., 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    union = pred_area + gt_area - inter
    iou = inter / union.clamp(min=1e-6)

    enc_x1 = torch.minimum(pred_xyxy[..., 0], gt_xyxy[..., 0])
    enc_y1 = torch.minimum(pred_xyxy[..., 1], gt_xyxy[..., 1])
    enc_x2 = torch.maximum(pred_xyxy[..., 2], gt_xyxy[..., 2])
    enc_y2 = torch.maximum(pred_xyxy[..., 3], gt_xyxy[..., 3])
    enclose = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0)

    giou = iou - (enclose - union) / enclose.clamp(min=1e-6)
    return (1.0 - giou).sum()


def oks_keypoint_loss(
    pred_off: torch.Tensor,  # (N_pos, K, 2) (dx, dy) in stride units
    gt_off: torch.Tensor,  # (N_pos, K, 2)
    gt_vis: torch.Tensor,  # (N_pos, K) 0/1
    bbox_diag_stride_units: torch.Tensor,  # (N_pos,) bbox diagonal in stride units
    sigma: float = 0.05,
) -> torch.Tensor:
    """OKS-style keypoint loss = 1 - exp(-d^2 / (2 * (sigma * diag)^2)).

    Penalises tiny offsets more on small wheels (diag smaller → ratio
    bigger). Visibility-masked: invisible ground-truth keypoints don't
    contribute. Returns a sum reduction; caller normalises by visible
    count.
    """
    d2 = ((pred_off - gt_off) ** 2).sum(dim=-1)  # (N_pos, K)
    scale = (sigma * bbox_diag_stride_units.clamp(min=1e-3)).unsqueeze(
        -1
    ) ** 2  # (N_pos, 1)
    oks = torch.exp(-d2 / (2.0 * scale))
    return ((1.0 - oks) * gt_vis).sum()


class PoseLoss(nn.Module):
    """Combined detection + pose loss.

    Weights match the proposal's recommended values; override at init
    time if a future experiment wants to retune them. The loss does
    *not* include a DFL term — keypoint regression is direct
    (dx, dy) and the proposal doesn't call for distribution focal loss
    on bbox either.
    """

    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_bbox: float = 2.0,
        lambda_kpt: float = 8.0,
        lambda_vis: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        oks_sigma: float = 0.05,
    ) -> None:
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_bbox = lambda_bbox
        self.lambda_kpt = lambda_kpt
        self.lambda_vis = lambda_vis
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.oks_sigma = oks_sigma

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        match: MatchResult,
    ) -> dict[str, torch.Tensor]:
        """Returns dict with `total` plus each component for logging."""
        # Flatten predictions to (B, H*W, ...) for alignment with matcher.
        B, _, H, W = preds["cls"].shape
        cls_pred = preds["cls"].permute(0, 2, 3, 1).reshape(B, H * W)
        bbox_pred = preds["bbox"].permute(0, 2, 3, 1).reshape(B, H * W, 4)
        kpt_pred = preds["kpt"].permute(0, 2, 3, 1).reshape(B, H * W, N_KEYPOINTS, 2)
        vis_pred = preds["vis"].permute(0, 2, 3, 1).reshape(B, H * W, N_KEYPOINTS)

        n_pos = match.pos_mask.sum().clamp(min=1).float()

        # Focal cls loss on all cells.
        l_cls = (
            focal_loss(
                cls_pred,
                match.cls_target,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
            )
            / n_pos
        )

        # Regression / keypoint / visibility on positive cells only.
        pos_mask = match.pos_mask
        if pos_mask.any():
            bbox_pred_pos = bbox_pred[pos_mask]
            bbox_gt_pos = match.bbox_target[pos_mask]
            l_bbox = giou_loss(bbox_pred_pos, bbox_gt_pos) / n_pos

            kpt_pred_pos = kpt_pred[pos_mask]
            kpt_gt_pos = match.kpt_target[pos_mask]
            vis_gt_pos = match.vis_target[pos_mask]
            # Bbox diagonal in stride units for OKS scaling.
            l, t, r, b = bbox_gt_pos.unbind(-1)
            w = l + r
            h = t + b
            diag = (w * w + h * h).clamp(min=1e-6).sqrt()
            visible_count = vis_gt_pos.sum().clamp(min=1.0)
            l_kpt = (
                oks_keypoint_loss(
                    kpt_pred_pos, kpt_gt_pos, vis_gt_pos, diag, sigma=self.oks_sigma
                )
                / visible_count
            )

            l_vis = F.binary_cross_entropy_with_logits(
                vis_pred[pos_mask], vis_gt_pos, reduction="mean"
            )
        else:
            zero = cls_pred.sum() * 0.0  # keep grad graph attached
            l_bbox = zero
            l_kpt = zero
            l_vis = zero

        total = (
            self.lambda_cls * l_cls
            + self.lambda_bbox * l_bbox
            + self.lambda_kpt * l_kpt
            + self.lambda_vis * l_vis
        )

        return {
            "total": total,
            "cls": l_cls.detach(),
            "bbox": l_bbox.detach(),
            "kpt": l_kpt.detach(),
            "vis": l_vis.detach(),
            "n_pos": n_pos.detach(),
        }
