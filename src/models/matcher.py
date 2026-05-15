"""Center-point matcher for the stride-32 single-scale head.

Each ground-truth wheel is assigned to exactly one cell — the one
whose center is closest to the wheel's bbox center. All other cells
are negatives for that image. With at most a handful of wheels per
image and a 20×20 grid (400 cells per 640² input), collisions are
rare and resolved by area-priority (smaller wheel wins).

This is the simpler stand-in for the SimOTA matcher referenced in
docs/MODEL_ARCHITECTURE_PROPOSAL.md §2.5. Center-point assignment
gives ~the same training signal for non-overlapping targets like
wheels; SimOTA mainly helps when objects overlap heavily. Wheels do
not.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.models.mobilenetv2_skipless_pose import FEATURE_STRIDE, N_KEYPOINTS


@dataclass
class MatchResult:
    """Per-image assignment of ground truth → grid cells.

    Shapes assume a single image and a (H, W) grid. Stack along batch
    in the caller.

    cls_target:  (H*W,) {0, 1} — 1 at positive cells
    bbox_target: (H*W, 4) — (l, t, r, b) in stride units, only valid
                 at positive cells (don't use loss elsewhere)
    kpt_target:  (H*W, K, 2) — (dx, dy) in stride units, only valid
                 at positive cells
    vis_target:  (H*W, K) — {0, 1} per keypoint visibility, only valid
                 at positive cells
    pos_mask:    (H*W,) bool — True at positive cells (used to gate
                 the regression / keypoint / visibility losses)
    """

    cls_target: torch.Tensor
    bbox_target: torch.Tensor
    kpt_target: torch.Tensor
    vis_target: torch.Tensor
    pos_mask: torch.Tensor


def _make_grid_centers(
    H: int, W: int, stride: int, device: torch.device
) -> torch.Tensor:
    """Return (H*W, 2) tensor of cell-center pixel coordinates."""
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    cx = (xs + 0.5) * stride
    cy = (ys + 0.5) * stride
    return torch.stack([cx, cy], dim=-1).reshape(-1, 2)


def assign_targets_single(
    gt_bboxes: torch.Tensor,  # (N, 4) xyxy in pixels
    gt_keypoints: torch.Tensor,  # (N, K, 2) (x, y) in pixels
    gt_visibility: torch.Tensor,  # (N, K) in {0, 1}
    grid_hw: tuple[int, int],
    stride: int = FEATURE_STRIDE,
) -> MatchResult:
    """Center-point assignment for one image."""
    H, W = grid_hw
    device = gt_bboxes.device if gt_bboxes.numel() > 0 else torch.device("cpu")
    n_cells = H * W

    cls_target = torch.zeros(n_cells, dtype=torch.float32, device=device)
    bbox_target = torch.zeros(n_cells, 4, dtype=torch.float32, device=device)
    kpt_target = torch.zeros(
        n_cells, N_KEYPOINTS, 2, dtype=torch.float32, device=device
    )
    vis_target = torch.zeros(n_cells, N_KEYPOINTS, dtype=torch.float32, device=device)
    pos_mask = torch.zeros(n_cells, dtype=torch.bool, device=device)

    if gt_bboxes.numel() == 0:
        return MatchResult(cls_target, bbox_target, kpt_target, vis_target, pos_mask)

    centers = _make_grid_centers(H, W, stride, device)  # (n_cells, 2)

    # Smaller wheels first → bigger wheels overwrite if they take the
    # same cell. This biases assignment toward the bigger overlapping
    # object on collision, which is the historically robust choice for
    # FCOS-style heads.
    bbox_w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
    bbox_h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
    bbox_area = bbox_w * bbox_h
    order = torch.argsort(bbox_area)  # ascending

    for idx in order.tolist():
        bx1, by1, bx2, by2 = gt_bboxes[idx].tolist()
        gt_cx = 0.5 * (bx1 + bx2)
        gt_cy = 0.5 * (by1 + by2)

        # Nearest cell by L2 distance from gt center.
        dists = (centers[:, 0] - gt_cx) ** 2 + (centers[:, 1] - gt_cy) ** 2
        cell = int(torch.argmin(dists).item())

        cls_target[cell] = 1.0
        pos_mask[cell] = True

        cell_cx, cell_cy = centers[cell].tolist()
        # bbox targets: (l, t, r, b) distances from cell center, in
        # *stride* units (i.e. the same units the head emits — divide
        # pixel distances by stride). Clamp to >= 0 to avoid negatives
        # from cell-centers slightly outside the bbox.
        bbox_target[cell, 0] = max(0.0, (cell_cx - bx1) / stride)
        bbox_target[cell, 1] = max(0.0, (cell_cy - by1) / stride)
        bbox_target[cell, 2] = max(0.0, (bx2 - cell_cx) / stride)
        bbox_target[cell, 3] = max(0.0, (by2 - cell_cy) / stride)

        # keypoint offsets in stride units.
        kpt_target[cell, :, 0] = (gt_keypoints[idx, :, 0] - cell_cx) / stride
        kpt_target[cell, :, 1] = (gt_keypoints[idx, :, 1] - cell_cy) / stride
        vis_target[cell] = gt_visibility[idx].float()

    return MatchResult(cls_target, bbox_target, kpt_target, vis_target, pos_mask)


def assign_targets_batched(
    gt_bboxes_list: list[torch.Tensor],
    gt_keypoints_list: list[torch.Tensor],
    gt_visibility_list: list[torch.Tensor],
    grid_hw: tuple[int, int],
    stride: int = FEATURE_STRIDE,
) -> MatchResult:
    """Stack per-image assignments along the batch dim.

    Each entry of the *_list is the GT for one image; the lists must
    all have the same length B. Outputs have a leading batch dim.
    """
    if not (len(gt_bboxes_list) == len(gt_keypoints_list) == len(gt_visibility_list)):
        raise ValueError("gt lists must have the same length (batch size)")

    results = [
        assign_targets_single(b, k, v, grid_hw, stride=stride)
        for b, k, v in zip(gt_bboxes_list, gt_keypoints_list, gt_visibility_list)
    ]
    return MatchResult(
        cls_target=torch.stack([r.cls_target for r in results]),
        bbox_target=torch.stack([r.bbox_target for r in results]),
        kpt_target=torch.stack([r.kpt_target for r in results]),
        vis_target=torch.stack([r.vis_target for r in results]),
        pos_mask=torch.stack([r.pos_mask for r in results]),
    )
