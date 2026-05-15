"""MobileNetV2-skipless wheel pose detector.

Single-scale anchor-free detector on top of MobileNetV2's C5
(stride-32) feature map. No FPN, no skip connections from earlier
stages — by design (see docs/MODEL_ARCHITECTURE_PROPOSAL.md §2.1).

Outputs per cell at stride 32:
  cls  — sigmoid logit, P(wheel) at the cell
  bbox — (l, t, r, b) distances from the cell center, scaled by stride
         (FCOS-style). Decode: x1 = cx - l*32, etc.
  kpt  — per-keypoint (dx, dy) center-offsets in [-0.5, 0.5] units of
         the cell width. Decode: kp_x = cx + dx*32. Sub-pixel friendly.
  vis  — per-keypoint visibility logit. Sigmoid > 0.5 → visible.

At 640² input the grid is 20×20 = 400 cells. Each ground-truth wheel
is assigned to exactly one cell (the one containing its bbox center);
all others are negatives. See `src/models/matcher.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

N_KEYPOINTS = 3  # a, b, c_disc_bottom — contract is frozen
FEATURE_STRIDE = 32  # MobileNetV2 last block downsampling


class WheelPoseHead(nn.Module):
    """FCOS-style decoupled head producing cls / bbox / kpt / vis.

    Shared projection + tower, then four 1×1 output heads. Tower uses
    depthwise-separable convs to keep parameter count low — matches
    the mobile-friendly philosophy of the backbone.
    """

    def __init__(self, in_channels: int = 1280, mid: int = 256) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(
            self._dw_block(mid),
            self._dw_block(mid),
        )
        self.cls = nn.Conv2d(mid, 1, kernel_size=1)
        self.bbox = nn.Conv2d(mid, 4, kernel_size=1)
        self.kpt = nn.Conv2d(mid, N_KEYPOINTS * 2, kernel_size=1)
        self.vis = nn.Conv2d(mid, N_KEYPOINTS, kernel_size=1)

        # Initialise cls bias so that focal loss starts at a sane prior
        # (prevents the loss exploding on the first batches because the
        # network would otherwise predict P(wheel)=0.5 everywhere).
        # bias = -log((1-p)/p) with p=0.01 → 99% negatives prior.
        nn.init.constant_(self.cls.bias, -4.595)

    @staticmethod
    def _dw_block(c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.tower(self.proj(feat))
        return {
            "cls": self.cls(x),  # (B, 1, H, W)
            "bbox": torch.relu(self.bbox(x)),  # >= 0 distances
            "kpt": self.kpt(x),  # (B, 2*K, H, W)
            "vis": self.vis(x),  # (B, K, H, W) logits
        }


class MobileNetV2SkiplessPose(nn.Module):
    """MobileNetV2 encoder, single-scale (stride 32) pose head.

    "Skipless" = we tap only the final encoder feature map and ignore
    the lateral connections that an FPN would build. Earlier-stage
    features (b3/b4/b5) still exist in the encoder graph; we simply
    do not route them to the head. This keeps the inference graph
    flat — better for INT8 quantization on Android NNAPI.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = mobilenet_v2(weights=weights)
        # torchvision's MobileNetV2.features is a Sequential that ends
        # with a 1×1 conv promoting to 1280 channels at stride 32.
        self.encoder = backbone.features
        self.head = WheelPoseHead(in_channels=1280)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(x)  # (B, 1280, H/32, W/32)
        return self.head(feat)


def decode_predictions(
    preds: dict[str, torch.Tensor],
    stride: int = FEATURE_STRIDE,
) -> dict[str, torch.Tensor]:
    """Decode raw head outputs into per-cell pixel-space predictions.

    Returns:
      cls_prob: (B, H*W) — sigmoid of cls logits, flattened
      bbox_xyxy: (B, H*W, 4) — decoded boxes in pixel coords
      kpt_xy: (B, H*W, K, 2) — decoded keypoints in pixel coords
      vis_prob: (B, H*W, K) — sigmoid of visibility logits

    Caller filters by cls_prob threshold + NMS to get final detections.
    """
    B, _, H, W = preds["cls"].shape
    device = preds["cls"].device

    # Cell-center grid in pixel coordinates.
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    cx = (xs + 0.5) * stride  # (H, W)
    cy = (ys + 0.5) * stride

    cls_prob = preds["cls"].sigmoid().flatten(2).squeeze(1)  # (B, H*W)

    # bbox: (l, t, r, b) distances in stride units → pixel xyxy.
    ltrb = preds["bbox"].permute(0, 2, 3, 1).reshape(B, H * W, 4) * stride
    cx_flat = cx.reshape(-1)  # (H*W,)
    cy_flat = cy.reshape(-1)
    bbox_xyxy = torch.stack(
        [
            cx_flat - ltrb[..., 0],
            cy_flat - ltrb[..., 1],
            cx_flat + ltrb[..., 2],
            cy_flat + ltrb[..., 3],
        ],
        dim=-1,
    )

    # kpt offsets (dx, dy) per kp in cell-width units, → pixel.
    kpt_offs = (
        preds["kpt"].permute(0, 2, 3, 1).reshape(B, H * W, N_KEYPOINTS, 2) * stride
    )
    kpt_xy = torch.stack(
        [
            cx_flat.unsqueeze(-1) + kpt_offs[..., 0],
            cy_flat.unsqueeze(-1) + kpt_offs[..., 1],
        ],
        dim=-1,
    )

    vis_prob = preds["vis"].permute(0, 2, 3, 1).reshape(B, H * W, N_KEYPOINTS).sigmoid()

    return {
        "cls_prob": cls_prob,
        "bbox_xyxy": bbox_xyxy,
        "kpt_xy": kpt_xy,
        "vis_prob": vis_prob,
    }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
