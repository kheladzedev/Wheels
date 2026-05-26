# Model Architecture Proposal — VSBL Wheel Pose Detector

Forward-looking design doc for the production wheel detector. Written
2026-05-14 while customer-side data is pending. Target backbone is
fixed per product direction: **MobileNetV2-skipless** (single feature
scale, no FPN). This doc specifies the full pipeline around that
backbone — head, loss, training recipe, data plan, KPIs — so that
once real plugin data lands, the team can train end-to-end without
re-litigating architecture.

## 0. TL;DR

- **Production architecture: MobileNetV2-skipless encoder + single-scale
  pose head.** ~3.4M backbone params + ~0.4M head = ~3.8M total. INT8
  TFLite target ~4 MB. Reason: mobile-friendly weight + runtime,
  proven NNAPI/Hexagon path on Android, clean quantization surface.
- **Today (this PR):** wheel_v4_real fine-tune of YOLO11n-pose on a
  seed of 8 hand-labelled real images + cartoon synthetic. This is
  **plumbing validation**, not the production model. Used to prove
  the data path works on non-cartoon inputs.
- **Real-data milestone (≥ 2 k labelled frames):** train the
  MobileNetV2-skipless model from scratch (ImageNet-pretrained
  encoder) on the real-only set. This is the first true production
  candidate.
- **Hard requirement before implementation:** lift the repository
  "no torch outside ultralytics" dependency rule. That's the gating
  migration step.

## 1. Current state (what we ship today)

`runs/pose/wheel_v3/weights/best.pt` — YOLO11n-pose, trained on
cartoon synthetic (`data/wheel_dataset`, 240 train / 60 val). val
mAP50 = 0.984 in-distribution, **fails badly on real photos**
(headlights/mirrors fire as wheels — see
`outputs/manual_real_predictions/`).

This PR adds `wheel_v4_real`, a fine-tune from wheel_v3 on a
6-real + 2-real-val seed batch mixed (×20 upsampled) with the
cartoon set. Used as the live model **only until the production
architecture is trained**.

**Important caveat — wheel_v4_real is NOT AR-mock-ready.** The seed
batch was labelled under legacy rim semantics (A/B = left/right rim
edges, C = lowest disc point), **not** the 2026-05-13 floor-ray
contract (A/B = floor-ray points near the wheel's ground footprint).
The geometry audit
(`outputs/real_infer_geometry_audit.md`,
`scripts/audit_geometry.py`) shows wheel_v4_real fails both audited
wheels on:

- A/B not in the lower band of the bbox (rel_y ≈ 0.47 vs target ≥ 0.80);
- C below A/B in image coordinates (inverted vs. the floor-ray rule).

This is by construction — the model emits what it was trained on.
Relabelling the seed under the floor-ray contract and re-fine-tuning
is a prerequisite for AR mock integration.

Additionally (resolved 2026-05-14): `src/infer_batch.py` now writes
the confirmed schema as primary `<stem>__frame_XXX.json`; the legacy
intermediate is `<stem>__frame_XXX_legacy.json` and only emitted when
`--emit-legacy` is passed. The previously emitted pre-confirmed
"target" draft JSON is no longer produced. The semantic A/B/C
mismatch on wheel_v4_real (legacy rim vs. floor-ray contract) still
stands — schema is aligned, geometry is not.

## 2. Production architecture — MobileNetV2-skipless

### 2.1 Why this choice (rationale recorded)

| Reason | Notes |
|---|---|
| **Weight** | MobileNetV2 backbone: ~3.4M params, ~300 MFLOPs @ 224². At 640² input the input size we actually use, ~2.5 GFLOPs total — under YOLO11n-pose's 6.7 GFLOPs and roughly half its activation memory. |
| **Mobile inference path** | MobileNetV2 was co-designed with Google's mobile inference stack. Maps cleanly to TFLite XNNPACK on CPU, NNAPI on Android, Hexagon DSP on Snapdragon. YOLO11's custom C3k2 blocks rely on ops that NNAPI sometimes falls back to CPU. |
| **No FPN = simpler graph** | Dropping the top-down + lateral skip aggregation removes the resize+add ops that are commonly the INT8 quantization-error hotspot. The exported TFLite is a flatter graph that quantizers handle without surprises. |
| **Scale-matched task** | In-AR-session wheels live in a narrow scale band (~50–400 px at 1080p input). Multi-scale FPN heads compete for the same instance and lose accuracy when scale is bounded. Single-scale head is the right inductive bias. |
| **Decoupled from ultralytics** | Owning the model surface means we can evolve A/B keypoint semantics (e.g. 2026-05-13 floor-ray contract change) without fighting upstream defaults on matcher/loss/augmentation. |

### 2.2 Topology

Input: 3 × 640 × 640 (resized + letterboxed from the source frame).

```
Stage          Op                                            Stride  Out shape       Params
input          rgb_normalize (imagenet mean/std)             1       3 × 640 × 640   —
encoder.s2     conv2d 3×3 s=2 + bn + relu6                    2       32 × 320 × 320   864
encoder.b1     InvertedResidual t=1 c=16  n=1 s=1             2       16 × 320 × 320   624
encoder.b2     InvertedResidual t=6 c=24  n=2 s=2             4       24 × 160 × 160   5 136
encoder.b3     InvertedResidual t=6 c=32  n=3 s=2             8       32 × 80  × 80   17 360
encoder.b4     InvertedResidual t=6 c=64  n=4 s=2            16       64 × 40  × 40   86 144
encoder.b5     InvertedResidual t=6 c=96  n=3 s=1            16       96 × 40  × 40  191 168
encoder.b6     InvertedResidual t=6 c=160 n=3 s=2            32      160 × 20  × 20  616 320
encoder.b7     InvertedResidual t=6 c=320 n=1 s=1            32      320 × 20  × 20  481 200
encoder.last   conv2d 1×1 + bn + relu6                       32     1280 × 20  × 20  411 520
                                                                                    -------
                                                                             total: ~3.4M
head.proj      conv2d 1×1 (1280→256) + bn + relu             32      256 × 20  × 20  327 936
head.tower     2 × (depthwise 3×3 + bn + relu) + (1×1 256)   32      256 × 20  × 20  ~150 000
head.cls       conv2d 1×1 → 1 (sigmoid)                      32        1 × 20  × 20      257
head.bbox      conv2d 1×1 → 4 (l, t, r, b offsets, ReLU)     32        4 × 20  × 20    1 028
head.kpt       conv2d 1×1 → 9 (3 × (x, y, vis))              32        9 × 20  × 20    2 313
                                                                                    -------
                                                                             total: ~3.8M end-to-end
```

Notes:
- "Skipless" = no top-down pathway from b6 to b3/b2 (no FPN). Only
  the C5/stride-32 feature is used by the head. b3/b4/b5 still exist
  in the encoder graph; we simply do not tap them.
- Output stride 32 → 20×20 feature grid at 640² input. Each cell
  predicts one anchor-free wheel instance.
- Head is decoupled (classification, bbox, keypoint towers share the
  feature but have separate output heads). FCOS-style: bbox is
  (l, t, r, b) distances from the cell center.
- Keypoints regress (x, y) offsets *from the cell center*, scaled by
  the bbox diagonal (sub-pixel friendly under INT8).
- Visibility is a per-keypoint sigmoid in [0, 1]. Below 0.5 → dropped
  before emission to the AR JSON (matches contract: no `visibility`
  field downstream).

### 2.3 Sub-pixel keypoint head — important detail

Stride 32 alone gives ~32 px keypoint quantisation, which would
break AR's RANSAC tolerance. We resolve this with **center-offset
regression**: at the assigned cell, the head predicts (dx, dy) in
units of cell width (i.e. (-0.5, 0.5) on each axis means the kpt is
anywhere in the cell). Decoded keypoint = (cell_cx + dx · 32,
cell_cy + dy · 32). This is a single-step refinement that costs
3 × 2 = 6 extra channels, no extra FLOPs of consequence, and gives
sub-pixel localisation matching what AR's RANSAC needs.

### 2.4 Loss

```
L_total = lambda_cls  · L_focal(cls)
        + lambda_bbox · L_giou(bbox)        (only on positive cells)
        + lambda_kpt  · L_oks(keypoints)    (only on positive cells)
        + lambda_vis  · L_bce(visibility)   (only on positive cells)
```

- `L_focal`: alpha=0.25, gamma=2.0 — handles the heavy negative-cell
  imbalance (1 positive per wheel out of 400 cells per image).
- `L_giou`: GIoU on decoded boxes. More robust than L1 when bbox
  predictions are far from ground truth (early training).
- `L_oks`: OKS-style keypoint loss (1 − OKS) with uniform sigma=0.05
  of bbox-diagonal. Penalises tiny offsets more on small wheels.
- `L_bce`: per-keypoint visibility supervision; v_gt = 1 for emitted
  points, 0 for "wheel labeled but this point missing" (contract
  says we never emit such wheels, but if real data contains partials
  we want the model to suppress them).
- Suggested weights: lambda_cls=1.0, lambda_bbox=2.0, lambda_kpt=8.0,
  lambda_vis=1.0. AR cares about keypoint precision more than bbox
  tightness.

### 2.5 Matcher

SimOTA / dynamic-k cost = focal_cls + 3·giou + 8·oks. One positive
cell per ground-truth wheel (k=1 is fine because wheels rarely
overlap in screen space). Negative cells outside any wheel's center
region (1.5 × cell radius) contribute only the focal-cls loss.

## 3. Training recipe

### 3.1 Data plan (the unblocker)

Per `docs/PLUGIN_DATA_EXPECTATION.md`, the Android plugin will drop
batches to `data/incoming/android_plugin/`. Before greenlighting the
production training run we need:

| Bucket | Target count | Notes |
|---|---|---|
| Real, AR-session-like | ≥ 2 000 | Phone camera, 1–3 m from car, indoor + outdoor mix |
| Real, occluded-wheel frames | ≥ 200 | Negatives — must NOT be emitted (contract §3) |
| Real, no-car frames | ≥ 200 | Hard negatives: floors, walls, lighting |
| Real, wrong-target frames | ≥ 200 | Bicycles, motorcycles — round-wheel-like NOT-cars |
| Real, hard backgrounds | ≥ 100 | Wet road, dappled shade, reflections, dirty rims |
| Synthetic (existing cartoon) | ≤ 10% | Optional; keep small to not bias the prior |

Labels follow the 2026-05-13 floor-ray contract: A/B are
screen-space floor points near the wheel's ground footprint, C is
the lowest visible point of the metal disc. Legacy rim semantics
(used in current synthetic + this PR's seed) MUST be re-labelled
under the new contract before production training.

### 3.2 Augmentation

- Mosaic ×4: on for first 80% of epochs, off for last 20%.
- HSV jitter h=0.015, s=0.7, v=0.4–0.6 (more aggressive than YOLO
  default — AR lighting varies more than dataset).
- Horizontal flip 0.5, flip_idx=[1, 0, 2] (A↔B swap, C fixed).
- Random scale 0.5×–1.5×. No vertical translation > 0.05 (wheels
  are ground-anchored).
- Mixup 0.1 — encourages decision boundary smoothing.
- Copy-paste 0.3 — paste real wheels onto wrong contexts (walls,
  no-car frames) → hard negatives for free.
- Random JPEG re-compress 30–80% quality 0.5 prob — AR captures may
  be lossy.

### 3.3 Schedule (production training)

| Phase | Epochs | LR | Notes |
|---|---|---|---|
| Warmup | 3 | 1e-4 → 1e-3 linear | encoder *frozen*, head only |
| Head ramp | 7 | 1e-3 cos to 5e-4 | encoder frozen |
| Full fine-tune | 40 | 5e-4 cos to 1e-5 | encoder unfrozen, full backprop |
| Polish | 10 | 1e-5 const, mosaic off | for stable val metric |

Optimizer: AdamW (betas=0.9, 0.999, wd=5e-4). Batch 32 on a single
A100; 16 on mps. Mixed-precision fp16. Early-stop patience=15 on
val pose mAP50-95.

### 3.4 Encoder pretraining

Initialize encoder from torchvision's ImageNet-1k MobileNetV2
weights. Two design questions to revisit once data is in:

1. **Re-pretrain encoder on a self-supervised wheel-context task?**
   E.g. SimCLR on unlabelled AR-session frames. Could help if the
   labelled real set is small (< 5 k frames). Not worth deciding
   abstractly — measure with and without.
2. **Distill from a heavier model?** Train a YOLO11s or larger
   teacher first, then distill into MobileNetV2-skipless. Common
   pattern when the deployment-target model is tiny but accuracy
   must be high.

### 3.5 Quantization & export

- Train in FP32 with FP16 mixed-precision.
- Calibrate INT8 with 500 real frames from the train set (not val).
- Export path: PyTorch → ONNX → TFLite. Ultralytics' export is not
  reusable here; we own the export pipeline. Use TFLite-converter
  with default optimisation + `representative_dataset` from the
  500 calibration frames.
- Acceptance: INT8 mAP50 drop ≤ 5pp vs FP32. If above, fall back
  to FP16 export (still mobile-friendly, ~2× model size).

## 4. KPIs / acceptance gates

| Metric | Target (v1) | Target (prod) | Notes |
|---|---|---|---|
| val box mAP50 | ≥ 0.85 | ≥ 0.95 | Standard COCO-style |
| val pose mAP50-95 | ≥ 0.55 | ≥ 0.75 | OKS-based |
| FP-rate on hard negatives | < 1.0 false wheel/img | < 0.1 false wheel/img | The actual real-world failure mode |
| On-device latency on target Android tier | within budget | within budget | Open Q for AR team |
| INT8 mAP50 drop vs FP32 | < 7pp | < 3pp | Quantization cleanliness |
| AR-side disc-height sigma (K=10 RANSAC) | < 3 cm | < 1 cm | The user-visible quality |

The disc-height sigma is the only metric that *actually* reflects
user-perceived quality. mAP without it is academic.

## 5. Stub: `src/models/mobilenetv2_skipless_pose.py`

Reference Python module. Lives behind a future `/goal` once the
torch dependency is approved. Included here so the team has a
concrete starting point.

```python
# src/models/mobilenetv2_skipless_pose.py (DRAFT — not yet wired in)
#
# Requires: torch >= 2.0, torchvision >= 0.15.
# NOT INSTALLED YET — see the repository dependency rule that keeps
# torch/torchvision isolated from the ultralytics path.
# Implementing this is a separate migration step that lifts that rule.

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


N_KEYPOINTS = 3  # a, b, c_disc_bottom (contract is frozen)


class WheelPoseHead(nn.Module):
    """Single-scale FCOS-style head on top of stride-32 features."""

    def __init__(self, in_channels: int = 1280, mid: int = 256) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(
            *(self._dw_block(mid) for _ in range(2))
        )
        self.cls = nn.Conv2d(mid, 1, 1)
        self.bbox = nn.Conv2d(mid, 4, 1)
        self.kpt = nn.Conv2d(mid, N_KEYPOINTS * 2, 1)
        self.vis = nn.Conv2d(mid, N_KEYPOINTS, 1)

    @staticmethod
    def _dw_block(c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.tower(self.proj(feat))
        return {
            "cls": self.cls(x),                 # (B, 1, 20, 20)
            "bbox": torch.relu(self.bbox(x)),    # >= 0 distances
            "kpt": self.kpt(x),                  # (B, 6, 20, 20)
            "vis": self.vis(x),                  # (B, 3, 20, 20) logits
        }


class MobileNetV2SkiplessPose(nn.Module):
    """MobileNetV2 encoder, single-scale (stride 32) pose head.

    The "skipless" property is implicit: we only tap the final feature
    map and ignore lateral connections that an FPN would build.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = mobilenet_v2(weights=weights)
        # Everything up to and including the 1280-channel last conv.
        self.encoder = backbone.features
        self.head = WheelPoseHead(in_channels=1280)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(x)  # (B, 1280, H/32, W/32)
        return self.head(feat)


def export_tflite(
    model: MobileNetV2SkiplessPose,
    calib_loader,            # iterable of FP32 tensors (B, 3, 640, 640)
    out_path: str,
) -> None:
    """Sketch of the int8 export path. Real code lives in a /goal."""
    import torch.onnx

    onnx_path = out_path.replace(".tflite", ".onnx")
    dummy = torch.randn(1, 3, 640, 640)
    model.train(False)  # inference mode
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["image"], output_names=["cls", "bbox", "kpt", "vis"],
        opset_version=17, dynamic_axes={"image": {0: "batch"}},
    )
    # ONNX -> TFLite + INT8 via onnx2tf or tflite-converter.
    # The representative_dataset comes from calib_loader; see
    # docs/MODEL_ARCHITECTURE_PROPOSAL.md section 3.5 for the recipe.
```

## 6. Risks and validations during implementation

Items I flag as a senior engineer — *not* blockers to the
architecture choice, but things the team must measure during
implementation. If any of these fail, the model evolves (e.g. add a
second head scale) rather than the backbone changes.

| Risk | What to measure | Mitigation if it bites |
|---|---|---|
| Real-data scale variance exceeds single-scale capacity | Histogram of wheel bbox-diagonal / image-diagonal on the first 500 real frames | Add a second head at stride 16 (still no FPN, just two parallel heads). Cheap fix. |
| Stride-32 + sub-pixel offset insufficient for AR's RANSAC sigma target | A/B pixel error on val set vs sigma-target (open Q for AR) | Add a refinement head at stride 8 (cropped per-instance). +0.2 GFLOPs, regains precision. |
| INT8 quantization drops mAP50 > 5pp | Run after every retrain; calibrate on real data | Per-channel quantization on conv weights; FP16 fallback for the head. |
| MobileNetV2 is 2018 architecture, MobileNetV3-Small or EfficientNet-Lite0 may be Pareto-better | Same training recipe, three runs in parallel once data is in | Swap backbone (head unchanged) — head is the load-bearing IP. |
| AR-team budget unknown | Latency + size measurements on target device once we have a trained model | Distil-from-larger (see §3.4.2) if undersized; head pruning if oversized |

## 7. Open questions — AR team / product

Block implementation until answered:

1. **Target Android tier** — lowest device + chipset. Determines
   latency budget by 10×.
2. **Per-frame latency budget** — fixed ms, or "as low as you can"?
3. **App size / memory budget** — total ML budget on the AR bundle?
4. **Quantization tolerance** — what mAP50 drop is acceptable?
   (Open as Q10 in `docs/QUESTIONS_FOR_TEAM.md`.)
5. **RANSAC sigma tolerance** — A/B pixel error AR can absorb before
   disc-height sigma exceeds the user-visible threshold (1 cm prod,
   3 cm v1)?

## 8. Action items / sequence

- [x] **This PR**: wheel_v4_real fine-tune of YOLO11n-pose on
      8-real-image seed. Proves data path. Not production.
- [x] **This PR**: this architecture document + stub model module.
- [x] **Phase 1 — migration `/goal` (executed)**: torch dependency
      rule lifted (scoped to `src/models/`), production pipeline
      implemented:
      - `src/models/mobilenetv2_skipless_pose.py` — model (2.69M params)
      - `src/models/matcher.py` — center-point assigner
      - `src/models/loss.py` — focal + GIoU + OKS + BCE
      - `scripts/train_mobilenetv2_skipless.py` — training skeleton,
        works on synthetic batches as a smoke
      - `tests/test_mobilenetv2_skipless_pose.py`, `tests/test_matcher.py`,
        `tests/test_loss.py` — 22 unit tests, all green
- [ ] **Next session**: file Q1–Q5 above with AR team in
      `docs/OPEN_QUESTIONS_AR_SPEC.md`. Block production training on
      Q1–Q3.
- [ ] **When plugin batch ≥ 500 frames arrives**: relabel under
      the 2026-05-13 floor-ray contract, run the histograms in
      §6 (scale variance, A/B error potential), confirm
      single-scale assumption holds.
- [ ] **When plugin batch ≥ 2 000 frames + AR Q1–Q3 answered —
      Phase 2 `/goal`**: real Dataset implementation, swap
      `_synthetic_batch` in the training script, run the full 60-epoch
      schedule from §3.3, validate KPIs (§4), export INT8 TFLite.
- [ ] **After first prod build**: AR-side disc-height sigma
      measurement under real RANSAC. The only KPI that matters.

## 9. See also

- Repository dependency policy — the "no torch outside ultralytics" rule
  must be lifted by the migration step.
- `docs/AR_ML_CONTRACT.md` — runtime JSON contract; this
  architecture must emit it byte-for-byte.
- `docs/KEYPOINT_SPEC.md` — A/B/C semantics under the 2026-05-13
  floor-ray contract.
- `docs/OPEN_QUESTIONS_AR_SPEC.md` — pending AR confirmations.
- `runs/pose/wheel_v3/args.yaml` — current training config.
- `runs/pose/wheel_v4_real/` — this PR's fine-tune outputs
  (plumbing validation, not production).
