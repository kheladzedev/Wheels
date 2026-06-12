# Web Floor Network Contract

This document defines the **web-only** Wheels ML output contract for the
network that predicts wheels plus lightweight floor state from one RGB image.
It does not change `docs/AR_ML_CONTRACT.md`: mobile AR remains the frozen
2D-only contract where AR owns raycasts, RANSAC, planes, and multi-frame state.

## Runtime scope

The web inference path is intentionally cheap:

- one RGB image in;
- one neural-network forward pass;
- simple tensor decode and JSON validation out.

Runtime inference must not require depth maps, segmentation masks, multi-frame
accumulation, RANSAC, or backend-side geometric postprocess. Existing 3D/floor
ray tooling in this repository is useful for offline validation, but it is not
part of the required web runtime path.

The canonical runtime scope string is:

```text
single_forward_no_depth_no_ransac
```

## Input

The first web handoff target uses RGB images resized/letterboxed to `512x512`.
Preprocessing is the same image-normalization family as the existing PyTorch
model path unless a later export manifest states otherwise.

The model input tensor is named `image` for export:

```text
image: float32[1, 3, 512, 512]
```

## Raw outputs

The stable export wrapper should expose named tensors for the wheel head and a
single floor tensor:

```text
cls:   float32[1, 1, 16, 16]
bbox:  float32[1, 4, 16, 16]
kpt:   float32[1, 6, 16, 16]
vis:   float32[1, 3, 16, 16]
floor: float32[1, 3]
```

The current implementation's `floor` tensor order is:

```text
[pitch, roll, distance]
```

`distance` maps to the legacy/internal `delta_z` slot used by
`src/models/web_multitask.py`. Public web handoff material should use
`distance` so consumers do not infer an unqualified metric-Z guarantee.

## Decoded JSON

One decoded payload per image:

```json
{
  "frame_id": "web-frame-0001",
  "runtime_scope": "single_forward_no_depth_no_ransac",
  "floor": {
    "pitch": 0.04,
    "roll": -0.01,
    "distance": 1.6,
    "distance_mode": "scale_relative",
    "fov_mode": "unknown"
  },
  "wheels": [
    {
      "bbox_xyxy": [100.0, 210.0, 220.0, 350.0],
      "confidence": 0.91,
      "points": {
        "a": [112.0, 340.0],
        "b": [208.0, 341.0],
        "c_disc_bottom": [160.0, 305.0]
      }
    }
  ]
}
```

### Top-level fields

| Field | Required | Notes |
|---|---:|---|
| `frame_id` | no | Optional echo/id for the caller. |
| `runtime_scope` | no | Defaults to `single_forward_no_depth_no_ransac`; if present it must match exactly. |
| `floor` | yes | Direct floor angles plus distance. |
| `wheels` | yes | Zero or more detected wheels. Empty list is valid. |

### Floor fields

| Field | Required | Notes |
|---|---:|---|
| `pitch` | yes | Angle in radians, camera/floor convention documented by the export manifest. |
| `roll` | yes | Angle in radians, camera/floor convention documented by the export manifest. |
| `distance` | yes | Direct scalar output. Metric only when `distance_mode` declares a metric anchor. |
| `distance_mode` | yes | One of `scale_relative`, `metric_anchor`, `normalized`, `unknown`. |
| `fov_mode` | no | One of `unknown`, `fixed`, `provided`, `predicted`; defaults to `unknown`. |

Absolute metric distance is not guaranteed from a monocular RGB image unless a
scale anchor or calibrated application contract is supplied. Until that exists,
`distance_mode=scale_relative` is the safe default.

### Wheel fields

Wheel point names deliberately match the confirmed mobile AR names:
