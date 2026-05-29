# Export Parity Audit

> **Two distinct "parity" axes — do not conflate.**
> 1. **PT-vs-exported runtime parity** (ONNX/TFLite vs PyTorch) — the
>    table below; still uncertified under the strict policy.
> 2. **UE camera-pose parity** (3D-eval) — **CERTIFIED 2026-05-29** on the
>    MCP `WheelsDataset_v0_2` export; see the section at the bottom.

## UE camera-pose parity (3D-eval) — CERTIFIED 2026-05-29

The long-standing blocker ("the UE Roll/Pitch sign/zero convention must
be confirmed against one clean UE export frame before the pose is
trusted") is **resolved with evidence**. The MCP rich annotations pair
3D `keypoints_world` with 2D `keypoints_image` and the full camera pose,
so the UE→OpenCV convention is verifiable by reprojection.

`scripts/certify_ue_export_parity.py` builds the camera with
`src/camera_from_ue_pose.py` and reprojects every exported world keypoint:

- **Result: `certified: true`, max reprojection ≈ 0.0002 px** over all
  1000 frames (`outputs/eval3d/export_parity_v0_2.json`).
- Certified convention: **FOV is horizontal**; UE world is left-handed →
  harness right-handed via a single **Y-negation**; forward from the
  `[roll, pitch, yaw]` rotator; roll about the optical axis.
- Regression-pinned by `tests/test_camera_from_ue_pose.py` (real-frame
  fixture, asserts sub-px).

This certifies camera **geometry only**. It does NOT make the export a
model gate: v0_2's `a`/`b` are rim spheres (z≈28 cm), not the floor-ray
points the 2026-05-14 contract requires, and the 2D points are ground
truth, not model predictions. See `docs/EVAL3D_AND_3D_LOSS_STATUS.md`
("Real export — what landed 2026-05-29") for the remaining blockers.

---

Diagnostic summary for PT-vs-exported parity drift reports.

- Audit OK: True
- Certified: False
- Recommendation: Do not certify exported artifacts as drop-in replacements under the current strict parity policy. ONNX and TFLite aggregate eval remain usable evidence, but production needs either fixed export parity or an explicitly approved aggregate/AR-holdout acceptance policy.

| Export | Strict OK | Matched | Max bbox px | Max kp px | Max conf | Failure categories |
|---|---:|---:|---:|---:|---:|---|
| onnx | False | 14/20 | 8.497 | 13.371 | 0.228 | bbox_drift=1, keypoint_drift=2, confidence_drift=4; scale_warnings=0 |
| tflite | False | 14/20 | 8.497 | 13.372 | 0.228 | bbox_drift=1, keypoint_drift=2, confidence_drift=4; scale_warnings=0 |

## Findings

- ONNX and TFLite have identical failure category counts; the issue is likely shared exported-backend/postprocess parity rather than a TFLite-only runtime bug.
- onnx: strict failures are drift-only; no detection-count mismatch.
- onnx: no coordinate-scale warnings in the official strict report.
- onnx: all sampled frames pass only at bbox=10.0px, kp=15.0px, conf=0.25.
- tflite: strict failures are drift-only; no detection-count mismatch.
- tflite: no coordinate-scale warnings in the official strict report.
- tflite: all sampled frames pass only at bbox=10.0px, kp=15.0px, conf=0.25.

## Tolerance Sweep

### onnx

| bbox px | kp px | conf | Passed |
|---:|---:|---:|---:|
| 2.000 | 3.000 | 0.050 | 14/20 |
| 3.000 | 4.000 | 0.050 | 15/20 |
| 5.000 | 8.000 | 0.100 | 16/20 |
| 10.000 | 15.000 | 0.250 | 20/20 |

### tflite

| bbox px | kp px | conf | Passed |
|---:|---:|---:|---:|
| 2.000 | 3.000 | 0.050 | 14/20 |
| 3.000 | 4.000 | 0.050 | 15/20 |
| 5.000 | 8.000 | 0.100 | 16/20 |
| 10.000 | 15.000 | 0.250 | 20/20 |
