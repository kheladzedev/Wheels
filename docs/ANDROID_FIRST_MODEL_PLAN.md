# Android-First Model Plan

Confirmed direction (AR team, 2026-05-13; re-confirmed 2026-05-18):
**Android is the first target platform**. We aim for an on-device
model that detects wheels + 3 keypoints (`a`, `b`, `c_disc_bottom`)
per frame and runs at AR-friendly latency on a real Android device.

This document tracks the two-stage model roadmap. Stage 1 validates
the data and pipeline with the simplest thing that already works;
Stage 2 swaps the encoder to something Android-class once we know the
data is sane and the keypoint task is learnable on it.

## Stage 1 — quick baseline

**Goal:** validate the data and the end-to-end pipeline. Quality at
this stage is a sanity check, not a production target.

Use the existing YOLO-pose pipeline as-is:

- Encoder: whatever Ultralytics ships (CSPDarknet-style backbone in
  `yolo11n-pose.pt` / `yolo11s-pose.pt`).
- Head: YOLO-pose head with 3 keypoints per wheel.
- Training: `src/train_yolo.py`, no code changes.
- Eval: `src/eval_keypoints.py` with the existing pixel-error + OKS +
  slice + failure-catalog reporting.

**Done when:**

- The first batch of plugin-collected (or Unreal-rendered) data
  ingests cleanly via `src/convert_incoming_to_yolo.py`.
- Training run on real data converges (bbox mAP50 ≥ 0.85 on val).
- Eval JSON shows non-empty `failure_samples`, `slices.by_bbox_area`.
- A few hand-inspected predictions visually look right (the script
  `src/visualize_predictions.py` is the fastest way to spot-check).

Quality on Stage 1 weights is **not** the AR-facing milestone. It is
a green light to invest in Stage 2 — i.e. evidence that the keypoint
task is learnable from the data we have.

## Stage 2 — lightweight Android-oriented model

**Goal:** model that runs on Android at AR-relevant latency and ships
to the on-device runtime (TFLite / LiteRT).

Likely shape, aligned with the Unreal-side recommendation
("super-light, MobileNetV2 as encoder, without lower skips"):

- **Encoder:** MobileNetV2, ImageNet-pretrained where available.
  Use the final encoder feature map only; no FPN / lower skip
  connections.
- **Detection + keypoint head:** small one-stage head for the single
  `wheel` class and 3 keypoints. The current repo implementation is
  `src/models/mobilenetv2_skipless_pose.py`; it is architecturally
  aligned but still needs a real dataset trainer/export path before it
  can replace the YOLO baseline.
- **Input size:** 320×320 or 416×416 — pick after measuring on-device
  latency.
- **Quantization:** INT8 post-training, calibrated on the real-data
  subset.

Decision criteria for moving Stage 1 → Stage 2:

- ≥ 500 real-data wheel instances ingested and converted.
- Stage 1 weights show that the keypoint task is learnable on this
  data (OKS mean ≥ 0.5 on val, per-keypoint median px error ≤ 5 px
  on 640-input).
- An AR-side latency budget is available (currently not pinned — see
  *Open items* below).

## Export pipeline (Android target)

Already covered by `src/export_model.py`:

```bash
python src/export_model.py \
  --model runs/pose/<run>/weights/best.pt \
  --format tflite --int8
```

The sanity check after export compares PyTorch vs TFLite outputs on
one sample image: 2 px on bbox xyxy, 3 px on keypoint xy, 0.05 abs on
detection confidence. The script exits non-zero if drift exceeds these
tolerances.

`tensorflow` must be installed in the venv for the `tflite` path. We
deliberately do not auto-install it — the version Ultralytics expects
moves often.

## Open items

These do not block Stage 1 but are needed before Stage 2 ships:

- **On-device latency budget.** What's the acceptable inference time
  per frame on the lowest target Android device? Without this we
  cannot pick input size or quantization aggressiveness.
- **Model-size budget.** Hard cap on the deployed `.tflite` file size?
- **Frame rate.** Does AR drive ML at video-rate (~30 fps) or only
  every N frames during accumulation? Affects whether streaming-batch
  shape matters.

These are tracked alongside the broader open items in
`docs/QUESTIONS_FOR_TEAM.md`.

## See also

- `docs/AR_ML_CONTRACT.md` — JSON contract.
- `docs/KEYPOINT_SPEC.md` — A/B/C definitions.
- `src/export_model.py` — multi-format export with sanity check.
- `src/eval_keypoints.py` — evaluation pipeline.
- `docs/TASK_PLAN.md` — stage-by-stage delivery plan.
