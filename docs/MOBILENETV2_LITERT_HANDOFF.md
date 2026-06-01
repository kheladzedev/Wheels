# MobileNetV2 LiteRT Handoff

> **CORRECTION 2026-05-30 (STALE / UNBACKED).** This document describes a
> PROPOSED MobileNetV2-skipless architecture that was NEVER trained. There are
> ZERO MN2 checkpoints, model packages, or `.tflite` on disk
> (`outputs/model_packages/` does not exist). The "Verified Metrics" and parity
> numbers below are not backed by any artifact. The actually-shipped champion is
> the YOLO11s-pose model at
> `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt`
> (+ `best.onnx` + float16/float32 TFLite). The MN2 module
> (`src/models/mobilenetv2_skipless_pose.py`) is kept as load-bearing code for
> the 3D-loss / web-multitask features, but it is untrained.

Status: `provisional_not_production`

This document is the Android/LiteRT handoff for the current provisional
MobileNetV2 runtime package. The package is technically runnable and has
PyTorch / ONNX / TFLite parity evidence, but it is not a production-approved
model. Production approval still depends on a cleaner accepted export and
human review of the remaining uncertain annotations.

## Runtime Package

- Package root:
  `outputs/model_packages/mn2_combined_0003_neuraldata1_review_patch_v3_e20_provisional`
- TFLite model:
  `outputs/model_packages/mn2_combined_0003_neuraldata1_review_patch_v3_e20_provisional/tflite_export/mn2_combined_0003_neuraldata1_review_patch_v3_e20.tflite`
- Package summary:
  `outputs/model_packages/mn2_combined_0003_neuraldata1_review_patch_v3_e20_provisional/PACKAGE_SUMMARY.md`
- Machine-readable manifest:
  `outputs/model_packages/mn2_combined_0003_neuraldata1_review_patch_v3_e20_provisional/package_manifest.json`
- Validation bundle:
  `outputs/model_packages/mn2_combined_0003_neuraldata1_review_patch_v3_e20_provisional/android_litert_validation_bundle`

## Model I/O

The exported `.tflite` is float32 and currently uses NCHW input.

| Tensor | Name | Shape | Type | Meaning |
| --- | --- | --- | --- | --- |
| input | `inputs_0` | `[1, 3, 640, 640]` | `float32` | RGB image, channel-first, normalized to `[0, 1]` |
| output 0 | `Identity` | `[1, 20, 20, 1]` | `float32` | wheel class logits |
| output 1 | `Identity_1` | `[1, 20, 20, 4]` | `float32` | bbox distances `(l, t, r, b)` in stride units |
| output 2 | `Identity_2` | `[1, 20, 20, 6]` | `float32` | keypoint offsets `(a.x, a.y, b.x, b.y, c.x, c.y)` in stride units |
| output 3 | `Identity_3` | `[1, 20, 20, 3]` | `float32` | keypoint visibility logits, debug only |

Android preprocessing must resize the source frame to `640 x 640`, convert
BGR/camera format to RGB, normalize to float `[0, 1]`, and feed the tensor as
`[batch, channel, y, x]`. If the Android camera pipeline produces NHWC, transpose
to NCHW before invoking the model.

## Decode Rules

Use stride `32`; the output grid is `20 x 20`.

For each grid cell `(row, col)`:

- `cx = (col + 0.5) * 32`
- `cy = (row + 0.5) * 32`
- `score = sigmoid(cls_logit)`
- discard cells with `score < 0.30`
- bbox decode:
  - `x1 = cx - l * 32`
  - `y1 = cy - t * 32`
  - `x2 = cx + r * 32`
  - `y2 = cy + b * 32`
- keypoint decode:
  - `a = [cx + dx_a * 32, cy + dy_a * 32]`
  - `b = [cx + dx_b * 32, cy + dy_b * 32]`
  - `c_disc_bottom = [cx + dx_c * 32, cy + dy_c * 32]`

Clamp decoded bbox/keypoint coordinates to `[0, 640]`, run NMS with IoU `0.5`,
keep at most `5` detections, then scale coordinates back to the original frame:

- `x_original = x_640 * original_width / 640`
- `y_original = y_640 * original_height / 640`

Before emitting confirmed AR JSON, apply the same geometry gate as
`src/postprocess_wheels.py`: `a` must be left of `b`, A/B must sit on the lower
floor-ray band of the bbox, and `c_disc_bottom` must be above the A/B floor-ray
line. Candidates that fail this gate are debug detections only and must not be
emitted in the confirmed schema.

## Confirmed AR JSON

Emit exactly this schema, one JSON object per frame:

```json
{
  "frame_id": "image_or_camera_frame_id",
  "wheels": [
    {
      "bbox_xyxy": [1043.9832, 1135.8210, 1196.2158, 1273.3578],
      "confidence": 0.8478,
      "points": {
        "a": [1059.8340, 1270.6634],
        "b": [1184.1739, 1267.9141],
        "c_disc_bottom": [1122.6055, 1247.7771]
      }
    }
  ]
}
```

Do not add `timestamp`, `visibility`, per-keypoint confidence, `track_id`, 3D
coordinates, raycast results, plane parameters, or RANSAC fields to this JSON.
Those belong to the AR layer.

## Verified Metrics

Runtime eval on the provisional validation split with `conf=0.30`:

- images: `325`
- GT wheels: `453`
- predictions: `470`
- matched: `382`
- precision: `0.8128`
- recall: `0.8433`
- mean IoU: `0.8847`
- mean keypoint error: `a=3.9216 px`, `b=3.9324 px`,
  `c_disc_bottom=2.8532 px`
- false positives on empty labels: `13`

Export/runtime checks:

- ONNX raw parity: `matched=True`, max abs diff `5.1021576e-05`
- ONNX decoded parity: `matched=True`
- TFLite export status: `PASS`
- TFLite raw parity: `matched=True`, max abs diff `2.4795532e-05`
- TFLite decoded parity: `matched=True`
- TFLite val40 runtime status: `PASS`
- TFLite val40 runtime failures: `0`
- PyTorch/TFLite val40 frame order and wheel counts: `matched=True`
- PyTorch/TFLite max numeric drift: `0.00029297`

## Local Smoke Command

```bash
./.venv/bin/python scripts/predict_mobilenetv2_tflite.py \
  --tflite-model outputs/model_packages/mn2_combined_0003_neuraldata1_review_patch_v3_e20_provisional/tflite_export/mn2_combined_0003_neuraldata1_review_patch_v3_e20.tflite \
  --source outputs/model_packages/mn2_combined_0003_neuraldata1_review_patch_v3_e20_provisional/android_litert_validation_bundle/sample_image.jpg \
  --runtime-python ./.tflite-venv/bin/python \
  --imgsz 640 \
  --conf 0.30 \
  --nms-iou 0.5 \
  --max-det 5 \
  --preview-count 1 \
  --out-dir outputs/model_packages/mn2_combined_0003_neuraldata1_review_patch_v3_e20_provisional/android_litert_validation_bundle/local_smoke
```

Expected local smoke result: `status=PASS`, `runtime_failure_count=0`, confirmed
JSON schema with `frame_id` and `wheels[].bbox_xyxy/confidence/points`.

## Known Limits

- This is a provisional model trained on provisional data.
- It is float32 TFLite, not INT8.
- Real Android device latency has not been measured yet.
- The model uses NCHW input; Android integration must preserve that layout.
- Production retraining should wait for a cleaner acceptance-backed export or a
  reviewed annotation set.
