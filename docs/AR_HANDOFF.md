# AR handoff — `wheel_real_v1_self_plus_ue_synthetic_s`

Date: 2026-05-27.

This is the package AR should integrate first. ML returns 2D pixels only;
AR owns raycast, RANSAC, plane recovery, K-frame accumulation, and
cross-frame association.

## Files to use

| Purpose | File |
|---|---|
| PyTorch weights | `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt` |
| ONNX export | `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.onnx` |
| Real-only eval report | `outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_on_self_val.json` |
| Mixed real+UE anchor eval report | `outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s.json` |
| Production audit | `docs/PRODUCTION_READINESS_AUDIT.md` |
| Model comparison | `outputs/eval/all_models_summary.csv` |
| Smoke outputs | `outputs/production_audit/smoke_single/`, `outputs/production_audit/smoke_batch/` |
| Contract | `docs/AR_ML_CONTRACT.md` |
| Keypoint semantics | `docs/KEYPOINT_SPEC.md` |

## Contract AR consumes

One JSON per frame:

```json
{
  "frame_id": "frame_0001",
  "wheels": [
    {
      "bbox_xyxy": [x1, y1, x2, y2],
      "confidence": 0.94,
      "points": {
        "a": [xa, ya],
        "b": [xb, yb],
        "c_disc_bottom": [xc, yc]
      }
    }
  ]
}
```

There is no `timestamp`, `track_id`, per-keypoint confidence,
`visibility`, 3D coordinate, plane, raycast result, or RANSAC metadata
in the ML response.

Point meanings:

- `points.a`: left floor-ray screen point.
- `points.b`: right floor-ray screen point.
- `points.c_disc_bottom`: lowest visible point of the metal rim / disc.

`bbox_xyxy` and `confidence` are metadata for filtering/debug overlays.
The AR math should use `frame_id` and `points`.

## Current production candidate metrics

Source: `outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_on_self_val.json`
for the real-only acceptance split, plus
`outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s.json` for the
mixed real+UE anchor.

| Metric | Value |
|---|---:|
| Box mAP50, real-only val | 0.912 |
| Box mAP50-95, real-only val | 0.813 |
| Pose mAP50, real-only val | 0.912 |
| Mean OKS, sigma 0.10 | 0.887 |
| False-negative rate, real-only val | 0.063 |
| False-positive rate, real-only val | 0.250 |
| `a` median pixel error | 7.5 px |
| `b` median pixel error | 7.7 px |
| `c_disc_bottom` median pixel error | 7.7 px |
| Mixed real+UE anchor Box mAP50 | 0.697 |

The real-only detection target `Box mAP50 >= 0.85` is met. The mixed
real+UE anchor split is harder and includes synthetic validation frames;
it is used as a regression signal, not as the production acceptance
split. The long-term keypoint target `<=5 px` is not met by the
champion.

> Note (2026-05-30): the champion's real-only self-val false-positive rate is
> `FP=0.25` (above the `real_only_fp_ceiling` default of 0.15). `production_gate.py`
> now enforces this FP ceiling (severity: fail) plus an FP ceiling on the human
> AR holdout, so integration mode honestly fails on `real_only_fp_ceiling` (and on
> `dataset_audit`).

## Single-frame smoke test

```bash
./.venv/bin/python src/infer_image.py \
  --image data/wheel_pose_dataset_real_v1_self/images/val/real_v1_self__wmc_0021_View_of_a_car_parked_on_the_side_of_a_road_AM_79469-1.jpg \
  --model runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt \
  --out-dir outputs/ar_handoff_smoke \
  --device cpu \
  --conf 0.25 --iou 0.45 --max-det 20 \
  --frame-id ar_handoff_frame_0001
```

Read `outputs/ar_handoff_smoke/<image_stem>.json`. That file is the
confirmed AR schema. `_legacy.json` and `_raw.json` are ML debug only.

## Batch replay for AR RANSAC tuning

For a folder of AR-captured frames:

```bash
./.venv/bin/python src/infer_batch.py \
  --source /path/to/ar_frames \
  --model runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt \
  --out-dir outputs/ar_batch_replay \
  --device cpu \
  --conf 0.25 --iou 0.45 --max-det 20
```

For a video:

```bash
./.venv/bin/python src/infer_batch.py \
  --source /path/to/ar_session.mp4 \
  --model runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt \
  --out-dir outputs/ar_batch_replay \
  --device cpu \
  --every-n-frames 1 \
  --conf 0.25 --iou 0.45 --max-det 20
```

Primary per-frame `.json` files and `.jsonl` output use the confirmed
schema. `_legacy` files include timestamps, image paths, thresholds,
and per-keypoint confidences for debugging only.

## Integration notes

- Pass a stable AR `frame_id` with each frame and use the echoed
  `frame_id` to recover the camera transform saved at capture time.
- Drop or down-weight low-confidence wheels AR-side if needed; the ML
  confidence threshold defaults to 0.25.
- Treat missing detections as an empty `wheels` list. Do not infer
  persistence from ML output; tracking is AR-side after 3D raycast.
- Partially occluded wheels are not represented in the confirmed schema.

## Known limitations

- Current metrics are on self-labelled real data plus synthetic anchors,
  not on an AR-device human-labelled holdout.
- `best.onnx` is available and aggregate eval is close to PyTorch, but
  strict PT-vs-ONNX parity currently fails on 6/20 sampled frames; see
  `outputs/production_audit/onnx_drift_20.json`.
- TFLite/LiteRT export is still an integration decision for Android
  on-device deployment.
- Keypoint median error is above the long-term 5 px target, even though
  aggregate detection quality clears the handoff target.
