# Wheel Pose Model Card

## Summary

- Task: single-class wheel detection with three keypoints: `a`, `b`, `c_disc_bottom`.
- Intended use: AR integration that raycasts wheel floor points and disc-bottom point into 3D.
- Current status: integration-ready, not full production-ready until external Android/AR evidence is present.
- Production ready: False
- Production blockers: android_litert_device_validation, human_labelled_ar_device_holdout, ar_3d_replay_validation, production_evidence_audit_ready, production_gate

## Champion

- PyTorch artifact: `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt`
- ONNX artifact: `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.onnx`
- TFLite artifact: `outputs/production_audit/tflite_export/best_float32.tflite`
- Training run: `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s`
- Training data config: `configs/pose_dataset_real_v1_self_plus_ue_synthetic.yaml`
- Source model: `runs/pose/runs/pose/wheel_real_v1_self_s/weights/best.pt`
- Epochs / image size / batch: 100 / 640 / 8

## Data

- Dataset audit OK: False
- Dataset configs checked: 22
- Total train images across configs: 5705
- Total val images across configs: 1287
- Total wheel labels across configs: 12148
- External 3D car pool: 300 clean GLBs from Sketchfab/Objaverse.
- UE clean geometry labels: 132 frames / 548 wheels after QA filtering.

## Metrics

| Eval | bbox mAP50 | bbox mAP50-95 | OKS | FN rate | FP rate | GT/pred/matched |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch real-only validation | 0.9118179979136706 | 0.813140425830221 | 0.8872704757196613 | 0.0625 | 0.25 | 64/80/60 |
| PyTorch mixed anchor validation | 0.696960312285691 | 0.620895520362303 | 0.8872704757196613 | 0.2857142857142857 | 0.25925925925925924 | 84/81/60 |
| ONNX mixed anchor validation | 0.6923761088840539 | 0.6170811216723833 | 0.8880996578601513 | 0.2857142857142857 | 0.2682926829268293 | 84/82/60 |
| TFLite mixed anchor validation | 0.6923761088840539 | 0.6170811216723833 | 0.8880997181541215 | 0.2857142857142857 | 0.2682926829268293 | 84/82/60 |

## Export And Runtime

- Export backend certification: True (`desktop_export_backend_certification_not_android_device`)
- TFLite package certification: True (`desktop_tflite_litert_package_not_android_device`)
- PyTorch CPU mean latency: 41.96455724968473 ms
- ONNX CPU mean latency: 39.09157031216637 ms
- LiteRT desktop smoke mean latency: 269.1269209004531 ms
- Android-device LiteRT validation is not yet present.

## Production Evidence

- Evidence ready: False
- Evidence blockers: android_litert_device_validation, human_labelled_ar_device_holdout, ar_3d_replay_validation
- Required evidence contract: `docs/PRODUCTION_EVIDENCE_CHECKLIST.md`.

## Limitations

- No human-labelled AR-device holdout has been evaluated yet.
- No AR 3D replay/RANSAC validation report is present yet.
- Android-device LiteRT latency/memory/output evidence is not present yet.
- Synthetic/UE geometry labels are useful for coverage, but are not a replacement for real AR-device validation.

## Release

- Release integrity OK: True
- Deterministic package manifest: `docs/RELEASE_PACKAGE.md` / `outputs/production_audit/release_integrity.json`.
