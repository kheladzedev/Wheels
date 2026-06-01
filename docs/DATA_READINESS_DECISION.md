# Data Readiness Decision

## Verdict

- Для тестовой передачи: можно отдавать.
- Для production: данных недостаточно.
- Confidence: `medium_for_integration_handoff_low_for_production`.

## Current Data

- Strict gate subset: train=423, val=106, wheels=602, labels_ok=True, leakage_ok=True.
- All configured datasets: configs=22, failed=20, train_images=5705, val_images=1287, wheels=12148.
- Operating point: conf=0.8, bbox_mAP50=0.9034846645803373, OKS=0.8879455801173212, FN=0.09375, FP=0.14705882352941177.

## Risk Flags

- Legacy/experimental configs: 20 failed out of 22; they are excluded from the production gate.
- Strict subset size: too_small_for_production (recommended 2000+ real frames for production retrain).
- Ground truth quality: Current strict validation data is clean structurally, but not a human-labelled AR-device holdout.
- External evidence: android_litert_device_validation, human_labelled_ar_device_holdout, ar_3d_replay_validation
- Retrain decision: не дообучать вслепую; first collect AR holdout, replay and hard negatives.

## Data Plan

- P0 `android`: Run exact TFLite artifact on physical Android LiteRT harness. Reason: Closes serving/runtime skew for Android before any retrain.
- P0 `ar_data_collection`: Collect at least gate minimum 50 AR frames / 80 wheels, recommended 300+ frames / 500+ wheels for confidence. Reason: The current val set is local/static, not a real AR-device holdout.
- P0 `ar_runtime`: Collect AR replay JSONL with floor hits, RANSAC and residuals. Reason: 2D keypoints are not enough; production requires 3D floor-hit behavior.
- P1 `ml`: Mine 300-1000 hard-negative frames from false positives and add them with empty labels. Reason: This directly targets FP risk instead of just increasing data volume.
- P2 `ml_ar`: Build a production training pool of 2000+ real labelled AR/app frames with WheelBBox/keypoints and scene/device/session groups. Reason: Needed for production retrain without leakage and domain skew.

## Do Now

- Отдать текущий bundle в тест.
- Собирать возврат от Android/iOS/AR по шаблонам.
- После внешнего evidence делать targeted retrain, not generic data download.
