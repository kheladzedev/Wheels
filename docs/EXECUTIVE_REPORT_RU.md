# Executive Report RU

## Короткий статус

- Integration ready: True
- Production ready: False
- Audit suite OK: True
- Production evidence audit ready: False
- Production blockers: android_litert_device_validation, human_labelled_ar_device_holdout, ar_3d_replay_validation, production_evidence_audit_ready, production_gate

Вывод: модель и export package готовы для интеграции и smoke-проверок. Полный production gate не закрыт, потому что не хватает внешних Android/AR evidence artifacts.

## Что сделано

- Собран пул 300 чистых GLB-моделей машин из Sketchfab/Objaverse.
- Через Unreal/MCP подготовлен synthetic/geometry поток: 192 кадра и 702 wheel labels до clean-фильтра, 152 кадра и 626 wheel labels после clean QA.
- Проведен model inventory: 11 train runs, 30 artifacts, 20 eval reports.
- Проведен dataset audit: 12 configs, 2082 train images, 489 val images, 4033 wheel labels.
- Champion model выбран через machine-readable promotion guard: ok=True, anchor candidates=5, promotion required=0.
- Соответствие AR technical spec проверено отдельным audit: ok=True, failures=[].
- ONNX/TFLite export package сертифицирован по calibrated backend policy.
- TFLite/LiteRT desktop package сертифицирован; Android-device runtime validation вынесен отдельным production blocker.

## Champion model

- PT: `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt`
- ONNX: `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.onnx`
- TFLite: `outputs/production_audit/tflite_export/best_float32.tflite`
- Training data: `configs/pose_dataset_real_v1_self_plus_ue_synthetic.yaml`

## Основные метрики

- Real-only PyTorch bbox mAP50: 0.912
- Real-only PyTorch OKS: 0.887
- Real-only PyTorch FN rate: 0.062
- TFLite mixed-anchor bbox mAP50: 0.692
- TFLite mixed-anchor OKS: 0.888
- Export backend certification: True (`desktop_export_backend_certification_not_android_device`)
- TFLite package certification: True (`desktop_tflite_litert_package_not_android_device`)

## Соответствие требованиям

- Requirements закрыто: 11 / 16
- Consolidated production evidence gate: False
- Traceability report: `docs/REQUIREMENTS_TRACEABILITY.md`
- Senior ML audit: `docs/SENIOR_ML_AUDIT.md`
- Model card: `docs/MODEL_CARD.md`

## Что не закрыто для production

- `android_litert_device_validation`: missing_source:data/incoming/android_litert_device_report.json, missing_report:outputs/production_audit/android_litert_device_eval.json
- `human_labelled_ar_device_holdout`: missing_source_dirs:data/incoming/ar_device_holdout, missing_provenance:data/incoming/ar_device_holdout/metadata/provenance.json, missing_report:outputs/production_audit/ar_device_holdout_eval.json, missing_pipeline:outputs/production_audit/ar_device_holdout_pipeline.json
- `ar_3d_replay_validation`: missing_source:data/incoming/ar_3d_replay/ar_replay.jsonl, missing_report:outputs/production_audit/ar_3d_replay_eval.json

## Что нужно от Android/AR команды

1. Заполнить `data/incoming/android_litert_device_report.json` по контракту `docs/ANDROID_LITERT_DEVICE_REPORT.md`.
2. Передать human-reviewed AR-device holdout в `data/incoming/ar_device_holdout` с `metadata/provenance.json`.
3. Передать AR replay JSONL в `data/incoming/ar_3d_replay/ar_replay.jsonl`.
4. Запустить единый intake: `./.venv/bin/python src/run_production_evidence_intake.py`.
5. После зеленого intake запустить `./.venv/bin/python src/production_audit_suite.py --with-pytest`.

## Release package

- Release integrity OK: True
- Deterministic package manifest: `docs/RELEASE_PACKAGE.md` / `outputs/production_audit/release_integrity.json`.

## Финальный вывод

На текущем evidence модель является integration-ready, но не full production-ready. Блокеры не связаны с отсутствием модели/export package: они связаны с отсутствием реального Android-device LiteRT отчета, human-labelled AR holdout и AR-side 3D replay validation.
