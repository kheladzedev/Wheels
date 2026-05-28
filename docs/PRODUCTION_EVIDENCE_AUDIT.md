# Production Evidence Audit

- Audit OK: True
- Production evidence ready: False
- Blockers: android_litert_device_validation, human_labelled_ar_device_holdout, ar_3d_replay_validation

## Required Evidence

| Evidence | Owner | Producer | Inputs | Command | Gate |
|---|---|---|---|---|---|
| android_litert_device_validation | android | `android_litert_harness/README.md`<br>`android_litert_harness/AndroidLiteRtDeviceValidationTest.kt` | `data/incoming/android_litert_device_report.json` | `./.venv/bin/python src/validate_android_litert_report.py --source data/incoming/android_litert_device_report.json --out outputs/production_audit/android_litert_device_eval.json --expected-artifact outputs/production_audit/tflite_export/best_float32.tflite --min-runs 20 --max-mean-latency-ms 120.0 --max-p95-latency-ms 180.0 --max-peak-memory-mb 512.0` | `{"ok": true}` |
| human_labelled_ar_device_holdout | ar_data_collection | `ar_holdout_harness/README.md`<br>`ar_holdout_harness/ArHoldoutAnnotationWriter.kt` | `data/incoming/ar_device_holdout/images`<br>`data/incoming/ar_device_holdout/annotations`<br>`data/incoming/ar_device_holdout/metadata/provenance.json` | `./.venv/bin/python src/evaluate_ar_holdout.py --source-root data/incoming/ar_device_holdout --eval-out outputs/production_audit/ar_device_holdout_eval.json --status-out outputs/production_audit/ar_device_holdout_pipeline.json --min-map50 0.85 --min-oks 0.8 --max-fn 0.1 --min-images 50 --min-gt-wheels 80` | `{"images": ">=50", "gt_wheels": ">=80", "bbox_mAP50": ">=0.85", "oks_mean": ">=0.8", "false_negative_rate": "<=0.1"}` |
| ar_3d_replay_validation | ar_runtime | `ar_replay_harness/README.md`<br>`ar_replay_harness/ArReplayLogger.kt` | `data/incoming/ar_3d_replay/ar_replay.jsonl` | `./.venv/bin/python src/validate_ar_replay.py --jsonl data/incoming/ar_3d_replay/ar_replay.jsonl --out outputs/production_audit/ar_3d_replay_eval.json --min-observations 30 --min-sessions 1 --min-floor-hit-rate 0.9 --min-inlier-rate 0.7 --max-median-residual 0.02 --max-p95-residual 0.05 --min-final-positions 1` | `{"ok": true}` |

## Current Checks

| Evidence | Ready | Source | Report | Failures |
|---|---:|---|---|---|
| android_litert_device_validation | False | `data/incoming/android_litert_device_report.json` | `outputs/production_audit/android_litert_device_eval.json` | missing_source:data/incoming/android_litert_device_report.json, missing_report:outputs/production_audit/android_litert_device_eval.json |
| human_labelled_ar_device_holdout | False | `data/incoming/ar_device_holdout` | `outputs/production_audit/ar_device_holdout_eval.json` | missing_source_dirs:data/incoming/ar_device_holdout, missing_provenance:data/incoming/ar_device_holdout/metadata/provenance.json, missing_report:outputs/production_audit/ar_device_holdout_eval.json, missing_pipeline:outputs/production_audit/ar_device_holdout_pipeline.json |
| ar_3d_replay_validation | False | `None` | `outputs/production_audit/ar_3d_replay_eval.json` | missing_source:data/incoming/ar_3d_replay/ar_replay.jsonl, missing_report:outputs/production_audit/ar_3d_replay_eval.json |
| external_evidence_custody | True | `outputs/production_audit/external_evidence_drop_import.json` | `outputs/production_audit/external_evidence_drop_import.json` | none |
