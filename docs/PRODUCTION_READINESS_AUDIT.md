# Production Readiness Audit

Date: 2026-05-27.

## Executive Decision

**Status: integration candidate, not full production-certified.**

The current PyTorch production candidate is:

- PT: `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt`
- ONNX: `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.onnx`

The model is usable for AR integration smoke tests and server/Python
inference. It is **not yet fully production-certified** for Android
on-device release because there is no human-labelled AR-device holdout,
no AR-side 3D raycast/RANSAC validation, and no Android-device LiteRT
production certification. ONNX/TFLite strict parity remains diagnostic,
but the calibrated desktop export certification now passes using
aggregate metric parity, no count mismatch, no coordinate-scale warnings,
and LiteRT smoke evidence.

Automated gate outputs:

- Integration gate: `PASS`
  (`outputs/production_audit/integration_gate.json`)
- Production gate: `FAIL`
  (`outputs/production_audit/production_gate.json`)

Model inventory:

- Report: `docs/MODEL_INVENTORY.md`
- Train runs: 11
- Artifacts: 30
  (`.pt`=22,
  `.onnx`=7,
  `.tflite`=1)
- Eval reports linked: 20
- Champion training data: `configs/pose_dataset_real_v1_self_plus_ue_synthetic.yaml`
- Champion source model: `runs/pose/runs/pose/wheel_real_v1_self_s/weights/best.pt`

Model selection audit:

- Report: `docs/MODEL_SELECTION_AUDIT.md`
- Selection OK: True
- Anchor candidates compared: 5
- Promotion required: 0
- Failures: none

Spec compliance audit:

- Report: `docs/SPEC_COMPLIANCE_AUDIT.md`
- Overall OK: True
- Checks: 11
- Failures: none

Dataset audit:

- Report: `docs/DATASET_AUDIT.md`
- Overall OK: True
- Dataset configs checked: 12
- Failed configs: 0
- Total train images across configs: 2082
- Total val images across configs: 489
- Total wheel label lines across configs: 4033

Release package integrity:

- Report: `docs/RELEASE_PACKAGE.md`
- Overall OK: True
- Artifacts: 85
- Total size: 126.342 MB

Runtime contract audit:

- Report: `outputs/production_audit/runtime_contract_audit.json`
- Overall OK: True
- Single-image wheels: 2
- Batch frames/wheels: 5 / 12

Performance audit:

- Report: `docs/PERFORMANCE_AUDIT.md`
- Overall OK: True
- Scope: `desktop_local_runtime_diagnostic_not_android_certification`
- Sample frames: 8
- PyTorch CPU mean/p95: 41.208 / 51.491 ms
- ONNX CPU mean/p95: 37.600 / 49.507 ms
- LiteRT smoke mean/p95: 269.127 / 271.012 ms

Senior ML audit:

- Report: `docs/SENIOR_ML_AUDIT.md`
- Audit OK: True
- Integration ready: True
- Production ready: False
- Requirements passed: 17 / 24
- Production blockers: android_litert_device_validation, human_labelled_ar_device_holdout, ar_3d_replay_validation, production_evidence_audit_ready, production_gate

Objective completion audit:

- Report: `docs/OBJECTIVE_COMPLETION_AUDIT.md`
- Objective complete: False
- Integration ready: True
- Production ready: False
- Failed requirements: production_evidence_present, production_gate_passed

Export parity audit:

- Report: `docs/EXPORT_PARITY_AUDIT.md`
- Certified: False
- ONNX failure categories: {'count_mismatch': 0, 'bbox_drift': 1, 'keypoint_drift': 2, 'confidence_drift': 4, 'other': 0}
- TFLite failure categories: {'count_mismatch': 0, 'bbox_drift': 1, 'keypoint_drift': 2, 'confidence_drift': 4, 'other': 0}

Export certification:

- Report: `docs/EXPORT_CERTIFICATION.md`
- Certified: True
- Scope: `desktop_export_backend_certification_not_android_device`

Production evidence audit:

- Report: `docs/PRODUCTION_EVIDENCE_AUDIT.md`
- Evidence ready: False
- Blockers: android_litert_device_validation, human_labelled_ar_device_holdout, ar_3d_replay_validation

Requirements traceability:

- Report: `docs/REQUIREMENTS_TRACEABILITY.md`
- Passed: 11 / 16
- Production ready: False

## Requirement Audit

| Requirement | Status | Evidence |
|---|---|---|
| Confirmed AR JSON contract | PASS | `tests/test_ar_contract.py`, `tests/test_confirmed_ar_schema_shape.py`, `src/infer_image.py`, `src/infer_batch.py` |
| Multi-wheel per-frame inference | PASS | Single smoke found 2 wheels; batch smoke found 12 wheels over 5 frames |
| 300 external car models collected | PASS | `data/sketchfab_cars`: 300 clean GLBs; import status has 300 tasks |
| UE/MCP geometry-label pipeline | PASS | Geometry status: groups=52, frames=192, wheels=702 |
| Clean Sketchfab/Objaverse labels | PASS | QA kept 152 frames / 626 wheels |
| Dataset format/leakage audit | PASS | `outputs/production_audit/dataset_audit.json`: 12 configs, failed=0 |
| Release package integrity | PASS | `outputs/production_audit/release_integrity.json`: 85 artifacts, total=126.342 MB |
| Runtime AR contract smoke | PASS | `outputs/production_audit/runtime_contract_audit.json`: single wheels=2, batch=5 frames / 12 wheels |
| Desktop performance audit | PASS | `outputs/production_audit/performance_audit.json`: samples=8, PT mean=41.208ms, ONNX mean=37.600ms, LiteRT smoke mean=269.127ms |
| Senior ML evidence matrix | PASS | `outputs/production_audit/senior_ml_audit.json`: integration_ready=True, production_ready=False |
| Objective completion audit | PASS | `outputs/production_audit/objective_completion_audit.json`: objective_complete=False, failed=2, missing=0 |
| Export parity diagnosis | PASS | `outputs/production_audit/export_parity_audit.json`: certified=False |
| Calibrated export certification | PASS | `outputs/production_audit/export_certification.json`: scope=desktop_export_backend_certification_not_android_device |
| Champion clears real-only bbox target | PASS | Real-only eval bbox mAP50=0.912 on `configs/pose_dataset_real_v1_self.yaml` |
| Champion on mixed real+UE anchor | WARN | Anchor bbox mAP50=0.697; this split includes synthetic validation frames and is harder/not the production acceptance split |
| New Sketchfab clean fine-tune improves champion | FAIL | Mixed clean fine-tune is below champion and is not promoted |
| ONNX export exists | PASS | `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.onnx` |
| ONNX strict parity vs PyTorch | WARN | Diagnostic strict policy: `outputs/production_audit/onnx_drift_20.json`: 14/20 samples matched, max kp drift=13.371px |
| ONNX aggregate eval | PASS | Calibrated export certification compares ONNX aggregate metrics against PyTorch champion |
| TFLite/LiteRT desktop package | PASS | `outputs/production_audit/tflite_certification.json`: scope=desktop_tflite_litert_package_not_android_device |
| Android LiteRT device validation | FAIL | Missing `outputs/production_audit/android_litert_device_eval.json` from the target app/device runtime |
| Human-labelled AR-device holdout | FAIL | No Android plugin holdout batch is present yet |
| AR-side 3D validation | FAIL | No recorded AR session with raycast + RANSAC error report is present |

## Model Comparison

| Model / eval split | bbox mAP50 | bbox mAP50-95 | OKS mean | FN rate | FP rate | GT / pred / matched |
|---|---:|---:|---:|---:|---:|---:|
| Champion PT on real-only self val | 0.912 | 0.813 | 0.887 | 0.062 | 0.250 | 64 / 80 / 60 |
| Champion PT on real+self+UE anchor val | 0.697 | 0.621 | 0.887 | 0.286 | 0.259 | 84 / 81 / 60 |
| Previous self_s on real+self+UE anchor val | 0.688 | 0.598 | 0.894 | 0.298 | 0.280 | 84 / 82 / 59 |
| Mixed real+self+UE+Sketchfab clean fine-tune | 0.680 | 0.591 | 0.860 | 0.310 | 0.256 | 84 / 78 / 58 |
| UE-only Sketchfab clean fine-tune | 0.113 | 0.066 | 0.176 | 0.857 | 0.571 | 84 / 28 / 12 |
| Champion ONNX on real+self+UE anchor val | 0.692 | 0.617 | 0.888 | 0.286 | 0.268 | 84 / 82 / 60 |
| Champion TFLite float32 on real+self+UE anchor val | 0.692 | 0.617 | 0.888 | 0.286 | 0.268 | 84 / 82 / 60 |

Decision: keep `wheel_real_v1_self_plus_ue_synthetic_s` as the current
PyTorch integration candidate. Do not promote the UE-only or
Sketchfab-clean mixed fine-tunes.

## Export Audit

Strict PT-vs-ONNX drift check:

- Report: `outputs/production_audit/onnx_drift_20.json`
- Samples matched: 14 / 20
- Max bbox drift: 8.497px
- Max keypoint drift: 13.371px
- Max confidence drift: 0.228

The strict parity report remains diagnostic. Calibrated export
certification is the authoritative desktop export policy and passes for
both ONNX and TFLite.

TFLite/LiteRT status:

- Certification: `outputs/production_audit/tflite_certification.json`
- Certified: True
- Artifact: `outputs/production_audit/tflite_export/best_float32.tflite`
- Aggregate eval: `outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_tflite_on_self_plus_ue_val.json`
- Aggregate bbox mAP50: 0.692
- Aggregate OKS: 0.888
- Scope: `desktop_tflite_litert_package_not_android_device`
- Strict 20-frame diagnostic drift: 14 / 20 matched, max keypoint drift=13.372px
- LiteRT Python smoke: ok=True, output shape=[1, 14, 8400], mean CPU latency=269.127 ms

Desktop TFLite/LiteRT package certification passes. It is not an Android
device certificate until the exact artifact is checked in the Android
app/device runtime, with output shape, finite output, latency, and memory
measurements on the target device.

## Confidence Threshold Sweep

Real-only validation split, current PyTorch candidate:

| conf | OKS mean | FN rate | FP rate | GT / pred / matched |
|---:|---:|---:|---:|---:|
| 0.15 | 0.887 | 0.062 | 0.277 | 64 / 83 / 60 |
| 0.20 | 0.887 | 0.062 | 0.268 | 64 / 82 / 60 |
| 0.25 | 0.887 | 0.062 | 0.250 | 64 / 80 / 60 |
| 0.30 | 0.887 | 0.062 | 0.250 | 64 / 80 / 60 |
| 0.40 | 0.887 | 0.062 | 0.221 | 64 / 77 / 60 |
| 0.50 | 0.887 | 0.062 | 0.211 | 64 / 76 / 60 |
| 0.60 | 0.887 | 0.078 | 0.213 | 64 / 75 / 59 |

Recommendation for AR smoke: use `conf=0.50` initially. On the current
real-only validation split it keeps FN at `0.063` while reducing FP from
`0.250` at conf `0.25` to `0.211`. `conf=0.60` starts losing recall.

## Production Blockers

1. Collect or receive a human-labelled Android/AR plugin holdout. The
   current real set is self-labelled and image-source biased.
2. Run AR-side replay through raycast + RANSAC and measure 3D disc-bottom
   error/stability. Pixel OKS alone is not the final product metric.
3. Certify the actual Android runtime with
   `src/validate_android_litert_report.py`; desktop TFLite/LiteRT package
   certification already passes.
4. Improve keypoint precision. Current median keypoint errors are around
   7-8 px; the old <=5 px line is not consistently met.

## Commands Re-run For This Audit

```bash
./.venv/bin/python src/production_audit_suite.py --with-pytest
```

```bash
MODEL=runs/pose/wheel_real_v1_self_s/weights/best.pt \
DATA=configs/pose_dataset_real_v1_self_plus_ue_synthetic.yaml \
OUT=outputs/eval/wheel_real_v1_self_s_on_self_plus_ue_val.json \
DEVICE=mps ./scripts/eval_baseline.sh
```

```bash
MODEL=runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt \
DATA=configs/pose_dataset_real_v1_self.yaml \
OUT=outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_on_self_val.json \
DEVICE=mps ./scripts/eval_baseline.sh
```

```bash
./.venv/bin/python src/check_export_drift.py \
  --pt-model runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt \
  --exported-model runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.onnx \
  --images-dir data/wheel_pose_dataset_real_v1_self_plus_ue_synthetic/images/val \
  --limit 20 --device cpu \
  --out outputs/production_audit/onnx_drift_20.json
```

```bash
./.venv_tflite/bin/python src/check_export_drift.py \
  --pt-model runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt \
  --exported-model outputs/production_audit/tflite_export/best_float32.tflite \
  --exported-task pose \
  --images-dir data/wheel_pose_dataset_real_v1_self_plus_ue_synthetic/images/val \
  --limit 20 --device cpu \
  --out outputs/production_audit/tflite_drift_20.json
```

```bash
./.venv/bin/python scripts/create_android_litert_report_template.py
```

Android device evidence producer:

```text
android_litert_harness/README.md
android_litert_harness/AndroidLiteRtDeviceValidationTest.kt
```

```bash
./.venv/bin/python scripts/create_ar_replay_log_template.py
```

AR replay evidence producer:

```text
ar_replay_harness/README.md
ar_replay_harness/ArReplayLogger.kt
```

```bash
./.venv/bin/python scripts/create_ar_holdout_provenance_template.py
```

AR holdout evidence producer:

```text
ar_holdout_harness/README.md
ar_holdout_harness/ArHoldoutAnnotationWriter.kt
```

```bash
./.venv/bin/python src/validate_android_litert_report.py   --source data/incoming/android_litert_device_report.json   --out outputs/production_audit/android_litert_device_eval.json
```

```bash
./.venv/bin/python src/evaluate_ar_holdout.py \
  --source-root data/incoming/ar_device_holdout \
  --eval-out outputs/production_audit/ar_device_holdout_eval.json
```

```bash
./.venv/bin/python src/validate_ar_replay.py \
  --jsonl path/to/ar_replay.jsonl \
  --out outputs/production_audit/ar_3d_replay_eval.json
```

```bash
./.venv/bin/python src/run_production_evidence_intake.py
```

```bash
./.venv/bin/python src/import_external_evidence_drop.py path/to/evidence_drop.zip --dry-run
./.venv/bin/python src/import_external_evidence_drop.py path/to/evidence_drop.zip --overwrite
```

```bash
./.venv/bin/python scripts/create_external_evidence_return_template.py
```

```bash
./.venv/bin/python scripts/build_external_evidence_handoff_bundle.py
```

```bash
./.venv/bin/python src/verify_external_evidence_handoff_bundle.py
```
