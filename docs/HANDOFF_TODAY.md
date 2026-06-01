# Handoff Today

## Current Status

- Car-body model pool: 300/300 clean GLBs, sketchfab=234, objaverse_fallback=66, missing=0, rejected=19
- UnrealMCP: 127.0.0.1:55557 not reachable
- Champion checkpoint: `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt`
- Champion ONNX: `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.onnx`
- Champion eval JSON: `outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s.json`
- Engine-keypoint incoming: data/incoming/ue_neuraldata_keypoint_full images=51/50 annotations=51 wheels=90/80
- Engine-keypoint YOLO: data/wheel_pose_dataset_ue_neuraldata_keypoint_full train=41/41 val=10/10
- Sketchfab/Objaverse render pool: outputs/ue_sketchfab_renders/images png=1200/1200
- Sketchfab/Objaverse geometry incoming: data/incoming/ue_sketchfab_geometry images=192/150 annotations=192 wheels=702/500
- Sketchfab/Objaverse geometry RGB content: data/incoming/ue_sketchfab_geometry/images nonblack_png=50/25 sampled=50
- Sketchfab/Objaverse geometry YOLO: data/wheel_pose_dataset_ue_sketchfab_geometry train=154/154 val=38/38
- Sketchfab/Objaverse clean geometry incoming: data/incoming/ue_sketchfab_geometry_clean images=152/120 annotations=152 wheels=626/500
- Sketchfab/Objaverse clean geometry RGB content: data/incoming/ue_sketchfab_geometry_clean/images nonblack_png=50/25 sampled=50
- Sketchfab/Objaverse clean geometry YOLO: data/wheel_pose_dataset_ue_sketchfab_geometry_clean train=122/122 val=30/30
- Real+self+UE+Sketchfab clean mixed YOLO: data/wheel_pose_dataset_real_self_ue_plus_sketchfab_clean train=354/354 val=58/58
- Real+self+UE+Sketchfab clean eval diagnostic: not_promoted candidate oks=0.860 fn=0.310 fp=0.256 bbox_mAP50=0.680; champion oks=0.887 fn=0.286 fp=0.259 bbox_mAP50=0.697
- Production readiness audit: docs/PRODUCTION_READINESS_AUDIT.md
- Model inventory: 13 train runs, 35 artifacts, 29 eval reports; report `docs/MODEL_INVENTORY.md`.
- Model selection audit: ok=True, anchor candidates=5, promotion required=0; report `docs/MODEL_SELECTION_AUDIT.md`.
- Spec compliance audit: ok=True, failures=[]; report `docs/SPEC_COMPLIANCE_AUDIT.md`.
- Dataset audit: ok=False, configs=22, failed=20, wheel labels=12148; report `docs/DATASET_AUDIT.md`.
- Release package integrity: ok=True, artifacts=103, size=196.211 MB; report `docs/RELEASE_PACKAGE.md`.
- Runtime contract audit: ok=True, single wheels=2, batch=5 frames / 12 wheels.
- Performance audit: ok=True, samples=8, PT mean=41.96455724968473 ms, ONNX mean=39.09157031216637 ms, LiteRT smoke mean=269.1269209004531 ms; report `docs/PERFORMANCE_AUDIT.md`.
- Senior ML audit: integration_ready=True, production_ready=False, production blockers=['android_litert_device_validation', 'human_labelled_ar_device_holdout', 'ar_3d_replay_validation', 'production_evidence_audit_ready', 'production_gate']; report `docs/SENIOR_ML_AUDIT.md`.
- Objective completion audit: objective_complete=False, integration_ready=True, production_ready=False; report `docs/OBJECTIVE_COMPLETION_AUDIT.md`.
- Export parity audit: certified=False, ONNX categories={'count_mismatch': 0, 'bbox_drift': 1, 'keypoint_drift': 2, 'confidence_drift': 4, 'other': 0}, TFLite categories={'count_mismatch': 0, 'bbox_drift': 1, 'keypoint_drift': 2, 'confidence_drift': 4, 'other': 0}; report `docs/EXPORT_PARITY_AUDIT.md`.
- Calibrated export certification: certified=True, scope=desktop_export_backend_certification_not_android_device; report `docs/EXPORT_CERTIFICATION.md`.
- Champion ONNX drift diagnostic: not_certified samples=14/20 max_bbox=8.497px max_kp=13.371px max_conf=0.228
- Champion TFLite certification diagnostic: certified artifact=outputs/production_audit/tflite_export/best_float32.tflite bbox_mAP50=0.692 oks=0.888 fn=0.286 fp=0.268
- Champion TFLite float32: `outputs/production_audit/tflite_export/best_float32.tflite`; aggregate eval `GT/pred/matched=84/82/60, OKS=0.8880997181541215, FN=0.2857142857142857, FP=0.2682926829268293, bbox mAP50=0.6923761088840539`; certified=True.
- Champion CoreML mlmodel: `outputs/production_audit/coreml_export/best.mlmodel`; certified=True; scope=desktop_coreml_package_not_ios_device.
- Integration gate: PASS; production gate: FAIL failed=['production_evidence_audit_ready', 'android_litert_device_eval', 'human_ar_holdout_eval', 'ar_3d_replay_eval'].
- Sketchfab/Objaverse pseudo-label diagnostic: data/incoming/ue_sketchfab_pseudo_conf005 images=2 annotations=2 wheels=2

## Champion Eval

- GT / predicted / matched wheels: 84 / 81 / 60
- OKS mean: 0.8872704757196613
- FN rate: 0.2857142857142857
- FP rate: 0.25925925925925924

## Synthetic Training Result

- UE-only clean geometry fine-tune: `GT/pred/matched=84/28/12, OKS=0.17641177000046013, FN=0.8571428571428571, FP=0.5714285714285714, bbox mAP50=0.11266599716664634`. Not promoted; it regressed badly on real validation.
- Mixed real+self+UE+Sketchfab clean fine-tune v2: `GT/pred/matched=84/78/58, OKS=0.8595977937738902, FN=0.30952380952380953, FP=0.2564102564102564, bbox mAP50=0.6801854591063443`. Not promoted; it uses the expanded clean geometry dataset but is still below the champion on OKS, FN, and bbox mAP50.
- Current production checkpoint remains `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt`.
- Recommended AR smoke confidence threshold: `0.50` (keeps FN at 0.063 on real-only val and reduces FP versus `0.25`).
- Full production is not certified yet: Android LiteRT device certification, human-labelled AR-device holdout, and AR-side 3D validation are still open. Strict ONNX/TFLite parity is now diagnostic; calibrated desktop export certification passes. See `docs/PRODUCTION_READINESS_AUDIT.md`.

## Ready Artifacts

- Sketchfab resumable downloader: `src/fetch_sketchfab_cars.py`
- Objaverse fallback downloader: `src/fetch_objaverse_cars.py`
- Autonomous fetch loop: `scripts/fetch_sketchfab_until_target.sh`
- UE import script: `scripts/ue/import_sketchfab_glbs.py`
- UE batch render script: `scripts/ue/render_sketchfab_cars.py`
- UE geometry-label export script: `scripts/ue/render_sketchfab_geometry_labels.py`
- UE geometry QA filter: `src/filter_geometry_incoming.py`
- YOLO dataset union builder: `src/build_yolo_pose_dataset_union.py`
- UE render pseudo-label bridge: `src/pseudo_label_images_to_incoming.py`
- End-to-end UE pseudo wrapper: `scripts/prepare_ue_sketchfab_pseudo_data.sh`
- UE grouped model render status: `outputs/ue_tasks/render_sketchfab_cars_status.json`
- UE geometry-label status: `outputs/ue_tasks/render_sketchfab_geometry_labels_status.json`
- UE geometry-label dataset config: `configs/pose_dataset_ue_sketchfab_geometry.yaml`
- UE clean geometry dataset config: `configs/pose_dataset_ue_sketchfab_geometry_clean.yaml`
- Mixed clean dataset config: `configs/pose_dataset_real_self_ue_plus_sketchfab_clean.yaml`
- Production readiness audit: `docs/PRODUCTION_READINESS_AUDIT.md`
- Model inventory report: `docs/MODEL_INVENTORY.md`
- Model selection audit report: `docs/MODEL_SELECTION_AUDIT.md`
- Spec compliance audit report: `docs/SPEC_COMPLIANCE_AUDIT.md`
- Model card: `docs/MODEL_CARD.md`
- Dataset audit report: `docs/DATASET_AUDIT.md`
- Release package report: `docs/RELEASE_PACKAGE.md`
- Performance audit report: `docs/PERFORMANCE_AUDIT.md`
- Senior ML audit report: `docs/SENIOR_ML_AUDIT.md`
- Objective completion audit report: `docs/OBJECTIVE_COMPLETION_AUDIT.md`
- Export parity audit report: `docs/EXPORT_PARITY_AUDIT.md`
- Export certification report: `docs/EXPORT_CERTIFICATION.md`
- Android LiteRT report contract: `docs/ANDROID_LITERT_DEVICE_REPORT.md`
- Production evidence checklist: `docs/PRODUCTION_EVIDENCE_CHECKLIST.md`
- Production evidence intake doc: `docs/PRODUCTION_EVIDENCE_INTAKE.md`
- External evidence handoff bundle doc: `docs/EXTERNAL_EVIDENCE_HANDOFF_BUNDLE.md`
- Production evidence audit: `docs/PRODUCTION_EVIDENCE_AUDIT.md`
- Production evidence intake status: `outputs/production_audit/production_evidence_intake_status.json`
- Production evidence preflight status: `outputs/production_audit/production_evidence_intake_preflight_status.json`
- Requirements traceability matrix: `docs/REQUIREMENTS_TRACEABILITY.md`
- Executive report RU: `docs/EXECUTIVE_REPORT_RU.md`
- Runtime contract audit: `outputs/production_audit/runtime_contract_audit.json`
- Model package manifest: `outputs/production_audit/model_package_manifest.json`
- TFLite certification report: `outputs/production_audit/tflite_certification.json`
- CoreML artifact: `outputs/production_audit/coreml_export/best.mlmodel`
- CoreML certification report: `outputs/production_audit/coreml_certification.json`
- CoreML certification doc: `docs/COREML_CERTIFICATION.md`
- LiteRT runtime smoke: `outputs/production_audit/litert_runtime_smoke.json`
- Multi-sample export drift checker: `src/check_export_drift.py`
- LiteRT runtime checker: `src/check_litert_runtime.py`
- Android LiteRT device report validator: `src/validate_android_litert_report.py`
- Android LiteRT report template writer: `scripts/create_android_litert_report_template.py`
- Android LiteRT validation harness doc: `android_litert_harness/README.md`
- Android LiteRT validation harness test: `android_litert_harness/AndroidLiteRtDeviceValidationTest.kt`
- AR holdout provenance template: `outputs/production_audit/ar_device_holdout_provenance.template.json`
- AR holdout provenance template writer: `scripts/create_ar_holdout_provenance_template.py`
- AR holdout annotation harness doc: `ar_holdout_harness/README.md`
- AR holdout annotation writer: `ar_holdout_harness/ArHoldoutAnnotationWriter.kt`
- AR replay log template: `outputs/production_audit/ar_3d_replay.template.jsonl`
- AR replay log template writer: `scripts/create_ar_replay_log_template.py`
- AR replay logging harness doc: `ar_replay_harness/README.md`
- AR replay logging harness: `ar_replay_harness/ArReplayLogger.kt`
- External evidence handoff bundle: `outputs/production_audit/external_evidence_handoff_bundle.zip`
- External evidence handoff bundle manifest: `outputs/production_audit/external_evidence_handoff_bundle_manifest.json`
- External evidence handoff bundle verification: `outputs/production_audit/external_evidence_handoff_bundle_verification.json`
- External evidence handoff bundle builder: `scripts/build_external_evidence_handoff_bundle.py`
- External evidence handoff bundle verifier: `src/verify_external_evidence_handoff_bundle.py`
- Production evidence audit runner: `src/production_evidence_audit.py`
- External evidence drop importer: `src/import_external_evidence_drop.py`
- Production evidence intake runner: `src/run_production_evidence_intake.py`
- External evidence return template: `outputs/production_audit/external_evidence_return_template.zip`
- External evidence return template manifest: `outputs/production_audit/external_evidence_return_template_manifest.json`
- External evidence return template writer: `scripts/create_external_evidence_return_template.py`
- Requirements traceability runner: `src/requirements_traceability.py`
- Executive report RU runner: `src/executive_report_ru.py`
- Objective completion audit runner: `src/objective_completion_audit.py`
- Model selection audit runner: `src/model_selection_audit.py`
- Spec compliance audit runner: `src/spec_compliance_audit.py`
- AR holdout evaluator: `src/evaluate_ar_holdout.py`
- AR replay validator for raycast/RANSAC logs: `src/validate_ar_replay.py`
- Production audit suite runner: `src/production_audit_suite.py`
- UE model import status: `outputs/ue_tasks/import_sketchfab_glbs_status.json`
- Final orchestrator: `scripts/finish_project_today.sh`
- Readiness gate: `src/project_readiness.py`

## Remaining Gates

1. Spot-QA the clean UE geometry labels: labels are mesh-part projected boxes/keypoints and are marked draft/review-needed.
2. Improve label precision/domain randomization before another promotion attempt; both UE-only and mixed fine-tunes have been measured and are not production replacements yet.

## Commands

```bash
TARGET_TOTAL=300 RATE_LIMIT_SLEEP=900 \
./scripts/fetch_sketchfab_until_target.sh
```

```bash
./.venv/bin/python src/fetch_objaverse_cars.py \
  --output-dir data/sketchfab_cars --target-total 300
```

```bash
RUN_FETCH=1 RUN_OBJAVERSE=1 RUN_UE=1 WAIT_FOR_MCP=1 \
MCP_WAIT_TIMEOUT=1800 MCP_WAIT_INTERVAL=10 \
./scripts/finish_project_today.sh
```

```bash
./.venv/bin/python src/project_readiness.py
```

```bash
./.venv/bin/python src/production_audit_suite.py --with-pytest
```
