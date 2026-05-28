# Senior ML Audit

Machine-readable production-readiness evidence matrix for the wheel-pose model.

- Audit OK: True
- Integration ready: True
- Production ready: False
- Requirements: 24
- Passed: 17
- Failed/missing: 7
- Integration blockers: none
- Production blockers: android_litert_device_validation, human_labelled_ar_device_holdout, ar_3d_replay_validation, production_evidence_audit_ready, production_gate

| Requirement | Category | Status | Integration | Production | Evidence | Detail |
|---|---|---:|---:|---:|---|---|
| champion_pytorch_artifact | artifact | pass | True | True | `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt` | present |
| champion_onnx_artifact | artifact | pass | True | False | `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.onnx` | present |
| champion_tflite_artifact | artifact | pass | False | True | `outputs/production_audit/tflite_export/best_float32.tflite` | present |
| model_inventory_lineage | lineage | pass | True | True | `outputs/production_audit/model_inventory.json` | train_runs=11, artifacts=30, eval_reports=20, champion_run=True |
| model_selection_promotion_guard | lineage | pass | True | True | `outputs/production_audit/model_selection_audit.json` | ok=True, failures=[] |
| external_3d_model_pool | data | pass | True | False | `data/sketchfab_cars` | clean_glb=300/300, rejected=19 |
| ue_geometry_label_yield | data | pass | True | False | `outputs/ue_tasks/render_sketchfab_geometry_labels_status.json` | frames=192/150, wheels=702/500 |
| dataset_format_and_leakage | data | pass | True | True | `outputs/production_audit/dataset_audit.json` | ok=True, failures=[] |
| champion_real_validation_quality | model_quality | pass | True | True | `outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_on_self_val.json` | bbox_mAP50=0.912>=0.850, OKS=0.887>=0.800, FN=0.062<=0.100 |
| runtime_contract | runtime_contract | pass | True | True | `outputs/production_audit/runtime_contract_audit.json` | ok=True, failures=[] |
| spec_compliance_contract | runtime_contract | pass | True | True | `outputs/production_audit/spec_compliance_audit.json` | ok=True, failures=[] |
| performance_audit | runtime_contract | pass | True | True | `outputs/production_audit/performance_audit.json` | ok=True, failures=[] |
| onnx_parity | export | fail | False | False | `outputs/production_audit/onnx_drift_20.json` | ok=False, matched=14/20, max_bbox=8.497px, max_kp=13.371px |
| tflite_parity | export | fail | False | False | `outputs/production_audit/tflite_drift_20.json` | ok=False, matched=14/20, max_bbox=8.497px, max_kp=13.372px |
| export_backend_certification | export | pass | False | True | `outputs/production_audit/export_certification.json` | certified=True, status=certified, artifact=n/a |
| tflite_litert_certification | export | pass | False | True | `outputs/production_audit/tflite_certification.json` | certified=True, status=certified, artifact=outputs/production_audit/tflite_export/best_float32.tflite |
| android_litert_device_validation | runtime_contract | missing | False | True | `outputs/production_audit/android_litert_device_eval.json` | ok=False, failures=[] |
| human_labelled_ar_device_holdout | production_validation | missing | False | True | `outputs/production_audit/ar_device_holdout_eval.json` | bbox_mAP50=0.000>=0.850, OKS=0.000>=0.800, FN=1.000<=0.100 |
| ar_3d_replay_validation | production_validation | missing | False | True | `outputs/production_audit/ar_3d_replay_eval.json` | ok=False, failures=[] |
| production_evidence_audit_ready | production_validation | fail | False | True | `outputs/production_audit/production_evidence_audit.json` | production_evidence_ready=False, blockers=['android_litert_device_validation', 'human_labelled_ar_device_holdout', 'ar_3d_replay_validation'] |
| ar_holdout_evaluation_pipeline | production_tooling | pass | True | False | `src/evaluate_ar_holdout.py` | present |
| ar_replay_validation_pipeline | production_tooling | pass | True | False | `src/validate_ar_replay.py` | present |
| integration_gate | gating | pass | True | False | `outputs/production_audit/integration_gate.json` | ok=True, failed=[] |
| production_gate | gating | fail | False | True | `outputs/production_audit/production_gate.json` | ok=False, failed=['production_evidence_audit_ready', 'android_litert_device_eval', 'human_ar_holdout_eval', 'ar_3d_replay_eval'] |
