# Requirements Traceability

- Production ready: False
- Requirements passed: 11/16
- Train runs inventoried: 11
- Eval reports inventoried: 20
- Release integrity OK: True

| Requirement | Status | Evidence | Detail | Gap |
|---|---|---|---|---|
| Champion model artifact exists | pass | `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt` | present |  |
| Exportable TFLite artifact exists | pass | `outputs/production_audit/tflite_export/best_float32.tflite` | present |  |
| Model lineage and inventory are documented | pass | `outputs/production_audit/model_inventory.json` | train_runs=11, artifacts=30, eval_reports=20, champion_run=True |  |
| Champion selection and promotion guard passes | pass | `outputs/production_audit/model_selection_audit.json` | ok=True, failures=[] |  |
| Training/evaluation datasets pass format and leakage audit | pass | `outputs/production_audit/dataset_audit.json` | ok=True, failures=[] |  |
| Champion meets real-validation quality targets | pass | `outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_on_self_val.json` | bbox_mAP50=0.912>=0.850, OKS=0.887>=0.800, FN=0.062<=0.100 |  |
| AR JSON runtime contract is implemented and smoke-tested | pass | `outputs/production_audit/runtime_contract_audit.json` | ok=True, failures=[] |  |
| ML deliverable matches the AR technical specification | pass | `outputs/production_audit/spec_compliance_audit.json` | ok=True, failures=[] |  |
| Local performance audit passes | pass | `outputs/production_audit/performance_audit.json` | ok=True, failures=[] |  |
| ONNX/TFLite export backends are certified | pass | `outputs/production_audit/export_certification.json` | certified=True, status=certified, artifact=n/a |  |
| Desktop TFLite/LiteRT package is certified | pass | `outputs/production_audit/tflite_certification.json` | certified=True, status=certified, artifact=outputs/production_audit/tflite_export/best_float32.tflite |  |
| Android-device LiteRT evidence is validated | missing | `outputs/production_audit/android_litert_device_eval.json` | ok=False, failures=[] | missing_source:data/incoming/android_litert_device_report.json, missing_report:outputs/production_audit/android_litert_device_eval.json |
| Human-labelled AR-device holdout passes production thresholds | missing | `outputs/production_audit/ar_device_holdout_eval.json` | bbox_mAP50=0.000>=0.850, OKS=0.000>=0.800, FN=1.000<=0.100 | missing_source_dirs:data/incoming/ar_device_holdout, missing_provenance:data/incoming/ar_device_holdout/metadata/provenance.json, missing_report:outputs/production_audit/ar_device_holdout_eval.json, missing_pipeline:outputs/production_audit/ar_device_holdout_pipeline.json |
| AR-side 3D replay/RANSAC validation passes | missing | `outputs/production_audit/ar_3d_replay_eval.json` | ok=False, failures=[] | missing_source:data/incoming/ar_3d_replay/ar_replay.jsonl, missing_report:outputs/production_audit/ar_3d_replay_eval.json |
| Consolidated production evidence audit passes | fail | `outputs/production_audit/production_evidence_audit.json` | production_evidence_ready=False, blockers=['android_litert_device_validation', 'human_labelled_ar_device_holdout', 'ar_3d_replay_validation'] | android_litert_device_validation, human_labelled_ar_device_holdout, ar_3d_replay_validation |
| Production gate passes | fail | `outputs/production_audit/production_gate.json` | ok=False, failed=['production_evidence_audit_ready', 'android_litert_device_eval', 'human_ar_holdout_eval', 'ar_3d_replay_eval'] |  |
