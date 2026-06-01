# Report Consistency Audit

- OK: True
- Failures: none

| Check | OK | Detail |
|---|---:|---|
| release_manifest_no_self_reference | True | self_references=[], release_ok=True |
| release_manifest_hashes_current | True | missing=[], mismatches=[] |
| suite_release_count_matches_manifest | True | suite_release_artifacts=103, release_artifact_count=103 |
| objective_status_matches_suite | True | mismatches=[] |
| objective_completion_consistent | True | objective_complete=False, production_ready=False, failed_requirements=['training_data_reviewed', 'production_evidence_present', 'production_gate_passed'] |
| suite_status_matches_gates | True | mismatches=[], integration_gate_ok=True, production_gate_ok=False |
| prefinal_reports_do_not_embed_release_counts | True | hits=[] |
| required_final_reports_present | True | missing=[] |
| model_package_manifest_artifacts_current | True | missing=[], mismatches=[], digest_ok=True, missing_declared=[] |
| external_evidence_return_template_artifact_current | True | failures=[], artifact_path=outputs/production_audit/tflite_export/best_float32.tflite, manifest_sha=c75f7173d97fab9e69bf71fa2d7cca9482d6beeff8bd4d647432545a4f496237, actual_sha=c75f7173d97fab9e69bf71fa2d7cca9482d6beeff8bd4d647432545a4f496237, zip_sha_match=True, zip_size_match=True |
| intake_finalization_contract | True | status_source=preflight, production_evidence_ready=False, status_finalization_required=True, status_finalization_command=['./.venv/bin/python', 'src/production_audit_suite.py', '--with-pytest'], status_finalization_ok=False, doc_command=True, doc_finalize=True, doc_finalization_required=True |
| intake_post_finalization_refresh_contract | True | required=False, production_evidence_ready=False, finalization_ok=False, refresh_count=0, failures=[] |
| handoff_bundle_verification_strict | True | verification_ok=True, required_artifact_count=29/29, entry_count=29, expected_entry_count=29, current_artifacts_ok=True, bundle_sha_match=True, bundle_size_match=True, missing_required=[], failures=[] |
| production_evidence_release_artifacts | True | production_evidence_ready=False, required_artifacts=6, missing=['outputs/production_audit/android_litert_device_eval.json', 'outputs/production_audit/ar_3d_replay_eval.json', 'outputs/production_audit/ar_device_holdout_eval.json', 'outputs/production_audit/ar_device_holdout_pipeline.json', 'outputs/production_audit/external_evidence_drop_import.json', 'outputs/production_audit/production_evidence_intake_status.json'] |
