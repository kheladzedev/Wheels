# Spec Compliance Audit

Executable audit for AR spec / ML contract compliance.

- OK: True
- Failures: none
- ML scope: per-frame, stateless 2D wheel detection with three keypoints
- AR scope: raycast, RANSAC, plane recovery, K-frame accumulation, tracking

| Check | OK | Evidence | Detail |
|---|---:|---|---|
| spec_compliance_document | True | `docs/SPEC_COMPLIANCE.md` | present and contains required spec anchors |
| ar_ml_contract_document | True | `docs/AR_ML_CONTRACT.md` | present and contains required spec anchors |
| canonical_keypoint_names | True | `src/postprocess_wheels.py` | KEYPOINT_NAMES=('rim_left', 'rim_right', 'disc_bottom'), N_KEYPOINTS=3 |
| confirmed_keypoint_mapping | True | `src/postprocess_wheels.py` | INTERNAL_TO_CONFIRMED_KP={'rim_left': 'a', 'rim_right': 'b', 'disc_bottom': 'c_disc_bottom'} |
| confirmed_top_level_schema | True | `src/postprocess_wheels.py::to_confirmed_schema` | keys=['frame_id', 'wheels'] |
| confirmed_wheel_schema | True | `src/postprocess_wheels.py::to_confirmed_schema` | keys=['bbox_xyxy', 'confidence', 'points'] |
| confirmed_points_schema | True | `src/postprocess_wheels.py::to_confirmed_schema` | points=['a', 'b', 'c_disc_bottom'] |
| occluded_wheels_are_dropped | True | `src/postprocess_wheels.py::to_confirmed_schema` | emitted_wheels=1 |
| confirmed_schema_has_no_forbidden_ml_fields | True | `src/postprocess_wheels.py::to_confirmed_schema` | forbidden_hits=[] |
| contract_tests_present | True | `tests/test_ar_contract.py; tests/test_confirmed_ar_schema_shape.py` | shape guards present |
| inference_wrappers_present | True | `src/infer_image.py; src/infer_batch.py` | single-frame and batch AR payload entrypoints present |
