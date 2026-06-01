"""Write a concise current-status handoff report for the ML pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import project_readiness as readiness

OUT = Path("docs/HANDOFF_TODAY.md")


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _metric_line(report: dict) -> str:
    counts = report.get("counts", {})
    oks = report.get("oks", {})
    rates = report.get("rates", {})
    bbox = report.get("metrics_bbox", {})
    return (
        f"GT/pred/matched={counts.get('gt_wheels', 'n/a')}/"
        f"{counts.get('pred_wheels_above_conf', 'n/a')}/"
        f"{counts.get('matched', 'n/a')}, "
        f"OKS={oks.get('mean', 'n/a')}, "
        f"FN={rates.get('false_negative_rate', 'n/a')}, "
        f"FP={rates.get('false_positive_rate', 'n/a')}, "
        f"bbox mAP50={bbox.get('mAP50', 'n/a')}"
    )


def _one_gate_status(report: dict) -> str:
    if report.get("ok") is True:
        return "PASS"
    failed = report.get("failed", [])
    if failed:
        return f"FAIL failed={failed}"
    return "FAIL"


def _gate_status_line(integration_gate: dict, production_gate: dict) -> str:
    return (
        "- Integration gate: "
        f"{_one_gate_status(integration_gate)}; production gate: "
        f"{_one_gate_status(production_gate)}."
    )


def main() -> int:
    args = readiness.parse_args([])
    checks = readiness.collect_checks(args)
    check_by_name = {check.name: check for check in checks}
    eval_report = _read_json(Path("outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s.json"))
    counts = eval_report.get("counts", {})
    oks = eval_report.get("oks", {})
    rates = eval_report.get("rates", {})
    ue_only_eval = _read_json(Path("outputs/eval/wheel_ue_sketchfab_geometry_clean_ft20_on_real.json"))
    mixed_eval = _read_json(
        Path("outputs/eval/wheel_real_self_ue_plus_sketchfab_clean_ft20_v2_on_real.json")
    )
    tflite_eval = _read_json(
        Path("outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_tflite_on_self_plus_ue_val.json")
    )
    tflite_cert = _read_json(Path("outputs/production_audit/tflite_certification.json"))
    coreml_cert = _read_json(Path("outputs/production_audit/coreml_certification.json"))
    model_inventory = _read_json(Path("outputs/production_audit/model_inventory.json"))
    model_selection = _read_json(Path("outputs/production_audit/model_selection_audit.json"))
    spec_compliance = _read_json(Path("outputs/production_audit/spec_compliance_audit.json"))
    dataset_audit = _read_json(Path("outputs/production_audit/dataset_audit.json"))
    release_integrity = _read_json(Path("outputs/production_audit/release_integrity.json"))
    runtime_contract = _read_json(Path("outputs/production_audit/runtime_contract_audit.json"))
    performance_audit = _read_json(Path("outputs/production_audit/performance_audit.json"))
    senior_audit = _read_json(Path("outputs/production_audit/senior_ml_audit.json"))
    objective_audit = _read_json(Path("outputs/production_audit/objective_completion_audit.json"))
    export_parity = _read_json(Path("outputs/production_audit/export_parity_audit.json"))
    export_certification = _read_json(Path("outputs/production_audit/export_certification.json"))
    integration_gate = _read_json(Path("outputs/production_audit/integration_gate.json"))
    production_gate = _read_json(Path("outputs/production_audit/production_gate.json"))
    perf_pt = performance_audit.get("benchmarks", {}).get("pytorch_cpu", {})
    perf_onnx = performance_audit.get("benchmarks", {}).get("onnx_cpu", {})
    perf_litert = performance_audit.get("benchmarks", {}).get("litert_cpu_smoke", {})

    text = f"""# Handoff Today

## Current Status

- Car-body model pool: {check_by_name['car_body_model_pool'].detail}
- UnrealMCP: {check_by_name['unreal_mcp'].detail}
- Champion checkpoint: `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt`
- Champion ONNX: `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.onnx`
- Champion eval JSON: `outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s.json`
- Engine-keypoint incoming: {check_by_name['ue_neuraldata_keypoint_full_incoming'].detail}
- Engine-keypoint YOLO: {check_by_name['ue_neuraldata_keypoint_full_yolo'].detail}
- Sketchfab/Objaverse render pool: {check_by_name['ue_sketchfab_render_pool'].detail}
- Sketchfab/Objaverse geometry incoming: {check_by_name['ue_sketchfab_geometry_incoming'].detail}
- Sketchfab/Objaverse geometry RGB content: {check_by_name['ue_sketchfab_geometry_rgb_content'].detail}
- Sketchfab/Objaverse geometry YOLO: {check_by_name['ue_sketchfab_geometry_yolo'].detail}
- Sketchfab/Objaverse clean geometry incoming: {check_by_name['ue_sketchfab_geometry_clean_incoming'].detail}
- Sketchfab/Objaverse clean geometry RGB content: {check_by_name['ue_sketchfab_geometry_clean_rgb_content'].detail}
- Sketchfab/Objaverse clean geometry YOLO: {check_by_name['ue_sketchfab_geometry_clean_yolo'].detail}
- Real+self+UE+Sketchfab clean mixed YOLO: {check_by_name['real_self_ue_plus_sketchfab_clean_yolo'].detail}
- Real+self+UE+Sketchfab clean eval diagnostic: {check_by_name['real_self_ue_plus_sketchfab_clean_eval_diagnostic'].detail}
- Production readiness audit: {check_by_name['production_readiness_audit'].detail}
- Model inventory: {model_inventory.get('counts', {}).get('train_runs', 'n/a')} train runs, {model_inventory.get('counts', {}).get('artifacts', 'n/a')} artifacts, {model_inventory.get('counts', {}).get('eval_reports', 'n/a')} eval reports; report `docs/MODEL_INVENTORY.md`.
- Model selection audit: ok={model_selection.get('ok', 'n/a')}, anchor candidates={model_selection.get('counts', {}).get('anchor_candidates', 'n/a')}, promotion required={model_selection.get('counts', {}).get('promotion_required', 'n/a')}; report `docs/MODEL_SELECTION_AUDIT.md`.
- Spec compliance audit: ok={spec_compliance.get('ok', 'n/a')}, failures={spec_compliance.get('failures', 'n/a')}; report `docs/SPEC_COMPLIANCE_AUDIT.md`.
- Dataset audit: ok={dataset_audit.get('ok', 'n/a')}, configs={dataset_audit.get('counts', {}).get('configs', 'n/a')}, failed={dataset_audit.get('counts', {}).get('failed', 'n/a')}, wheel labels={dataset_audit.get('counts', {}).get('total_wheel_labels', 'n/a')}; report `docs/DATASET_AUDIT.md`.
- Release package integrity: ok={release_integrity.get('ok', 'n/a')}, artifacts={release_integrity.get('artifact_count', 'n/a')}, size={release_integrity.get('total_size_mb', 'n/a')} MB; report `docs/RELEASE_PACKAGE.md`.
- Runtime contract audit: ok={runtime_contract.get('ok', 'n/a')}, single wheels={runtime_contract.get('counts', {}).get('single_wheels', 'n/a')}, batch={runtime_contract.get('counts', {}).get('batch_frames', 'n/a')} frames / {runtime_contract.get('counts', {}).get('batch_wheels', 'n/a')} wheels.
- Performance audit: ok={performance_audit.get('ok', 'n/a')}, samples={performance_audit.get('sample_count', 'n/a')}, PT mean={perf_pt.get('latency_ms', {}).get('mean', 'n/a')} ms, ONNX mean={perf_onnx.get('latency_ms', {}).get('mean', 'n/a')} ms, LiteRT smoke mean={perf_litert.get('latency_ms', {}).get('mean', 'n/a')} ms; report `docs/PERFORMANCE_AUDIT.md`.
- Senior ML audit: integration_ready={senior_audit.get('integration_ready', 'n/a')}, production_ready={senior_audit.get('production_ready', 'n/a')}, production blockers={senior_audit.get('production_blockers', 'n/a')}; report `docs/SENIOR_ML_AUDIT.md`.
- Objective completion audit: objective_complete={objective_audit.get('objective_complete', 'n/a')}, integration_ready={objective_audit.get('integration_ready', 'n/a')}, production_ready={objective_audit.get('production_ready', 'n/a')}; report `docs/OBJECTIVE_COMPLETION_AUDIT.md`.
- Export parity audit: certified={export_parity.get('certified', 'n/a')}, ONNX categories={export_parity.get('summary', {}).get('onnx', {}).get('category_counts', 'n/a')}, TFLite categories={export_parity.get('summary', {}).get('tflite', {}).get('category_counts', 'n/a')}; report `docs/EXPORT_PARITY_AUDIT.md`.
- Calibrated export certification: certified={export_certification.get('certified', 'n/a')}, scope={export_certification.get('scope', 'n/a')}; report `docs/EXPORT_CERTIFICATION.md`.
- Champion ONNX drift diagnostic: {check_by_name['champion_onnx_drift_diagnostic'].detail}
- Champion TFLite certification diagnostic: {check_by_name['champion_tflite_certification_diagnostic'].detail}
- Champion TFLite float32: `{tflite_cert.get('artifact', {}).get('path', 'n/a')}`; aggregate eval `{_metric_line(tflite_eval)}`; certified={tflite_cert.get('certified', False)}.
- Champion CoreML mlmodel: `{coreml_cert.get('artifact', {}).get('path', 'n/a')}`; certified={coreml_cert.get('certified', False)}; scope={coreml_cert.get('scope', 'n/a')}.
{_gate_status_line(integration_gate, production_gate)}
- Sketchfab/Objaverse pseudo-label diagnostic: {check_by_name['ue_sketchfab_pseudo_yield_diagnostic'].detail}

## Champion Eval

- GT / predicted / matched wheels: {counts.get('gt_wheels', 'n/a')} / {counts.get('pred_wheels_above_conf', 'n/a')} / {counts.get('matched', 'n/a')}
- OKS mean: {oks.get('mean', 'n/a')}
- FN rate: {rates.get('false_negative_rate', 'n/a')}
- FP rate: {rates.get('false_positive_rate', 'n/a')}

## Synthetic Training Result

- UE-only clean geometry fine-tune: `{_metric_line(ue_only_eval)}`. Not promoted; it regressed badly on real validation.
- Mixed real+self+UE+Sketchfab clean fine-tune v2: `{_metric_line(mixed_eval)}`. Not promoted; it uses the expanded clean geometry dataset but is still below the champion on OKS, FN, and bbox mAP50.
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
TARGET_TOTAL=300 RATE_LIMIT_SLEEP=900 \\
./scripts/fetch_sketchfab_until_target.sh
```

```bash
./.venv/bin/python src/fetch_objaverse_cars.py \\
  --output-dir data/sketchfab_cars --target-total 300
```

```bash
RUN_FETCH=1 RUN_OBJAVERSE=1 RUN_UE=1 WAIT_FOR_MCP=1 \\
MCP_WAIT_TIMEOUT=1800 MCP_WAIT_INTERVAL=10 \\
./scripts/finish_project_today.sh
```

```bash
./.venv/bin/python src/project_readiness.py
```

```bash
./.venv/bin/python src/production_audit_suite.py --with-pytest
```
"""
    OUT.write_text(text, encoding="utf-8")
    print(f"[handoff] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
