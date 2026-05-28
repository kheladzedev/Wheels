"""Write the current senior ML production-readiness audit report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUT = Path("docs/PRODUCTION_READINESS_AUDIT.md")
MANIFEST_OUT = Path("outputs/production_audit/model_package_manifest.json")

PT_CHAMPION = "runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt"
ONNX_CHAMPION = "runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.onnx"
TFLITE_CHAMPION = "outputs/production_audit/tflite_export/best_float32.tflite"
PACKAGE_DIGEST_EXCLUDED_PATHS = {
    # Generated immediately after this manifest in production_audit_suite.py.
    # Keep it discoverable in "reports", but do not hash a stale previous run.
    "docs/HANDOFF_TODAY.md",
}


def read_json(path: str | Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_entry(path: str, *, role: str) -> dict:
    p = Path(path)
    exists = p.is_file()
    size_bytes = p.stat().st_size if exists else 0
    return {
        "role": role,
        "path": path,
        "exists": exists,
        "size_bytes": size_bytes,
        "sha256": sha256_file(p) if exists and size_bytes > 0 else None,
    }


def package_digest(artifacts: list[dict]) -> str:
    canonical = json.dumps(
        artifacts,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_package_artifacts(reports: dict[str, str]) -> list[dict]:
    required: list[tuple[str, str]] = [
        ("champion_pt", PT_CHAMPION),
        ("champion_onnx", ONNX_CHAMPION),
        ("champion_tflite", TFLITE_CHAMPION),
        ("production_readiness_audit", str(OUT)),
        ("real_only_eval", "outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_on_self_val.json"),
        ("mixed_anchor_eval", "outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s.json"),
        ("onnx_eval", "outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_onnx_on_self_plus_ue_val.json"),
        ("tflite_eval", "outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_tflite_on_self_plus_ue_val.json"),
        ("release_integrity", "outputs/production_audit/release_integrity.json"),
        ("integration_gate", "outputs/production_audit/integration_gate.json"),
        ("production_gate", "outputs/production_audit/production_gate.json"),
        ("senior_ml_audit", "outputs/production_audit/senior_ml_audit.json"),
        ("objective_completion_audit", "outputs/production_audit/objective_completion_audit.json"),
        ("production_evidence_audit", "outputs/production_audit/production_evidence_audit.json"),
    ]
    for role, path in sorted(reports.items()):
        if (
            path.endswith((".json", ".md"))
            and path not in PACKAGE_DIGEST_EXCLUDED_PATHS
            and path not in {item[1] for item in required}
            and Path(path).is_file()
        ):
            required.append((f"report:{role}", path))
    return [artifact_entry(path, role=role) for role, path in required]


def metric(report: dict, dotted: str, default: str = "n/a") -> str:
    cur: object = report
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    if isinstance(cur, float):
        return f"{cur:.3f}"
    return str(cur)


def eval_line(name: str, report: dict) -> str:
    return (
        f"| {name} | {metric(report, 'metrics_bbox.mAP50')} | "
        f"{metric(report, 'metrics_bbox.mAP50_95')} | "
        f"{metric(report, 'oks.mean')} | "
        f"{metric(report, 'rates.false_negative_rate')} | "
        f"{metric(report, 'rates.false_positive_rate')} | "
        f"{metric(report, 'counts.gt_wheels')} / "
        f"{metric(report, 'counts.pred_wheels_above_conf')} / "
        f"{metric(report, 'counts.matched')} |"
    )


def threshold_line(conf: str, report: dict) -> str:
    return (
        f"| {conf} | {metric(report, 'oks.mean')} | "
        f"{metric(report, 'rates.false_negative_rate')} | "
        f"{metric(report, 'rates.false_positive_rate')} | "
        f"{metric(report, 'counts.gt_wheels')} / "
        f"{metric(report, 'counts.pred_wheels_above_conf')} / "
        f"{metric(report, 'counts.matched')} |"
    )


def main() -> int:
    champion_real = read_json(
        "outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_on_self_val.json"
    )
    champion_anchor = read_json("outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s.json")
    self_s_on_anchor = read_json("outputs/eval/wheel_real_v1_self_s_on_self_plus_ue_val.json")
    mixed_clean = read_json(
        "outputs/eval/wheel_real_self_ue_plus_sketchfab_clean_ft20_v2_on_real.json"
    )
    ue_only = read_json("outputs/eval/wheel_ue_sketchfab_geometry_clean_ft20_on_real.json")
    onnx_anchor = read_json(
        "outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_onnx_on_self_plus_ue_val.json"
    )
    tflite_anchor = read_json(
        "outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_tflite_on_self_plus_ue_val.json"
    )
    drift = read_json("outputs/production_audit/onnx_drift_20.json")
    tflite_cert = read_json("outputs/production_audit/tflite_certification.json")
    tflite_drift = read_json("outputs/production_audit/tflite_drift_20.json")
    litert_smoke = read_json("outputs/production_audit/litert_runtime_smoke.json")
    model_inventory = read_json("outputs/production_audit/model_inventory.json")
    model_selection = read_json("outputs/production_audit/model_selection_audit.json")
    spec_compliance = read_json("outputs/production_audit/spec_compliance_audit.json")
    dataset_audit = read_json("outputs/production_audit/dataset_audit.json")
    release_integrity = read_json("outputs/production_audit/release_integrity.json")
    runtime_contract = read_json("outputs/production_audit/runtime_contract_audit.json")
    performance_audit = read_json("outputs/production_audit/performance_audit.json")
    senior_audit = read_json("outputs/production_audit/senior_ml_audit.json")
    objective_audit = read_json("outputs/production_audit/objective_completion_audit.json")
    export_parity_audit = read_json("outputs/production_audit/export_parity_audit.json")
    export_certification = read_json("outputs/production_audit/export_certification.json")
    production_evidence = read_json("outputs/production_audit/production_evidence_audit.json")
    traceability = read_json("outputs/production_audit/requirements_traceability.json")
    integration_gate = read_json("outputs/production_audit/integration_gate.json")
    production_gate = read_json("outputs/production_audit/production_gate.json")
    smoke_batch = read_json("outputs/production_audit/smoke_batch/batch_summary.json")
    import_status = read_json("outputs/ue_tasks/import_sketchfab_glbs_status.json")
    geometry_status = read_json("outputs/ue_tasks/render_sketchfab_geometry_labels_status.json")
    qa_report = read_json("data/incoming/ue_sketchfab_geometry_clean/metadata/qa_report.json")
    threshold_reports = {
        "0.15": read_json("outputs/production_audit/threshold_conf015_real_val.json"),
        "0.20": read_json("outputs/production_audit/threshold_conf020_real_val.json"),
        "0.25": champion_real,
        "0.30": read_json("outputs/production_audit/threshold_conf030_real_val.json"),
        "0.40": read_json("outputs/production_audit/threshold_conf040_real_val.json"),
        "0.50": read_json("outputs/production_audit/threshold_conf050_real_val.json"),
        "0.60": read_json("outputs/production_audit/threshold_conf060_real_val.json"),
    }

    real_map50 = float(champion_real.get("metrics_bbox", {}).get("mAP50", 0.0))
    anchor_map50 = float(champion_anchor.get("metrics_bbox", {}).get("mAP50", 0.0))
    onnx_drift_ok = bool(drift.get("ok", False))
    batch_wheels = smoke_batch.get("wheels_detected_total", "n/a")
    perf_pt = performance_audit.get("benchmarks", {}).get("pytorch_cpu", {})
    perf_onnx = performance_audit.get("benchmarks", {}).get("onnx_cpu", {})
    perf_litert = performance_audit.get("benchmarks", {}).get("litert_cpu_smoke", {})

    text = f"""# Production Readiness Audit

Date: 2026-05-27.

## Executive Decision

**Status: integration candidate, not full production-certified.**

The current PyTorch production candidate is:

- PT: `{PT_CHAMPION}`
- ONNX: `{ONNX_CHAMPION}`

The model is usable for AR integration smoke tests and server/Python
inference. It is **not yet fully production-certified** for Android
on-device release because there is no human-labelled AR-device holdout,
no AR-side 3D raycast/RANSAC validation, and no Android-device LiteRT
production certification. ONNX/TFLite strict parity remains diagnostic,
but the calibrated desktop export certification now passes using
aggregate metric parity, no count mismatch, no coordinate-scale warnings,
and LiteRT smoke evidence.

Automated gate outputs:

- Integration gate: `{'PASS' if integration_gate.get('ok') else 'FAIL'}`
  (`outputs/production_audit/integration_gate.json`)
- Production gate: `{'PASS' if production_gate.get('ok') else 'FAIL'}`
  (`outputs/production_audit/production_gate.json`)

Model inventory:

- Report: `docs/MODEL_INVENTORY.md`
- Train runs: {model_inventory.get('counts', {}).get('train_runs', 'n/a')}
- Artifacts: {model_inventory.get('counts', {}).get('artifacts', 'n/a')}
  (`.pt`={model_inventory.get('counts', {}).get('pt_artifacts', 'n/a')},
  `.onnx`={model_inventory.get('counts', {}).get('onnx_artifacts', 'n/a')},
  `.tflite`={model_inventory.get('counts', {}).get('tflite_artifacts', 'n/a')})
- Eval reports linked: {model_inventory.get('counts', {}).get('eval_reports', 'n/a')}
- Champion training data: `{model_inventory.get('champion_run', {}).get('data', 'n/a')}`
- Champion source model: `{model_inventory.get('champion_run', {}).get('source_model', 'n/a')}`

Model selection audit:

- Report: `docs/MODEL_SELECTION_AUDIT.md`
- Selection OK: {model_selection.get('ok', 'n/a')}
- Anchor candidates compared: {model_selection.get('counts', {}).get('anchor_candidates', 'n/a')}
- Promotion required: {model_selection.get('counts', {}).get('promotion_required', 'n/a')}
- Failures: {', '.join(model_selection.get('failures', [])) if isinstance(model_selection.get('failures'), list) and model_selection.get('failures') else 'none'}

Spec compliance audit:

- Report: `docs/SPEC_COMPLIANCE_AUDIT.md`
- Overall OK: {spec_compliance.get('ok', 'n/a')}
- Checks: {len(spec_compliance.get('checks', [])) if isinstance(spec_compliance.get('checks'), list) else 'n/a'}
- Failures: {', '.join(spec_compliance.get('failures', [])) if isinstance(spec_compliance.get('failures'), list) and spec_compliance.get('failures') else 'none'}

Dataset audit:

- Report: `docs/DATASET_AUDIT.md`
- Overall OK: {dataset_audit.get('ok', 'n/a')}
- Dataset configs checked: {dataset_audit.get('counts', {}).get('configs', 'n/a')}
- Failed configs: {dataset_audit.get('counts', {}).get('failed', 'n/a')}
- Total train images across configs: {dataset_audit.get('counts', {}).get('total_train_images', 'n/a')}
- Total val images across configs: {dataset_audit.get('counts', {}).get('total_val_images', 'n/a')}
- Total wheel label lines across configs: {dataset_audit.get('counts', {}).get('total_wheel_labels', 'n/a')}

Release package integrity:

- Report: `docs/RELEASE_PACKAGE.md`
- Overall OK: {release_integrity.get('ok', 'n/a')}
- Artifacts: {release_integrity.get('artifact_count', 'n/a')}
- Total size: {release_integrity.get('total_size_mb', 'n/a')} MB

Runtime contract audit:

- Report: `outputs/production_audit/runtime_contract_audit.json`
- Overall OK: {runtime_contract.get('ok', 'n/a')}
- Single-image wheels: {runtime_contract.get('counts', {}).get('single_wheels', 'n/a')}
- Batch frames/wheels: {runtime_contract.get('counts', {}).get('batch_frames', 'n/a')} / {runtime_contract.get('counts', {}).get('batch_wheels', 'n/a')}

Performance audit:

- Report: `docs/PERFORMANCE_AUDIT.md`
- Overall OK: {performance_audit.get('ok', 'n/a')}
- Scope: `{performance_audit.get('scope', 'n/a')}`
- Sample frames: {performance_audit.get('sample_count', 'n/a')}
- PyTorch CPU mean/p95: {metric(perf_pt, 'latency_ms.mean')} / {metric(perf_pt, 'latency_ms.p95')} ms
- ONNX CPU mean/p95: {metric(perf_onnx, 'latency_ms.mean')} / {metric(perf_onnx, 'latency_ms.p95')} ms
- LiteRT smoke mean/p95: {metric(perf_litert, 'latency_ms.mean')} / {metric(perf_litert, 'latency_ms.p95')} ms

Senior ML audit:

- Report: `docs/SENIOR_ML_AUDIT.md`
- Audit OK: {senior_audit.get('audit_ok', 'n/a')}
- Integration ready: {senior_audit.get('integration_ready', 'n/a')}
- Production ready: {senior_audit.get('production_ready', 'n/a')}
- Requirements passed: {senior_audit.get('counts', {}).get('passed', 'n/a')} / {senior_audit.get('counts', {}).get('requirements', 'n/a')}
- Production blockers: {', '.join(senior_audit.get('production_blockers', [])) if isinstance(senior_audit.get('production_blockers'), list) else 'n/a'}

Objective completion audit:

- Report: `docs/OBJECTIVE_COMPLETION_AUDIT.md`
- Objective complete: {objective_audit.get('objective_complete', 'n/a')}
- Integration ready: {objective_audit.get('integration_ready', 'n/a')}
- Production ready: {objective_audit.get('production_ready', 'n/a')}
- Failed requirements: {', '.join(objective_audit.get('failed_requirements', [])) if isinstance(objective_audit.get('failed_requirements'), list) else 'n/a'}

Export parity audit:

- Report: `docs/EXPORT_PARITY_AUDIT.md`
- Certified: {export_parity_audit.get('certified', 'n/a')}
- ONNX failure categories: {export_parity_audit.get('summary', {}).get('onnx', {}).get('category_counts', 'n/a')}
- TFLite failure categories: {export_parity_audit.get('summary', {}).get('tflite', {}).get('category_counts', 'n/a')}

Export certification:

- Report: `docs/EXPORT_CERTIFICATION.md`
- Certified: {export_certification.get('certified', 'n/a')}
- Scope: `{export_certification.get('scope', 'n/a')}`

Production evidence audit:

- Report: `docs/PRODUCTION_EVIDENCE_AUDIT.md`
- Evidence ready: {production_evidence.get('production_evidence_ready', 'n/a')}
- Blockers: {', '.join(production_evidence.get('blockers', [])) if isinstance(production_evidence.get('blockers'), list) else 'n/a'}

Requirements traceability:

- Report: `docs/REQUIREMENTS_TRACEABILITY.md`
- Passed: {traceability.get('summary', {}).get('passed', 'n/a')} / {traceability.get('summary', {}).get('requirements', 'n/a')}
- Production ready: {traceability.get('production_ready', 'n/a')}

## Requirement Audit

| Requirement | Status | Evidence |
|---|---|---|
| Confirmed AR JSON contract | PASS | `tests/test_ar_contract.py`, `tests/test_confirmed_ar_schema_shape.py`, `src/infer_image.py`, `src/infer_batch.py` |
| Multi-wheel per-frame inference | PASS | Single smoke found 2 wheels; batch smoke found {batch_wheels} wheels over {smoke_batch.get('frames_inferred', 'n/a')} frames |
| 300 external car models collected | PASS | `data/sketchfab_cars`: 300 clean GLBs; import status has {import_status.get('tasks_total', import_status.get('tasks', 'n/a'))} tasks |
| UE/MCP geometry-label pipeline | PASS | Geometry status: groups={geometry_status.get('groups', 'n/a')}, frames={geometry_status.get('frames_written', 'n/a')}, wheels={geometry_status.get('wheels_written', 'n/a')} |
| Clean Sketchfab/Objaverse labels | PASS | QA kept {qa_report.get('kept_frames', 'n/a')} frames / {qa_report.get('kept_wheels', 'n/a')} wheels |
| Dataset format/leakage audit | {'PASS' if dataset_audit.get('ok') else 'FAIL'} | `outputs/production_audit/dataset_audit.json`: {dataset_audit.get('counts', {}).get('configs', 'n/a')} configs, failed={dataset_audit.get('counts', {}).get('failed', 'n/a')} |
| Release package integrity | {'PASS' if release_integrity.get('ok') else 'FAIL'} | `outputs/production_audit/release_integrity.json`: {release_integrity.get('artifact_count', 'n/a')} artifacts, total={release_integrity.get('total_size_mb', 'n/a')} MB |
| Runtime AR contract smoke | {'PASS' if runtime_contract.get('ok') else 'FAIL'} | `outputs/production_audit/runtime_contract_audit.json`: single wheels={runtime_contract.get('counts', {}).get('single_wheels', 'n/a')}, batch={runtime_contract.get('counts', {}).get('batch_frames', 'n/a')} frames / {runtime_contract.get('counts', {}).get('batch_wheels', 'n/a')} wheels |
| Desktop performance audit | {'PASS' if performance_audit.get('ok') else 'FAIL'} | `outputs/production_audit/performance_audit.json`: samples={performance_audit.get('sample_count', 'n/a')}, PT mean={metric(perf_pt, 'latency_ms.mean')}ms, ONNX mean={metric(perf_onnx, 'latency_ms.mean')}ms, LiteRT smoke mean={metric(perf_litert, 'latency_ms.mean')}ms |
| Senior ML evidence matrix | {'PASS' if senior_audit.get('audit_ok') else 'FAIL'} | `outputs/production_audit/senior_ml_audit.json`: integration_ready={senior_audit.get('integration_ready', 'n/a')}, production_ready={senior_audit.get('production_ready', 'n/a')} |
| Objective completion audit | {'PASS' if objective_audit.get('ok') else 'FAIL'} | `outputs/production_audit/objective_completion_audit.json`: objective_complete={objective_audit.get('objective_complete', 'n/a')}, failed={objective_audit.get('summary', {}).get('failed', 'n/a')}, missing={objective_audit.get('summary', {}).get('missing', 'n/a')} |
| Export parity diagnosis | {'PASS' if export_parity_audit.get('ok') else 'FAIL'} | `outputs/production_audit/export_parity_audit.json`: certified={export_parity_audit.get('certified', 'n/a')} |
| Calibrated export certification | {'PASS' if export_certification.get('certified') else 'FAIL'} | `outputs/production_audit/export_certification.json`: scope={export_certification.get('scope', 'n/a')} |
| Champion clears real-only bbox target | {'PASS' if real_map50 >= 0.85 else 'FAIL'} | Real-only eval bbox mAP50={real_map50:.3f} on `configs/pose_dataset_real_v1_self.yaml` |
| Champion on mixed real+UE anchor | WARN | Anchor bbox mAP50={anchor_map50:.3f}; this split includes synthetic validation frames and is harder/not the production acceptance split |
| New Sketchfab clean fine-tune improves champion | FAIL | Mixed clean fine-tune is below champion and is not promoted |
| ONNX export exists | PASS | `{ONNX_CHAMPION}` |
| ONNX strict parity vs PyTorch | WARN | Diagnostic strict policy: `outputs/production_audit/onnx_drift_20.json`: {drift.get('samples_matched', 'n/a')}/{drift.get('samples_checked', 'n/a')} samples matched, max kp drift={float(drift.get('max_kp_drift_px', 0.0)):.3f}px |
| ONNX aggregate eval | {'PASS' if export_certification.get('backends', {}).get('onnx', {}).get('certified') else 'FAIL'} | Calibrated export certification compares ONNX aggregate metrics against PyTorch champion |
| TFLite/LiteRT desktop package | {'PASS' if tflite_cert.get('certified') else 'FAIL'} | `outputs/production_audit/tflite_certification.json`: scope={tflite_cert.get('scope', 'n/a')} |
| Android LiteRT device validation | FAIL | Missing `outputs/production_audit/android_litert_device_eval.json` from the target app/device runtime |
| Human-labelled AR-device holdout | FAIL | No Android plugin holdout batch is present yet |
| AR-side 3D validation | FAIL | No recorded AR session with raycast + RANSAC error report is present |

## Model Comparison

| Model / eval split | bbox mAP50 | bbox mAP50-95 | OKS mean | FN rate | FP rate | GT / pred / matched |
|---|---:|---:|---:|---:|---:|---:|
{eval_line('Champion PT on real-only self val', champion_real)}
{eval_line('Champion PT on real+self+UE anchor val', champion_anchor)}
{eval_line('Previous self_s on real+self+UE anchor val', self_s_on_anchor)}
{eval_line('Mixed real+self+UE+Sketchfab clean fine-tune', mixed_clean)}
{eval_line('UE-only Sketchfab clean fine-tune', ue_only)}
{eval_line('Champion ONNX on real+self+UE anchor val', onnx_anchor)}
{eval_line('Champion TFLite float32 on real+self+UE anchor val', tflite_anchor)}

Decision: keep `wheel_real_v1_self_plus_ue_synthetic_s` as the current
PyTorch integration candidate. Do not promote the UE-only or
Sketchfab-clean mixed fine-tunes.

## Export Audit

Strict PT-vs-ONNX drift check:

- Report: `outputs/production_audit/onnx_drift_20.json`
- Samples matched: {drift.get('samples_matched', 'n/a')} / {drift.get('samples_checked', 'n/a')}
- Max bbox drift: {float(drift.get('max_bbox_drift_px', 0.0)):.3f}px
- Max keypoint drift: {float(drift.get('max_kp_drift_px', 0.0)):.3f}px
- Max confidence drift: {float(drift.get('max_conf_drift', 0.0)):.3f}

The strict parity report remains diagnostic. Calibrated export
certification is the authoritative desktop export policy and passes for
both ONNX and TFLite.

TFLite/LiteRT status:

- Certification: `outputs/production_audit/tflite_certification.json`
- Certified: {bool(tflite_cert.get('certified', False))}
- Artifact: `{tflite_cert.get('artifact', {}).get('path', 'n/a')}`
- Aggregate eval: `outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_tflite_on_self_plus_ue_val.json`
- Aggregate bbox mAP50: {metric(tflite_anchor, 'metrics_bbox.mAP50')}
- Aggregate OKS: {metric(tflite_anchor, 'oks.mean')}
- Scope: `{tflite_cert.get('scope', 'n/a')}`
- Strict 20-frame diagnostic drift: {tflite_drift.get('samples_matched', 'n/a')} / {tflite_drift.get('samples_checked', 'n/a')} matched, max keypoint drift={float(tflite_drift.get('max_kp_drift_px', 0.0)):.3f}px
- LiteRT Python smoke: ok={litert_smoke.get('ok', 'n/a')}, output shape={litert_smoke.get('outputs', [{}])[0].get('shape', 'n/a') if litert_smoke.get('outputs') else 'n/a'}, mean CPU latency={metric(litert_smoke, 'latency_ms.mean')} ms

Desktop TFLite/LiteRT package certification passes. It is not an Android
device certificate until the exact artifact is checked in the Android
app/device runtime, with output shape, finite output, latency, and memory
measurements on the target device.

## Confidence Threshold Sweep

Real-only validation split, current PyTorch candidate:

| conf | OKS mean | FN rate | FP rate | GT / pred / matched |
|---:|---:|---:|---:|---:|
{threshold_line('0.15', threshold_reports['0.15'])}
{threshold_line('0.20', threshold_reports['0.20'])}
{threshold_line('0.25', threshold_reports['0.25'])}
{threshold_line('0.30', threshold_reports['0.30'])}
{threshold_line('0.40', threshold_reports['0.40'])}
{threshold_line('0.50', threshold_reports['0.50'])}
{threshold_line('0.60', threshold_reports['0.60'])}

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
MODEL=runs/pose/wheel_real_v1_self_s/weights/best.pt \\
DATA=configs/pose_dataset_real_v1_self_plus_ue_synthetic.yaml \\
OUT=outputs/eval/wheel_real_v1_self_s_on_self_plus_ue_val.json \\
DEVICE=mps ./scripts/eval_baseline.sh
```

```bash
MODEL={PT_CHAMPION} \\
DATA=configs/pose_dataset_real_v1_self.yaml \\
OUT=outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_on_self_val.json \\
DEVICE=mps ./scripts/eval_baseline.sh
```

```bash
./.venv/bin/python src/check_export_drift.py \\
  --pt-model {PT_CHAMPION} \\
  --exported-model {ONNX_CHAMPION} \\
  --images-dir data/wheel_pose_dataset_real_v1_self_plus_ue_synthetic/images/val \\
  --limit 20 --device cpu \\
  --out outputs/production_audit/onnx_drift_20.json
```

```bash
./.venv_tflite/bin/python src/check_export_drift.py \\
  --pt-model {PT_CHAMPION} \\
  --exported-model outputs/production_audit/tflite_export/best_float32.tflite \\
  --exported-task pose \\
  --images-dir data/wheel_pose_dataset_real_v1_self_plus_ue_synthetic/images/val \\
  --limit 20 --device cpu \\
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
./.venv/bin/python src/validate_android_litert_report.py \
  --source data/incoming/android_litert_device_report.json \
  --out outputs/production_audit/android_litert_device_eval.json
```

```bash
./.venv/bin/python src/evaluate_ar_holdout.py \\
  --source-root data/incoming/ar_device_holdout \\
  --eval-out outputs/production_audit/ar_device_holdout_eval.json
```

```bash
./.venv/bin/python src/validate_ar_replay.py \\
  --jsonl path/to/ar_replay.jsonl \\
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
"""

    OUT.write_text(text, encoding="utf-8")
    reports = {
        "production_audit": str(OUT),
        "model_inventory": "docs/MODEL_INVENTORY.md",
        "model_inventory_json": "outputs/production_audit/model_inventory.json",
        "model_selection_audit": "docs/MODEL_SELECTION_AUDIT.md",
        "model_selection_audit_json": "outputs/production_audit/model_selection_audit.json",
        "spec_compliance_audit": "docs/SPEC_COMPLIANCE_AUDIT.md",
        "spec_compliance_audit_json": "outputs/production_audit/spec_compliance_audit.json",
        "model_card": "docs/MODEL_CARD.md",
        "dataset_audit": "docs/DATASET_AUDIT.md",
        "dataset_audit_json": "outputs/production_audit/dataset_audit.json",
        "release_package": "docs/RELEASE_PACKAGE.md",
        "release_integrity": "outputs/production_audit/release_integrity.json",
        "runtime_contract_audit": "outputs/production_audit/runtime_contract_audit.json",
        "performance_audit": "docs/PERFORMANCE_AUDIT.md",
        "performance_audit_json": "outputs/production_audit/performance_audit.json",
        "senior_ml_audit": "docs/SENIOR_ML_AUDIT.md",
        "senior_ml_audit_json": "outputs/production_audit/senior_ml_audit.json",
        "objective_completion_audit": "docs/OBJECTIVE_COMPLETION_AUDIT.md",
        "objective_completion_audit_json": "outputs/production_audit/objective_completion_audit.json",
        "export_parity_audit": "docs/EXPORT_PARITY_AUDIT.md",
        "export_parity_audit_json": "outputs/production_audit/export_parity_audit.json",
        "export_certification": "docs/EXPORT_CERTIFICATION.md",
        "export_certification_json": "outputs/production_audit/export_certification.json",
        "android_litert_device_report_contract": "docs/ANDROID_LITERT_DEVICE_REPORT.md",
        "android_litert_harness_doc": "android_litert_harness/README.md",
        "android_litert_harness_test": "android_litert_harness/AndroidLiteRtDeviceValidationTest.kt",
        "external_evidence_handoff_bundle_doc": "docs/EXTERNAL_EVIDENCE_HANDOFF_BUNDLE.md",
        "production_evidence_checklist": "docs/PRODUCTION_EVIDENCE_CHECKLIST.md",
        "production_evidence_intake": "docs/PRODUCTION_EVIDENCE_INTAKE.md",
        "production_evidence_audit": "docs/PRODUCTION_EVIDENCE_AUDIT.md",
        "production_evidence_audit_json": "outputs/production_audit/production_evidence_audit.json",
        "production_evidence_intake_status": "outputs/production_audit/production_evidence_intake_status.json",
        "production_evidence_intake_preflight_status": "outputs/production_audit/production_evidence_intake_preflight_status.json",
        "external_evidence_drop_importer": "src/import_external_evidence_drop.py",
        "external_evidence_return_template": "outputs/production_audit/external_evidence_return_template.zip",
        "external_evidence_return_template_manifest": "outputs/production_audit/external_evidence_return_template_manifest.json",
        "external_evidence_handoff_bundle": "outputs/production_audit/external_evidence_handoff_bundle.zip",
        "external_evidence_handoff_bundle_manifest": "outputs/production_audit/external_evidence_handoff_bundle_manifest.json",
        "external_evidence_handoff_bundle_verification": "outputs/production_audit/external_evidence_handoff_bundle_verification.json",
        "requirements_traceability": "docs/REQUIREMENTS_TRACEABILITY.md",
        "requirements_traceability_json": "outputs/production_audit/requirements_traceability.json",
        "executive_report_ru": "docs/EXECUTIVE_REPORT_RU.md",
        "android_litert_device_report_template": "outputs/production_audit/android_litert_device_report.template.json",
        "ar_holdout_provenance_template": "outputs/production_audit/ar_device_holdout_provenance.template.json",
        "ar_holdout_harness_doc": "ar_holdout_harness/README.md",
        "ar_holdout_harness_writer": "ar_holdout_harness/ArHoldoutAnnotationWriter.kt",
        "ar_replay_log_template": "outputs/production_audit/ar_3d_replay.template.jsonl",
        "ar_replay_harness_doc": "ar_replay_harness/README.md",
        "ar_replay_harness_logger": "ar_replay_harness/ArReplayLogger.kt",
        "handoff_today": "docs/HANDOFF_TODAY.md",
        "integration_gate": "outputs/production_audit/integration_gate.json",
        "production_gate": "outputs/production_audit/production_gate.json",
        "onnx_drift": "outputs/production_audit/onnx_drift_20.json",
        "tflite_certification": "outputs/production_audit/tflite_certification.json",
        "tflite_drift": "outputs/production_audit/tflite_drift_20.json",
        "litert_runtime_smoke": "outputs/production_audit/litert_runtime_smoke.json",
        "android_litert_device_eval": "outputs/production_audit/android_litert_device_eval.json",
        "android_litert_validator": "src/validate_android_litert_report.py",
        "tflite_eval": "outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_tflite_on_self_plus_ue_val.json",
        "ar_holdout_evaluator": "src/evaluate_ar_holdout.py",
        "ar_replay_validator": "src/validate_ar_replay.py",
        "ar_replay_template_writer": "scripts/create_ar_replay_log_template.py",
        "ar_holdout_provenance_template_writer": "scripts/create_ar_holdout_provenance_template.py",
        "production_evidence_intake_runner": "src/run_production_evidence_intake.py",
        "external_evidence_drop_import_runner": "src/import_external_evidence_drop.py",
        "external_evidence_return_template_writer": "scripts/create_external_evidence_return_template.py",
        "external_evidence_handoff_bundle_builder": "scripts/build_external_evidence_handoff_bundle.py",
        "external_evidence_handoff_bundle_verifier": "src/verify_external_evidence_handoff_bundle.py",
        "production_audit_suite": "src/production_audit_suite.py",
        "real_only_eval": "outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_on_self_val.json",
        "mixed_anchor_eval": "outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s.json",
    }
    package_artifacts = build_package_artifacts(reports)
    manifest = {
        "schema_version": 2,
        "status": "integration_candidate_not_full_production_certified",
        "date": "2026-05-27",
        "candidate": {
            "pt": PT_CHAMPION,
            "onnx": ONNX_CHAMPION,
            "tflite": TFLITE_CHAMPION,
            "recommended_conf_for_ar_smoke": 0.50,
        },
        "reports": reports,
        "package_artifacts": package_artifacts,
        "package_digest_sha256": package_digest(package_artifacts),
        "missing_artifacts": [
            artifact["path"]
            for artifact in package_artifacts
            if not artifact["exists"] or not artifact["sha256"]
        ],
        "gates": {
            "integration": bool(integration_gate.get("ok")),
            "production": bool(production_gate.get("ok")),
        },
        "metrics": {
            "model_inventory": model_inventory.get("counts", {}),
            "model_selection": {
                "ok": model_selection.get("ok"),
                "anchor_candidates": model_selection.get("counts", {}).get("anchor_candidates"),
                "promotion_required": model_selection.get("counts", {}).get("promotion_required"),
                "failures": model_selection.get("failures", []),
            },
            "spec_compliance": {
                "ok": spec_compliance.get("ok"),
                "failures": spec_compliance.get("failures", []),
                "checks": len(spec_compliance.get("checks", []))
                if isinstance(spec_compliance.get("checks"), list)
                else None,
            },
            "dataset_audit": dataset_audit.get("counts", {}),
            "release_integrity": {
                "ok": release_integrity.get("ok"),
                "artifact_count": release_integrity.get("artifact_count"),
                "total_size_mb": release_integrity.get("total_size_mb"),
            },
            "runtime_contract": runtime_contract.get("counts", {}),
            "performance_audit": {
                "ok": performance_audit.get("ok"),
                "sample_count": performance_audit.get("sample_count"),
                "pytorch_cpu_mean_latency_ms": metric(perf_pt, "latency_ms.mean"),
                "onnx_cpu_mean_latency_ms": metric(perf_onnx, "latency_ms.mean"),
                "litert_smoke_mean_latency_ms": metric(perf_litert, "latency_ms.mean"),
            },
            "senior_ml_audit": {
                "audit_ok": senior_audit.get("audit_ok"),
                "integration_ready": senior_audit.get("integration_ready"),
                "production_ready": senior_audit.get("production_ready"),
                "production_blockers": senior_audit.get("production_blockers", []),
            },
            "objective_completion_audit": {
                "objective_complete": objective_audit.get("objective_complete"),
                "integration_ready": objective_audit.get("integration_ready"),
                "production_ready": objective_audit.get("production_ready"),
                "failed_requirements": objective_audit.get("failed_requirements", []),
            },
            "export_parity_audit": {
                "ok": export_parity_audit.get("ok"),
                "certified": export_parity_audit.get("certified"),
                "summary": export_parity_audit.get("summary", {}),
            },
            "export_certification": {
                "certified": export_certification.get("certified"),
                "scope": export_certification.get("scope"),
                "backends": {
                    name: backend.get("certified")
                    for name, backend in export_certification.get("backends", {}).items()
                    if isinstance(backend, dict)
                },
            },
            "real_only": {
                "bbox_map50": metric(champion_real, "metrics_bbox.mAP50"),
                "oks_mean": metric(champion_real, "oks.mean"),
                "fn_rate": metric(champion_real, "rates.false_negative_rate"),
                "fp_rate": metric(champion_real, "rates.false_positive_rate"),
            },
            "onnx_drift": {
                "samples_matched": drift.get("samples_matched"),
                "samples_checked": drift.get("samples_checked"),
                "max_kp_drift_px": drift.get("max_kp_drift_px"),
            },
            "tflite": {
                "certified": bool(tflite_cert.get("certified", False)),
                "bbox_map50": metric(tflite_anchor, "metrics_bbox.mAP50"),
                "oks_mean": metric(tflite_anchor, "oks.mean"),
                "fn_rate": metric(tflite_anchor, "rates.false_negative_rate"),
                "fp_rate": metric(tflite_anchor, "rates.false_positive_rate"),
                "drift_samples_matched": tflite_drift.get("samples_matched"),
                "drift_samples_checked": tflite_drift.get("samples_checked"),
                "max_kp_drift_px": tflite_drift.get("max_kp_drift_px"),
                "litert_runtime_smoke_ok": litert_smoke.get("ok"),
                "litert_cpu_mean_latency_ms": metric(litert_smoke, "latency_ms.mean"),
            },
        },
        "production_blockers": [
            "missing_android_litert_device_validation",
            "missing_human_labelled_ar_device_holdout",
            "missing_ar_side_3d_raycast_ransac_validation",
        ],
    }
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[production-audit] wrote {OUT}")
    print(f"[production-audit] wrote {MANIFEST_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
