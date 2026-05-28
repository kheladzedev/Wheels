# Production Evidence Checklist

Current integration evidence is complete, but production readiness still
requires three external AR/Android artifacts.

## 1. Android LiteRT Device Validation

Input from Android:

```text
data/incoming/android_litert_device_report.json
```

Template and schema:

- `docs/ANDROID_LITERT_DEVICE_REPORT.md`
- `outputs/production_audit/android_litert_device_report.template.json`

Validation command:

```bash
./.venv/bin/python src/validate_android_litert_report.py
```

Required output:

```text
outputs/production_audit/android_litert_device_eval.json
```

The validator records `source_sha256`; the production evidence audit
rejects the report if `data/incoming/android_litert_device_report.json`
changes after validation.

The report must include `source_type: android_litert_device_validation`,
a real `test_session_id`, non-placeholder device model/manufacturer/
Android version/SoC, `artifact.format: tflite_float32`, finite output
`min`/`max`/`mean`, measured peak memory `> 0 MB`, and the exact shipped
TFLite SHA256.

Production gate requirement: `ok: true`.

## 2. Human-Labelled AR-Device Holdout

Input from AR/data collection:

```text
data/incoming/ar_device_holdout/
  images/<frame>.jpg
  annotations/<frame>.json
  metadata/provenance.json
```

Required provenance:

- Template: `outputs/production_audit/ar_device_holdout_provenance.template.json`
- Template writer: `scripts/create_ar_holdout_provenance_template.py`
- AR holdout annotation writer: `ar_holdout_harness/ArHoldoutAnnotationWriter.kt`

```json
{
  "source_type": "android_ar_device_human_labelled",
  "label_type": "human_reviewed",
  "capture_device": "Pixel 8 Pro",
  "review_status": "accepted",
  "capture_app_version": "1.2.3",
  "capture_date_utc": "2026-05-27",
  "annotator": "labeler_a",
  "reviewer": "reviewer_b"
}
```

`capture_device` must be a real device name, not `FILL_ME`, `TODO`,
`TBD`, or `unknown`. `capture_app_version`, real `capture_date_utc` in
`YYYY-MM-DD` format, `annotator`, and `reviewer` are also required; the
annotator and reviewer must be different people/accounts.

Validation/eval command:

```bash
./.venv/bin/python src/evaluate_ar_holdout.py
```

Required output:

```text
outputs/production_audit/ar_device_holdout_eval.json
outputs/production_audit/ar_device_holdout_pipeline.json
```

The pipeline records `source_manifest_sha256` over `images/`,
`annotations/`, and provenance metadata. The production evidence audit
rejects stale holdout reports if any image, annotation, or provenance
file changes after evaluation. It also ties
`outputs/production_audit/ar_device_holdout_pipeline.json` to the exact
`outputs/production_audit/ar_device_holdout_eval.json` by `eval_report`
path, `eval_returncode`, and `eval_report_sha256`; replacing the eval
JSON after the pipeline run is rejected.

Production gate thresholds:

- at least `50` evaluated AR-device frames
- at least `80` labelled wheels
- bbox mAP50 `>= 0.85`
- OKS `>= 0.80`
- false negative rate `<= 0.10`

## 3. AR 3D Replay Validation

Input from AR replay:

```text
path/to/ar_replay.jsonl
```

Schema:

- `docs/AR_MOCK_LOG_CONTRACT.md`
- Template: `outputs/production_audit/ar_3d_replay.template.jsonl`
- Template writer: `scripts/create_ar_replay_log_template.py`
- AR replay logger harness: `ar_replay_harness/ArReplayLogger.kt`

Production logs must replace all template `FILL_ME` values and carry
`source_type: android_ar_device_replay` (or another allowed AR-device
replay source type) plus a real `capture_device` on every observation.
The production audit also rejects reports with relaxed or missing replay
thresholds: fewer than 30 observations, no production-source requirement,
floor-hit rate below 90%, disabled RANSAC checks, inlier rate below 70%,
median residual above 0.02, p95 residual above 0.05, or missing final
disc-bottom 3D positions. The report must carry complete counters for
floor hits, RANSAC labels, residuals, and final positions; `ok: true`
alone is not sufficient.

Validation command:

```bash
./.venv/bin/python src/validate_ar_replay.py \
  --jsonl path/to/ar_replay.jsonl \
  --out outputs/production_audit/ar_3d_replay_eval.json
```

Required output:

```text
outputs/production_audit/ar_3d_replay_eval.json
```

The validator records `source_sha256`; the production evidence audit
rejects the report if the replay JSONL changes after validation.

Production gate requirement: `ok: true`.

## Final Gate

After all three artifacts are present:

If they arrive as a zip or folder:

```bash
./.venv/bin/python scripts/create_external_evidence_return_template.py
./.venv/bin/python src/import_external_evidence_drop.py path/to/evidence_drop.zip --dry-run
./.venv/bin/python src/import_external_evidence_drop.py path/to/evidence_drop.zip --overwrite
```

```bash
./.venv/bin/python src/run_production_evidence_intake.py --dry-run
```

```bash
./.venv/bin/python src/run_production_evidence_intake.py
```

Then run the full suite, or include `--finalize` in the intake command to
run this automatically after a green intake:

```bash
./.venv/bin/python src/production_audit_suite.py --with-pytest
./.venv/bin/python src/run_production_evidence_intake.py --evidence-drop path/to/evidence_drop.zip --evidence-drop-overwrite --finalize
```

Use `--finalize` only with the canonical production paths. If evidence
was validated from custom source/eval paths, import or copy it into
`data/incoming/...` first so the final suite certifies the same files.

Expected final state:

- `outputs/production_audit/integration_gate.json`: `ok: true`
- `outputs/production_audit/production_gate.json`: `ok: true`
- `outputs/production_audit/senior_ml_audit.json`: `production_ready: true`
