# Production Readiness Plan

Definitions, gates, and current status for the VSBL wheel-fitting
project. The point of this document is to make the difference between
"we have something to show" and "this is shippable" load-bearing in
process, not in tone of voice.

The machine-checkable counterpart is `scripts/prod_readiness_check.py`,
which reads the artefacts listed below and emits
`outputs/prod_readiness/REPORT.{md,json}`. Run it before making any
"production / AR-ready" claim.

## 1. Four levels of readiness

Use these terms exactly. They are not synonyms.

### DEMO

- Curated visual showcase only (see `outputs/awe_demo/`).
- May use model predictions, auto-annotation drafts, synthetic-smoke
  frames, or mock AR overlays — each tagged by provenance.
- **No quality claim.** A demo passing does not mean the model is good.
- A demo is allowed even when every higher gate is failing.

### BASELINE

- A model has been trained and evaluated on a held-out validation
  split (e.g. `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/` —
  current real-only val mAP50=0.912, median A/B/C ~7.5 px).
- The model emits the confirmed AR JSON schema and passes
  `tests/test_ar_contract.py`.
- **Not AR-ready.** Baseline says "the pipeline trains and produces
  contract-compliant output". It does not say "AR can use this for
  3D reconstruction on device".

### PRODUCTION_CANDIDATE

All of:

1. Training data was **explicitly accepted**: a per-batch
   `data/incoming/<batch>/metadata/acceptance_status.json` with
   `status: ACCEPT_FOR_TRAINING`, real `WheelBBox` / `BBox` provenance,
   human preview review.
2. A human-labelled AR-device **holdout** exists, validated by
   `src/evaluate_ar_holdout.py` against production-provenance rules
   in `docs/AR_ML_CONTRACT.md`.
3. **Geometry / bbox audit** PASS (`outputs/full_pipeline_audit/REPORT.json`,
   or equivalent), with `geometry_audit_pass: true` and
   `bbox_audit_pass: true`.
4. **AR-replay metric** Stage 3 PASS on a real device session
   (`docs/AR_REPLAY_METRIC_PLAN.md`).
5. **Export / runtime parity** acceptable: `src/export_parity_audit.py`
   and `src/check_litert_runtime.py` green for the platform(s) under
   audit.
6. **Latency** measured on target devices (or at least bench numbers
   recorded in the model card).
7. A model card from `docs/MODEL_CARD_TEMPLATE.md` is filled in.

### PRODUCTION

Production_candidate **plus**:

1. Validated on target devices in the field, not only in a lab.
2. AR-side 3D replay is stable across more than one device family.
3. Monitoring / error logging plan documented.
4. Versioning / rollback plan documented.
5. Documented limitations and the failure modes a user is allowed
   to hit (e.g., "no support for indoor low-light").

These are intentionally process gates, not metric gates. The metric
gates live inside the previous level.

## 2. Current status (2026-05-28)

| Level | Status | Evidence |
|---|---|---|
| DEMO | satisfied | `outputs/awe_demo/demo_summary.json`, `docs/AWE_DEMO_PLAN.md`. |
| BASELINE | satisfied for `wheel_real_v1_self_plus_ue_synthetic_s` | Trained checkpoint, val report `outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_on_self_val.json`. Contract tests `tests/test_ar_contract.py` green. |
| PRODUCTION_CANDIDATE | **blocked** | Gates 1–6 all failing. See blockers below. |
| PRODUCTION | **blocked** | Upstream level not reached. |

## 3. Known blockers

In the order they need to be cleared.

1. **Plugin export with real `WheelBBox` / `BBox`.** Current real
   plugin batches do not carry trustworthy bounding boxes; the
   exporter side fix is the next thing to land
   (`docs/EXPORT_PARITY_AUDIT.md`, `docs/EXPORT_CERTIFICATION.md`).
2. **Per-batch acceptance metadata.** Today's `data/incoming/*` lacks
   `metadata/acceptance_status.json` for any real batch. Without it,
   `prod_readiness_check.py` keeps `training_allowed=false`. The
   metadata format is templated in `docs/DATASET_ACCEPTANCE_TEMPLATE.md`.
3. **Human-labelled AR-device holdout** with the provenance required
   by `docs/AR_ML_CONTRACT.md` ("Production Holdout Provenance").
4. **Geometry / bbox audit pass.** No
   `outputs/full_pipeline_audit/REPORT.json` exists today; gate
   defaults to fail.
5. **AR-replay metric Stage 3** on at least one real device session
   (`docs/AR_REPLAY_METRIC_PLAN.md`).
6. **Mobile / TFLite / LiteRT validation.** Deferred until the
   blockers above are closed.

## 4. Required gates before training

`prod_readiness_check.py` enforces all of these. Each maps to a
machine-readable field on disk; missing files default to "blocking".

| Gate | File / signal | Required value |
|---|---|---|
| Per-batch acceptance | `data/incoming/<batch>/metadata/acceptance_status.json` | `status: ACCEPT_FOR_TRAINING` |
| Real bbox source | same file | `bbox_source` not in `{PLACEHOLDER, NEEDS_FIX, UNKNOWN}` |
| Plugin bbox fixed | same file | `requires_plugin_bbox: false` |
| Human preview | same file | `human_preview_accepted: true` |
| Validation result | same file | `validation_result: PASS` |

If any of these is missing or fails, training is **not allowed**. This
is by design — synthetic plumbing and demo overlays must not be a
route into a real training run.

## 5. Required gates before AR-ready claim

| Gate | File / signal | Required value |
|---|---|---|
| Real floor-ray model trained | `runs/pose/<model>/SEMANTICS.md` | `semantics_version: floorray_v1`, `trained_on_real_data: true`, `stale: false` |
| Geometry audit | `outputs/full_pipeline_audit/REPORT.json` | `geometry_audit_pass: true` |
| Bbox audit | same | `bbox_audit_pass: true` |
| AR-replay metric (Stage 3) | same | `ar_replay_metric_pass: true` |
| Export / runtime parity | same | `export_parity_pass: true` |

Until every row is green for at least one model, the AR-ready claim is
forbidden, including in README, model card, slide decks, and demo
narration.

## 6. Required gates before mobile / MobileNetV2 / TFLite work

Mobile work is **not started until the baseline-on-real path passes**.
Specifically:

1. A model satisfying all gates in §5 exists for the desktop /
   PyTorch path.
2. `src/check_litert_runtime.py` smoke passes on the current
   `best_saved_model` / ONNX outputs.
3. A capacity plan is recorded in the model card: latency / memory
   targets per device class.

Premature mobile work multiplies the failure surface against a
non-AR-ready baseline and tends to mask the real blockers.

## 7. Status format conventions

To avoid bikeshedding, all gate files use the same vocabulary.

- `status` values for dataset acceptance:
  `ACCEPT_FOR_TRAINING`, `ACCEPT_ONLY_AS_DEBUG`, `REJECT_NEEDS_FIX`.
  Anything outside this set is treated as `REJECT_NEEDS_FIX` by the
  readiness check.
- `validation_result`, `preview_result`, `bbox_audit_result`:
  `PASS`, `WARN`, `FAIL`, `NOT_RUN`. Anything outside is `FAIL`.
- `semantics_version` for `runs/pose/<model>/SEMANTICS.md`:
  `floorray_v1` is the current contract; older legacy `rim_v0` models
  are explicitly **not** AR-ready (A / B carry rim semantics, not
  floor-ray).
- `stale` (bool): set true when the underlying training data has been
  superseded, the schema has changed, or the export pipeline has been
  reworked since the model was trained.

## 8. Exact next steps

In order, each gating the next.

1. Ship the exporter fix so real `WheelBBox` / `BBox` lands in plugin
   batches. Then write
   `data/incoming/android_plugin_real/metadata/acceptance_status.json`
   from `docs/DATASET_ACCEPTANCE_TEMPLATE.md`, with `bbox_source`
   pointing at the new exporter, `requires_plugin_bbox: false`, and
   `validation_result: PASS`.
2. Run human preview on the corrected batch. Mark
   `human_preview_accepted: true` only after a reviewer signs off.
3. Re-run `prod_readiness_check.py`. Expect `training_allowed: true`
   once gates 1–2 are green.
4. Train on the accepted batch. Save the resulting run with a
   `SEMANTICS.md` filled in from the model card template
   (`semantics_version: floorray_v1`, `trained_on_real_data: true`,
   `stale: false`, `trained_at: <date>`).
5. Run geometry + bbox audit, write
   `outputs/full_pipeline_audit/REPORT.json` with
   `geometry_audit_pass`, `bbox_audit_pass`,
   `export_parity_pass`.
6. Capture a real-device replay session and validate with
   `src/validate_ar_replay.py`. Then implement and run Stage 3 of
   `docs/AR_REPLAY_METRIC_PLAN.md`, write
   `ar_replay_metric_pass: true` into the audit report.
7. Re-run `prod_readiness_check.py`. Only at this point is the
   AR-ready claim allowed.
8. Mobile / TFLite work begins only after step 7.

Each step is a separate `/goal`.

## 9. Non-goals (load-bearing)

- **Not** a substitute for the AR ↔ ML contract; the contract still
  lives in `docs/AR_ML_CONTRACT.md` and is contract-frozen.
- **Not** a release-engineering doc; rollout / monitoring detail
  belongs in the production-level checklist once we are close to it.
- **Not** an unblocker for partial claims — there is no "AR-ready
  on synthetic only" path here.

## 10. See also

- `docs/AR_ML_CONTRACT.md` — frozen ML / AR JSON contract.
- `docs/AR_HANDOFF.md` — current integration package.
- `docs/AR_REPLAY_METRIC_PLAN.md` — 3D-aware eval plan that feeds
  the geometry / replay gates.
- `docs/AWE_DEMO_PLAN.md` — DEMO-level scope; cannot satisfy higher
  gates.
- `docs/EXPORT_CERTIFICATION.md`, `docs/EXPORT_PARITY_AUDIT.md` —
  the upstream blocker.
- `docs/PRODUCTION_READINESS_AUDIT.md` — current production audit
  output (read-only; this plan tells you why it is still red).
- `docs/MODEL_CARD_TEMPLATE.md` — the per-model card the
  production_candidate gate requires.
- `docs/DATASET_ACCEPTANCE_TEMPLATE.md` — the per-batch acceptance
  metadata the training gate requires.
- `scripts/prod_readiness_check.py` — the gate-checker;
  fails closed by default.
