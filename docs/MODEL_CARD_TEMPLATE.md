# Model Card Template

Fill in one card per trained model that will leave this repo (handoff,
demo, or production candidate). The filled card lives next to the
weights at `runs/pose/<model>/MODEL_CARD.md`. A companion
`runs/pose/<model>/SEMANTICS.md` carries the machine-readable
gate signals (see §11 below).

Mark fields `UNKNOWN` rather than guessing. `prod_readiness_check.py`
treats `UNKNOWN` and missing fields as failing gates.

---

## 1. Model identity

- **name:** e.g. `wheel_real_v1_self_plus_ue_synthetic_s`
- **architecture:** e.g. `yolo11s-pose`, 3 keypoints per wheel
- **weights:** `runs/pose/<name>/weights/best.pt` (+ ONNX / TFLite if
  exported)
- **trained_at:** ISO date, e.g. `2026-05-27`
- **trained_by:** owner / handle
- **semantics_version:** `floorray_v1` if A / B are floor-ray points
  (post-2026-05-14 contract). `rim_v0` for legacy rim-edge models —
  these are **not** AR-ready.

## 2. Training data

- **batches:** list of `data/incoming/<batch>/` paths actually used.
- **acceptance:** per-batch `acceptance_status.json` `status` value
  (`ACCEPT_FOR_TRAINING` is the only one that justifies the run).
- **image_count:** training-split image count.
- **wheel_count:** training-split wheel count.
- **bbox_source:** how `WheelBBox` / `BBox` was produced (exporter
  build, version, or fallback). Must not be `PLACEHOLDER`.
- **keypoint_mapping:** `floorray_v1` or `rim_v0`.
- **augmentations:** brief.
- **synthetic fraction:** explicit. UE / cartoon fractions must be
  separately reported.

## 3. Validation data

- **path:** dataset / split path used for the val report.
- **val_image_count / val_wheel_count.**
- **provenance:** human-labelled, auto-drafted, or synthetic. Must
  match the production rules in `docs/AR_ML_CONTRACT.md` if used as a
  production gate.
- **metrics:** Box mAP50, Box mAP50-95, Pose mAP50, mean OKS,
  median A / B / C px error, false-negative rate, false-positive
  rate. Cite the exact report file path.

## 4. Human-labelled AR-device holdout

- **path:** `data/incoming/ar_device_holdout/` if present.
- **provenance:** `source_type`, `label_type`, `capture_device`,
  `capture_app_version`, `capture_date_utc`, `annotator`, `reviewer`,
  `review_status`. All required for the production_candidate gate.
- **report:** `outputs/eval/<model>_on_ar_device_holdout.json`.
- **status:** `PASS / WARN / FAIL / NOT_RUN`.

## 5. Geometry audit

- **report:** `outputs/full_pipeline_audit/REPORT.json` path.
- **geometry_audit_pass:** bool.
- **bbox_audit_pass:** bool.
- **notes:** what was inspected, what failed.

## 6. AR replay

- **session_logs:** list of `data/incoming/ar_3d_replay/*.jsonl`
  validated by `src/validate_ar_replay.py`.
- **stage_3_implemented:** bool — refers to
  `docs/AR_REPLAY_METRIC_PLAN.md` Stage 3.
- **ar_replay_metric_pass:** bool.
- **metrics:** inlier ratio, plane residual, plane stability,
  C-projection stability, failure rate; final 3D error vs GT when
  available.

## 7. Export formats

> Note (2026-05-30): v1 ships Android-only (LiteRT/TFLite). iOS/CoreML is
> deferred; expect `CoreML: NOT_EXPORTED` (zero CoreML artifacts today).

- **PT:** `runs/pose/<name>/weights/best.pt` (always).
- **ONNX:** path + opset.
- **CoreML:** path or `NOT_EXPORTED`.
- **TFLite / LiteRT:** path or `NOT_EXPORTED`.
- **export_parity_pass:** bool (`src/export_parity_audit.py` output).

## 8. Latency

- **device class:** e.g. `Pixel 8 Pro`, `iPhone 15`, `Apple M2`.
- **runtime:** `PT (CPU)`, `PT (MPS)`, `ONNX`, `TFLite GPU delegate`,
  `LiteRT`.
- **median / p95 latency (ms).**
- **input resolution.**
- **batch:** always 1 (per ML / AR contract).
- **notes:** thermal / quantisation / quirks.

## 9. Limitations

Spell out everything the model is **not** good at, in plain
language. Example items:

- Partially occluded wheels are dropped (contract).
- Indoor low-light not validated.
- Side-on wheels in motion blur not validated.
- Trailers / motorcycles not validated.

## 10. Allowed use

- AR client per the contract in `docs/AR_ML_CONTRACT.md`.
- Demo packs under `outputs/awe_demo/` with provenance labels.
- Auto-annotation drafts via `src/auto_annotate_wheels.py`.

## 11. Forbidden claims

Write these in the card so future readers can quote them back.

- **Not** "production-ready" unless every gate in
  `docs/PRODUCTION_READINESS_PLAN.md` §5 is green for this exact
  model.
- **Not** "AR-ready" unless `runs/pose/<name>/SEMANTICS.md`
  has `semantics_version: floorray_v1`, `trained_on_real_data: true`,
  `stale: false`, AND geometry + AR-replay gates pass.
- **Not** "TFLite production" unless `src/check_litert_runtime.py`
  passes on a target device.
- **Not** "keypoint accuracy meets 5 px" unless the val report shows
  median A / B / C ≤ 5 px (current champion is ~7.5 px).
- **Not** "tracks wheels across frames" — tracking is AR-side.

## 12. SEMANTICS.md companion (machine-readable gate signals)

Save next to the card at `runs/pose/<name>/SEMANTICS.md`. Parsed by
`scripts/prod_readiness_check.py` as plain `key: value` lines
(stdlib only — no YAML dep). Example:

```text
semantics_version: floorray_v1
trained_on_real_data: true
stale: false
trained_at: 2026-05-27
acceptance_batches: data/incoming/android_plugin_real
notes: post-exporter-fix retrain; floor-ray A/B; matches confirmed contract.
```

Allowed keys:

- `semantics_version`: `floorray_v1` (current) or `rim_v0` (legacy).
- `trained_on_real_data`: `true` / `false`.
- `stale`: `true` / `false`.
- `trained_at`: `YYYY-MM-DD`.
- `acceptance_batches`: comma-separated paths the run consumed.
- `notes`: free text on one line.

Absence of any required key, or `stale: true`, blocks the AR-ready
claim.
