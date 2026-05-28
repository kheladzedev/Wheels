# Dataset Acceptance Template

Per-batch acceptance metadata. One JSON per incoming batch, at
`data/incoming/<batch>/metadata/acceptance_status.json`. Parsed by
`scripts/prod_readiness_check.py`. Missing file or missing required
field defaults to `REJECT_NEEDS_FIX` — training stays blocked.

## Required fields

```json
{
  "schema_version": 1,
  "source": "android_plugin",
  "batch_id": "android_plugin_real",
  "export_date": "YYYY-MM-DD",
  "image_count": 0,
  "wheel_count": 0,
  "bbox_source": "PLUGIN_BUILD_v2.4.1",
  "keypoint_mapping": "floorray_v1",
  "requires_plugin_bbox": false,

  "validation_result": "PASS",
  "preview_result": "PASS",
  "bbox_audit_result": "PASS",

  "human_reviewer": "handle_or_email",
  "human_preview_accepted": true,
  "review_date": "YYYY-MM-DD",
  "review_notes": "free text",

  "status": "ACCEPT_FOR_TRAINING"
}
```

## Field semantics

- `schema_version`: integer; must be `1`. Unknown values are rejected.
- `source`: producer of the batch. Examples: `android_plugin`,
  `manual_sample`, `unreal_synthetic`, `wikimedia_auto_draft`.
- `batch_id`: stable identifier, typically the directory name under
  `data/incoming/`.
- `export_date`: ISO date the export was produced.
- `image_count`, `wheel_count`: integer counts after format validation.
  Sanity: both > 0 for a usable batch.
- `bbox_source`: how `WheelBBox` / `BBox` was produced. Must not be in
  `{PLACEHOLDER, NEEDS_FIX, UNKNOWN}` for a training-grade batch. Use
  the exporter build / plugin version that emitted the bbox.
- `keypoint_mapping`: `floorray_v1` (post-2026-05-14 contract) or
  `rim_v0` (legacy). Only `floorray_v1` is allowed for training-grade
  status.
- `requires_plugin_bbox`: bool. `true` means this batch lacks
  trustworthy bboxes and is waiting on the plugin / exporter fix. Any
  batch with `requires_plugin_bbox: true` is forced to
  `ACCEPT_ONLY_AS_DEBUG` or `REJECT_NEEDS_FIX` regardless of other
  fields.
- `validation_result`: result of the format / schema validator
  (`src/check_keypoint_incoming.py` or sibling).
  Allowed: `PASS`, `WARN`, `FAIL`, `NOT_RUN`. Outside set → `FAIL`.
- `preview_result`: result of the human preview pass. Allowed values
  match `validation_result`.
- `bbox_audit_result`: result of the bbox sanity audit. Same allowed
  values.
- `human_reviewer`: stable identifier of the person who signed off.
  `null` or placeholder values invalidate `human_preview_accepted`.
- `human_preview_accepted`: bool. Must be `true` AND
  `human_reviewer` non-placeholder AND `preview_result == PASS` for
  the human-preview gate to clear.
- `review_date`: ISO date the human review concluded.
- `review_notes`: free text.
- `status`: one of:
  - `ACCEPT_FOR_TRAINING` — batch is training-grade. **All** of the
    following must hold: `requires_plugin_bbox: false`,
    `bbox_source` not in the forbidden set, `keypoint_mapping:
    floorray_v1`, `validation_result == PASS`, `preview_result ==
    PASS`, `bbox_audit_result == PASS`, `human_preview_accepted:
    true`.
  - `ACCEPT_ONLY_AS_DEBUG` — batch may be used for plumbing tests
    only. Never feeds a training run. Typical reason:
    `requires_plugin_bbox: true`, or `keypoint_mapping: rim_v0`.
  - `REJECT_NEEDS_FIX` — batch is broken or unaudited. Skip entirely.

`prod_readiness_check.py` does not trust `status` blindly — it
re-derives it from the field-level signals above. Inconsistencies
(e.g., `status: ACCEPT_FOR_TRAINING` with `requires_plugin_bbox:
true`) collapse the batch to `REJECT_NEEDS_FIX`.

## Example: blocked batch (current state — waiting on exporter)

`requires_plugin_bbox: true` collapses the derived status to
`ACCEPT_ONLY_AS_DEBUG` even if a different `status` is declared.

```json
{
  "schema_version": 1,
  "source": "android_plugin",
  "batch_id": "android_plugin_real",
  "export_date": "2026-05-27",
  "image_count": 0,
  "wheel_count": 0,
  "bbox_source": "PLACEHOLDER",
  "keypoint_mapping": "floorray_v1",
  "requires_plugin_bbox": true,

  "validation_result": "NOT_RUN",
  "preview_result": "NOT_RUN",
  "bbox_audit_result": "NOT_RUN",

  "human_reviewer": null,
  "human_preview_accepted": false,
  "review_date": null,
  "review_notes": "Waiting on exporter fix for real WheelBBox / BBox; plumbing-only.",

  "status": "ACCEPT_ONLY_AS_DEBUG"
}
```

## Example: accepted batch

```json
{
  "schema_version": 1,
  "source": "android_plugin",
  "batch_id": "android_plugin_real_v2",
  "export_date": "2026-06-10",
  "image_count": 412,
  "wheel_count": 1483,
  "bbox_source": "PLUGIN_BUILD_v2.4.1",
  "keypoint_mapping": "floorray_v1",
  "requires_plugin_bbox": false,

  "validation_result": "PASS",
  "preview_result": "PASS",
  "bbox_audit_result": "PASS",

  "human_reviewer": "reviewer_b",
  "human_preview_accepted": true,
  "review_date": "2026-06-11",
  "review_notes": "Spot-checked 50 frames; A/B clearly on floor.",

  "status": "ACCEPT_FOR_TRAINING"
}
```

## See also

- `docs/PRODUCTION_READINESS_PLAN.md` — the gates this file feeds.
- `docs/MODEL_CARD_TEMPLATE.md` — per-model card the training run
  produces.
- `docs/AR_ML_CONTRACT.md` — confirmed JSON contract; the
  `keypoint_mapping` field follows the contract's A / B / C
  semantics.
- `scripts/prod_readiness_check.py` — the gate checker.
