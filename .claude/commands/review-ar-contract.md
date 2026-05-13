---
description: Audit pending changes against the confirmed ML→AR JSON contract — forbidden fields, schema drift, occlusion policy.
argument-hint: [optional path or diff scope, e.g. "src/infer_image.py" or "HEAD~3"]
---

# /review-ar-contract — audit changes against the confirmed AR contract

Invoke the `vsbl-ar-contract` skill, then audit pending changes for
contract violations. Use this **before** committing anything that
touches inference, postprocess, the AR JSON, or related tests.

Argument: `$ARGUMENTS` — optional scope. If empty, audit the full
current diff (`git diff` + untracked files). If a path or commit range
is given, scope the audit to that.

## Steps

### 1. Load the contract

Invoke the skill explicitly so the confirmed schema, forbidden fields,
and responsibility split are loaded into context:

> Use the `vsbl-ar-contract` skill.

### 2. Determine scope

If `$ARGUMENTS` is non-empty, scope the audit to it. Otherwise:

```bash
git status --short
git diff --stat
```

Identify files in any of these contract-bearing groups:

- `src/infer_image.py`, `src/infer_batch.py`
- `src/postprocess_wheels.py`, `src/visualize_predictions.py`
- `tests/test_ar_contract.py`
- `docs/AR_ML_CONTRACT.md`, `docs/KEYPOINT_SPEC.md`,
  `docs/OPEN_QUESTIONS_AR_SPEC.md`
- `README.md` schema sections
- Any new file that emits or consumes per-frame wheel JSON

### 3. Check against the contract

For every changed contract-bearing file, verify:

- **Top-level keys** in any emitted JSON are exactly `frame_id` +
  `wheels[]`. Nothing else, nothing renamed.
- **Per-wheel keys** are exactly `bbox_xyxy`, `confidence`, `points`
  (with `points = {a, b, c_disc_bottom}` only).
- **No forbidden field appears**: `track_id`, `timestamp`,
  `keypoints_confidence`, `points.*.confidence`, `visibility`,
  any 3D coordinate (`depth`, `world_xyz`, `plane_*`), camera
  intrinsics, IMU snapshots.
- **No legacy/transitional schema** is being added back: no
  `bbox_xywh`, no flat `keypoints` array, no `rim_left` / `rim_right`
  / `disc_bottom` names in the **response** (those names live only in
  training labels).
- **Occlusion policy**: no path emits a wheel with `visibility=1` or
  with a missing point — wheels with unlabelled points are dropped
  upstream, period.
- **No 3D logic added on the ML side**: no RANSAC, no plane fitting,
  no raycasting, no `numpy.linalg.lstsq` on point clouds.

### 4. Run the contract-specific tests

```bash
./.venv/bin/pytest -q tests/test_ar_contract.py
```

Must exit `0`.

### 5. Report

For each finding, classify and report:

- **VIOLATION** — change conflicts with the confirmed contract. State
  the file:line, the offending field/code, and the contract rule it
  breaks. **Block done.**
- **SUSPECT** — change is contract-adjacent but might be acceptable
  (e.g. a new internal helper that doesn't reach the response). State
  why you flagged it and what would clear the suspicion.
- **OK** — change in scope but compliant. One-line confirmation.

If any VIOLATION is found, the recommendation is one of:

1. Revert the change.
2. Open an item in `docs/OPEN_QUESTIONS_AR_SPEC.md` and wait for AR
   sign-off before merging.

End the report with:

- Contract status: `unchanged` / `changed (signed off in OPEN_QUESTIONS)`
  / `VIOLATED — do not merge`.
- pytest test_ar_contract: pass / fail + count.

## Notes

- This is an audit, not an automated rewrite. Do not modify code
  unless asked — flag the issues and stop.
- For the dataset-side training label format (which is intentionally
  different from the response), consult the `yolo-pose-dataset` skill
  instead. The two formats coexist legally.
