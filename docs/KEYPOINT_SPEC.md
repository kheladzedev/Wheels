# Keypoint Spec — A / B / C per wheel

Confirmed keypoint contract for the wheel-detection model, per the AR
team's 2026-05-13 response to `docs/OPEN_QUESTIONS_AR_SPEC.md` and the
**2026-05-14 semantic clarification** of the AR mock-system spec. ML
emits exactly three screen-space points per **fully-visible** wheel;
AR raycasts them and recovers the wheel's 3D plane plus the
disc-bottom position.

> **Semantic revision 2026-05-14.** Earlier iterations of this doc
> described A and B as the *left / right edge of the metal rim*. The
> AR mock-system section of the spec makes the actual role explicit:
> A/B are **screen positions whose raycasts must hit the floor** near
> the wheel footprint, so the two floor projections define the base
> direction of the vertical wheel plane. C is unchanged — it still
> sits on the metal rim / disc. Old "rim_left / rim_right" wording
> is forbidden when describing A/B going forward; the legacy literal
> label strings in code persist for backward compatibility only and
> describe **drifted** semantics.

## The three keypoints

Per wheel, ML emits exactly three 2D points in screen-space (pixels,
top-left origin). Names below are the final AR-facing keys in the JSON
output; the "descriptive name" column is the canonical wording for
docs and UI labels.

| Index | JSON key          | Descriptive name        | Geometric role |
|-------|-------------------|-------------------------|----------------|
| 0     | `a`               | `point_a_floor_ray`     | **Left floor-ray point.** Screen pixel from which AR shoots a ray onto the floor plane near the wheel's ground footprint. The resulting floor point is one anchor of the wheel's vertical plane. **Not** on the metal rim. |
| 1     | `b`               | `point_b_floor_ray`     | **Right floor-ray point.** Mirror of A. AR raycasts `b` onto the floor; the line through the two floor anchors carries the vertical wheel plane. **Not** on the metal rim. |
| 2     | `c_disc_bottom`   | `c_disc_bottom`         | **Lowest visible point of the metal rim / disc** (not the tire rubber, not the hub centre). AR raycasts this onto the already-recovered vertical plane to compute the disc's installation height. C unchanged from prior revisions. |

### Visual placement guide

Picture a wheel from the camera's point of view:

```
         car body
       ┌──────────┐
       │          │
       │  metal   │
       │  rim   ● │ ← C (c_disc_bottom): lowest visible point on the metal rim/disc
       │ ●●●●●●● │
       │ ● tire ● │
       │ ●●●●●●● │
       └──────────┘
    ●               ●    ← A (left)            ← B (right)
   floor near footprint  floor near footprint
   (raycast hits floor)  (raycast hits floor)
```

A and B should land on the **floor / base around the wheel** so that
their raycasts intersect the floor plane (used to recover the vertical
wheel plane). C should sit on the **metal disc**, at the lowest visible
point of the rim.

### Why three points specifically

AR's pipeline runs in two stages:

1. **Vertical plane recovery.** Floor projections of `a` and `b` (via
   raycast onto the AR-detected floor plane) give two coplanar points
   on the floor. The vertical plane through them — perpendicular to
   the floor — is the wheel's plane. Accumulated across K frames and
   filtered with RANSAC.
2. **Disc anchor.** With the plane known, one ray from `c_disc_bottom`
   onto that plane gives the 3D disc-bottom position. Averaged across
   the valid frames.

Two floor-ray points are the minimum to constrain a wheel-plane line
on the floor; one disc-bottom point is enough once the plane is known
(reduces 3 DoF to 1 DoF along the ray).

## Coordinate convention

- Units: **pixels**, image native resolution.
- Origin: **top-left**, `+x` right, `+y` down.
- No normalization to `[0, 1]` in the JSON output. Internal training
  labels (YOLO format on disk) are normalized per Ultralytics
  convention; the converter handles that transform.

## Occluded wheels: dropped

Partially occluded wheels are **excluded from the training set** and
are not emitted at inference time. Confirmed AR-team decision
(`docs/OPEN_QUESTIONS_AR_SPEC.md` §3, response 2026-05-13).

Practical consequences:

- Annotators skip a wheel entirely if any of `a` / `b` /
  `c_disc_bottom` is not visible. No "guess" for the occluded point.
- The model is trained only on fully-visible wheels. The detection
  head may still emit a bbox for a partially-occluded wheel at
  inference, but the keypoint output for such wheels will be
  unreliable — AR is expected to drop wheels whose wheel-level
  `confidence` is below threshold, and AR-side 3D consistency checks
  (RANSAC residuals) filter further.
- There is **no `visibility` flag** in the JSON output. Every emitted
  wheel is implicitly "all three points visible".

## No per-keypoint confidence

ML does not emit per-keypoint confidence in the JSON. Only the
wheel-level `confidence` is exposed. Confirmed AR-team decision §4.

If post-inference filtering needs per-point reliability later, it can
be re-added behind a feature flag without breaking the current
contract.

## What ML does NOT emit

These belong to the AR client and must not appear in the ML JSON:

- 3D world positions of `a` / `b` / `c_disc_bottom`.
- Recovered plane parameters (normal, point, equation, RANSAC
  residuals).
- Cross-frame `track_id`. AR clusters wheels across frames by 3D
  position; ML stays stateless per frame.
- Camera intrinsics / extrinsics.
- Per-keypoint confidence or visibility (see above).
- Timestamps.

## Mapping to internal training labels (and the 2026-05-14 drift)

The training pipeline still carries the legacy literal label strings
`rim_left`, `rim_right`, `disc_bottom` internally (in
`configs/dataset.yaml`, `src/convert_incoming_to_yolo.py`,
`src/postprocess_wheels.py`, label files on disk for the legacy
`manual_sample` flow). **As of 2026-05-14 the A/B semantics drift:**
the string `rim_left` literally said "left edge of the metal rim",
but A under the new contract is a floor-ray point, not a rim edge
point. The literal strings remain in code for backward compatibility
with the converters and `postprocess_wheels.KEYPOINT_NAMES`; the
*meaning* of indices 0 and 1 has changed.

| YOLO label index | Legacy literal string | Current geometric meaning                | AR-facing JSON key |
|------------------|-----------------------|------------------------------------------|--------------------|
| 0                | `rim_left`            | **Left floor-ray point** (NOT rim edge)  | `a`                |
| 1                | `rim_right`           | **Right floor-ray point** (NOT rim edge) | `b`                |
| 2                | `disc_bottom`         | Lowest visible metal-rim / disc point     | `c_disc_bottom`    |

Practical consequences:

- The plugin flow (`a` / `b` / `c_disc_bottom` JSON keys via
  `src/convert_keypoint_incoming_to_yolo_pose.py`) carries the new
  semantics directly; new annotations land in the right place.
- The legacy flow (`rim_left` / `rim_right` JSON keys via
  `src/convert_incoming_to_yolo.py`) preserves the old string names
  but its A/B *content* must be re-annotated as floor-ray points
  before that data can be trained against the new contract.
- Bundles annotated before 2026-05-14 (including any
  `data/incoming/manual_sample/`, `data/incoming/manual_real/`,
  `data/incoming/manual_real_draft/`, and the synthetic
  `create_sample_*` outputs) carry old "rim edge" geometry and are
  **invalid for training the production contract** without
  re-annotation.
- Renaming the internal literal strings to `a` / `b` /
  `c_disc_bottom` is a code-side follow-up; the label *order* is
  the load-bearing contract and is unchanged.

## See also

- `docs/AR_ML_CONTRACT.md` — full responsibility split and target JSON.
- `docs/ANDROID_FIRST_MODEL_PLAN.md` — model roadmap (baseline → Android-first).
- `docs/PLUGIN_DATA_EXPECTATION.md` — collection-plugin input format.
- `docs/OPEN_QUESTIONS_AR_SPEC.md` — resolved + remaining items.
- `docs/DATASET_SPEC.md` — internal label format on disk.
