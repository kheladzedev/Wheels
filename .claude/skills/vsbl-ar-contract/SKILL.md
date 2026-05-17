---
name: vsbl-ar-contract
description: Use whenever a change might touch the ML→AR JSON contract — adding/renaming fields in inference output, postprocessor, response schemas, or anything that AR consumes. Activates on edits to src/infer_image.py, src/postprocess_wheels.py, AR-payload tests, docs/AR_ML_CONTRACT.md, docs/KEYPOINT_SPEC.md, or any code path that serialises a wheel response. Confirms what fields are allowed, what is forbidden, and where the ML↔AR responsibility line sits.
---

# VSBL — confirmed AR/ML contract

Source of truth for the ML→AR JSON shape. Anchored at **2026-05-13**
spec confirmation and the **2026-05-18** Unreal-side follow-up. If a
change conflicts with what's below, the change is wrong unless the AR
team signed off in
`docs/OPEN_QUESTIONS_AR_SPEC.md`.

## Required output (per inference call)

ML returns a per-frame result. No batching across frames, no streams,
no internal state retained between calls.

```json
{
  "frame_id": "frame_0001",
  "wheels": [
    {
      "bbox_xyxy": [x1, y1, x2, y2],
      "confidence": 0.94,
      "points": {
        "a":             [xa, ya],
        "b":             [xb, yb],
        "c_disc_bottom": [xc, yc]
      }
    }
  ]
}
```

Field-by-field:

- `frame_id` (string, required) — echoed from AR. Used to match the
  camera transform AR saved at capture time.
- `wheels` (array, required) — empty array is valid (means "nothing
  detected this frame"). Do not omit the field.
- `wheels[].bbox_xyxy` (`[x1, y1, x2, y2]`, required) — pixels. Top-left
  + bottom-right corners. Integer or float, both acceptable.
- `wheels[].confidence` (number in `[0, 1]`, required) — wheel-level
  detection confidence.
- `wheels[].points.a` (`[x, y]`, required) — **A: left floor-ray
  point.** Screen-space pixel position that AR raycasts onto the
  floor plane near the wheel's ground footprint. One of the two
  anchors of the wheel's vertical plane. **NOT a metal-rim edge
  point** — wording "rim left / rim right" predates the 2026-05-14
  semantics revision and must not be used to describe A/B going
  forward.
- `wheels[].points.b` (`[x, y]`, required) — **B: right floor-ray
  point.** Mirror of A. Same role: screen-space raycast source for
  the right anchor of the vertical wheel plane. **NOT a metal-rim
  edge point.**
- `wheels[].points.c_disc_bottom` (`[x, y]`, required) — **C: lowest
  visible point of the metal rim / disc.** Distinct from A/B: C still
  sits on the metal. AR raycasts C onto the recovered vertical plane
  to get the disc's installation height.

ML returns A/B/C **in screen space only**. AR alone performs the
raycasts and reconstructs anything 3D — see "Responsibility split"
below.

## Forbidden fields

Do **not** add any of the following without prior AR-team confirmation
recorded in `docs/OPEN_QUESTIONS_AR_SPEC.md`:

- `track_id` — tracking is AR's job. No persistent IDs across frames.
- `timestamp` — AR matches the camera transform via `frame_id`. Adding
  a timestamp duplicates state and invites drift bugs.
- Per-keypoint confidence (`keypoints_confidence`, `points_confidence`,
  `points.a.confidence`, etc.). Wheel-level `confidence` is the only
  confidence surface AR consumes.
- Per-keypoint `visibility` flag in the **response**. (Visibility lives
  inside training labels — see `yolo-pose-dataset` skill — but never
  leaks out to AR. Occluded wheels are dropped at inference and not
  emitted.)
- 3D coordinates of any kind — depths, world-space positions,
  reconstructed planes. Pixel space only.
- Per-frame metadata (camera intrinsics, exposure, IMU snapshot). AR
  owns all of that.
- Renamed alternates: `bbox_xywh` instead of `bbox_xyxy`, `keypoints`
  list instead of named `points`, snake/camel case alternatives. The
  confirmed names above are exact.

## Occlusion policy

Partially occluded wheels are **dropped at annotation time** and never
emitted at inference. There is no `visibility=1` path in the response.
If a wheel cannot have all three A/B/C points labelled reliably, it is
excluded from the dataset; correspondingly, inference must not emit a
wheel with missing or fabricated points.

## Responsibility split

| Concern                                                                | Owner |
|------------------------------------------------------------------------|-------|
| Wheel detection (bbox)                                                  | ML    |
| Screen-space A (left floor-ray point) regression                        | ML    |
| Screen-space B (right floor-ray point) regression                       | ML    |
| Screen-space C (lower visible metal rim / disc point) regression        | ML    |
| Wheel-level confidence                                                  | ML    |
| Dropping fully-occluded / unlabelled wheels                             | ML    |
| Per-frame response in pixel space                                       | ML    |
| Echoing `frame_id`                                                      | ML    |
| Raycasting `a` and `b` onto the floor plane                             | AR    |
| Building the vertical wheel plane from the A/B floor projections        | AR    |
| RANSAC across K frames → stable wheel plane                             | AR    |
| Raycasting `c_disc_bottom` onto the recovered vertical plane            | AR    |
| Averaging disc height across K valid frames                             | AR    |
| Tracking / association across frames                                    | AR    |
| Camera transform storage and matching                                   | AR    |
| K-frame accumulation and smoothing                                      | AR    |
| Anything 3D                                                             | AR    |

## When this skill applies

Invoke this skill BEFORE making changes that could touch the contract:

- Editing `src/infer_image.py`, `src/infer_batch.py`,
  `src/postprocess_wheels.py`, `src/visualize_predictions.py`.
- Updating `docs/AR_ML_CONTRACT.md`, `docs/KEYPOINT_SPEC.md`,
  `docs/PLUGIN_DATA_EXPECTATION.md`.
- Adding or modifying tests under `tests/test_ar_contract.py`.
- Renaming exported keys in any serialiser.
- Considering "we should also emit X" — first check against the
  forbidden list above.

If a proposed change conflicts with the confirmed contract:

1. Stop. Do not implement.
2. Flag it as an open question candidate.
3. Suggest the user add it to `docs/OPEN_QUESTIONS_AR_SPEC.md` for the
   AR team to sign off.

## See also

- `docs/AR_ML_CONTRACT.md` — full narrative of the split.
- `docs/OPEN_QUESTIONS_AR_SPEC.md` — items still pending AR sign-off.
- `docs/KEYPOINT_SPEC.md` — A/B/C definitions.
- Skill `yolo-pose-dataset` — internal training-side keypoint format
  (which is allowed to differ from the response shape).
