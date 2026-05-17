# AR ↔ ML Contract

Defines who does what between the ML side (this repo) and the AR client
in the wheel-fitting pipeline. The boundary is deliberately narrow: ML
emits per-frame, per-wheel 2D screen-space points; AR owns everything
3D.

The decisions below are **confirmed with the AR team** (initial
responses received 2026-05-13 against the question set in
`docs/OPEN_QUESTIONS_AR_SPEC.md`, then re-confirmed by the
Unreal-side follow-up on 2026-05-18). Schema changes from here on
require explicit AR sign-off — they are pinned by
`tests/test_ar_contract.py`.

## Responsibility split

### ML side (this repo)

For each input RGB frame, ML produces a JSON payload describing every
fully-visible wheel in the frame:

- detects wheels (single class, multiple per frame);
- returns each wheel's `bbox` in pixel coordinates;
- returns wheel-level detection `confidence` in `[0, 1]`;
- returns three **screen-space pixel points** per wheel:
  - `a` — left floor-ray point (raycast source for the wheel-plane
    base; **not** a metal rim edge);
  - `b` — right floor-ray point (raycast source for the wheel-plane
    base; **not** a metal rim edge);
  - `c_disc_bottom` — lowest visible point of the metal rim / disc;
- echoes the `frame_id` provided by the AR client so AR can match the
  response back to the camera transform it saved at capture time.

ML returns **2D screen-space pixels only.** No 3D, no raycasts, no
plane geometry leaves this side. Full A/B/C geometric definitions
live in `docs/KEYPOINT_SPEC.md`.

ML is **per-frame and stateless**. ML does **not**:

- raycast `a`, `b`, or `c_disc_bottom`;
- run RANSAC;
- build any 3D plane (vertical, floor, or otherwise);
- track wheels across frames;
- emit timestamps, world coordinates, or camera intrinsics/extrinsics;
- emit a `visibility` flag or per-keypoint confidence.

Partially occluded wheels are excluded at annotation/training time and
should not appear in inference output either. Every emitted wheel is
implicitly "all three points visible".

The primary inference exporter also filters out wheels whose A/B/C
predictions violate the confirmed 2D floor-ray geometry: `a` must be
left of `b`, A/B must sit on the lower floor-ray band of the wheel
bbox, and `c_disc_bottom` must be above that A/B floor-ray line. Such
candidates remain inspectable in legacy/debug artifacts, but they are
not safe to emit in the confirmed AR JSON because the confirmed schema
has no visibility or "needs review" field.

### AR side (client)

For each captured frame, AR owns the full 3D pipeline:

- stores the camera transform indexed by its own `frame_id`;
- forwards the frame to ML;
- receives the ML JSON and matches it back via `frame_id`;
- **raycasts A** (`points.a`) into the scene using the saved
  transform; the ray's intersection with the floor plane is the
  **left base anchor** of the wheel's vertical plane;
- **raycasts B** (`points.b`) similarly → **right base anchor**;
- accumulates A/B floor projections across **K frames** and runs
  **RANSAC** → fits one stable **vertical wheel plane** through the
  two floor anchors per wheel;
- **raycasts C** (`points.c_disc_bottom`) onto that recovered vertical
  plane → 3D disc-bottom position;
- **averages C's 3D position across the K valid frames** → final
  disc installation height;
- clusters wheels across frames by their 3D position — this is how
  cross-frame association happens. No `track_id` from ML;
- drives the AR visualisation (temporary cube, final virtual wheel).

The mock-system section of the AR spec uses three fixed screen
points to drive the same raycast + RANSAC flow without ML — A/B/C
from ML simply replace those fixed points with per-wheel,
per-frame screen positions.

## JSON shape

One JSON per frame.

```json
{
  "frame_id": "frame_0001",
  "wheels": [
    {
      "bbox_xyxy": [x1, y1, x2, y2],
      "confidence": 0.94,
      "points": {
        "a": [xa, ya],
        "b": [xb, yb],
        "c_disc_bottom": [xc, yc]
      }
    }
  ]
}
```

### Field meanings

| Field | Type | Notes |
|---|---|---|
| `frame_id` | string | Echoed from AR. Stable identifier used to retrieve the matching camera transform on the AR side. Required. |
| `wheels` | array | Zero or more detected wheels in this frame. |
| `wheels[].bbox_xyxy` | `[x1, y1, x2, y2]` | Pixels, top-left origin. Top-left and bottom-right corners. |
| `wheels[].confidence` | float in `[0, 1]` | Wheel-level detection confidence. |
| `wheels[].points` | object | Keys: `a`, `b`, `c_disc_bottom`. Each value is `[x, y]` in pixels. |

### What the AR layer actually consumes

The contract-critical fields are:

- `frame_id` → matches the camera transform.
- `wheels[*].points.a` and `points.b` → raycast onto floor → RANSAC
  across K frames → vertical wheel plane.
- `wheels[*].points.c_disc_bottom` → raycast onto recovered plane →
  averaged 3D disc height.

`bbox_xyxy` and `confidence` are supporting metadata for AR-side
filtering and debug overlays; they are not used by the 3D math.

## Resolved AR-team decisions (2026-05-13)

These were `docs/OPEN_QUESTIONS_AR_SPEC.md` items; recording the
confirmed answer here so the contract is self-contained.

| Item | Decision |
|---|---|
| §1 Definition of `c_disc_bottom` | Lowest visible point of the metal rim / disc (not the tire, not the hub centre). |
| §2 Per-wheel A/B (not screen-fixed) | Yes, per-wheel. A/B are floor-plane post-process points; after AR raycasts / filters them on the floor, the vertical plane through the two floor projections is the wheel plane. |
| §3 Field names / shape | `a`, `b`, `c_disc_bottom` under `points`. Flat `[x, y]` lists. |
| §3 Occlusion handling | Partially occluded wheels are **dropped** from annotation and inference. No `visibility` flag in the JSON. |
| §4 Per-keypoint confidence | Not emitted. AR weights observations 3D-side. |
| §5 `track_id` | Not emitted. AR associates wheels across frames by 3D position after raycast. |
| §6 frame_id vs timestamp | `frame_id` only. `timestamp` is not part of the contract. |
| §10 First target platform | Android first (TFLite / LiteRT). The production model path should be lightweight, MobileNetV2-class, skipless where possible. See `docs/ANDROID_FIRST_MODEL_PLAN.md`. |

## Out of scope for ML

These belong to AR and must not appear in the ML JSON:

- 3D world or camera-space coordinates of `a` / `b` / `c_disc_bottom`.
- Plane parameters (normal, point, equation, RANSAC inliers).
- Cross-frame wheel association / `track_id`.
- Camera intrinsics / extrinsics.
- Per-keypoint confidence or visibility.
- Timestamps.

## Contract tests

The confirmed shape pinned in this document is enforced by
`tests/test_confirmed_ar_schema_shape.py`, `tests/test_ar_contract.py`,
`tests/test_infer_batch_schema.py`, and the primary output paths in
`src/infer_image.py` / `src/infer_batch.py`.

Legacy/debug payloads still exist for ML-side inspection, but they are
not the AR contract. Any change that adds `timestamp`, `visibility`,
per-keypoint confidence, `track_id`, 3D, raycast, plane, or RANSAC
fields to the confirmed JSON must update the contract docs and tests
explicitly.

## See also

- `docs/KEYPOINT_SPEC.md` — A/B/C geometric definitions.
- `docs/ANDROID_FIRST_MODEL_PLAN.md` — model roadmap (baseline → Android-first).
- `docs/PLUGIN_DATA_EXPECTATION.md` — input format from the upcoming collection plugin.
- `docs/OPEN_QUESTIONS_AR_SPEC.md` — original question set and resolved / open items.
- `README.md` — repo overview.
