# AR Mock-System Observation Log Contract

When the AR client runs the wheel-fitting mock pipeline, it must
record one observation per captured frame so the AR team can tune
RANSAC, inspect tracking noise, and replay sessions offline. The
**ML side is not responsible** for producing this log — ML only
emits the per-frame screen-space response defined in
`docs/AR_ML_CONTRACT.md`. This document defines what the AR-side log
contains so that downstream tooling on both sides can read it.

The log is **JSON Lines** (`.jsonl`): one observation per line, no
top-level wrapper. Append-only. One file per session keyed by
`session_id`.

## Per-observation schema

### Required fields (must be present from frame 0)

| Field | Type | Description |
|---|---|---|
| `session_id`         | string | Stable identifier for one capture session (one car, one user). Used to group observations into a RANSAC batch. |
| `frame_id`           | string | The exact `frame_id` AR forwarded to ML (and received back unchanged in the ML response). Used to join the log row to the ML response and to the saved camera transform. |
| `capture_index`      | integer | Monotonic index of this observation within the session (`0, 1, 2, …`). |
| `camera_transform`   | object \| null | Camera transform (pose) at capture time. Shape is AR-internal but should be sufficient to repeat the raycast offline. If a separate pose store is used, set this to `null` and populate `camera_pose_ref`. |
| `camera_pose_ref`    | string \| null | Stable key into an AR-side camera-pose store. Mutually exclusive with `camera_transform`. |
| `screen_points.a`             | `[x, y]` (pixels) | The exact A point ML returned for one wheel in this frame — the left screen-space floor / raycast point near the wheel's footprint. |
| `screen_points.b`             | `[x, y]` (pixels) | Right floor / raycast screen point from ML. |
| `screen_points.c_disc_bottom` | `[x, y]` (pixels) | Lower visible metal-rim / disc screen point from ML. |
| `floor_raycast_hits.a`        | `[x, y, z]` (world) | World-space point where the ray from A hit the AR floor plane. `null` if the ray missed (e.g. no floor in view). |
| `floor_raycast_hits.b`        | `[x, y, z]` (world) | World-space point where the ray from B hit the AR floor plane. `null` if the ray missed. |

Notes:

- One log row describes **one wheel observation**, not one ML frame.
  If ML returned N wheels in a frame, AR writes N log rows with the
  same `frame_id` / `capture_index` and a per-wheel identity carried
  via an additional AR-side index field if needed (`wheel_index`,
  `wheel_track_id`, etc.).
- `screen_points` mirror what ML emitted under `wheels[].points.{a, b,
  c_disc_bottom}`. They are stored verbatim so the log can be
  replayed independently of ML.

### Optional fields (populated after RANSAC / plane recovery)

| Field | Type | Description |
|---|---|---|
| `inlier`                  | boolean | RANSAC inlier flag for this observation (`true` = used to build the recovered plane, `false` = rejected). |
| `residual`                | number  | Geometric residual of this observation against the recovered plane (units: world distance, RANSAC-internal). |
| `recovered_plane.normal`  | `[nx, ny, nz]` | Unit normal of the recovered vertical wheel plane. |
| `recovered_plane.point`   | `[x, y, z]`    | A point on the recovered plane. |
| `recovered_plane.support` | integer        | Number of A/B floor hits used to fit the plane. |
| `c_plane_hit`             | `[x, y, z]` \| null | World-space intersection of the ray from C with the recovered vertical plane. `null` if no plane recovered yet or the ray missed. |
| `c_height_value`          | number \| null      | Disc-bottom height derived from `c_plane_hit` (units: AR-side; typically metres). |
| `final_disc_bottom_position` | `[x, y, z]` \| null | Final averaged disc-bottom 3D position across the session's valid frames. Set only on the row(s) that produced the final estimate, or echoed onto every row, AR's choice. |

Optional fields stay absent (or `null`) until RANSAC has run. A
batched RANSAC pipeline may rewrite earlier rows once it has
finalised inliers — either by appending a finalised
`session_id`-scoped row, or by updating in place; that is an
AR-internal decision and does not affect ML.

## Forbidden fields

Per the ML/AR responsibility split (`docs/AR_ML_CONTRACT.md`), the
log must NOT include:

- Per-keypoint confidence (ML never emits it — there is nothing to log).
- `visibility` flags (occluded wheels are dropped at annotation /
  inference time).
- Anything from inside the ML model — class activations, raw heatmaps,
  detection-head logits. The log is for the AR-side mock pipeline,
  not for ML training telemetry.

## JSONL example

Two observations from one session. First row before RANSAC has fired;
second row after RANSAC has finalised inliers and computed disc
height.

```jsonl
{"session_id":"s_2026_05_14_001","frame_id":"frame_0000","capture_index":0,"camera_transform":{"R":[[0.999,0.01,-0.04],[-0.01,0.999,0.02],[0.04,-0.02,0.999]],"t":[0.12,1.55,-0.30]},"camera_pose_ref":null,"screen_points":{"a":[612.4,742.1],"b":[861.7,743.0],"c_disc_bottom":[738.2,690.5]},"floor_raycast_hits":{"a":[1.18,0.00,-2.05],"b":[1.51,0.00,-2.06]}}
{"session_id":"s_2026_05_14_001","frame_id":"frame_0023","capture_index":23,"camera_transform":null,"camera_pose_ref":"pose_s_2026_05_14_001_23","screen_points":{"a":[618.1,738.8],"b":[859.0,740.6],"c_disc_bottom":[741.9,689.7]},"floor_raycast_hits":{"a":[1.19,0.00,-2.07],"b":[1.49,0.00,-2.06]},"inlier":true,"residual":0.0036,"recovered_plane":{"normal":[0.998,0.000,0.062],"point":[1.34,0.00,-2.06],"support":18},"c_plane_hit":[1.34,0.41,-2.06],"c_height_value":0.41,"final_disc_bottom_position":[1.34,0.41,-2.06]}
```

## See also

- `docs/AR_ML_CONTRACT.md` — full responsibility split between ML and AR.
- `docs/KEYPOINT_SPEC.md` — A/B/C definitions (load-bearing).
- `docs/OPEN_QUESTIONS_AR_SPEC.md` — confirmed and outstanding contract items.
