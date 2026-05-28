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
| `schema_version`    | integer | Must be `1` for production replay evidence. Missing/unsupported versions are rejected. |
| `session_id`         | string | Stable identifier for one capture session (one car, one user). Used to group observations into a RANSAC batch. |
| `frame_id`           | string | The exact `frame_id` AR forwarded to ML (and received back unchanged in the ML response). Used to join the log row to the ML response and to the saved camera transform. |
| `capture_index`      | integer | Non-negative index of this observation within the session. Rows must be non-decreasing per `session_id` in append order. Multiple rows may share one `capture_index` only when they are the same `frame_id` and each row has a unique wheel identity. |
| `source_type`        | string | Production logs must use `android_ar_device_replay`, `ios_ar_device_replay`, or `ar_device_replay`. Template/synthetic values are rejected by the production validator. |
| `capture_device`     | string | Physical AR capture device name, e.g. `Pixel 8 Pro`. Required for production validation; placeholder values such as `FILL_ME`, `TODO`, `TBD`, and `unknown` are rejected. |
| `capture_app_version` | string | AR app/build version that produced the replay. Required for custody; placeholders are rejected. |
| `capture_date_utc`   | string | Real UTC capture date in `YYYY-MM-DD` format. Impossible dates such as `2026-99-99` and future dates are rejected. |
| `camera_transform`   | object \| null | Camera transform (pose) at capture time. When inline, it must contain finite numeric `R` 3x3 and `t` vec3 fields. If a separate pose store is used, set this to `null` and populate `camera_pose_ref`. |
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
  `wheel_track_id`, etc.). Production validation accepts duplicate
  `frame_id` / `capture_index` rows only when every duplicate row has
  a unique `wheel_index` or `wheel_track_id`.
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
| `c_height_value`          | non-negative number \| null | Disc-bottom height derived from `c_plane_hit` (units: AR-side; typically metres). |
| `final_disc_bottom_position` | `[x, y, z]` \| null | Final averaged disc-bottom 3D position across the session's valid frames. Set only on the row(s) that produced the final estimate, or echoed onto every row, AR's choice. |

Optional fields stay absent (or `null`) until RANSAC has run. A
batched RANSAC pipeline may rewrite earlier rows once it has
finalised inliers — either by appending a finalised
`session_id`-scoped row, or by updating in place; that is an
AR-internal decision and does not affect ML.

## Production validation

After AR records a replay JSONL, validate it before treating the model
as production-ready:

```bash
./.venv/bin/python src/validate_ar_replay.py \
  --jsonl path/to/ar_replay.jsonl \
  --out outputs/production_audit/ar_3d_replay_eval.json
```

The production gate expects `outputs/production_audit/ar_3d_replay_eval.json`
to contain `ok: true`. By default the validator requires at least 30
valid observations, complete A/B floor-raycast hits for at least 90% of
observations, RANSAC inlier labels, non-negative residuals, recovered plane evidence
(`recovered_plane.normal` as a unit vector, `recovered_plane.point`, and positive
`recovered_plane.support`), `c_plane_hit`, non-negative `c_height_value`, and at least
one final disc-bottom 3D position. It also requires every observation to carry camera pose
evidence: either a finite numeric inline `camera_transform` with `R` 3x3
and `t` vec3 fields, or a non-placeholder `camera_pose_ref`.
Every observation must also carry a
`schema_version=1`, production `source_type`, non-placeholder `capture_device`,
non-placeholder `capture_app_version`, and a real `capture_date_utc`; use
`--no-require-production-source` only for synthetic/template smoke checks.
Thresholds are CLI flags, but any relaxed thresholds must be recorded in
the generated JSON.

Template:

```bash
./.venv/bin/python scripts/create_ar_replay_log_template.py
```

This writes `outputs/production_audit/ar_3d_replay.template.jsonl`.
Replace all `FILL_ME` values with real AR-device session values before
using the log as production evidence.

Android/AR-side logging harness:

```text
ar_replay_harness/README.md
ar_replay_harness/ArReplayLogger.kt
```

The harness writes production-shaped `ar_replay.jsonl` rows from the AR
app after ML inference, floor raycasts, and RANSAC/plane recovery.

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
{"source_type":"android_ar_device_replay","capture_device":"Pixel 8 Pro","capture_app_version":"1.2.3","capture_date_utc":"2026-05-14","session_id":"s_2026_05_14_001","frame_id":"frame_0000","capture_index":0,"camera_transform":{"R":[[0.999,0.01,-0.04],[-0.01,0.999,0.02],[0.04,-0.02,0.999]],"t":[0.12,1.55,-0.30]},"camera_pose_ref":null,"screen_points":{"a":[612.4,742.1],"b":[861.7,743.0],"c_disc_bottom":[738.2,690.5]},"floor_raycast_hits":{"a":[1.18,0.00,-2.05],"b":[1.51,0.00,-2.06]}}
{"source_type":"android_ar_device_replay","capture_device":"Pixel 8 Pro","capture_app_version":"1.2.3","capture_date_utc":"2026-05-14","session_id":"s_2026_05_14_001","frame_id":"frame_0023","capture_index":23,"camera_transform":null,"camera_pose_ref":"pose_s_2026_05_14_001_23","screen_points":{"a":[618.1,738.8],"b":[859.0,740.6],"c_disc_bottom":[741.9,689.7]},"floor_raycast_hits":{"a":[1.19,0.00,-2.07],"b":[1.49,0.00,-2.06]},"inlier":true,"residual":0.0036,"recovered_plane":{"normal":[0.998,0.000,0.062],"point":[1.34,0.00,-2.06],"support":18},"c_plane_hit":[1.34,0.41,-2.06],"c_height_value":0.41,"final_disc_bottom_position":[1.34,0.41,-2.06]}
```

## See also

- `docs/AR_ML_CONTRACT.md` — full responsibility split between ML and AR.
- `docs/KEYPOINT_SPEC.md` — A/B/C definitions (load-bearing).
- `docs/OPEN_QUESTIONS_AR_SPEC.md` — confirmed and outstanding contract items.
