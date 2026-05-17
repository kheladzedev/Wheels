# Open Questions — AR Mock Spec Clarification

Originally a focused list of items ML needed the AR team to confirm
before the JSON contract and annotation conventions could be frozen.
Most items were resolved by the AR-team response on **2026-05-13** and
then re-confirmed / tightened by the Unreal-side follow-up on
**2026-05-18**. This document now tracks the resolved decisions plus
the items still open after that round.

For older, broader questions (data volume, K-frame accumulation
parameter, ground-plane area `N`, etc.) see
`docs/QUESTIONS_FOR_TEAM.md`. This doc is the narrow follow-up to the
spec-clarification round.

---

## Latest AR-side confirmation (2026-05-18)

The Unreal/AR developer answered the current ML question set directly:

- `c_disc_bottom` is the visually lowest visible point of the metal
  rim / disc.
- A/B are the two side floor-plane post-process points: AR uses them
  on the floor plane, then builds the vertical wheel plane
  perpendicular to the floor through those points.
- Partially closed / occluded wheels are dropped.
- No per-keypoint confidence.
- No ML `track_id`; AR filters/associates by coordinates after its
  3D/raycast stage.
- `frame_id` is enough; no timestamp in the contract.
- A new wheel-collection plugin is in progress; the developer said
  Unreal can output essentially anything, but the collector may take
  extra implementation time depending on requested fields.
- First platform remains Android.

The old limited raw Unreal batch (`0001.zip`) is therefore useful as
debug / preview data only unless it also contains the agreed
`bbox_xyxy` and true `c_disc_bottom` semantics. Do not promote that
batch to production training merely because it parses.

---

## Resolved (2026-05-13, re-confirmed 2026-05-18)

### §1 — Definition of `c_disc_bottom` ✅

**Decision:** lowest visible point of the metal rim / disc (not the
tire, not the hub centre). Confirmed visually against the
reference image supplied by AR.

Codified in: `docs/KEYPOINT_SPEC.md` ("The three keypoints"), and
`docs/AR_ML_CONTRACT.md` (Resolved-decisions table §1).

### §2 — Per-wheel A and B (not screen-fixed) ✅

**Decision:** per-wheel. The model emits `a` and `b` per detected
wheel. After AR raycasts each onto the floor plane, the vertical
plane through the two floor-projected points is the wheel plane.

This drops mock-pipeline option (b) (screen-fixed A/B) entirely.

Codified in: `docs/KEYPOINT_SPEC.md` (geometric roles), and
`docs/AR_ML_CONTRACT.md` (Resolved-decisions §2).

### §3 — Field names, shape, occlusion, visibility, per-kp confidence ✅

**Decision:**

- Final keypoint names: `a`, `b`, `c_disc_bottom`, under a `points`
  object (flat `{name: [x, y]}` dict).
- Per-keypoint shape: just `[x, y]`. No richer object with confidence
  or visibility.
- **No `visibility` flag at all.** Partially occluded wheels are
  dropped from annotation and inference — not emitted with
  `visibility=0`.
- **No per-keypoint confidence.** Only wheel-level `confidence`.

Codified in: `docs/AR_ML_CONTRACT.md` ("JSON shape" + Resolved
§3), `docs/KEYPOINT_SPEC.md` ("Occluded wheels: dropped",
"No per-keypoint confidence").

### §4 — Per-keypoint confidence ✅

**Decision:** not emitted. AR does its own weighting on the 3D side.

Codified in: `docs/AR_ML_CONTRACT.md` (Resolved §4),
`docs/KEYPOINT_SPEC.md` ("No per-keypoint confidence").

### §5 — `track_id` ownership ✅

**Decision:** AR-side. ML stays stateless per-frame. AR clusters
wheels across frames by 3D position after raycast. No `track_id` in
the JSON.

Codified in: `docs/AR_ML_CONTRACT.md` (Resolved §5).

### §6 — frame_id vs timestamp ✅

**Decision:** `frame_id` only. `timestamp` is **not** part of the
contract. AR uses `frame_id` to retrieve the camera transform it
saved at capture time.

Codified in: `docs/AR_ML_CONTRACT.md` (Resolved §6, "Field
meanings"). NB: this renames the previously-listed §6 ("acceptable
keypoint error") to §9 below to avoid confusion — see *Still open*.

### §8 — Bbox format ✅

**Decision:** `bbox_xyxy` (top-left and bottom-right corners). This
flips the previously-stated default of `bbox_xywh`.

Codified in: `docs/AR_ML_CONTRACT.md` (JSON shape).

### §10 — Fields removed from the target ✅

**Decision:** confirmed. `image`, `image_size`, `thresholds`,
`stats.n_wheels`, `wheels[*].warnings` are all dropped. AR does not
rely on any of them. No request to retain `image_size`.

Codified in: `docs/AR_ML_CONTRACT.md` (no such fields in the new
shape).

### §11 — First target platform ✅

**Decision:** Android first (TFLite / LiteRT). Eventually likely a
MobileNetV2-class lightweight model.

Codified in: `docs/ANDROID_FIRST_MODEL_PLAN.md`.

---

## Still open

### §7 — Unreal export capabilities ⏳ partially resolved

AR developer is finishing a collection plugin. The latest answer says
Unreal can output essentially anything, but each extra collector field
may cost implementation time.

**Still required for the next accepted training batch:**

- `bbox_xyxy` around the full visible wheel (tyre + rim).
- `points.a`, `points.b`, `points.c_disc_bottom` in final image pixel
  coordinates.
- Occluded wheels omitted entirely.
- `frame_id` matching the image stem.

**Still useful but not required in the ML JSON:** 3D world positions of
`a` / `b` / `c_disc_bottom` plus camera intrinsics + extrinsics. If the
Unreal export can include these, we can audit 2D labels against 3D
projection and build an AR-side error metric. They must stay in
metadata/debug artifacts, not in the confirmed ML -> AR response.

Documented expectations for the plugin:
`docs/PLUGIN_DATA_EXPECTATION.md`.

### §9 — Acceptable keypoint error budget ⏳

AR response: *"сложно сказать, тк её похорошему надо в 3D считать"*
("hard to say — it should really be measured in 3D").

**Interpretation:** no pixel-error budget. Final quality is judged
on the AR side after raycast + RANSAC, not on raw 2D keypoint pixel
error. ML's pixel-error metrics (in `src/eval_keypoints.py` output)
remain useful as **informational signal**, but they are not the
acceptance criterion.

A 3D error budget would still help — e.g. "max 1 cm drift between
the AR-recovered disc-bottom position and ground truth across the
RANSAC pipeline". Pending an AR-side measurement once real data is
flowing.

---

## See also

- `docs/AR_ML_CONTRACT.md` — current confirmed contract and the
  per-decision table mirroring this resolution log.
- `docs/KEYPOINT_SPEC.md` — A/B/C definitions used here.
- `docs/QUESTIONS_FOR_TEAM.md` — broader open questions (data, K,
  error budget, platforms) pre-dating this round.
- `docs/PLUGIN_DATA_EXPECTATION.md` — what we expect from the plugin
  data feed once §7 lands.
- `docs/ANDROID_FIRST_MODEL_PLAN.md` — model roadmap implied by §11.
- `tests/test_ar_contract.py` and
  `tests/test_confirmed_ar_schema_shape.py` — guard the confirmed
  schema while keeping legacy/debug converter behavior explicit.
