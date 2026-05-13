# Open Questions — AR Mock Spec Clarification

Originally a focused list of items ML needed the AR team to confirm
before the JSON contract and annotation conventions could be frozen.
Most items were resolved by the AR-team response on **2026-05-13**;
this document now tracks the resolved decisions plus the items still
open after that round.

For older, broader questions (data volume, K-frame accumulation
parameter, ground-plane area `N`, etc.) see
`docs/QUESTIONS_FOR_TEAM.md`. This doc is the narrow follow-up to the
spec-clarification round.

---

## Resolved (2026-05-13)

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

### §7 — Unreal export capabilities ⏳ in progress

AR developer working on a collection plugin landing **evening of
2026-05-13**. The plugin author said they can output essentially
anything if we tell them what we want.

**Asked but not yet committed in writing:** does the export include
3D world positions of `a` / `b` / `c_disc_bottom` plus camera
intrinsics + extrinsics? If yes, we can auto-generate 2D keypoint
labels by projection (zero manual labelling for synthetic frames).

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
- `tests/test_ar_contract.py` — currently still pins the
  transitional + earlier-target schemas; will be migrated to the
  confirmed shape alongside the code port of
  `src/postprocess_wheels.py` / `src/infer_image.py`.
