# AR-Replay Metric Plan

Forward-looking design for a 3D-aware evaluation layer on top of the
current 2D YOLO-pose pipeline. This document is a **plan**, not an
implemented system. Nothing here is part of the AR / ML contract, the
training gate, or the production-readiness audit yet.

Status as of 2026-05-28:

- 2D pipeline is the current production target (`docs/AR_ML_CONTRACT.md`,
  `tests/test_ar_contract.py`).
- AR-side JSONL replay log is already specified
  (`docs/AR_MOCK_LOG_CONTRACT.md`) and gated by a plumbing validator
  (`src/validate_ar_replay.py`).
- Current blocker for further training: clean export with real
  `WheelBBox` / `BBox` (see `docs/EXPORT_CERTIFICATION.md`,
  `docs/EXPORT_PARITY_AUDIT.md`). No training, no full 3D loss, and no
  AR-ready claim until export acceptance passes.

## 1. Why this plan exists

Pure 2D keypoint loss on A and B has a known failure mode: the wheel
disc has stronger gradient and texture cues than the ground around it,
so the L2 / OKS objective can pull A and B onto the rim instead of the
floor footprint. The pixel-error number stays small, but A and B no
longer lie on the floor when AR raycasts them. Floor projection drifts,
RANSAC inlier ratio collapses, and the recovered vertical wheel plane
ends up tilted — and the 2D acceptance metrics never see it.

The fix has two complementary directions:

1. **Validate A and B against floor / ground evidence**, ideally from a
   pretrained segmentation model run at evaluation time (Stage 2). This
   does not change the JSON contract, only the dataset / eval-time
   labels and filters.
2. **Score the full AR reconstruction offline** by replaying AR-captured
   sessions through the same RANSAC + plane-fit pipeline that runs on
   the device (Stage 3), and surface a small set of geometric metrics
   that catch wheel-attraction drift even when 2D pixel error is
   acceptable.

Both should land as offline evaluation **before** any 3D-aware
auxiliary loss touches training. The reasons are spelled out in §6.

## 2. Staged approach

| Stage | Layer | Status | Owner of work | Acceptance |
|---|---|---|---|---|
| **1** | 2D baseline: bbox + A / B / C in pixel space, confirmed AR JSON contract | implemented | ML | `tests/test_ar_contract.py` green; `src/eval_keypoints.py` pixel-error metrics on real-only val pass the documented thresholds |
| **2** | Floor / ground mask validation for A and B at dataset / eval time | not started | ML | A and B from drafts / labels fall inside a floor mask for ≥ X % of annotated wheels; failing rows are surfaced (not silently dropped) |
| **3** | Offline AR-replay metric over `ar_replay.jsonl` sessions | not started | ML (consumer), AR (producer of the log — already specified) | Metric report at `outputs/ar_replay/<session_id>_metric.json` with the metrics in §5; reproducible from a fixed `ar_replay.jsonl` |
| **4** | Optional 3D-aware auxiliary training loss | deferred | ML | Only after Stage 3 stabilises and a 3D error budget is agreed with the AR team (§9 of `docs/OPEN_QUESTIONS_AR_SPEC.md` is still open) |

Each stage strictly subsumes the previous one. None of the stages
beyond Stage 1 changes the ML → AR JSON contract.

### Stage 1 — 2D baseline (today)

Already in place. Documented in `docs/AR_ML_CONTRACT.md`,
`docs/KEYPOINT_SPEC.md`, `docs/KEYPOINT_DATASET_FORMAT.md`. The
training, inference, and acceptance metrics live in
`src/train_yolo.py`, `src/infer_image.py`, `src/infer_batch.py`,
`src/eval_keypoints.py`. The only invariants this plan must respect
from Stage 1 are:

- Inference output shape is frozen: `frame_id` + `wheels[]`, with
  `bbox_xyxy`, `confidence`, and `points.{a, b, c_disc_bottom}`.
- ML stays per-frame and stateless. AR owns 3D.
- No `track_id`, no `timestamp`, no per-keypoint confidence, no
  `visibility`, no 3D coordinates leaking from ML.

This plan adds an **evaluation layer outside that contract**: AR
records a log, ML consumes the log offline to compute a metric.
Nothing crosses into inference.

### Stage 2 — Floor / ground mask validation for A and B

Goal: detect "A / B has drifted onto the wheel" without waiting for an
AR raycast.

What it produces:

- For each annotated wheel, a boolean `a_on_floor` and `b_on_floor`
  derived from a precomputed ground / floor segmentation mask of the
  source image.
- For each model prediction, the same booleans computed at eval time.

Where it slots in (without touching the contract):

- Dataset-side: extend the validator path
  (`src/check_keypoint_incoming.py` or a new sibling) to flag drafts
  whose A / B fall outside a floor mask. Flagged drafts go to manual
  review via `src/manual_keypoint_annotator.py --prefill-from …`; they
  are not silently dropped from training data.
- Eval-side: extend `src/eval_keypoints.py` (or a new
  `src/eval_floor_consistency.py`) to report `floor_consistency_a`,
  `floor_consistency_b` on the real-only val split.

Mask sources to evaluate (pick one, do not stack without measurement):

- COCO-Stuff / ADE20K-pretrained segmenter for `road`, `floor`,
  `pavement`, `ground`, `earth`, `grass` classes. Implementable via
  existing `ultralytics` segmentation models without enlarging the
  dep surface.
- SAM 2 with a point prompt placed at the midpoint between A and B
  on the floor side, reused from `src/auto_annotate_wheels.py`.
- A simple OpenCV heuristic (Hough lines + colour) as a fallback for
  diagnostic-only use; not production-grade.

This stage is **opt-in at the dataset / eval boundary**. No mask data,
no segmenter weights, and no extra confidence fields leak into the
inference JSON. Acceptance is a per-batch report only.

### Stage 3 — Offline AR-replay metric

Goal: given an AR-captured session log, replay the RANSAC + plane
recovery offline and emit metrics that quantify how well the recovered
geometry holds together. This is the layer that catches
wheel-attraction drift even when Stage 2 looks fine.

Inputs:

- `ar_replay.jsonl` recorded on device by `ar_replay_harness/`
  (`ArReplayLogger.kt`), already shape-validated by
  `src/validate_ar_replay.py`.
- Optional ground-truth disc-bottom 3D position per session, where
  available (manual measurement; not part of the device log).

Outputs:

- `outputs/ar_replay/<session_id>_metric.json` — per-session metrics
  (§5).
- `outputs/ar_replay/<session_id>_per_frame.csv` — per-observation
  residuals and inlier flags for debugging.

Position in the codebase:

- New script `src/eval_ar_replay_metric.py`. Consumes the validated
  JSONL. Pure stdlib + numpy. No model weights, no inference, no
  schema changes.
- The plumbing validator (`src/validate_ar_replay.py`) stays the gate
  on log shape and evidence completeness; the new script is the
  **quality scorer** on top of a log that has already passed the
  validator.

### Stage 4 — Optional 3D-aware auxiliary loss

Deferred until Stage 3 produces enough signal to (a) define a 3D
error budget with the AR team and (b) demonstrate that Stage 2 is
insufficient on its own.

When triggered, the candidate is an auxiliary loss term added to the
YOLO-pose training objective that re-projects predicted A and B onto a
session-derived floor plane and penalises distance from a target plane
(or directly from a target 3D disc-bottom position when GT is
available). This requires per-frame camera transform and a stable
session grouping — both of which are already collected by the AR-side
log. The loss does **not** change the model output shape or the AR
contract; only the training-time supervision signal changes.

No code, no plan-of-record, and no design freeze for Stage 4 in this
document. It is listed here only to fix the order: replay-metric
first, training-loss later.

## 3. Data needed from the AR side

Most of the required fields are already specified in
`docs/AR_MOCK_LOG_CONTRACT.md`. This section restates the subset that
Stage 3 actually reads and flags anything additional that is currently
optional in the log but load-bearing for the metric.

### Already required (Stage 3 reads as-is)

| Field | Used for |
|---|---|
| `schema_version` (=1) | Gate: reject unsupported logs upstream. |
| `session_id` | Grouping observations into one RANSAC batch. |
| `frame_id` | Joining log rows to ML responses; ordering check. |
| `capture_index` | Detecting frame ordering and duplicate-row collisions. |
| `source_type`, `capture_device`, `capture_app_version`, `capture_date_utc` | Custody: distinguishing real device replays from templates / synthetic smoke. The metric refuses to run on non-production sources by default. |
| `camera_transform` or `camera_pose_ref` | Future Stage 4 input. Stage 3 does not re-raycast (AR did it on device); the transform is recorded for reproducibility and diagnostic re-projections. |
| `screen_points.{a, b, c_disc_bottom}` | The exact ML output the AR session consumed. Used to (1) check ML / log agreement and (2) compute pixel-space stability. |
| `floor_raycast_hits.{a, b}` | The A / B floor projections in world space — the core input to the metric. |

### Already optional in the log, required by the metric

These fields are listed in `docs/AR_MOCK_LOG_CONTRACT.md` under
"Optional fields (populated after RANSAC / plane recovery)". For
Stage 3 they are **not** optional — `src/eval_ar_replay_metric.py`
must reject sessions that lack them.

| Field | Used for |
|---|---|
| `inlier` | Inlier ratio per session. |
| `residual` | Residual distribution per session, plus inlier filtering when AR's `inlier` flag is disputed. |
| `recovered_plane.{normal, point, support}` | Plane stability across observations within a session, plane orientation sanity checks (vertical-up). |
| `c_plane_hit` | C-projection stability. |
| `c_height_value` | Disc-height stability, final-error vs GT when available. |
| `final_disc_bottom_position` | Final 3D error vs GT when available, session-level summary. |

### Additional data the metric needs that is NOT in the AR log

| Field | Source | Required? |
|---|---|---|
| Per-session group key linking K observations to one wheel | `session_id` + `wheel_index` or `wheel_track_id` rows in the existing log | already present, no new field needed |
| Ground-truth 3D disc-bottom position (optional) | Manual measurement during capture, supplied as a separate per-session JSON sidecar (`<session_id>.gt.json`) | optional, only required for final-error metric |
| Ground-truth wheel plane (optional) | Same sidecar, e.g. a measured plane normal and point | optional, only required for plane-quality metric vs GT |

The GT sidecar lives outside the JSONL on purpose: it is not produced
by the AR client, it is not part of the contract, and not every
session will have it. The metric is fully usable without GT (it then
reports stability / consistency metrics only; see §5).

## 4. AR-replay flow (what the metric replays)

The on-device flow is documented in `docs/AR_ML_CONTRACT.md` (§
"Responsibility split", AR side). The offline metric replays the
same flow against the recorded log. Per session:

1. **Read** all observations with this `session_id`, sorted by
   `capture_index`.
2. **Filter** to observations with valid `floor_raycast_hits.a` and
   `floor_raycast_hits.b` (the AR side wrote `null` when the ray
   missed).
3. **Accumulate K observations** per `wheel_index` (or
   `wheel_track_id`). K is read from the log itself — the metric is
   not allowed to invent a new K. If the AR side has not recorded
   enough observations for a wheel (`< 30` by current production
   threshold), the session is reported as "insufficient evidence" and
   the metric stops without faking a result.
4. **Refit RANSAC** on the floor projections of A and B using the
   same algorithm the AR side uses (the parameters must be a faithful
   replay; see "Reproducibility note" below). Compare against the AR
   log's `inlier` / `residual` / `recovered_plane.*` columns to detect
   replay drift between device and offline runs.
5. **Recover** the vertical wheel plane (the plane that is
   perpendicular to the floor normal and passes through the
   RANSAC-fitted base line through the A / B floor projections).
6. **Project / raycast C** onto the recovered vertical plane to get
   `c_plane_hit` per observation. Compare with the AR-side
   `c_plane_hit` recorded in the log.
7. **Average** `c_plane_hit` across the inlier observations for the
   wheel → `final_disc_bottom_position` (replayed). Compare against
   the AR-side `final_disc_bottom_position`.
8. **Score** using the metrics in §5.

Reproducibility note: the metric does **not** redefine RANSAC params.
It either (a) replays the AR-side algorithm with the same params, or
(b) reports only the stability metrics that do not require refitting.
Adding a separate offline RANSAC variant would create a second
ground-truth-by-disagreement, which is the wrong tool for evaluating
the device behaviour.

## 5. Metrics produced per session

All metrics are per-session, derived from the replayed flow in §4.
The report at `outputs/ar_replay/<session_id>_metric.json` carries
every field below; the eval CLI summarises across sessions for a
batch.

| Metric | Definition | Without GT | With GT |
|---|---|---|---|
| **Inlier ratio** | `inlier=true` observations divided by total floor-raycast-valid observations, per `wheel_index`. | reported | reported |
| **Plane residual** | Median and 95th-percentile `residual` over inlier observations. | reported | reported |
| **Plane stability across frames** | Standard deviation of `recovered_plane.normal` direction (angular, in degrees) across observations within the session. Captures whether RANSAC converges or wobbles. | reported | reported |
| **Plane verticality** | Angle between `recovered_plane.normal` and the device-floor normal (recovered from the camera transform's gravity vector when available). Should be near 90°. | reported when transform is available | reported |
| **C projection stability** | Standard deviation of `c_plane_hit` in 3D across inlier observations; captures whether C is mapping consistently across K frames or jumping. | reported | reported |
| **Final disc-bottom 3D error** | Euclidean distance from replayed `final_disc_bottom_position` to GT 3D position from `<session_id>.gt.json`. | not reported | reported |
| **Final plane error vs GT** | Angle between replayed `recovered_plane.normal` and GT plane normal. | not reported | reported |
| **Failure rate** | Fraction of wheels in the session for which the replay could not produce a final disc-bottom position (insufficient inliers, all rays missed, missing optional fields, etc.). Wheels that fail are not silently excluded from the inlier-ratio numerator. | reported | reported |

Threshold values are deliberately not pinned in this plan. They will
be set after the first real-device replay batch is captured, in
coordination with the AR team and against the open 3D error budget
(`docs/OPEN_QUESTIONS_AR_SPEC.md` §9). Until thresholds exist, the
metric is **informational**, not a gate.

## 6. Why this is offline eval first, not training loss

Reasons to keep Stage 3 strictly offline and out of the training loop
for now:

1. **Schema is frozen.** Anything 3D in the ML side currently violates
   the confirmed contract (§3, §5, §6 of
   `docs/OPEN_QUESTIONS_AR_SPEC.md`). An offline metric does not change
   the contract; a 3D loss would require per-frame camera transforms
   inside the training pipeline and a renegotiation of where the 3D
   responsibility line sits.
2. **No 3D error budget yet.** §9 of the open questions still reads
   *"hard to say — it should really be measured in 3D"*. A training
   loss with no agreed error budget would optimise an arbitrary
   surrogate.
3. **The current ML blocker is upstream.** Export with real
   `WheelBBox` / `BBox` is not yet clean
   (`docs/EXPORT_PARITY_AUDIT.md`). Adding 3D supervision before
   export acceptance multiplies the failure surface; a divergence
   between PT and ONNX on a 2D model is small relative to a
   divergence on a 3D-aware model.
4. **Replay first, train second.** A loss term should be motivated by
   a measured failure mode on real device replays, not by a synthetic
   intuition. Stage 3 produces that motivation; without it, Stage 4 is
   guesswork.
5. **Synthetic data is not 3D-truth.** The UE plugin can in principle
   project ground-truth A / B / C / camera intrinsics + extrinsics
   (§7 in `docs/OPEN_QUESTIONS_AR_SPEC.md`), but synthetic 3D loss has
   the same domain-gap risk as synthetic detection loss — it can
   reduce real-device 2D pixel error while making real-device 3D
   geometry worse. Replay metric will quantify that gap before any
   weight is tuned against synthetic 3D.

## 7. Floor / ground segmentation as A / B validation (Stage 2 detail)

The wheel-attraction failure mode shows up as A or B sitting on the
rim instead of the floor / pavement / road. A floor / ground
segmentation mask catches that at the pixel level, before any
raycast.

Implementation directions (none of them touch the contract):

1. **Eval-time only**: run a frozen segmenter on the real-only val
   images, intersect the predicted A / B with the union of
   floor / road / pavement / earth / grass classes, and report
   `floor_consistency_a`, `floor_consistency_b` next to the existing
   pixel-error metrics in `src/eval_keypoints.py` output. This is the
   minimum viable Stage 2.
2. **Draft-validation**: run the same segmenter on
   `data/incoming/<batch>/images`, flag drafts whose A / B fall
   outside the mask, and surface them to the manual annotator via
   `src/manual_keypoint_annotator.py --prefill-from …` for human
   review. Failing rows are not deleted from the batch — they are
   re-labelled. This converts the failure mode into labelling signal
   instead of training noise.
3. **Synthetic GT cross-check** (only when UE plugin §7 lands): the
   plugin can emit a per-frame floor mask alongside A / B / C, in
   which case the segmenter dependency disappears entirely.

A floor mask can also be used as a **soft training signal** without
changing the contract: e.g. an auxiliary segmentation head trained
jointly with the pose head, dropped at inference. That is a Stage 4
variant and is out of scope for this plan.

## 8. Dependencies, sequencing, and what stays blocked

Sequencing constraints:

- Stage 1 must remain green: `pytest -q`, `tests/test_ar_contract.py`,
  and `src/eval_keypoints.py` thresholds on the real-only val split.
- Stage 2 starts only after the current export blocker (real
  `WheelBBox` / `BBox`) is closed — otherwise Stage 2 reports against
  a model that is about to be re-exported and the results will be
  thrown away.
- Stage 3 starts only after AR records at least one real-device
  replay session that passes `src/validate_ar_replay.py` against the
  current production thresholds (30+ valid observations, full
  evidence, real `capture_*` fields). The metric script is cheap to
  build; without a real session log there is nothing to score.
- Stage 4 starts only after Stage 3 surfaces a reproducible failure
  mode AND the open 3D error budget (§9) is closed.

What stays blocked until Stage 3 reports green on real-device replays:

- Marking the model as AR-ready / production-certified.
- Updating `docs/PRODUCTION_READINESS_AUDIT.md` to claim 3D-validated.
- Any AR-side claim that recovered disc-bottom heights meet a target
  error.

## 9. Non-goals

- **Not** a contract change. `points.a` / `.b` / `.c_disc_bottom`
  remain 2D pixel coordinates in the ML JSON.
- **Not** a tracker. Cross-frame association stays on the AR side.
- **Not** an on-device runtime change. The metric runs in this repo,
  offline, against a recorded JSONL.
- **Not** a training-loss change. Stage 4 is deferred.
- **Not** a replacement for `src/validate_ar_replay.py`. The validator
  guards log shape and evidence presence; this metric scores quality
  on a log that has already passed the validator.
- **Not** an excuse to relax the export gate. Export with clean
  `WheelBBox` / `BBox` remains a precondition for any new training run.

## 10. See also

- `docs/AR_ML_CONTRACT.md` — confirmed 2D contract and ML / AR
  responsibility split (load-bearing for §1).
- `docs/AR_MOCK_LOG_CONTRACT.md` — AR-side JSONL schema this plan
  consumes (load-bearing for §3 and §4).
- `docs/AR_HANDOFF.md` — current AR integration package and metrics.
- `docs/KEYPOINT_SPEC.md` — A / B / C geometric definitions.
- `docs/OPEN_QUESTIONS_AR_SPEC.md` — §7 (UE export), §9 (3D error
  budget). Both are referenced as open dependencies of Stage 3 and
  Stage 4.
- `docs/PRODUCTION_READINESS_AUDIT.md` — current production status
  this plan extends but does not modify.
- `docs/EXPORT_CERTIFICATION.md`, `docs/EXPORT_PARITY_AUDIT.md` —
  current upstream blocker (clean export with real
  `WheelBBox` / `BBox`).
- `src/validate_ar_replay.py` — plumbing validator on top of which
  Stage 3 builds.
- `src/evaluate_ar_holdout.py` — 2D holdout evaluator on AR-device
  human-labelled data; complementary to Stage 3, not replaced by it.
- `ar_replay_harness/` — Android logging harness that emits the
  JSONL Stage 3 consumes.
