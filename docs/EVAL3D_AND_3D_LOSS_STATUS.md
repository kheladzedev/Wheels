# 3D-Eval Harness, Reconstruction Loss & Web Multi-Task — Status

Status as of 2026-05-29. This document covers three ML-side components
added to move the goal's items **#2 / #3 / #4** from "not in the code"
to "implemented and unit-tested", while being explicit about what stays
**blocked on real data** (items #1 / #5) and what stays **gated** by the
staged plan in `docs/AR_REPLAY_METRIC_PLAN.md`.

None of this changes the frozen 2D AR↔ML contract
(`docs/AR_ML_CONTRACT.md`): the model still emits only 2D screen-space
points. The new code **measures** the 3D quality of those 2D points and
adds **optional, off-by-default** training signal — it does not move the
3D responsibility line.

## What is implemented (plumbing validated on synthetic round-trip)

### 1. 3D-eval harness — item #2

- `src/eval3d_floorray.py` — deterministic, numpy-only replay of the
  AR-side 3D pipeline: pixel→ray (`K⁻¹·[u,v,1]`), raycast A/B onto the
  floor, RANSAC a vertical wheel plane through the floor anchors,
  raycast C onto that plane, average across frames → disc-bottom.
- `src/eval3d_report.py` — val-set driver. Reads a per-`scene_id`
  *frames manifest* (UE `Ground` intrinsics+pose + predicted A/B/C),
  runs the harness per scene, aggregates **disc-height sigma** and
  (when 3D GT is present) **disc-height error**, against the acceptance
  budget (<3 cm accept, <1 cm target — 3D error budget still open,
  `docs/OPEN_QUESTIONS_AR_SPEC.md` §9).
- Tests: `tests/test_eval3d_floorray.py` (19), `tests/test_eval3d_report.py` (7).

Correctness is pinned by **round-trip**: a known 3D scene is forward
projected to pixels, the harness recovers it to ~0 error; injected
pixel noise raises sigma; A/B drifted off the floor (the
wheel-attraction failure of §1) blows up the disc-height error. The
metric was found to need a **GT-error gate, not sigma alone** — a
*systematic* A/B drift keeps cross-frame sigma low while the
reconstruction is biased, so acceptance gates on both.

### 2. Differentiable 3D reconstruction loss — item #3

- `src/models/reconstruction_loss.py` — torch counterpart of the
  harness. Reprojects predicted A/B→floor and C→recovered vertical
  plane, Huber on the 3D residual vs GT. Per-sample intrinsics `K`,
  pose `R`/`C` are inputs. Grazing angles are handled (sign-preserving
  denominator clamp + smooth grazing weight + Huber). `ramp` warm-up and
  `detach_plane` (isolate C's gradient) are first-class.
- Tests: `tests/test_reconstruction_loss.py` (10) — round-trip zero,
  monotonic in pixel error, gradient flow, Huber bound, grazing
  finiteness, ramp scaling, detach isolation (+ positive control),
  mixed-dtype safety, ramp=0 NaN suppression.

### 3. Web multi-task model — item #4

- `src/models/web_multitask.py` — shared MobileNetV2 trunk at 512² →
  the reused FCOS pose head (cls/bbox/kpt/vis) + a `FloorHead` that
  regresses `{pitch, roll, delta_z}` from global features.
  `MultiTaskLoss` uses learnable homoscedastic uncertainty weighting
  (Kendall et al.). `set_stage(2d|floor|recon|joint)` freezes heads for
  the staged schedule; `detach_floor` keeps the floor task off the
  shared trunk during the 2D stage. The floor loss is **scale-normalised
  per-DoF** (`FLOOR_SCALE`) so the metric-scale `delta_z` does not drown
  the radian-scale pitch/roll under a shared Huber.
- Tests: `tests/test_web_multitask.py` (10).

> All four modules were put through an adversarial multi-agent review
> (5 dimensions × 2 skeptics/finding). Confirmed substantive fixes
> applied: per-DoF floor-loss normalisation, `ramp=0` NaN
> short-circuit, dtype-safe `pixel_to_ray`, report provenance /
> `gate_status` and single-frame `sigma_estimable` honesty, degenerate
> anchor guard, RANSAC threshold guard, an independent (non-round-trip)
> UE pose-convention test, and `yaw`/`position_xy` future-proofing.

### 4. 3D promotion gate — item #1 acceptance (defined + enforced)

- `src/promotion_gate_3d.py` — turns the harness report into the
  disc-height **acceptance criterion** for promoting skipless over the
  champion, next to the 2D KPIs in `src/model_selection_audit.py`.
  Load-bearing invariant: a report whose `gate_status != "gate"`
  (synthetic / unverified UE pose) returns **insufficient_evidence** and
  can never pass — `compare_candidate_vs_champion` refuses to promote a
  synthetic candidate even with a lower sigma. CLI exits non-zero unless
  the candidate genuinely passes, so CI can gate on it.
- Tests: `tests/test_promotion_gate_3d.py` (8).

This defines #1's 3D "done" so it is executable the instant real data +
a clean UE export arrive. It does **not** make #1 done — promotion still
requires training on real floor-ray data and beating the champion on a
real val, which stays data-blocked below.

## What stays BLOCKED (cannot be closed in code)

- **Items #1 / #5 — real data.** Promoting skipless over the YOLO11n
  champion, OKS / FP-rate / INT8-acceptance KPIs, and any "production
  quality" claim require ≥2000 real labelled frames + hard-negative
  buckets + A/B re-annotation to the floor-ray contract + scene-split.
  None of that exists; the synthetic round-trip here validates **format
  and math**, never accuracy.
- **Running the harness on real model val.** Needs a clean UE export
  with per-frame intrinsics + pose + 3D GT **and model-predicted A/B/C**.
  Two of those three landed with the MCP `WheelsDataset_v0_2` export (see
  "Real export — what landed 2026-05-29" below); the missing pieces are a
  floor-ray-correct A/B and model inference on the exported frames.
- **Reconstruction loss in training.** Gated by
  `docs/AR_REPLAY_METRIC_PLAN.md` §6: offline metric first, training
  loss later, after a 3D error budget is agreed. The module is built and
  tested but stays **off by default** in `PoseLoss`.

## Real export — what landed 2026-05-29 (MCP `WheelsDataset_v0_2`)

The MCP renderer's rich annotations carry full camera pose (`location` +
`rotation` + `fov`), per-actor 3D `keypoints_world`, paired 2D
`keypoints_image`, and `stencil_id`. The capture is a turntable — 4
static `WheelMarker` actors, orbiting camera — so each actor is a real
multi-view scene. This moved three things from "blocked" to "done", and
sharpened what remains:

1. **Camera-pose parity CERTIFIED (was the documented blocker).**
   `src/camera_from_ue_pose.py` builds the OpenCV camera from the UE pose;
   reprojecting the exported `keypoints_world` reproduces `keypoints_image`
   to **< 1e-3 px** across the whole orbit
   (`scripts/certify_ue_export_parity.py` →
   `outputs/eval3d/export_parity_v0_2.json`, `certified: true`). Certified
   convention: FOV is **horizontal**; UE world is left-handed → harness RH
   via a single **Y-negation**; forward from the `[roll, pitch, yaw]`
   rotator. Pinned by `tests/test_camera_from_ue_pose.py` against a real
   frame fixture.

2. **Harness runs on REAL geometry + multi-view scenes.**
   `scripts/make_eval3d_manifest_from_ue_v0_2.py` groups frames by actor
   into scenes (4 scenes, ~85–450 inlier frames each) and feeds real
   cameras into `eval3d_report.py`. This is the #2 harness running on real
   data, not a synthetic round-trip.

3. **The GT-error gate fires on real data — the export's A/B are still
   rim-drifted.** v0_2 maps `a`=SphereLeft / `b`=SphereRight, which sit on
   the rim (world z≈28 cm), NOT on the floor (z=0) as the 2026-05-14
   floor-ray contract (`docs/KEYPOINT_SPEC.md`) requires. Result on the
   real manifest: median **sigma 0.55 cm** (would pass <3 cm) but median
   disc-height **error 27 cm** (fails) — exactly the *systematic A/B
   drift* the GT-error gate exists to catch, now demonstrated on real
   geometry (`outputs/eval3d/disc_height_report_ue_v0_2.json`).

Provenance is explicit and load-bearing: this manifest is
`source: "real_geometry_gt2d"`, `points_source: "ue_ground_truth"`,
`ab_contract: "rim_spheres_not_floor_ray"`. Because the 2D points are
ground truth (not model predictions), `gate_status` stays
`informational` and `promotion_gate_3d` returns **insufficient_evidence**
— a green disc-height number here measures harness geometry on real data,
never model quality.

**Still blocked for a true model gate:** (a) an export that emits
floor-contact A/B (or an AR-side footprint projection), and (b)
`points_source: "model_prediction"` from running inference on the
exported images. The 3D acceptance criterion is now wired next to the 2D
KPIs in `src/model_selection_audit.py` (`disc_height_3d` block,
`--eval3d-report`), so it flips green automatically the instant both land.

```bash
# real-geometry pipeline (informational; GT-2D, a/b rim-drift):
python scripts/certify_ue_export_parity.py \
    --dataset-root /path/to/WheelsDataset_v0_2 \
    --out outputs/eval3d/export_parity_v0_2.json
python scripts/make_eval3d_manifest_from_ue_v0_2.py \
    --dataset-root /path/to/WheelsDataset_v0_2 \
    --out outputs/eval3d/frames_manifest_ue_v0_2.json
python src/eval3d_report.py \
    --manifest outputs/eval3d/frames_manifest_ue_v0_2.json \
    --out outputs/eval3d/disc_height_report_ue_v0_2.json --ransac-threshold 5.0
python src/model_selection_audit.py \
    --eval3d-report outputs/eval3d/disc_height_report_ue_v0_2.json
```

## How to run (synthetic smoke today; real when export lands)

```bash
# build a synthetic frames manifest, then:
python src/eval3d_report.py \
    --manifest outputs/eval3d/frames_manifest.json \
    --out outputs/eval3d/disc_height_report.json
```

## See also

- `docs/AR_ML_CONTRACT.md`, `docs/KEYPOINT_SPEC.md` — frozen 2D contract.
- `docs/AR_REPLAY_METRIC_PLAN.md` — staged 3D-eval plan this implements
  the ML-side simulator slice of (complementary to the device-replay
  Stage 3).
- `docs/EXPORT_PARITY_AUDIT.md` — the upstream UE export blocker.
