# Task Plan — AR Wheel Fitting (ML)

Stage-by-stage plan from current state to a detector the AR client can
integrate. Each stage has a concrete deliverable.

The AR spec
(https://docs.google.com/document/d/1HwMfJYc3eWaovN183370iWYmLjTosF9UMconj-UawFg/)
fixes the ML deliverable as one `wheel` class with 3 keypoints per
instance: `a`, `b`, `c_disc_bottom`. A/B are floor-ray points, not rim
edges; C is the lowest visible metal rim / disc point. All 3D logic
(raycast, RANSAC, plane fit, K-frame accumulation, tracking) lives on
the AR side.

## Stage 0 — Environment and smoke test  ✅

**Goal:** local pipeline runs end-to-end on synthetic data.

- Python 3.11 venv, `pip install -r requirements.txt`.
- `python src/create_sample_incoming.py --count 20 --overwrite`.
- `python src/convert_incoming_to_yolo.py --source-root data/incoming/manual_sample --dataset-root data/wheel_dataset --overwrite`.
- `python src/check_dataset.py --dataset-root data/wheel_dataset`.
- `python src/preview_labels.py --dataset-root data/wheel_dataset --split train --count 10`.
- `python src/postprocess_wheels.py --demo`.

**Done when:** all five commands succeed. Detection quality is **not**
judged here — synthetic cartoon data does not generalize.

## Stage 1 — AR-team contract closure  ✅

**Goal:** lock down the JSON contract and annotation conventions.

See `docs/OPEN_QUESTIONS_AR_SPEC.md`. Critical answers are now pinned:

- `c_disc_bottom` = visually lowest visible point of the metal rim / disc.
- A/B = floor-plane post-process points used to recover the wheel plane.
- Occluded / partially closed wheels are dropped.
- No per-keypoint confidence.
- No ML `track_id`; AR filters/associates by coordinates after raycast.
- `frame_id` only; no timestamp.
- Android first.

**Done when:** any remaining item is explicitly non-contract-critical.
Current remaining item: exact Unreal collector richness / 3D debug
metadata. It does not change the confirmed ML JSON.

## Stage 2 — Real / synthetic data collection

**Goal:** assemble images representative of the target distribution with
the 3 keypoints labelled.

- Sources: see `docs/REAL_DATA_INGESTION.md` §1.
- Variety: front/side/3-quarter, day/night, indoor/outdoor, different rim
  styles, sizes, occlusion levels.
- Target ~500–1000 wheel instances for the v1 baseline; can grow.
- Drop staging files into `data/incoming/<source_name>/` per
  `docs/ANNOTATION_JSON_FORMAT.md`. Originals stay in `data/raw/` if needed.

**Done when:** `data/incoming/` has at least one source with non-trivial
volume ready for conversion.

## Stage 3 — Annotation and conversion

**Goal:** convert raw sources into the canonical YOLO-pose dataset.

- Annotation tooling for keypoints: CVAT (built-in keypoint support),
  Label Studio (with `KeyPointLabels`), Roboflow (Pose project type).
- Custom export to the interim JSON format
  (`docs/ANNOTATION_JSON_FORMAT.md`).
- Run `convert_incoming_to_yolo.py` with the appropriate `--split-by`
  strategy (`prefix` for video frames, default for independent photos).
- Validate with `check_dataset.py` and `preview_labels.py` before training.

**Done when:**
- `python src/check_dataset.py --dataset-root data/wheel_dataset` passes.
- `python src/preview_labels.py ...` shows clean keypoint placement on a
  human spot-check of 10 random samples per split.

## Stage 4 — YOLO-pose training baseline

**Goal:** fine-tune YOLO-pose on the keypoint dataset.

- Start from `yolo11n-pose.pt`; escalate to `yolo11s-pose.pt` if recall is
  insufficient on rim-occluded wheels.
- Run:
  ```bash
  python src/train_yolo.py \
    --data configs/pose_dataset.yaml \
    --model yolo11n-pose.pt \
    --epochs 50 \
    --device mps \
    --project runs/pose \
    --name wheel_baseline
  ```
- Track curves under `runs/pose/wheel_baseline/`.

**Done when:** weights exist at `runs/pose/wheel_baseline/weights/best.pt`
and val metrics are stable. Targets (revisit after Q7 error budget lands):

- mAP50 (bbox) ≥ 0.85 on real-data val.
- OKS-based pose mAP at OKS sigma 0.1 ≥ 0.5.
- Median per-keypoint pixel error ≤ 5 px at 640×640 input.

## Stage 5 — End-to-end AR-payload evaluation  ⏳ partial

**Goal:** quantify the payload as the AR layer will see it.

- Detection: mAP50, mAP50-95, per-keypoint OKS via `model.val()`.
- AR-specific custom metrics on a held-out real set:
  - per-keypoint pixel error (separately for `rim_left`, `rim_right`,
    `disc_bottom`).
  - false-negative rate (missed wheels).
  - false-positive rate (phantom wheels).
- Compare against the error budget once Q7 is answered.
- Failure-mode catalogue: small wheels, heavy occlusion, motion blur,
  unusual rim styles.

**Status (2026-05-13):** pipeline shipped — `src/eval_keypoints.py`
covers all four metrics + bbox-area / occlusion slices + worst-N
failure samples; `scripts/eval_baseline.sh` is the one-command
wrapper; `outputs/eval/wheel_baseline_v1.json` +
`outputs/eval/wheel_baseline_v1_summary.md` exist for the baseline.
**Still blocked on the QA pass through real_v1 auto-drafts** before
the numbers become a real generalisation signal — current val
labels are themselves auto-drafts (see summary's "Caveat" section
and `docs/REAL_V1_RETRAIN.md`).

**Done when:** the same eval run on `wheel_real_v1` (post-QA)
reaches the Stage 4 targets and the failure catalogue is committed
in `outputs/eval/`.

## Stage 6 — Android-first export and integration handoff

**Goal:** AR/web team can call the model on a fresh image and get the
documented JSON without any ML-side hand-holding.

- First export target is Android (TFLite / LiteRT). ONNX/CoreML stay
  useful for debug or later platform work, but are not the first
  integration milestone.
- Document the JSON contract version (already in README) and freeze it.
- Provide either a hosted endpoint or an inference wrapper script.
- One round of end-to-end test with the AR client: their `frame_id` →
  our response with matching `frame_id` and 3 keypoints per wheel.

**Done when:** the AR team has hooked our output into their raycast +
RANSAC pipeline and is able to place virtual wheels on a real photo.

## Stage 7 — Continuous data improvement (post-MVP)

Once the v1 detector ships, the pipeline supports incremental data drops
via `data/incoming/<source_name>/` → `convert_incoming_to_yolo.py` without
code changes. Production hardening to consider:

- Active learning: surface low-confidence predictions from production for
  re-labelling.
- Hard-negative mining: spare wheels, painted murals of cars, motorcycle
  wheels.
- Domain-specific augmentation tuning once we see Unreal-vs-real
  divergence on val.
- Tracker module if Q5 flips and AR wants per-wheel stable IDs.
