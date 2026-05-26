# VSBL — AR Wheel Fitting (ML side)

ML pipeline for the AR "Примерка колес" mechanic. The ML side runs YOLO-pose
on a camera frame and returns each detected wheel with **3 keypoints** in
pixel coordinates. The AR client does raycast, RANSAC, plane reconstruction,
and K-frame accumulation — ML stays out of 3D.

See `docs/OPEN_QUESTIONS_AR_SPEC.md` for the focused list of confirmation
items from the most recent AR-team spec clarification, and
`docs/QUESTIONS_FOR_TEAM.md` for the broader open-questions list.

## Current confirmed target

Confirmed by the AR team 2026-05-13. This is the authoritative shape;
older `point_*` / `bbox_xywh` drafts are obsolete and retained only in
legacy/debug artifacts.

ML returns per-frame, per-wheel detections in pixel coordinates. AR
owns everything 3D.

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

- `frame_id` — string, echoed from AR. Load-bearing for matching with
  the camera transform AR saved at capture time.
- `wheels[].bbox_xyxy` — pixels, top-left + bottom-right corners.
- `wheels[].confidence` — wheel-level detection confidence, `[0, 1]`.
- `wheels[].points.a` / `.b` — **left / right floor-ray points**.
  Screen-space pixels that AR raycasts onto the floor plane near the
  wheel's footprint; the two floor projections define the base of the
  vertical wheel plane (recovered via RANSAC across K frames).
  **NOT** metal-rim edge points — old "rim_left / rim_right" wording
  predates the 2026-05-14 spec revision and is forbidden.
- `wheels[].points.c_disc_bottom` — lowest visible point of the metal
  rim / disc. AR raycasts onto the recovered vertical plane for disc
  height.

No `timestamp`, no `track_id`, no per-keypoint confidence, no
`visibility` flag, no 3D coordinates from ML. Partially occluded
wheels are dropped at annotation time and not emitted at inference.
Inference also drops wheels whose predicted A/B/C violate the confirmed
floor-ray geometry (A left of B, A/B in the lower bbox band, C above
the A/B floor-ray line), because the confirmed JSON has no uncertainty
field that could safely carry a "needs review" state.

**First target platform: Android** (TFLite / LiteRT). See
`docs/ANDROID_FIRST_MODEL_PLAN.md`.

Full responsibility split: `docs/AR_ML_CONTRACT.md`. Keypoint
definitions: `docs/KEYPOINT_SPEC.md`. Plugin data we expect to
ingest: `docs/PLUGIN_DATA_EXPECTATION.md`.

## Historical pre-confirmation context

Earlier drafts used `point_a` / `point_b` /
`point_c_disc_bottom`, `bbox_xywh`, `timestamp`, `visibility`, and
per-keypoint confidence. Those fields are obsolete for AR consumption.
They may still appear in legacy/debug artifacts such as
`<stem>_legacy.json`, but the deprecated `--target-schema` preview path
has been removed. The production contract is the confirmed JSON shown
above.

The important semantic revision remains: legacy literal training names
`rim_left` / `rim_right` persist in some YOLO-pose label files for
backward compatibility, but their content must now be **floor-ray A/B
points**, not metal-rim edges. Any data or model trained before that
semantic change is schema-compatible smoke only, not AR-ready.

## Current baseline (`wheel_baseline_v1`, 2026-05-13)

> **STALE for A/B evaluation.** This run, and any other run trained
> before the **2026-05-14 keypoint semantic correction**, was fitted
> against the old "rim edge" interpretation of A/B. Under the current
> contract A/B are screen-space **floor / raycast points** near the
> wheel footprint, not rim edges. Bbox numbers, confidence numbers,
> and C alone remain meaningful; A/B predictions from these runs must
> not be used to score the keypoint pipeline. The synthetic smoke
> proof for the new semantics is `runs/pose/wheel_pose_semantic_v1/`
> (see its `SEMANTICS.md`). A real-data retrain against the corrected
> contract is still pending.

First YOLO-pose model that emits the confirmed AR schema end to end.
Trained on auto-labelled data, not human-verified — treat as a
plumbing-grade baseline that exercises the pipeline, not as the
production target.

| What                      | Value |
|---------------------------|-------|
| Base architecture         | `yolo11n-pose`, 5.6 MB checkpoint |
| Training data             | 221 wheel candidates over 399 photos from `data/incoming/real_v1/` (auto-drafts from `auto_annotate_wheels.py`, YOLO + SAM-2 on Wikimedia Commons) |
| Train / val split         | 319 / 80 images |
| Epochs                    | 50, mps |
| Pose mAP50 (val)          | **0.619** |
| Pose mAP50-95 (val)       | **0.598** |
| Box mAP50 (val)           | 0.612 (TZ target ≥0.85) |
| Recall (B / P)            | 0.841 / 0.841 |
| Inference (mps, 640px)    | ~100 ms / frame |
| Weights                   | `runs/pose/wheel_baseline_v1/weights/best.pt` |
| ONNX export               | `runs/pose/wheel_baseline_v1/weights/best.onnx` (drift <2 px keypoints / <0.05 conf vs PyTorch) |
| Eval report               | `outputs/eval/wheel_baseline_v1.json` + `outputs/eval/wheel_baseline_v1_summary.md` (regen via `./scripts/eval_baseline.sh`) |
| Demo presentation         | `docs/DEMO.md` |

Reproduce:

```bash
./.venv/bin/python src/check_yolo_pose_dataset.py --dataset-root data/wheel_pose_dataset
./.venv/bin/python src/train_yolo.py \
    --data configs/pose_dataset.yaml --model yolo11n-pose.pt \
    --epochs 50 --device mps \
    --project runs/pose --name wheel_baseline_v1
./.venv/bin/python src/export_model.py \
    --model runs/pose/wheel_baseline_v1/weights/best.pt \
    --format onnx --device cpu --simplify
```

Render a demo gallery of predictions on the 30 real photos:

```bash
./.venv/bin/python scripts/build_demo_gallery.py \
    --images-dir data/manual_real/images --pattern 'real_*.jpg' \
    --model runs/pose/wheel_baseline_v1/weights/best.pt \
    --out-dir outputs/demo --device cpu
```

Full presentation guide (what to show, sound bite, blockers):
`docs/DEMO.md`.

### Known limitations

- Trained on heuristic A/B/C labels derived from SAM-2 wheel masks, not
  human verified. Box mAP50 sits at 0.61 instead of the TZ target 0.85
  because precision suffers on noisy labels — recall is already 0.84.
- Wikimedia source pool skews towards parking-lot aerial views where
  individual wheels are <40 px; ~half of those images yield zero
  detections after the size filter.
- No real frames from the Android plugin yet — there is no
  human-verified hold-out, so the metrics above are val-on-auto-labels.
- TFLite / CoreML exports blocked on the locked dep surface; ONNX
  works through ONNX Runtime Mobile on Android.

### How to push it forward

The two unblockers, in order:

1. **QA the auto-drafts** via `manual_keypoint_annotator.py
   --prefill-from data/incoming/real_v1/annotations` (review the
   132 wheels flagged `_needs_review`, drag/drop where wrong). A
   single ~40-minute pass converts the dataset to human-verified and
   makes a v2 retrain meaningful.
2. **Real plugin frames**, even 50–100. The pipeline already accepts
   them through `data/incoming/<source_name>/` per
   `docs/KEYPOINT_DATASET_FORMAT.md`.

## Repo layout

```
VSBL/
  README.md
  requirements.txt
  configs/
    dataset.yaml              # YOLO-pose config: 1 class, 3 keypoints
  data/
    raw/                      # Original, unprocessed images / videos
    incoming/                 # Source batches awaiting conversion
    wheel_dataset/            # Canonical YOLO-pose dataset (auto-generated)
    samples/                  # Demo images for quick inference
  src/
    create_sample_incoming.py # Synthetic incoming batch generator
    convert_incoming_to_yolo.py # Incoming JSON → YOLO-pose labels
    check_dataset.py          # Validate dataset layout before training
    preview_labels.py         # Render a few labelled samples
    train_yolo.py             # Train YOLO-pose on the wheel dataset
    infer_image.py            # Run inference, emit AR JSON + viz
    postprocess_wheels.py     # Pose detections → AR JSON payload
    visualize_predictions.py  # Re-render a saved AR JSON onto an image
  docs/
    KEYPOINT_SPEC.md          # A/B/C keypoint definitions (target)
    AR_ML_CONTRACT.md         # ML ↔ AR responsibility split + target JSON
    OPEN_QUESTIONS_AR_SPEC.md # Confirmation items after spec clarification
    QUESTIONS_FOR_TEAM.md     # Broader open contract questions
    ANNOTATION_JSON_FORMAT.md # Interim incoming annotation schema
    DATASET_SPEC.md           # Dataset layout / label format
    REAL_DATA_INGESTION.md    # How real batches are ingested
    TASK_PLAN.md              # Stage plan
  outputs/                    # Inference artifacts (auto-created)
```

## Setup on macOS Apple Silicon

Tested on Python 3.11.

```bash
cd ~/Desktop/VSBL
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch comes in via `ultralytics`. On Apple Silicon it ships with MPS
support — pass `--device mps` to use the GPU. `--device cpu` always works.

## Quick inference

```bash
# Drop a car photo into data/samples/, then:
python src/infer_image.py --image data/samples/car.jpg --device mps
```

Artifacts written to `outputs/`:

- `outputs/<stem>.json` — **AR-team confirmed schema (PRIMARY)**.
  Exact shape: `{frame_id, wheels[].{bbox_xyxy, confidence,
  points.{a, b, c_disc_bottom}}}`. Per the "Current confirmed target"
  section above (response 2026-05-13). This is the load-bearing output
  consumers should read.
- `outputs/<stem>_legacy.json` — Intermediate legacy payload with
  `wheel_bbox` / per-keypoint `visibility` + `confidence` / `warnings`
  / `stats` / `image` / `image_size` / `thresholds`. Debug + backward
  compatibility for tools that pre-date the confirmed schema. Not the
  AR contract.
- `outputs/<stem>_raw.json` — Flat list of raw pose detections (after
  the YOLO confidence + NMS filters). Useful for ML-side debugging only.
- `outputs/<stem>_final_pred.jpg` — AR-final view (default): wheel bbox
  + named A/B/C keypoints. Drawn from the **confirmed** payload, so any
  wheel filtered out for occlusion or invalid floor-ray geometry is not
  shown here.
- `outputs/<stem>_raw_pred.jpg` — ML-debug view (when `--viz-mode raw|both`).

To run with a fine-tuned model and explicit frame metadata:

```bash
python src/infer_image.py \
  --image data/wheel_dataset/images/val/manual_sample__sample_0000.jpg \
  --model runs/pose/wheel_baseline/weights/best.pt \
  --device mps --conf 0.25 --iou 0.45 --max-det 20 \
  --frame-id frame_001 --timestamp 1736700000.0
```

`--frame-id` defaults to the image stem (required by the confirmed
schema — see `determine_frame_id` in `src/infer_image.py`).
`--timestamp` defaults to wall-clock at inference start and only lands
in `<stem>_legacy.json`; the confirmed schema drops it. In production
the AR client supplies the explicit `--frame-id`.

> **Migration note.** Before 2026-05-13 `<stem>.json` carried the
> legacy/transitional shape. After the AR team confirmed the contract,
> `<stem>.json` is the confirmed schema and the legacy shape moved to
> `<stem>_legacy.json`. Consumers should switch to `<stem>.json` and
> the confirmed field names — see `tests/test_confirmed_ar_schema_shape.py`
> and `tests/test_ar_contract.py` for the authoritative invariants.

### Inference thresholds

| Flag         | Default | Meaning |
|--------------|---------|---------|
| `--conf`     | `0.25`  | Minimum wheel-level detection confidence. |
| `--iou`      | `0.45`  | NMS IoU threshold. |
| `--max-det`  | `20`    | Hard cap on detections per image. |

A `--conf 0.25` run cannot produce a candidate below 0.25 — three layers
enforce it (`predict()` filter, manual re-filter in `infer_image.py`, final
`assert`).

If `infer_image.py` reports `Detections kept: 0`, the model is undertrained,
the image is out of distribution, or you pointed it at a non-pose checkpoint.

### Pose model warning

`infer_image.py` checks `model.task == "pose"` after loading and prints a
warning if not. A COCO-pretrained `yolo11n.pt` is a *detect* model, has no
keypoint head, and will produce empty `keypoints` arrays. Use a `-pose`
checkpoint or a model fine-tuned by `train_yolo.py` (which loads pose
weights by default).

## End-to-end smoke test (no real data needed)

The synthetic incoming generator + converter exercise the full ingestion
path. The synthetic images are cartoon cars with randomised yaw / tilt.
Per the 2026-05-14 contract revision, the literal label strings
`rim_left` / `rim_right` now carry floor / raycast semantics — A / B
sit in the lower band of each wheel bbox near the tyre footprint,
not on the rim. `disc_bottom` stays on the rim's lowest visible point.

```bash
# 1. Generate 20 synthetic incoming images + JSON annotations with 3 keypoints each
python src/create_sample_incoming.py --count 20 --overwrite

# 2. Convert to canonical YOLO-pose layout
python src/convert_incoming_to_yolo.py \
  --source-root data/incoming/manual_sample \
  --dataset-root data/wheel_dataset \
  --overwrite

# 3. Validate the resulting dataset (checks pose label format: 14 fields/line)
python src/check_dataset.py --dataset-root data/wheel_dataset

# 4. Tiny training run to confirm train_yolo.py works end-to-end
python src/train_yolo.py \
  --data configs/pose_dataset.yaml \
  --model yolo11n-pose.pt \
  --epochs 3 \
  --device mps \
  --project runs/pose \
  --name wheel_smoke
```

> **Synthetic dataset is NOT a quality signal.** Cartoon cars don't
> generalize. The point is to validate wiring before the team hands over a
> real labelled batch.

## First real plugin batch acceptance

For raw Unreal/plugin exports shaped like the `0002` trial
(`Images/`, `keyPoint/`, optional `Ground/`), use the raw-export
acceptance runner first. It inspects the raw files, imports them into
the confirmed plugin JSON contract, validates, converts to YOLO-pose,
renders previews, and writes a single report.

```bash
python scripts/accept_unreal_export.py \
  --source-root ~/Downloads/0002 \
  --source-name unreal_0002_trial \
  --overwrite

# Optional one-epoch smoke train after the gates pass:
python scripts/accept_unreal_export.py \
  --source-root ~/Downloads/0002 \
  --source-name unreal_0002_trial \
  --overwrite \
  --smoke-train --device mps
```

Artifacts land under
`outputs/unreal_export_acceptance/<source-name>/`:

- `acceptance_report.md` / `.json` — counts, drop reasons, paths, status.
- `inspection/` — raw keyPoint status report and raw overlays.
- `inspection/previews/by_status/` — status-specific galleries for
  invalid objects (`OUT_OF_BOUNDS`, `PARTIAL_ZERO`, `EMPTY_ALL_ZERO`, etc.).
- `incoming/` — imported `images/annotations/metadata` contract.
- `pose_dataset/` — converted YOLO-pose dataset.
- `previews/incoming/` and `previews/pose/train/` — visual review gates.
- `logs/` — per-step stdout/stderr logs.

The acceptance report has two separate gates:

- **Technical status** — does the raw archive parse, import, validate,
  convert, preview, and optionally smoke-train?
- **ML data-quality gate** — does the archive look clean enough to train
  without first fixing the exporter? Defaults require `usable_ratio >= 0.60`,
  `invalid_required_ratio <= 0.20`, `bad_geometry_ratio <= 0.15`, and low
  bbox fallback / empty-label rates. A trial can have technical `PASS` while
  still being `NOT_APPROVED_FOR_TRAINING_DATA_QUALITY_GATE_FAILED`.

When the **first** real Android-plugin batch lands, run the dedicated
acceptance workflow before anything else. It validates the incoming
format, renders previews, runs the YOLO-pose converter with the quality
gate enforced, and stops there. **It does not train.**

```bash
# Default source root: data/incoming/android_plugin_real
./scripts/accept_first_plugin_batch.sh

# Or with an explicit path:
./scripts/accept_first_plugin_batch.sh data/incoming/<batch_dir>
```

The script chains:

```
check_keypoint_incoming.py            -> incoming format invariants
preview_keypoint_annotations.py       -> outputs/keypoint_preview/
convert_keypoint_incoming_to_yolo_pose.py --fail-on-quality-gate
check_yolo_pose_dataset.py            -> YOLO-pose layout invariants
preview_yolo_pose_labels.py           -> outputs/pose_label_preview/train/
```

> **Do not train until a human has inspected both preview folders.**
> Confirm bbox covers the full wheel, A/B are screen-space floor-ray
> points near the wheel footprint (not rim edges), C is the lowest
> visible point of the metal disc, and no occluded wheels are
> annotated. Then mark the batch `ACCEPT_FOR_TRAINING`,
> `REJECT_NEEDS_PLUGIN_FIX`, or `ACCEPT_ONLY_AS_DEBUG` per the
> decision rubric in `docs/FIRST_PLUGIN_BATCH_ACCEPTANCE.md`.

Full step-by-step, expected layout, and the failure-mode rubric:
`docs/FIRST_PLUGIN_BATCH_ACCEPTANCE.md`.

### NeuralData1 Unreal-project capture

Igor's `NeuralData1` handoff is a full Unreal project, not a training
dataset until the plugin writes non-empty `Images/` and `keyPoint/`
folders. Use the capture wrapper to copy only the generated export
folders and run the raw-export acceptance pipeline:

```bash
python scripts/accept_neuraldata1_capture.py \
  --overwrite
```

The wrapper refuses empty captures, leaves the legacy 3D quarantine
folder out of training, and writes `capture_report.md` / `.json` next
to the usual acceptance report. It never marks a batch training-ready
unless the automated gates pass and `--human-preview-accepted` is used
after visual review.

Full workflow: `docs/NEURALDATA1_CAPTURE_WORKFLOW.md`.

## Convert real incoming annotations

When a real batch arrives, place it under `data/incoming/<source_name>/`
following `docs/ANNOTATION_JSON_FORMAT.md`:

```
data/incoming/<source_name>/
  images/image_001.jpg
  annotations/image_001.json
  metadata/                   # optional
```

Example annotation (literal label strings still `rim_left` /
`rim_right` for backward compat with the legacy converter; the
*content* follows the 2026-05-14 floor-ray semantics for A/B):

```json
{
  "image": "image_001.jpg",
  "objects": [
    {
      "class_name": "wheel",
      "bbox_xyxy": [100, 220, 200, 320],
      "keypoints": [
        {"name": "rim_left",    "xy": [130, 308], "visibility": 2},
        {"name": "rim_right",   "xy": [170, 308], "visibility": 2},
        {"name": "disc_bottom", "xy": [150, 290], "visibility": 2}
      ]
    }
  ]
}
```

Then run the converter:

```bash
python src/convert_incoming_to_yolo.py \
  --source-root data/incoming/<source_name> \
  --dataset-root data/wheel_dataset \
  --overwrite

python src/check_dataset.py --dataset-root data/wheel_dataset
python src/preview_labels.py --dataset-root data/wheel_dataset --split train --count 10
```

> **Split-strategy caveat:** the converter does a random per-image split.
> Unsafe for video frames or repeated shots of the same car. For
> production, pre-group by scene/car upstream, or extend the converter —
> see `docs/REAL_DATA_INGESTION.md` §6.

## Train on the dataset

```bash
python src/train_yolo.py \
  --data configs/pose_dataset.yaml \
  --model yolo11n-pose.pt \
  --epochs 50 \
  --device mps \
  --project runs/pose \
  --name wheel_baseline
```

`train_yolo.py` now runs the confirmed floor-ray dataset preflight before
constructing YOLO. Legacy rim-edge datasets fail fast instead of starting
training. It also guards against accidentally passing a non-pose checkpoint:
keypoint training requires `-pose` weights (`yolo11n-pose.pt`,
`yolo11s-pose.pt`, etc.).

Outputs land in `runs/pose/wheel_baseline/`. The best weights are at
`runs/pose/wheel_baseline/weights/best.pt`.

```
runs/pose/wheel_baseline/
  weights/
    best.pt           # Best epoch by val mAP — use this for inference
    last.pt           # Last epoch — for resuming
  results.png / results.csv
  confusion_matrix.png
  args.yaml
  train_batch*.jpg / val_batch*.jpg
```

`--project` is resolved to an absolute path, so outputs never leak into
Ultralytics' global `~/runs/...` directory.

> **Repeated runs:** if `runs/pose/wheel_baseline/` already exists,
> Ultralytics appends an auto-incrementing suffix (`wheel_baseline2`, etc.).
> Delete the directory or pass a different `--name` to overwrite.

## Re-render a saved JSON

```bash
python src/visualize_predictions.py \
  --image data/samples/car.jpg \
  --json outputs/car.json
```

Renders wheel bboxes and the 3 keypoints with names and per-keypoint
confidence.

## Export for AR / Web / iOS / Android

Convert a trained checkpoint to a target runtime format and verify that
the exported model still predicts the same wheels and keypoints as the
PyTorch original. The exported file lands next to the input `.pt`
(override with `--out-dir`).

```bash
# ONNX (the lightest, no extra deps)
python src/export_model.py \
  --model runs/pose/wheel_v3/weights/best.pt --format onnx --device cpu

# CoreML (iOS)
python src/export_model.py \
  --model runs/pose/wheel_v3/weights/best.pt --format coreml

# TFLite (Android)
python src/export_model.py \
  --model runs/pose/wheel_v3/weights/best.pt --format tflite --int8
```

Final production format is pending Q10 in `docs/QUESTIONS_FOR_TEAM.md`
(target platforms + order).

After export, the script reloads both the `.pt` and the exported model,
runs inference on one image (defaults to the first val image; override
with `--sample-image`), and compares numerically. Tolerances are
deliberately loose because quantization drifts a few px: 2 px on bbox
xyxy, 3 px on keypoint xy, 0.05 absolute on detection confidence. A
count mismatch (PT finds N wheels, exported finds M) is also a failure.
Exit code is non-zero on any tolerance miss. Skip with `--no-sanity`.

> **TFLite needs `tensorflow` in the venv.** If
> `model.export(format='tflite')` fails with an import error, install
> TF yourself (we do not auto-install) — see Ultralytics' export docs
> for the version Ultralytics expects.

## Keypoint incoming format for Android plugin

The Android collection plugin drops batches into
`data/incoming/android_plugin/` following the format documented in
`docs/KEYPOINT_DATASET_FORMAT.md` (`frame_id` + `image` +
`wheels[].bbox_xyxy` + `wheels[].points.{a, b, c_disc_bottom}`).
Three helper scripts manage that flow:

```bash
# Generate a synthetic batch (smoke-test the format end-to-end)
python src/create_sample_keypoint_incoming.py --count 50 --overwrite

# Validate a real plugin batch (or the synthetic one above)
python src/check_keypoint_incoming.py --source-root data/incoming/android_plugin

# Render bbox + A/B/C overlay on 10 random samples
python src/preview_keypoint_annotations.py \
  --source-root data/incoming/android_plugin --count 10

# Open the preview directory (macOS)
open outputs/keypoint_preview
```

- The generator produces 2- or 4-wheeled cars with the bbox + 3
  keypoints on each wheel. Per the 2026-05-14 contract revision, the
  generator places `a` / `b` in the lower band of each wheel bbox
  near the tyre footprint (floor / raycast semantics), and
  `c_disc_bottom` at the lowest visible point of the metal rim. The
  bundle is intended for plumbing smoke-tests only, not for training
  — cartoon geometry doesn't generalise.
- The validator checks: matching image↔annotation stems, required
  fields, `points` dict has exactly `{a, b, c_disc_bottom}`,
  coordinates inside the image, points within bbox ±5 px tolerance,
  bbox order `x1 < x2` / `y1 < y2`. Exits non-zero if any ERROR
  fires. WARNINGs (e.g. orphan annotations, point just outside bbox)
  do not fail.
- The previewer draws orange bbox + red A / blue B / green C
  circles with labels, matching the AR mock-spec board.

See `docs/KEYPOINT_DATASET_FORMAT.md` for the full schema and rules.

## Auto-annotate real photos with a foundation-model pipeline (pre-label)

`src/auto_annotate_wheels.py` produces a plugin-format draft bundle from
unannotated real photos by combining a COCO-pretrained vehicle detector
(`yolo11n.pt`, already in the repo) with **SAM 2** mask prompts and a
geometric postprocess for A/B/C. This is a 2026-style human-in-the-loop
pre-label: foundation model proposes, human reviews. Every emitted JSON
carries `_draft: true` and `_warning: "NOT_GROUND_TRUTH_REQUIRES_HUMAN_REVIEW"`,
and per-wheel `_needs_review: true` (with `_review_reasons[]`) when the
heuristic looks shaky.

The pipeline reuses only the dependency surface already allowed
(`ultralytics` + stdlib + numpy + opencv). It is intentionally **not** a
trained wheel detector — drafts must be reviewed in
`src/manual_keypoint_annotator.py` before training.

```bash
# 1. Generate a draft bundle from real photos. SAM 2 weights auto-download
#    on first run (~160 MB). On Apple Silicon use --device mps.
./.venv/bin/python src/auto_annotate_wheels.py \
    --images-dir   data/manual_real/images \
    --output-root  data/incoming/manual_real_auto \
    --device       mps \
    --overwrite

# 2. Validate the draft against the plugin contract.
./.venv/bin/python src/check_keypoint_incoming.py \
    --source-root data/incoming/manual_real_auto

# 3. Render bbox + A/B/C overlays for visual sanity.
./.venv/bin/python src/preview_keypoint_annotations.py \
    --source-root data/incoming/manual_real_auto \
    --count       5 \
    --output-root outputs/manual_real_auto_preview

# 4. Hand-correct flagged wheels and any false positives.
./.venv/bin/python src/manual_keypoint_annotator.py \
    --images-dir       data/incoming/manual_real_auto/images \
    --annotations-dir  data/incoming/manual_real_auto/annotations \
    --output-root      data/incoming/manual_real_reviewed
```

Notable knobs (defaults in the source):

- `--detect-conf` — vehicle detector confidence floor (`0.25`).
- `--drop-conf`   — wheel pseudo-confidence floor below which the wheel
  is silently dropped (`0.20`).
- `--review-conf` — wheels with pseudo-confidence below this are kept
  but flagged `_needs_review` (`0.50`).
- `--device`      — `mps` / `cuda` / `cpu`.

Hard filters applied to every SAM-2 mask candidate before it can become
a wheel (any failure → silent drop, no flag). Thresholds tuned against
`docs/KEYPOINT_SPEC.md`:

| Filter | Threshold | Rejects |
|---|---|---|
| Mask area | ≥ 1500 px (`HARD_MIN_MASK_PX`) | LED slivers, badges, distant reflected wheels |
| Min bbox side | ≥ 40 px (`MIN_BBOX_SIDE_PX`) | thin headlight strips, intake-grille shadows |
| Bbox aspect | 0.55–1.8 (`ASPECT_HARD_LO/HI`) | bumpers, door handles, mirror strips |
| Mask area / vehicle area | ≤ 13 % (`WHEEL_AREA_MAX_FRACTION_OF_VEHICLE`) | whole-car-body masks when SAM 2 over-segments |
| Centroid Y inside vehicle bbox | ≥ 55 % down (`CENTROID_MIN_FRACTION`) | grilles, fog lights, mid-height trim |
| Centroid Y inside full frame | ≥ 45 % down (`IMAGE_CENTROID_MIN_FRACTION`) | wheels of background vehicles whose bbox sits in the upper frame (auto-show stands, reflected cars) |
| Circularity 4πA/P² | ≥ 0.62 (`MIN_CIRCULARITY`) | headlight ovals, license plate rectangles, half-moon masks of partially occluded wheels (the spec's "occluded wheels are dropped" rule) |
| Mean BT.601 luminance inside mask | ≤ 130 (`MAX_TIRE_BRIGHTNESS`) | chrome, lit headlights, bright body paint, white plates |

A/B/C geometry inside `keypoints_from_mask` (matches `docs/KEYPOINT_SPEC.md`):

- **A** — leftmost mask pixel in the bottom 10 % band of the bbox (floor-ray near tyre footprint).
- **B** — rightmost mask pixel in the same band.
- **C** — central column's lowest mask pixel, then shifted up by **17.5 % of bbox height** (`C_OFFSET_FRACTION = 0.175`). The fraction follows the standard rim-to-tyre radius ratio (0.65), placing C on the metal disc rather than on the rubber sidewall.

Soft flags (kept in the bundle but marked `_needs_review` with
`_review_reasons`):

- `low_detector_conf`, `mask_small` (< 2500 px), `mask_touches_edge`,
  `bbox_touches_edge`, `extreme_aspect` (close to the hard band),
  `small_bbox` (< 60 px on a side), `low_circularity` (< 0.75),
  `light_mask` (> 110 luma).

Honest limits:

- The heuristic A/B (floor-ray) and C (disc-bottom) approximations are
  pixel-accurate only when the mask is clean and side-on. Three-quarter
  and front-view wheels get reasonable bboxes but A/B/C may drift by
  several pixels — those wheels are typically auto-flagged `_needs_review`.
- Reflections in shop windows / car bodies still slip through when the
  reflection itself reads as a vehicle to the COCO detector. Hand-remove
  during review.
- The hard filters above cut a lot of FPs but also drop some legitimate
  wheels under partial occlusion or unusual angles. If recall matters
  more than precision for your pass, loosen `MIN_CIRCULARITY`,
  `MAX_TIRE_BRIGHTNESS`, and `CENTROID_MIN_FRACTION` and re-run.
- Bundles produced by this script are not training-ready until a human
  passes them through `manual_keypoint_annotator.py`.

## Convert an Android-plugin batch into a YOLO-pose dataset

Once a plugin batch validates clean, `convert_keypoint_incoming_to_yolo_pose.py`
materialises a canonical YOLO-pose dataset at `data/wheel_pose_dataset/`.
Unlike `convert_incoming_to_yolo.py` (which consumes the legacy
`manual_sample` format with named keypoints + per-kp visibility), this
converter speaks the plugin contract directly: `wheels[].bbox_xyxy` +
`wheels[].points.{a, b, c_disc_bottom}`. Every kept point is emitted
with YOLO visibility `v=2`.

```bash
# 1. Generate a synthetic plugin batch (use real data once it lands).
python src/create_sample_keypoint_incoming.py --count 50 --overwrite

# 2. Convert plugin batch → YOLO-pose dataset.
python src/convert_keypoint_incoming_to_yolo_pose.py \
  --source-root data/incoming/android_plugin \
  --dataset-root data/wheel_pose_dataset \
  --overwrite

# 3. Validate the resulting dataset (label fields, class id, ranges).
python src/check_yolo_pose_dataset.py --dataset-root data/wheel_pose_dataset

# 4. Render bbox + A/B/C overlays to outputs/pose_label_preview/.
python src/preview_yolo_pose_labels.py \
  --dataset-root data/wheel_pose_dataset --split train --count 10

# 5. Open the preview directory (macOS).
open outputs/pose_label_preview
```

Output layout:

```
data/wheel_pose_dataset/
  images/{train,val}/<source_name>__<stem>.<ext>
  labels/{train,val}/<source_name>__<stem>.txt   # 14 fields/line
  metadata/split_manifest.json
  metadata/conversion_report.json
```

Label line:

```
<class_id=0> <cx> <cy> <w> <h> <a_x> <a_y> 2 <b_x> <b_y> 2 <c_x> <c_y> 2
```

All bbox / keypoint coordinates are normalized to `[0, 1]`. The matching
training config is `configs/pose_dataset.yaml` (1 class `wheel`,
`kpt_shape: [3, 3]`, `flip_idx: [1, 0, 2]`).

### Quality gate for real batches

The converter computes a per-batch quality gate. Defaults — `5%` of source
images may be skipped (`--max-skip-ratio 0.05`) and `10%` warnings per
source image (`--max-warning-ratio 0.10`). Without `--fail-on-quality-gate`
the converter prints a `WARNING` but exits `0`; with the flag, a failed
gate exits `1` and the report is still written to
`<dataset-root>/metadata/conversion_report.json`.

`conversion_report.json` carries `source_images`, `converted_images`,
`skipped_images`, `skipped_ratio`, `warnings_count`, `warnings_ratio`,
and a `quality_gate` block (`max_skip_ratio`, `max_warning_ratio`,
`passed`, `reasons`).

When a real plugin batch lands, run the chain in order — incoming check,
incoming preview, then the converter with the gate flag and a unique
`--source-name` so the batch doesn't clobber prior data:

```bash
python src/check_keypoint_incoming.py --source-root data/incoming/android_plugin
python src/preview_keypoint_annotations.py --source-root data/incoming/android_plugin --count 20
python src/convert_keypoint_incoming_to_yolo_pose.py \
  --source-root data/incoming/android_plugin \
  --dataset-root data/wheel_pose_dataset \
  --source-name android_plugin_first_batch \
  --overwrite \
  --fail-on-quality-gate
```

Train with:

```bash
python src/train_yolo.py \
  --data configs/pose_dataset.yaml \
  --model yolo11n-pose.pt \
  --epochs 50 --device mps \
  --project runs/pose --name wheel_plugin_baseline
```

## Manual real-photo sanity check

While we wait for the Android plugin to start sending real batches,
`src/manual_keypoint_annotator.py` lets you bootstrap a tiny real-photo
dataset in the same on-disk shape the plugin would produce. Take 10–30
phone shots of cars (different distances, angles, wheel types), drop
them under `data/manual_real/images/`, then click through them.

Click sequence per wheel (semantic revision 2026-05-14 — A/B are
**floor / raycast points**, not rim edges; the bbox must enclose the
**full wheel including tire**):

1. **bbox corner 1** — any corner of the wheel bbox covering the
   entire wheel (tire + rim). The annotator normalises to
   `x1 < x2 ∧ y1 < y2`.
2. **bbox corner 2** — the opposite corner.
3. **A — left floor/raycast point.** Левая screen-space точка на
   полу / основании около колеса (рядом с footprint колеса). AR
   raycast-ит её на плоскость пола. **НЕ** на металлическом диске,
   **НЕ** на резине.
4. **B — right floor/raycast point.** Правая screen-space точка на
   полу / основании около колеса. AR raycast-ит её на плоскость
   пола. **НЕ** на металлическом диске, **НЕ** на резине.
5. **C — c_disc_bottom.** Нижняя видимая точка металлического
   обода / диска. **НЕ** резина, **НЕ** пол, **НЕ** точка касания
   шины с землёй.

Keys:

- `n` / `Enter` — save the wheels staged so far and advance to next image
- `a` — add another wheel on the same image (after 5 clicks complete one)
- `r` — reset the in-progress wheel (clear the last unfinished clicks)
- `s` — skip image (saves `wheels: []` — useful for "no wheel visible")
- `q` — quit; the bundle is finalised on exit

Annotations save incrementally — quitting mid-batch never loses prior
work, and re-running the annotator skips any image that already has a
JSON in `--annotations-dir` (override with `--rerun`).

```bash
# 1. Drop photos here, then annotate.
python src/manual_keypoint_annotator.py \
  --images-dir       data/manual_real/images \
  --annotations-dir  data/manual_real/annotations \
  --output-root      data/incoming/manual_real \
  --start-index      0

# 2. Validate the incoming bundle (plugin-format checker).
python src/check_keypoint_incoming.py --source-root data/incoming/manual_real

# 3. Visually inspect the annotations on the original photos.
python src/preview_keypoint_annotations.py \
  --source-root data/incoming/manual_real --count 20
# → outputs/keypoint_preview/

# 4. Convert into the canonical YOLO-pose dataset (quality gate ON).
python src/convert_keypoint_incoming_to_yolo_pose.py \
  --source-root data/incoming/manual_real \
  --dataset-root data/wheel_pose_dataset \
  --source-name manual_real_smoke \
  --overwrite --fail-on-quality-gate

# 5. Validate the YOLO-pose label format.
python src/check_yolo_pose_dataset.py --dataset-root data/wheel_pose_dataset

# 6. Visually inspect YOLO-pose labels (bbox + A/B/C overlay).
python src/preview_yolo_pose_labels.py \
  --dataset-root data/wheel_pose_dataset --split train --count 20
# → outputs/pose_label_preview/train/
```

The annotator produces the exact plugin-contract JSON shape
(`frame_id`, `image`, `wheels[].{bbox_xyxy, points.{a, b, c_disc_bottom}}`)
plus `metadata/source_info.json` (`source_name: manual_real`, free-form
note, `annotation_method: manual clicks`). Anything downstream that
consumes a plugin batch consumes this bundle transparently.

> **A/B semantic note.** Under the 2026-05-14 spec revision, A/B are
> floor-ray points (raycast sources onto the floor plane for
> wheel-plane recovery). Bundles annotated *before* that date — even
> via this annotator — used the old "rim edge" wording and need to
> be re-clicked. See `docs/KEYPOINT_SPEC.md` for the canonical
> definition.

> **Status caveat.** Manual smoke data is for sanity-checking the full
> ingestion → preview → convert → training-format chain on real
> imagery, **not** a substitute for the plugin's labelled batch.
> Detection quality on 10–30 hand-clicked wheels is not a production
> signal — it just tells you nothing in the pipeline blows up when fed
> real (non-cartoon) pixels.

## Auto-draft annotations are not ground truth

`src/auto_draft_keypoint_annotations.py` exists for one narrow case: you
have real photos in `data/manual_real/images/`, no human-clicked labels
yet, and you want to push something through the rest of the manual-real
pipeline (validator → preview → converter → YOLO-pose preview) just to
make sure nothing blows up on non-cartoon pixels. **That is the only
thing this script is for.**

It does **not** run a model. It does **not** detect wheels. It picks
two synthetic wheel positions in the lower third of every image whose
filename hints at a car (`sboku`, `side`, `avto`, `car`, `mashin`,
`vid`, `wheel`, `koleso`) and emits `wheels: []` for the rest. The
geometry is fixed, identical across images, and obviously wrong as soon
as you eyeball the preview.

Every drafted annotation is flagged in two places:

- The per-image JSON carries `"_draft": true` and
  `"_warning": "NOT_GROUND_TRUTH_REQUIRES_HUMAN_REVIEW"`.
- `metadata/source_info.json` carries
  `"annotation_method": "auto_draft_heuristic"` and the same warning.

```bash
# 1. Draft a bundle (NOT ground truth).
python src/auto_draft_keypoint_annotations.py \
  --images-dir   data/manual_real/images \
  --output-root  data/incoming/manual_real_draft \
  --overwrite

# 2. Sanity-check the plugin format.
python src/check_keypoint_incoming.py \
  --source-root data/incoming/manual_real_draft

# 3. Inspect the drafted A/B/C overlays on the originals.
python src/preview_keypoint_annotations.py \
  --source-root data/incoming/manual_real_draft --count 20
# → outputs/keypoint_preview/
```

> **Do not train on this bundle.** Auto-drafted annotations cannot
> teach the model where wheels are; they will actively poison training
> with ~constant wrong A/B/C locations. The intended workflow is:
> draft → preview → discard the draft → re-annotate properly with
> `src/manual_keypoint_annotator.py` (or wait for the plugin batch).

## Postprocess in isolation

The pose-payload builder is independent of YOLO:

```bash
python src/postprocess_wheels.py --demo
```

Prints a worked example with two wheels, one fully visible, one with an
occluded `rim_right` (visibility=1) and a hidden `disc_bottom` (visibility=0).
