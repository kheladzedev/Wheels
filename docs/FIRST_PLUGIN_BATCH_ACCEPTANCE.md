# First Real Plugin Batch — Acceptance Workflow

Strict workflow for accepting the **first** real Android-plugin batch
into the ML pipeline. The synthetic smoke baseline is already done;
this document covers what happens when actual annotated frames land in
`data/incoming/android_plugin_real/`.

This workflow does **not** touch training, inference, model weights,
or any 3D / raycast / RANSAC / accumulation work — all of that is
either out of scope for ML (raycast, RANSAC, plane recovery → AR
side per `docs/AR_ML_CONTRACT.md`) or deliberately deferred until a
human has visually approved the batch.

> **Do not train on a new batch until previews have been inspected by
> a human and the batch has been explicitly marked ACCEPT_FOR_TRAINING.**
> Garbage in → silent garbage model. The smoke baseline already
> proved the plumbing learns whatever it's fed; that's exactly why the
> human-in-the-loop gate matters.

## Expected input layout

The plugin drops a batch under `data/incoming/android_plugin_real/`:

```
data/incoming/android_plugin_real/
  images/
    frame_0001.jpg
    frame_0002.jpg
    ...
  annotations/
    frame_0001.json
    frame_0002.json
    ...
  metadata/
    source_info.json        # optional, but expected
```

- Image extensions: any of `.jpg, .jpeg, .png, .bmp, .webp`.
- Stem matching is by filename (no extension): `frame_0001.jpg` ↔
  `frame_0001.json`.
- `metadata/source_info.json` records device, capture date, plugin
  build, per-batch settings. Optional but strongly preferred.

## Expected annotation JSON

One JSON per image. Schema (input side; matches plugin contract):

```json
{
  "frame_id": "frame_0001",
  "image": "frame_0001.jpg",
  "wheels": [
    {
      "bbox_xyxy": [x1, y1, x2, y2],
      "points": {
        "a":             [x, y],
        "b":             [x, y],
        "c_disc_bottom": [x, y]
      }
    }
  ]
}
```

Semantics of the three points (2026-05-14 spec revision — load-bearing):

- `points.a` — left screen-space **floor-ray point** near the wheel's
  footprint / base. AR raycasts this onto the floor plane to anchor
  the vertical wheel plane. **Not** a metal-rim edge point.
- `points.b` — right screen-space floor-ray point, mirror of A.
  **Not** a metal-rim edge point.
- `points.c_disc_bottom` — lowest visible point of the metal rim / disc.
  Not the tire, not the floor, not the tire/ground contact.

Forbidden in the annotation file: per-keypoint confidence, visibility
flag, `track_id`, `timestamp`, any 3D / world coordinates. Plugin
contract has no occlusion flag — partially occluded wheels must be
dropped at annotation time, never emitted as a partial wheel.

## Acceptance flow

Run the steps in order. If any step exits non-zero, stop, investigate,
and ping the plugin team. **Do not proceed to training until a human
has eyeballed the previews and made a written decision.**

### 1. Inspect folder structure

```bash
ls data/incoming/android_plugin_real/
ls data/incoming/android_plugin_real/images   | wc -l
ls data/incoming/android_plugin_real/annotations | wc -l
cat data/incoming/android_plugin_real/metadata/source_info.json 2>/dev/null || \
    echo "WARNING: no metadata/source_info.json (allowed but discouraged)"
```

Expect: equal counts of images and annotations; non-empty
`source_info.json` if present.

### 2. Validate incoming format

```bash
./.venv/bin/python src/check_keypoint_incoming.py \
    --source-root data/incoming/android_plugin_real
```

The validator (`src/check_keypoint_incoming.py`) enforces plugin-format
invariants: required keys, bbox order, points keys, points-inside-bbox
tolerance, valid image extensions, matching stems. Must exit 0.

### 3. Generate incoming preview (pre-conversion)

```bash
./.venv/bin/python src/preview_keypoint_annotations.py \
    --source-root data/incoming/android_plugin_real \
    --count 20
```

Writes 20 sampled overlays to `outputs/keypoint_preview/`. These
render directly on the original images so you can sanity-check the
annotation geometry **before** any conversion or training.

### 4. Manually inspect the previews

Open `outputs/keypoint_preview/` and step through every rendered image.
This is the load-bearing human gate. Confirm, on each previewed wheel:

- **bbox** covers the **full wheel including the tire** — not just the
  rim, not just the disc. If the bbox is glued to the rim and the
  tire pokes out, the annotation is wrong.
- **A / B (red / blue)** are near the **floor / base / footprint**
  of the wheel — at or just below the tire-ground contact line, in
  the lower band of the bbox. They are **not** rim-edge points and
  must not be sitting on the metal rim or up on the disc.
- **C (green)** is the **lowest visible point of the metal rim / disc**
  — on the metal, not on the tire, not on the floor.
- **Occluded wheels are not annotated.** If a wheel is half hidden
  behind a bumper / curb / other car and you can still see its
  annotation, the batch violates the contract.
- **Bounding rect is sane**: not zero-area, not flipped, not negative
  width/height, fits inside the image.

If any of these fails for a non-trivial fraction of frames, jump to
the decision step and choose REJECT or DEBUG.

### 5. Convert to YOLO-pose with the quality gate

```bash
./.venv/bin/python src/convert_keypoint_incoming_to_yolo_pose.py \
    --source-root data/incoming/android_plugin_real \
    --dataset-root data/wheel_pose_dataset \
    --source-name android_plugin_first_real_batch \
    --overwrite \
    --fail-on-quality-gate
```

Notes:

- `--source-name android_plugin_first_real_batch` namespaces the
  output filenames so this batch never collides with synthetic or
  later real batches.
- `--overwrite` clears the existing `data/wheel_pose_dataset/` before
  writing. If you want to merge with a previous batch instead, drop
  `--overwrite` (out of scope for the first batch — start clean).
- `--fail-on-quality-gate` makes the converter exit non-zero if the
  default skip / warning ratios are exceeded. We want that.

### 6. Validate the converted YOLO-pose dataset

```bash
./.venv/bin/python src/check_yolo_pose_dataset.py \
    --dataset-root data/wheel_pose_dataset
```

Enforces YOLO-pose layout invariants (14-field label lines, normalised
coordinates, train/val sane).

### 7. Generate YOLO-pose preview

```bash
./.venv/bin/python src/preview_yolo_pose_labels.py \
    --dataset-root data/wheel_pose_dataset \
    --split train \
    --count 20
```

Writes to `outputs/pose_label_preview/train/`. Cross-check against the
incoming preview from step 3 — geometry must match (the converter just
normalises coordinates, not their meaning).

### 8. Read the conversion report

```bash
cat data/wheel_pose_dataset/metadata/conversion_report.json
```

Look at:

- `skipped` / `wheels` counts.
- `quality_gate.passed`.
- `drop_reasons` distribution.

If `passed: false`, the converter has already exited non-zero (because
of `--fail-on-quality-gate`) and we never reached this step in a
healthy run.

## Decision

After steps 1–8, pick exactly one:

### `ACCEPT_FOR_TRAINING`

- All previews look correct, bbox / A / B / C geometry matches the
  semantics above, no occluded wheels are annotated.
- `check_keypoint_incoming.py` exited 0.
- Converter exited 0, quality gate passed.
- `check_yolo_pose_dataset.py` exited 0.
- YOLO-pose preview matches incoming preview.

Action: record the decision in `docs/REAL_V1_RETRAIN.md` (or the
follow-up tracker), then proceed to training as a separate task. **Do
not train from inside this workflow.**

### `REJECT_NEEDS_PLUGIN_FIX`

- Validator failed; or
- A non-trivial fraction of previews show wrong A/B geometry (e.g.
  A/B sitting on the rim instead of the floor), wrong bbox (tire cut
  off), or annotated occluded wheels.

Action: file the concrete failure (with sample image stems and
specific defect, e.g. "A on rim in 12/20 previewed frames") in
`docs/QUESTIONS_FOR_TEAM.md` and ping the plugin team. Do not
convert / train. Leave `data/incoming/android_plugin_real/` in place
for repro.

### `ACCEPT_ONLY_AS_DEBUG`

- Validator passes; previews are mostly ok but a minority is
  questionable.
- Useful for plumbing / smoke / regression testing but **not** for
  training a candidate production model.

Action: tag the converted dataset clearly (e.g. by leaving the
`source-name android_plugin_first_real_batch_debug`-suffixed copy
under `data/wheel_pose_dataset_debug/`) and explicitly call this out
in the next status report. Do not train a production candidate on
debug-only data.

## Quick reference — one-shot script

The whole non-decision part of this flow (steps 1–8) is wrapped in:

```bash
./scripts/accept_first_plugin_batch.sh
# or with an explicit source root:
./scripts/accept_first_plugin_batch.sh data/incoming/android_plugin_real
```

The script fails fast on any validation or conversion error and prints
preview paths + the conversion-report path at the end. It does **not**
train, **does not** run inference, and **does not** make the accept /
reject decision for you — that's the human's job.

## Out of scope for this workflow

- Training (`src/train_yolo.py`) — separate task after a human accept.
- Inference (`src/infer_image.py`, `src/infer_batch.py`) — separate
  task.
- 3D reconstruction, raycast, RANSAC, plane recovery, K-frame
  accumulation — these belong to the AR client per
  `docs/AR_ML_CONTRACT.md`. ML never touches them.
- Schema changes — any change to `frame_id` / `bbox_xyxy` /
  `confidence` / `points.{a, b, c_disc_bottom}` goes through
  `docs/OPEN_QUESTIONS_AR_SPEC.md` and AR-team sign-off first.

## See also

- `docs/KEYPOINT_DATASET_FORMAT.md` — full plugin input format spec.
- `docs/KEYPOINT_SPEC.md` — A/B/C definitions in detail.
- `docs/AR_ML_CONTRACT.md` — AR/ML responsibility split.
- `docs/REAL_DATA_INGESTION.md` — historical notes on earlier real
  batches and caveats.
