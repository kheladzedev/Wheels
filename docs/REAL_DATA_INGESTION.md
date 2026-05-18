# Real Data Ingestion — wheel + keypoints

How real and synthetic data flows from raw sources into the training-ready
`data/wheel_dataset/` layout used by `configs/dataset.yaml`. Operational
complement to `docs/DATASET_SPEC.md` (on-disk format) and
`docs/TASK_PLAN.md` (stage-by-stage plan).

## 1. Accepted data sources

We treat all sources uniformly downstream — each ends up in YOLO-pose
format under `data/wheel_dataset/`. Differences live only at the ingestion
stage.

| Source              | What we get                                            | Notes |
|---------------------|--------------------------------------------------------|-------|
| Real photos         | JPG/PNG of cars in the wild, dealership, studio        | Highest distribution match for production. Preferred. |
| Real video frames   | MP4 sampled at low FPS to avoid near-duplicates        | Cheap volume. Mind the duplicate-frame rule. |
| Unreal renders      | Synthetic RGB + engine-side annotations                | Easy to scale; can yield 3D keypoints essentially free — see §7. |
| 3D scans / renders  | Studio renders of specific rims and tires              | Best for rim-detail variety; backgrounds may be unrealistic. |
| Public datasets     | COCO car crops, Stanford Cars, Roboflow community sets | Usable only if license allows commercial training. Public labels never include our 3 keypoints — re-annotate. |

For each source, record origin and licensing in `metadata/SOURCE.md`.
Future model cards will need it.

## 2. Required annotations

The model learns one class with three keypoints. Required per wheel:

- `wheel` bbox covering the whole tire + rim.
- 3 keypoints in fixed order: `a`, `b`, `c_disc_bottom`.

There is no separate "MVP" annotation tier anymore — the AR spec requires
keypoints, and the model is trained with keypoints from day one. Bbox-only
annotations are not directly usable.

Annotator-facing rules live in `docs/ANNOTATION_GUIDELINES.md`; labelling
tool setup (CVAT project config, COCO → interim-JSON path) lives in
`docs/ANNOTATION_TOOLING.md`.

No per-keypoint confidence or visibility flag is part of the confirmed
plugin/ML contract. Occluded wheels are omitted entirely. Instance masks are
still useful for debugging or future experiments, but they are not required
by the current training path.

## 3. Incoming data layout

Every batch lands in its own subtree under `data/incoming/`. Nothing inside
`data/incoming/` is read by the trainer — it is a staging area.

```
data/incoming/<source_name>/
  images/         # raw images (or extracted video frames)
  annotations/    # raw annotations as exported by the source tool
  metadata/       # SOURCE.md, license, capture notes, camera info, etc.
```

`<source_name>` is a short slug, e.g. `dealership_munich_2026_03`,
`unreal_rim_pack_v2`, `roboflow_wheels_public`. Keep it stable.

`annotations/` is format-agnostic at the incoming stage. Conversion to
the canonical incoming JSON (see `docs/ANNOTATION_JSON_FORMAT.md`) and then
to YOLO-pose happens in separate steps.

## 4. Conversion target

`src/convert_incoming_to_yolo.py` writes:

```
data/wheel_dataset/
  images/train/    images/val/
  labels/train/    labels/val/
  metadata/split_manifest.json
  metadata/conversion_report.json
```

Conversion rules:

- One image per file (videos must already be sampled to frames upstream).
- Filename pattern: `<source_name>__<original_stem>.<ext>` — keeps stems
  globally unique and survives splitting.
- Labels in 14-field YOLO-pose format per `docs/DATASET_SPEC.md`.
- Class ID is fixed: `0 = wheel`.
- After conversion run `check_dataset.py` and `preview_labels.py` before
  training.

If a source cannot be auto-converted, escalate. Do not paper over with
one-off scripts that won't survive a second batch.

## 5. Data quality rules

Applied at ingestion / conversion. Failures are dropped or sent back, not
silently included.

- **Camera angle:** side or 3-quarter views preferred. Pure head-on or rear
  shots are de-prioritized — wheels are heavily foreshortened and AR rarely
  cares about those angles.
- **Wheel visibility:** ≥ 50% of the rim/disc unoccluded for at least one
  wheel.
- **Blur:** no extreme motion blur or out-of-focus shots.
- **Lighting:** mix daylight, overcast, indoor showroom, golden hour, dusk.
  Avoid an all-studio dataset.
- **Car variety:** sedan, SUV, hatchback, coupe, pickup, sports car. Avoid
  building a single-brand dataset.
- **Wheel variety:** different rim styles (multi-spoke, mesh, deep-dish,
  black/silver/painted), different sizes, with and without dirt.
- **Near-duplicates:** when sampling video, target ≤ 1 frame per second and
  reject perceptual-hash collisions.
- **Resolution:** drop images with the longer side below 480 px — too small
  to label rim keypoints reliably.
- **PII:** blur faces, license plates, and other identifying details if the
  source license requires it.

## 6. Train / val split rule

**Never split by random shuffle of individual frames** when the source has
upstream structure. It leaks information between splits and inflates
apparent metrics.

Split by the *upstream unit* — the coarsest grouping that guarantees no
shared visual content between splits:

- **Video:** split by **scene / clip**. All frames from one clip → one split.
- **Photo sessions:** split by **car**. All shots of one car → one split.
- **Synthetic batches:** split by **scene seed / generation batch**. All
  renders sharing background, lighting setup, or vehicle instance → one
  split.
- **Public datasets:** respect their upstream split if one exists.

A short note `metadata/split_strategy.md` per `<source_name>` records the
exact unit used. Recommended ratio: **80 / 20 train / val**.

> Current converter implements two strategies: `random_per_image` (default,
> safe only for independent single-car photos) and `prefix` (groups
> filenames by a configurable stem-prefix, e.g. `scene_001_*`). For sources
> that don't fit those, escalate before ingestion.

## 7. Unreal-specific expected outputs

The Unreal pipeline can give us much richer annotations than human
labellers. Define the export schema up front.

For every rendered frame, the Unreal export should produce:

**Required:**
- `rgb` image — PNG or high-quality JPG.
- `wheel` 2D bboxes.
- 3 keypoints per wheel in the canonical order (`a`, `b`,
  `c_disc_bottom`): A/B are floor-ray points, C is the lowest visible
  metal-rim/disc point.

**Strongly preferred** (cheap from Unreal, expensive to re-derive):

- Camera intrinsics (`fx, fy, cx, cy`) and extrinsics (camera-to-world).
- **3D world positions** of each of the 3 keypoints. We project them to 2D
  for label files; the 3D positions stay in the incoming JSON for future
  evaluation or weak supervision experiments. They are not emitted by ML
  inference.
- 3D world position of the wheel hub (centre of rotation) — useful as a
  derived signal.

If 3D keypoint positions are available, we can synthesize keypoint labels
without any human-in-the-loop annotation. This is the single largest
labour-saving in the pipeline and the reason Unreal is high-value.

**Optional:**
- Per-wheel orientation (axis-angle or quaternion).
- Wheel radius in world units.
- Material / rim style ID — for stratified evaluation across rim styles.

Each frame's annotations live alongside the image:
`frame_000123.json` next to `frame_000123.png`. The conversion step
extracts bboxes + 3 keypoints into YOLO-pose format; richer fields stay in
the incoming JSON.

**Domain-gap mitigation:**

- Mix Unreal with real at training time — do not train on Unreal alone.
- Photometric augmentation on Unreal frames more aggressively than on real.
- Track real-vs-synthetic ratio in `metadata/SOURCE.md` so we can re-weight
  later if val numbers diverge between domains.
