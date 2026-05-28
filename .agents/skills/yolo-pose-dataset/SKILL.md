---
name: yolo-pose-dataset
description: Use when working with the YOLO-pose dataset pipeline — generating, converting, validating, or previewing incoming batches (android_plugin or legacy manual_sample). Activates on edits to src/convert_*.py, src/check_*.py, src/preview_*.py, src/create_sample_*.py, configs/*dataset.yaml, or tests/test_convert_*.py / test_check_*.py. Documents the on-disk format, label spec, validation invariants, and command chain.
---

# VSBL — YOLO-pose dataset pipeline

There are two parallel flows in this repo. Pick the one that matches
the incoming batch shape.

## Flow A — Android plugin (preferred for new data)

**Incoming format** (`docs/KEYPOINT_DATASET_FORMAT.md`):

```
data/incoming/android_plugin/
  images/<stem>.jpg | .jpeg | .png | .bmp | .webp
  annotations/<stem>.json
  metadata/source_info.json
```

Each `<stem>.json`:

```json
{
  "frame_id": "frame_0001",
  "image":    "frame_0001.jpg",
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

Plugin contract carries **no `visibility` flag**. Every emitted point is
treated as fully visible.

**Keypoint semantics (2026-05-14 spec revision — load-bearing):**

- `points.a` — **left screen-space floor / raycast point** near the
  wheel's footprint / base. AR raycasts onto the floor plane to
  anchor the vertical wheel plane. **NOT** a metal-rim edge point.
- `points.b` — **right screen-space floor / raycast point**, mirror
  of A. **NOT** a metal-rim edge point.
- `points.c_disc_bottom` — **lowest visible point of the metal rim /
  disc**. Not the tire, not the floor, not the tire/ground contact.

Old wording that called A "rim_left" / "left rim source" / "left
point of metal rim" (and the symmetric B forms) is **obsolete** and
must not be reintroduced. The literal label strings `rim_left` /
`rim_right` survive in legacy code (`postprocess_wheels.KEYPOINT_NAMES`,
`convert_incoming_to_yolo.py`, `configs/dataset.yaml`) for backward
compatibility only — their *content* under the new contract is
floor / raycast points, not rim edges.

**Output (canonical YOLO-pose dataset)**:

```
data/wheel_pose_dataset/
  images/{train,val}/<source_name>__<stem>.<ext>
  labels/{train,val}/<source_name>__<stem>.txt
  metadata/split_manifest.json
  metadata/conversion_report.json
```

**Label line** (one per wheel, 14 fields, normalized to `[0, 1]` except `v`):

```
<class_id=0> <cx> <cy> <w> <h> <a_x> <a_y> <a_v=2> <b_x> <b_y> <b_v=2> <c_x> <c_y> <c_v=2>
```

- `class_id` = 0 (single class `wheel`).
- All bbox / keypoint coordinates normalised to `[0, 1]`.
- Visibility = 2 for every kept point (plugin has no occlusion flag).
- Keypoint order is fixed: `[a, b, c_disc_bottom]`.
- `flip_idx: [1, 0, 2]` — under horizontal flip A↔B swap, C stays.

**Config**: `configs/pose_dataset.yaml`.

**Commands**:

```bash
# Generate synthetic batch (smoke fixture, not training data)
./.venv/bin/python src/create_sample_keypoint_incoming.py --count 50 --overwrite

# Validate incoming format before converting
./.venv/bin/python src/check_keypoint_incoming.py \
    --source-root data/incoming/android_plugin

# Preview incoming annotations
./.venv/bin/python src/preview_keypoint_annotations.py \
    --source-root data/incoming/android_plugin --count 10
# → outputs/keypoint_preview/

# Convert plugin batch → YOLO-pose dataset
./.venv/bin/python src/convert_keypoint_incoming_to_yolo_pose.py \
    --source-root data/incoming/android_plugin \
    --dataset-root data/wheel_pose_dataset --overwrite

# Validate the converted dataset
./.venv/bin/python src/check_yolo_pose_dataset.py \
    --dataset-root data/wheel_pose_dataset

# Preview YOLO-pose labels (bbox + A/B/C overlay)
./.venv/bin/python src/preview_yolo_pose_labels.py \
    --dataset-root data/wheel_pose_dataset --split train --count 10
# → outputs/pose_label_preview/<split>/
```

## Flow B — Legacy manual_sample (still supported)

**Incoming format** (`docs/ANNOTATION_JSON_FORMAT.md`):

```
data/incoming/manual_sample/
  images/<stem>.jpg
  annotations/<stem>.json
  metadata/   # optional
```

Each annotation uses `objects[].class_name` + an explicit `keypoints`
list with `name`, `xy`, `visibility`:

```json
{
  "image": "img.jpg",
  "objects": [
    {
      "class_name": "wheel",
      "bbox_xyxy": [...],
      "keypoints": [
        {"name": "rim_left",    "xy": [x, y], "visibility": 0 | 1 | 2},
        {"name": "rim_right",   "xy": [x, y], "visibility": 0 | 1 | 2},
        {"name": "disc_bottom", "xy": [x, y], "visibility": 0 | 1 | 2}
      ]
    }
  ]
}
```

Visibility on this flow is real (`0 = not labelled`, `1 = occluded`,
`2 = visible`) and propagates into the YOLO labels.

**Output**: `data/wheel_dataset/`. **Config**: `configs/dataset.yaml`.

**Commands**:

```bash
./.venv/bin/python src/create_sample_incoming.py --count 20 --overwrite
./.venv/bin/python src/convert_incoming_to_yolo.py \
    --source-root data/incoming/manual_sample \
    --dataset-root data/wheel_dataset --overwrite
./.venv/bin/python src/check_dataset.py --dataset-root data/wheel_dataset
./.venv/bin/python src/preview_labels.py \
    --dataset-root data/wheel_dataset --split train --count 10
```

## Validation rules (both flows)

The dataset checkers (`check_dataset.py`, `check_yolo_pose_dataset.py`)
enforce, exit non-zero on any failure:

- `images/{train,val}` and `labels/{train,val}` exist.
- Every image has a matching label file (by stem).
- No orphan label files (label without matching image).
- Every non-empty label line has exactly **14 fields**
  (`5 + N_KEYPOINTS * 3`).
- `class_id ∈ {0}`.
- All bbox values `cx, cy, w, h ∈ [0, 1]`.
- Visibility flag `v ∈ {0, 1, 2}`.
- If `v > 0`: `kp_x, kp_y ∈ [0, 1]`.

The incoming-batch checker (`check_keypoint_incoming.py`) additionally
checks plugin-format invariants (point keys, bbox order, point-inside-
bbox tolerance, etc.). Run it before converting if the batch is real.

## Preview rules

- Plugin previewer (`preview_yolo_pose_labels.py`): orange bbox, green
  `a`, yellow `b`, red `c_disc_bottom`. Filled circle for `v=2`,
  hollow for `v=1`. Output: `outputs/pose_label_preview/<split>/`.
- Legacy previewer (`preview_labels.py`): same colour scheme; on-disk
  literal label names remain `rim_left` / `rim_right` / `disc_bottom`,
  but those strings are **legacy artefacts** — their geometric
  meaning under the 2026-05-14 contract is floor / raycast points
  (A, B) plus the lower metal rim / disc point (C), not rim edges.
  Output: `outputs/dataset_preview/<split>/`.
- Incoming previewer (`preview_keypoint_annotations.py`): renders on
  the original incoming images (pre-conversion). Output:
  `outputs/keypoint_preview/`.

## Common pitfalls

- **Mixing flows**: a plugin batch fed to `convert_incoming_to_yolo.py`
  will silently fail (wrong top-level keys). Use the matching converter.
- **Split leakage**: both converters do a random per-image split. For
  video frames or multi-shot photo sessions of the same car this is
  unsafe — group upstream by scene, or extend the converter (legacy
  has `--scene-regex`, plugin does not yet).
- **Visibility in plugin flow**: do not invent a `visibility` field for
  plugin data — the contract drops occluded wheels upstream. Emitting
  `v=1` would falsify the dataset.
- **Coordinate normalisation**: bbox and keypoint coordinates must be in
  `[0, 1]`. The converters do this; if you hand-roll a label file,
  remember.
- **Source-name collisions**: passing `--source-name` (or letting the
  source-root name collide) overwrites existing files in the dataset
  output. Use unique slugs per real batch.

## See also

- `docs/KEYPOINT_DATASET_FORMAT.md` — plugin format spec.
- `docs/ANNOTATION_JSON_FORMAT.md` — legacy format spec.
- `docs/DATASET_SPEC.md` — YOLO-pose dataset layout.
- `docs/REAL_DATA_INGESTION.md` — caveats for real batches.
- Skill `vsbl-ar-contract` — the response-side contract (separate from
  the training-label format).
