# Annotation Tooling — labelling setup for the wheel + 3-keypoints task

Audience: whoever sets up the labelling pipeline for the team. Annotators
themselves should read `docs/ANNOTATION_GUIDELINES.md` instead.

## 1. Tool recommendation

**Pick CVAT.** Justification below; alternatives listed for completeness.

| Tool | Verdict | Why |
|---|---|---|
| **CVAT** | **Recommended** | Native keypoint + skeleton support, free, self-hostable via `docker compose`, exports COCO-keypoints JSON cleanly. Mature multi-user workflow with review stage. |
| Label Studio | Alternative | Workable for keypoints; heavier than CVAT. Pick this only if the same team also does classification / multi-task labelling on the same project. |
| Roboflow | Managed | Fastest to start, no infra. Cost grows per-image at scale and data leaves our infra. Use only for a small pilot if CVAT setup is delayed. |

Assume the team can `docker compose up` CVAT on its own — installation
guide intentionally not duplicated here. Project config is below.

## 2. CVAT project setup

### 2.1 Label schema (one label, one skeleton)

Single label `wheel` with:

- A **bbox** shape.
- A **3-point skeleton** attached to that label, points in fixed index
  order:
  - `kp0` → `rim_left`
  - `kp1` → `rim_right`
  - `kp2` → `disc_bottom`

The point order **is** the contract. The names are working names and may
remain legacy strings for converter compatibility. `rim_left` and
`rim_right` now represent confirmed floor-ray A/B points near the wheel
footprint, not obsolete edge points.

### 2.2 Skeleton edges (visual aid only)

Edges are drawn for the annotator's visual feedback only — they have no
effect on the YOLO label format. Use:

- `rim_left ↔ rim_right`
- `rim_right ↔ disc_bottom`

**Do not** connect `rim_left ↔ disc_bottom`. A and B are floor-ray points
near the wheel footprint, while C is the lower visible rim/disc point; a
direct A-to-C edge teaches annotators to think of a triangle on the
wheel, which is wrong.

### 2.3 Visibility flag

Each keypoint carries a visibility value. CVAT exposes three states; map
them to our `0 / 1 / 2` encoding (COCO-pose):

| CVAT term | Our value | Meaning |
|---|---|---|
| `visible` | `2` | Visible, labelled. |
| `occluded` | `1` | Labelled but occluded — inferred xy. |
| `outside` | `0` | Not in frame / not inferrable. xy ignored downstream. |

The COCO-keypoints exporter emits the integer `0 / 1 / 2` per keypoint.

### 2.4 Label config to paste into CVAT

CVAT's label schema is configured via JSON in the project settings. The
exact field layout (especially `svg` skeleton coordinates) is
documented at <https://docs.cvat.ai>. The config below is the **working
configuration**; verify it parses in your CVAT version before mass
import (see §6 "Verify when setting up").

```json
[
  {
    "name": "wheel",
    "color": "#33ddff",
    "type": "skeleton",
    "attributes": [],
    "sublabels": [
      {
        "name": "rim_left",
        "color": "#ff3333",
        "type": "points",
        "attributes": []
      },
      {
        "name": "rim_right",
        "color": "#33ff33",
        "type": "points",
        "attributes": []
      },
      {
        "name": "disc_bottom",
        "color": "#3333ff",
        "type": "points",
        "attributes": []
      }
    ],
    "svg": "<line x1=\"20\" y1=\"50\" x2=\"80\" y2=\"50\" stroke=\"black\" data-type=\"edge\" data-node-from=\"1\" data-node-to=\"2\"></line><line x1=\"80\" y1=\"50\" x2=\"80\" y2=\"90\" stroke=\"black\" data-type=\"edge\" data-node-from=\"2\" data-node-to=\"3\"></line><circle r=\"1.5\" fill=\"#ff3333\" cx=\"20\" cy=\"50\" data-type=\"element node\" data-element-id=\"1\" data-node-id=\"1\" data-label-name=\"rim_left\"></circle><circle r=\"1.5\" fill=\"#33ff33\" cx=\"80\" cy=\"50\" data-type=\"element node\" data-element-id=\"2\" data-node-id=\"2\" data-label-name=\"rim_right\"></circle><circle r=\"1.5\" fill=\"#3333ff\" cx=\"80\" cy=\"90\" data-type=\"element node\" data-element-id=\"3\" data-node-id=\"3\" data-label-name=\"disc_bottom\"></circle>"
  },
  {
    "name": "wheel_bbox",
    "color": "#ffcc00",
    "type": "rectangle",
    "attributes": []
  }
]
```

Notes:

- The skeleton above defines the three keypoints and the two edges
  (rim_left → rim_right, rim_right → disc_bottom). These edges are visual
  aids only; the legacy names `rim_left` / `rim_right` still mean floor-ray
  A/B points, not obsolete edge points. The `wheel_bbox` shape is a separate
  rectangle label so the bbox and skeleton are drawn as independent shapes
  — easier for annotators than the combined-shape mode.
- If your CVAT version supports combined skeleton-with-bbox in a single
  shape, prefer that and drop the separate `wheel_bbox` label. Either
  way the export produces the bbox + 3 keypoints needed downstream.
- Visibility per keypoint uses CVAT's built-in `outside` / `occluded`
  attributes — no custom attribute config needed.

## 3. Export and conversion pipeline

### 3.1 Export from CVAT

- Format: **COCO Keypoints 1.0** (CVAT's "COCO Keypoints" exporter).
- One JSON per task, plus the original images.
- Save the export tarball or unpacked folder.

### 3.2 Drop location in the repo

Place each export under its own source slug:

```
data/incoming/<source_name>/
  images/                # the exported images
  annotations/           # COCO JSON from CVAT (single file or per-image)
  metadata/SOURCE.md     # origin, licence, capture notes
```

Naming convention for `<source_name>`: short, stable, dated where
relevant. Examples: `dealership_munich_2026_03`, `pilot_batch_2026_05`,
`unreal_rim_pack_v2`. Same convention as `docs/REAL_DATA_INGESTION.md`
§3.

### 3.3 Convert to YOLO-pose

```bash
./.venv/bin/python src/convert_incoming_to_yolo.py \
  --source-root data/incoming/<source_name> \
  --dataset-root data/wheel_dataset \
  --overwrite
```

The converter writes `data/wheel_dataset/images/{train,val}/` and
`data/wheel_dataset/labels/{train,val}/` plus a conversion report in
`data/wheel_dataset/metadata/conversion_report.json`. Output label
format is the 14-field YOLO-pose line documented in
`docs/DATASET_SPEC.md`.

### 3.4 COCO → interim-JSON gap

The converter currently consumes the **interim per-image JSON** schema
defined in `docs/ANNOTATION_JSON_FORMAT.md` — **not** raw COCO. CVAT
exports COCO. Two options to bridge:

- **(a) Extend the converter** to read COCO directly. Cleaner long-term;
  one fewer hop and the converter stays the single ingestion entry point.
- **(b) Write a thin adapter** `src/adapters/cvat_coco_to_interim.py`
  that reads the COCO JSON and writes one per-image JSON file under
  `data/incoming/<source_name>/annotations/` matching
  `docs/ANNOTATION_JSON_FORMAT.md`. Leave the converter unchanged.

Recommendation: **(b)** for the first real batch (fast, minimal risk),
then fold into **(a)** once we know the COCO export quirks of our CVAT
version. The adapter file does not exist yet — `src/adapters/` is the
suggested path so it lives alongside other ingestion code.

Whichever path is taken, the contract upstream of the converter stays
the same — `rim_left`, `rim_right`, `disc_bottom` in that order, plus
visibility `0 / 1 / 2`.

## 4. QA workflow

After each conversion, before any training run:

```bash
# Mandatory — 14-field format + value ranges.
./.venv/bin/python src/check_dataset.py --dataset-root data/wheel_dataset

# Mandatory — 10-sample human spot check (rendered bbox + keypoints).
./.venv/bin/python src/preview_labels.py --dataset-root data/wheel_dataset --split train --count 10
```

Both are described in `docs/DATASET_SPEC.md` §Validation. `preview_labels.py`
draws filled markers for visible keypoints and hollow markers for occluded
ones — a quick eyeball pass catches systematic mislabels.

### Optional — inter-annotator agreement

Have a second annotator re-label ~5% of the batch independently. For each
re-labelled wheel, compute per-keypoint pixel distance between the two
annotations. The distribution gives a **noise floor** — the model should
not be expected to beat it.

This is not wired into a script yet; a simple pandas notebook over the
two exported COCO JSONs is sufficient for the first batch.

## 5. Throughput and planning

- **30–60 seconds per wheel** is realistic with practice.
- Multi-wheel frames are faster per wheel (annotator stays oriented on
  the same car).
- **500-wheel batch ≈ 4–8 hours** one-pass labelling. Plan accordingly.
- Double-annotation QA (§4) adds ~5% time overhead.
- For video frames, sample first (≤ 1 fps, perceptual-hash dedup) before
  labelling — see `docs/REAL_DATA_INGESTION.md` §5.

## 6. Train / val splits

- Hold out **15–20%** for validation.
- **Split by upstream unit, not by random frame shuffle** — group
  video frames by clip / scene, multi-shot photo sessions by car,
  synthetic batches by scene seed. This prevents leakage between train
  and val.
- The current converter supports `random_per_image` (default) and
  `prefix` strategies. Sources that don't fit either should be
  pre-grouped upstream before ingestion. See
  `docs/REAL_DATA_INGESTION.md` §6 for the rules and the manifest
  produced (`data/wheel_dataset/metadata/split_manifest.json`).

## 7. Verify when setting up

Working assumptions in this doc that should be confirmed against the
actual CVAT version the team installs:

- The exact field names / casing in CVAT's label-config JSON (the SVG
  schema for skeletons has changed across CVAT versions; treat §2.4 as
  a starting point, not a guarantee).
- Whether CVAT's COCO-Keypoints exporter encodes visibility as
  `0 / 1 / 2` directly (it should — COCO standard — but confirm in a
  test export).
- Whether the combined "skeleton + bbox in one shape" mode is available
  and preferable to the two-label workaround in §2.4.
- Whether keyboard shortcuts for "set keypoint occluded" / "set keypoint
  outside" are bound conveniently — if not, rebind before annotators
  start.
- Whether CVAT's per-task review stage is enabled and at least one
  reviewer is assigned (mandatory for any non-pilot batch).

## See also

- `docs/ANNOTATION_GUIDELINES.md` — what annotators actually do.
- `docs/ANNOTATION_JSON_FORMAT.md` — the per-image JSON the converter
  consumes after the COCO → interim adapter step.
- `docs/DATASET_SPEC.md` — final on-disk YOLO-pose layout and 14-field
  label format.
- `docs/REAL_DATA_INGESTION.md` — source-to-dataset flow at the repo
  level, split-by-unit rules, Unreal-specific expectations.
- `docs/KEYPOINT_SPEC.md` — geometric definitions of A / B / C.
- `docs/OPEN_QUESTIONS_AR_SPEC.md` — items still open with the AR team.
