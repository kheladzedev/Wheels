# Plugin Data Expectation

What we expect from the upcoming collection plugin (Android / Unreal
side). Latest AR-side answer on **2026-05-18**: the plugin can output
essentially any collector field we request, but extra fields cost
implementation time. This document tells the plugin author the minimal
shape ML needs to ingest a batch directly without a custom adapter.

## Directory layout

Each batch lands in its own subtree:

```
data/incoming/android_plugin/
  images/
    frame_0001.jpg
    frame_0002.jpg
    ...
  annotations/
    frame_0001.json
    frame_0002.json
    ...
  metadata/
    SOURCE.md
    capture_info.json
```

- `images/` and `annotations/` filenames must share the same stem
  (`frame_0001.jpg` ↔ `frame_0001.json`). The stem is also the
  `frame_id` used downstream.
- `metadata/SOURCE.md` records origin, licensing, device, capture
  date. Free-form but required.
- `metadata/capture_info.json` (optional) records camera intrinsics
  and any per-batch settings the plugin captured.

## Annotation JSON shape

One JSON file per image. Suggested shape (one entry in `wheels[]` per
detected wheel):

```json
{
  "frame_id": "frame_0001",
  "image": "frame_0001.jpg",
  "wheels": [
    {
      "bbox_xyxy": [x1, y1, x2, y2],
      "points": {
        "a": [x, y],
        "b": [x, y],
        "c_disc_bottom": [x, y]
      }
    }
  ]
}
```

Coordinates are in **pixels**, top-left origin.

## Minimal acceptable collector output

For a batch to be eligible for training, every emitted wheel must have:

- `bbox_xyxy` around the full visible wheel: tyre + rim.
- `points.a` / `points.b` as the left/right floor-ray points in the
  final image coordinates.
- `points.c_disc_bottom` as the visually lowest visible point of the
  metal rim / disc, not the tyre contact patch.
- `frame_id` matching the image stem.

The earlier limited raw Unreal export (`0001.zip`) is not enough for
production training if it lacks full-wheel bbox or if its `Center`
point is a contact point rather than the true `c_disc_bottom`. Such a
batch may be kept for parser, preview, and service smoke tests only.

The later `0002` trial format is technically acceptable for intake when
it keeps this raw layout:

```
<batch>/
  Images/<frame_id>.jpg
  keyPoint/<frame_id>/<object_id>.txt
```

Each keyPoint object should contain:

- `Left` — preferred raw name for `points.a` (left floor-ray point).
- `Right` — preferred raw name for `points.b` (right floor-ray point).
- `Center` — maps to `points.c_disc_bottom`.
- `LeftTop` and `RightTop` — optional bbox helper points. When both are
  present, non-zero, and inside the image, the importer builds the
  full-wheel bbox from all five points. If the Unreal collector can add
  an explicit `BBox: x1,y1,x2,y2`, that is still preferred.

Igor's current Blueprint docs name these point actors `SphereLeft`,
`SphereRight`, `SphereLeftTop`, and `SphereRightTop`. The raw inspector and
importer accept those names as aliases and normalize them to `Left`,
`Right`, `LeftTop`, and `RightTop` before validation. This is an input
compatibility layer only; downstream AR JSON still emits only
`points.a`, `points.b`, and `points.c_disc_bottom`.

Observed historical note: the `0002` trial export used inverted raw names
(`Right` was the left screen-space point and `Left` was the right
screen-space point), while `0003` uses literal screen-side names. The raw
adapter now defaults to batch-level auto mapping from x-order and records
the resolved mapping in `metadata/source_info.json` and the acceptance
report. For new exports, prefer the literal screen-side convention:
`Left -> points.a`, `Right -> points.b`.

Coordinates must be pixels in the final exported image coordinate system
(`0..2048` for the current square renders). Objects with `0,0`, missing
required points, or out-of-bounds required points are treated as
invisible/invalid and dropped during import.

### Required fields

- `frame_id` — string, must match the image stem.
- `image` — filename of the corresponding image, relative to
  `images/`.
- `wheels` — array; zero or more entries.
- `wheels[].bbox_xyxy` — `[x1, y1, x2, y2]`, top-left and bottom-right
  corners, pixels.
- `wheels[].points.a` — `[x, y]` of the left ray source point.
- `wheels[].points.b` — `[x, y]` of the right ray source point.
- `wheels[].points.c_disc_bottom` — `[x, y]` of the lowest visible
  point of the metal rim / disc.

### Rules

- **Occluded wheels: omit entirely.** If `a`, `b`, or `c_disc_bottom`
  is not visible (blocked by car body, another wheel, scene element),
  do not include the wheel in `wheels[]`. Do not guess. Confirmed AR
  decision 2026-05-13 (`docs/OPEN_QUESTIONS_AR_SPEC.md` §3).
- **No per-keypoint confidence.** The plugin does not provide any
  confidence value alongside `points` — neither annotator certainty
  nor a model-derived score. Confirmed AR decision §4.
- **No visibility flag.** Because occluded wheels are omitted, there
  is no `visibility` key. Every emitted point is implicitly visible.
- **No `track_id` or temporal links.** Each frame is independent.
- **No 3D coordinates and no camera intrinsics inline.** Camera info,
  if the plugin captures it, goes in `metadata/capture_info.json`,
  not in the per-frame annotation.

### Empty frames

If a frame contains no fully-visible wheels, emit a JSON with
`"wheels": []`. Do not omit the JSON file — pairing by stem still
needs to find it.

## Ingestion path

Once the batch is dropped in place:

```bash
./.venv/bin/python scripts/accept_unreal_export.py \
  --source-root ~/Downloads/0002 \
  --source-name unreal_0002_trial \
  --overwrite
```

For already-normalized plugin JSON under `data/incoming/<source_name>/`,
use `src/convert_keypoint_incoming_to_yolo_pose.py` directly. For raw
Unreal exports (`Images/keyPoint/Ground`), always start with
`scripts/accept_unreal_export.py`; it runs inspection, import,
validation, conversion, preview generation, and optional smoke training
from one command.

## Quality rules (carried over)

From `docs/REAL_DATA_INGESTION.md` §5:

- Minimum resolution: longer side ≥ 480 px.
- Camera angle: side or 3-quarter preferred; pure head-on or rear
  de-prioritized.
- Wheel visibility per the occluded-drop rule above.
- Avoid extreme motion blur or out-of-focus shots.
- Mix lighting conditions; avoid an all-studio batch.

## See also

- `docs/AR_ML_CONTRACT.md` — JSON contract returned by the model.
- `docs/KEYPOINT_SPEC.md` — A/B/C definitions.
- `docs/REAL_DATA_INGESTION.md` — full ingestion-stage details.
- `docs/ANNOTATION_JSON_FORMAT.md` — legacy incoming format the
  converter currently reads.
- `src/convert_incoming_to_yolo.py` — converter implementation.
