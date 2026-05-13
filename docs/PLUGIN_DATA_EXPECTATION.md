# Plugin Data Expectation

What we expect from the upcoming collection plugin (Android side,
landing ~2026-05-13 evening per the AR team). This document tells the
plugin author what shape of data ML wants to receive so we can ingest
it directly without a custom adapter.

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
./.venv/bin/python src/convert_incoming_to_yolo.py \
  --source-root data/incoming/android_plugin \
  --dataset-root data/wheel_dataset \
  --overwrite
```

The current converter consumes a slightly different annotation shape
(the legacy `class_name` + `keypoints` array layout in
`docs/ANNOTATION_JSON_FORMAT.md`). Aligning the converter to the
plugin shape above is a small follow-up — once the first batch
arrives we will either:

1. adapt the converter to accept the plugin's native format, OR
2. add a tiny normaliser
   (`src/normalize_plugin_to_incoming.py`) that rewrites plugin JSON
   to the legacy format before conversion.

The choice depends on whether the plugin shape is final or expected
to evolve.

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
