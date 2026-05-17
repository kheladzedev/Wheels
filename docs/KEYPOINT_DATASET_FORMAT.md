# Keypoint Dataset Format — Android Plugin Incoming

The Android / Unreal collection plugin drops batches of labelled frames
into `data/incoming/android_plugin/` following the schema below. This
document is the on-disk contract: the plugin author writes this, the ML
side reads it. The latest AR-side confirmation is **2026-05-18**.

For the runtime JSON contract (what ML emits at inference), see
`docs/AR_ML_CONTRACT.md`. The two are intentionally similar but not
identical — input has `wheels[].points.a/b/c_disc_bottom` plus
`image`, output has the same plus `confidence`.

## Directory layout

Each batch:

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
    source_info.json
```

- `images/` and `annotations/` filenames share the same stem
  (`frame_0001.jpg` ↔ `frame_0001.json`). The stem is the `frame_id`
  used downstream.
- `metadata/source_info.json` records origin, device, capture date,
  any per-batch settings the plugin captured.

Plugin authors may use any extension under `images/` from
`.jpg, .jpeg, .png, .bmp, .webp`. The validator
(`src/check_keypoint_incoming.py`) checks for matching stems
regardless of extension.

## Annotation JSON

One JSON file per image.

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

All coordinates are in **pixels**, top-left origin, native image
resolution.

### Required fields

| Field | Type | Notes |
|---|---|---|
| `frame_id` | string | Must equal the image stem. |
| `image` | string | Filename of the corresponding image, relative to `images/`. |
| `wheels` | array | Zero or more entries. May be empty. |
| `wheels[].bbox_xyxy` | `[x1, y1, x2, y2]` | Top-left + bottom-right, pixels. Must cover the entire wheel (tyre + rim). |
| `wheels[].points.a` | `[x, y]` | Left floor-plane post-process point used for raycast / wheel-plane recovery. |
| `wheels[].points.b` | `[x, y]` | Right floor-plane post-process point used for raycast / wheel-plane recovery. |
| `wheels[].points.c_disc_bottom` | `[x, y]` | Lowest **visible** point of the metal disc / rim — not the tyre, not the hub centre. |

### Rules

1. `frame_id` is required. Must match the image stem exactly. The
   stem becomes the AR-side lookup key for the camera transform.
2. `image` is required and must equal the actual filename in `images/`.
3. `wheels` may be empty (`[]`). An image with no fully-visible wheels
   is a valid frame — the file must still be present so frame pairing
   by stem works.
4. **Occluded wheels are not added.** If any of `a` / `b` /
   `c_disc_bottom` is not visible (blocked by car body, another
   wheel, scene element), omit the whole wheel. Do not guess.
   Confirmed AR decision 2026-05-13, re-confirmed 2026-05-18.
5. `bbox_xyxy` must contain the entire wheel — tyre included. Use
   the same bbox an annotator would draw for the whole wheel
   silhouette.
6. `points.a` / `points.b` / `points.c_disc_bottom` should lie **inside
   the bbox** or within a small tolerance of it (the validator allows
   a 5 px slack so a slightly-clipped C disc-bottom on a tyre edge
   doesn't fail validation). Anything beyond that is a labelling
   mistake.
7. `c_disc_bottom` is the lowest visible point of the **metal disc /
   rim**. NOT the tyre's contact patch with the ground. NOT the wheel
   hub centre.
8. All coordinates are in pixels of the source image, top-left origin.
   No normalisation, no `[0, 1]` scaling.
9. There is no `visibility` flag and no per-keypoint confidence.
   Every emitted point is implicitly visible. Confirmed AR decision §3, §4.
10. There is no `track_id`, no `timestamp`, no 3D coordinates, no
    camera intrinsics inline. Camera metadata, if captured, goes in
    `metadata/source_info.json`.
11. A limited Unreal/debug batch that lacks `bbox_xyxy` or a true
    `c_disc_bottom` must not be marked training-approved even if the
    raw points parse successfully.

### Example: empty frame

```json
{
  "frame_id": "frame_0042",
  "image": "frame_0042.jpg",
  "wheels": []
}
```

Still required if no wheels are visible — pairing by stem needs to
find the JSON.

### Example: two-wheel frame

```json
{
  "frame_id": "frame_0001",
  "image": "frame_0001.jpg",
  "wheels": [
    {
      "bbox_xyxy": [120, 280, 220, 380],
      "points": {
        "a": [125, 320],
        "b": [215, 320],
        "c_disc_bottom": [170, 370]
      }
    },
    {
      "bbox_xyxy": [420, 280, 520, 380],
      "points": {
        "a": [425, 320],
        "b": [515, 320],
        "c_disc_bottom": [470, 370]
      }
    }
  ]
}
```

## `metadata/source_info.json`

Free-form but recommended fields:

```json
{
  "source_name": "android_plugin_v1",
  "captured_with": "Pixel 7, Android 14",
  "captured_at": "2026-05-13T18:00:00Z",
  "image_count": 50,
  "notes": "indoor showroom, mixed daylight"
}
```

The validator does not enforce these — they are for downstream
provenance only.

## Working with the format

| Action | Command |
|---|---|
| Generate a synthetic batch in this format | `python src/create_sample_keypoint_incoming.py --count 50 --overwrite` |
| Validate a real plugin batch | `python src/check_keypoint_incoming.py --source-root data/incoming/android_plugin` |
| Preview labels with bbox + A/B/C overlay | `python src/preview_keypoint_annotations.py --source-root data/incoming/android_plugin --count 10` |

## See also

- `docs/AR_ML_CONTRACT.md` — runtime JSON contract returned by the model.
- `docs/KEYPOINT_SPEC.md` — A/B/C geometric definitions.
- `docs/PLUGIN_DATA_EXPECTATION.md` — higher-level overview of plugin expectations.
- `docs/REAL_DATA_INGESTION.md` — broader ingestion-stage rules (resolution, blur, lighting variety).
- `src/check_keypoint_incoming.py` — validator (this format).
- `src/create_sample_keypoint_incoming.py` — generator (this format).
- `src/preview_keypoint_annotations.py` — visualiser (this format).
