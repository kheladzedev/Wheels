# Annotation JSON Format (Interim)

This is the **interim** annotation format consumed by
`src/convert_incoming_to_yolo.py`. It exists so we can start ingesting
batches before the final Unreal / labelling-tool export schema is agreed
with the team (see `docs/QUESTIONS_FOR_TEAM.md` Q9).

When the canonical schema lands, this format will be deprecated and the
converter extended. Do not bake long-lived tooling against it.

## Per-image file

One JSON file per image, named with the same stem as the image. Both live
under the source's incoming directory:

```
data/incoming/<source_name>/
  images/image_001.jpg
  annotations/image_001.json
```

## Schema

```json
{
  "image": "image_001.jpg",
  "objects": [
    {
      "class_name": "wheel",
      "bbox_xyxy": [100, 220, 200, 320],
      "keypoints": [
        {"name": "rim_left",     "xy": [150, 225], "visibility": 2},
        {"name": "rim_right",  "xy": [150, 315], "visibility": 2},
        {"name": "disc_bottom", "xy": [150, 315], "visibility": 2}
      ]
    }
  ]
}
```

## Fields

| Field                              | Type             | Required | Meaning |
|------------------------------------|------------------|----------|---------|
| `image`                            | string           | optional | Filename of the matching image. Informational — the converter matches by file stem. |
| `objects`                          | list             | required | Annotations for this image. Empty list = no objects. |
| `objects[].class_name`             | string           | required | Must be `"wheel"`. Anything else is skipped with a warning. |
| `objects[].bbox_xyxy`              | list[4 number]   | required | `[x1, y1, x2, y2]` in pixels, top-left origin, with `x2 > x1`, `y2 > y1`. |
| `objects[].keypoints`              | list[3 object]   | required | Exactly 3 keypoints in this order: `rim_left`, `rim_right`, `disc_bottom`. |
| `objects[].keypoints[].name`       | string           | optional | Informational name. The order in the list is authoritative. |
| `objects[].keypoints[].xy`         | list[2 number]   | required | `[x, y]` in pixels. Ignored if `visibility == 0`. |
| `objects[].keypoints[].visibility` | int (0, 1, 2)    | required | `0` = not labelled / not visible, `1` = labelled but occluded, `2` = visible. |

## Keypoint definitions

The three keypoints encode the spec's "two rim points + disc bottom"
(see `docs/KEYPOINT_SPEC.md` for the AR-facing target naming
`point_a` / `point_b` / `point_c_disc_bottom`):

- **`rim_left`** (target name `point_a`) — left-most visible point on the
  metallic rim (the disc, not the tire). AR uses this as the **left ray
  source** for plane recovery.
- **`rim_right`** (target name `point_b`) — right-most visible point on
  the metallic rim. Together with `rim_left`, AR uses these to RANSAC
  the wheel's vertical plane across K frames.
- **`disc_bottom`** (target name `point_c_disc_bottom`) — physical
  lowest point of the metal disc. AR uses this to determine the height
  at which the virtual disc sits above the reconstructed plane. For a
  straight-on view, `disc_bottom` and `rim_right` may coincide; for
  angled views they differ.

These definitions are our working defaults — see
`docs/OPEN_QUESTIONS_AR_SPEC.md` §1 and §3 for AR-team confirmation
status (semantics of C; final field names; visibility encoding).

## Class mapping

| `class_name` | YOLO `class_id` |
|--------------|-----------------|
| `wheel`      | `0`             |

The old `rim` class is gone — rim is now represented by the two rim
keypoints, not a separate bbox.

## Validation rules applied by the converter

- A JSON file must exist for every image. Images without annotations are
  **skipped** (logged in `metadata/conversion_report.json`).
- `bbox_xyxy` must be a list of 4 numbers with `x2 > x1` and `y2 > y1`.
- Bboxes that extend outside the image are **clipped** with a warning.
- Each wheel must have exactly **3** keypoints in the canonical order.
  Wrong-length keypoint lists are dropped from that image with a warning.
- Keypoints outside the image are clipped to image bounds. Keypoints with
  `visibility == 0` are emitted as `(0, 0, 0)` in the YOLO label.
- Unknown `class_name` values are dropped with a warning.
- An image with zero valid objects after filtering produces an **empty**
  label file — valid YOLO (means "no objects").

## What is intentionally NOT in this format

These may show up in the final schema but are out of scope here:

- Instance masks / segmentation polygons.
- Ellipse parameters for perspective-aware wheels.
- Camera intrinsics / extrinsics.
- 3D world positions of keypoints.
- Per-object IDs for tracking across frames (see Q5).

If your source already produces any of these, keep them in
`data/incoming/<source>/annotations/` as you receive them — they survive
in the staging area and we'll wire them up when the converter is upgraded.
