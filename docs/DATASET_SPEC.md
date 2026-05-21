# Dataset Specification — wheel + 3 keypoints (YOLO-pose)

This document fixes the on-disk layout, class IDs, label format, and
keypoint conventions used by `configs/dataset.yaml`, `src/check_dataset.py`,
`src/preview_labels.py`, and `src/train_yolo.py`.

## Directory layout

```
data/wheel_dataset/
  images/
    train/   # *.jpg | *.jpeg | *.png | *.bmp | *.webp
    val/
  labels/
    train/   # *.txt — one file per image, same stem
    val/
  metadata/
    split_manifest.json
    conversion_report.json
```

Rules:

- Every image in `images/<split>/` must have a sibling label at
  `labels/<split>/<same_stem>.txt`.
- An empty `.txt` means "no objects in this image" — allowed.
- A missing `.txt` is treated as an error by `check_dataset.py`.
- Image stems must be unique within a split. Cross-split repeats are
  strongly discouraged.

## Classes

| id | name  | meaning                                                       |
|----|-------|---------------------------------------------------------------|
| 0  | wheel | Full wheel (tire + rim) visible from outside.                 |

Single class. The rim is **not** a separate class anymore. Wheel pose is
encoded by the three keypoints below.

## Keypoints

Three keypoints per wheel, in a **fixed order** (the order is the contract,
not the names):

| Index | Internal name (code) | AR-facing name | Definition |
|-------|----------------------|----------------|------------|
| 0     | `rim_left`           | `a`            | Left screen-space floor-ray point near the wheel footprint/base. AR raycasts this pixel onto the floor; it is **not** a metal-rim point. |
| 1     | `rim_right`          | `b`            | Right screen-space floor-ray point near the wheel footprint/base. With `a`, AR recovers the wheel's vertical plane after floor raycast + RANSAC; it is **not** a metal-rim point. |
| 2     | `disc_bottom`        | `c_disc_bottom` | Lower visible point of the metal rim / disc where the rim meets the tire — the height anchor for AR placement. |

The AR-facing names and keypoint semantics are confirmed. See
`docs/KEYPOINT_SPEC.md` and `docs/AR_ML_CONTRACT.md`. The label *order*
and *count* remain locked for the YOLO-pose labels.

### Visibility flags

| Value | Meaning |
|-------|---------|
| 0     | Not labelled / not in frame. Loss skipped during training. xy ignored. |
| 1     | Labelled but occluded (annotator inferred position behind another object). Loss contributes. |
| 2     | Labelled and clearly visible. Loss contributes. |

## YOLO-pose label format

One line per wheel. Fields are space-separated, all numeric except the
final visibility flags which are integers.

```
<class_id> <cx> <cy> <w> <h> <kp0_x> <kp0_y> <v0> <kp1_x> <kp1_y> <v1> <kp2_x> <kp2_y> <v2>
```

Total: **14 fields** per line.

- `class_id` — `0` (only `wheel` exists).
- `cx`, `cy`, `w`, `h` — bbox center and size, normalized to `[0, 1]` by
  image width/height.
- `kp{i}_x`, `kp{i}_y` — keypoint position, normalized to `[0, 1]`. When
  `v{i} == 0`, the converter emits `0 0 0` regardless of any source xy.
- `v{i}` — visibility flag (`0`, `1`, or `2`).

Example (`labels/train/IMG_0123.txt`):

```
0 0.512 0.687 0.118 0.214 0.512 0.580 2 0.512 0.794 2 0.512 0.800 2
```

One wheel near the bottom-center, with all three keypoints fully visible.
The first two keypoints are floor-ray/base points; `disc_bottom` is the
lower visible rim/disc point.

## Annotator quality checklist

- Bbox tight around the visible wheel — no extra background.
- Keypoint placement is the contract — be consistent across wheels:
  - `rim_left` and `rim_right` are legacy internal label strings for
    `a` and `b`; place them as left/right screen-space floor-ray points
    near the wheel footprint/base, **not** on the metal rim.
  - `disc_bottom` is the lower visible point of the metal rim / disc
    where the rim meets the tire.
- Annotate partially visible wheels if at least ~50% of the disc is in
  frame. Mark occluded keypoints with `visibility = 1` and the inferred
  position. Mark off-frame keypoints with `visibility = 0`.
- Skip heavily motion-blurred wheels.
- Do not annotate spare wheels mounted on the back door / roof — keeps the
  class semantics consistent for AR.

## Validation

Before launching a training run:

```bash
python src/check_dataset.py --dataset-root data/wheel_dataset
python src/preview_labels.py --dataset-root data/wheel_dataset --split train --count 10
```

`check_dataset.py` enforces the 14-field format, single class, and value
ranges (bbox + visible keypoints in `[0, 1]`, visibility in `{0,1,2}`).
`preview_labels.py` renders bbox + keypoints (filled for visible, hollow
for occluded) so a human can eyeball annotation quality before spending
GPU time.
