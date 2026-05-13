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

Single class. The rim is **not** a separate class anymore — it is encoded
by two of the three keypoints below.

## Keypoints

Three keypoints per wheel, in a **fixed order** (the order is the contract,
not the names):

| Index | Internal name (code) | AR-facing name | Definition |
|-------|----------------------|----------------|------------|
| 0     | `rim_left`           | `point_a`      | Left point on the metal rim. AR raycasts from here as the **left ray source** for plane recovery. |
| 1     | `rim_right`          | `point_b`      | Right point on the metal rim. With `rim_left`, defines the wheel's vertical plane after AR-side raycast + RANSAC. |
| 2     | `disc_bottom`        | `point_c_disc_bottom` | Physical lowest point of the metal disc — the height anchor for AR placement. For straight-on views, may coincide with `rim_right`; for angled views, slightly below it. |

The AR-facing rename and the `disc_bottom` semantics are pending
AR-team confirmation — see `docs/OPEN_QUESTIONS_AR_SPEC.md` §1 and §3
and `docs/KEYPOINT_SPEC.md`. The label *order* and *count* are locked.

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
`disc_bottom` slightly below `rim_right`.

## Annotator quality checklist

- Bbox tight around the visible wheel — no extra background.
- Keypoint placement is the contract — be consistent across wheels:
  - `rim_left` and `rim_right` lie on the metallic rim (not the tire
    rubber), at the **left-most** and **right-most** visible points of
    the rim respectively.
  - `disc_bottom` is the physical lowest point of the metal disc. For
    perfectly straight-on views you may place it on top of `rim_right`.
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
