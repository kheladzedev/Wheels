# Annotation Guidelines — wheel + 3 keypoints

Audience: human annotator. You have never seen this project before. Read
this end to end, then start labelling. Expect ~10 minutes to internalise
the rules.

> **Confirmed contract.** The AR-facing JSON keys are `a`, `b`, and
> `c_disc_bottom`; the internal annotation labels remain `rim_left`,
> `rim_right`, and `disc_bottom` only for converter compatibility. Read
> `docs/KEYPOINT_SPEC.md` and `docs/AR_ML_CONTRACT.md` as the source of
> truth. The label order and count are locked: three 2D screen-space
> keypoints per wheel.

## 1. What gets labelled

| Object | Label? | Notes |
|---|---|---|
| Full wheel (tire rubber + visible disc/rim area) visible from outside the car | **Yes** | Single class `wheel`. |
| Spare wheel mounted on rear door / roof of an SUV or pickup | **No** | Keeps the class semantics clean for AR — spares are not driveable wheels. |
| Wheel through the cabin window (interior shot through glass) | **No** | Optical distortion, not the use case. |
| Motorcycle / bicycle / scooter wheel | **No** | Wrong vehicle class. |
| Painted wheels on a mural, wheel toys, decorative wheels | **No** | Not real wheels. |
| Wheel disconnected from a vehicle (e.g. spare on the ground) | **No** | Out of scope for AR fitting. |

Class id is fixed: `0 = wheel`. See `docs/DATASET_SPEC.md` for the
authoritative class table.

## 2. Bbox

- Tight around the visible wheel (tire + rim together).
- No extra background, no padding.
- Include occluded portions of the wheel if you can infer the silhouette
  (e.g. wheel partly behind a fender). Match the visible silhouette
  where the occlusion is, not the imagined full circle.
- The annotation tool exports `xyxy` (two corners). The downstream
  converter normalises to YOLO format and the AR-facing JSON eventually
  uses `xywh`. You do not need to think about formats — just draw tight.

## 3. The three keypoints

Three keypoints per wheel, in a **fixed order**. The order is the
contract. The `rim_left` / `rim_right` names are legacy internal strings;
their confirmed meaning is floor-ray A/B points, not obsolete edge points.

| Index | Internal label | Confirmed JSON key | Where to place it |
|---|---|---|---|
| 0 | `rim_left` | `a` | Left screen-space floor-ray point near the wheel footprint/base. Place it on the floor/ground near the left side of the wheel base: **not** on the metal rim, **not** on the tire rubber, **not** on the wheel itself. |
| 1 | `rim_right` | `b` | Right screen-space floor-ray point near the wheel footprint/base. Place it on the floor/ground near the right side of the wheel base: **not** on the metal rim, **not** on the tire rubber, **not** on the wheel itself. |
| 2 | `disc_bottom` | `c_disc_bottom` | Lower visible point of the metal rim / disc where the rim meets the tire. |

Definitions are aligned to the AR-team mock pipeline; see
`docs/KEYPOINT_SPEC.md` for the geometric rationale. ML returns only 2D
screen-space pixel points. AR raycasts A/B onto the floor, runs RANSAC
and plane recovery, then uses C for height estimation and 3D
visualization.

> Earlier iterations described `rim_left` / `rim_right` as points on the
> rim. That wording is obsolete. Under the confirmed contract, kp0 and
> kp1 are `a` / `b`: floor-ray points near the wheel footprint.

### Concrete placement rules

- A (`rim_left`) belongs on the floor/ground near the left side of the
  wheel footprint/base. It is a screen pixel that AR raycasts onto the
  floor plane.
- B (`rim_right`) belongs on the floor/ground near the right side of the
  wheel footprint/base. It is also a screen pixel for AR floor raycast.
- A/B should be near the bottom/footprint area of the wheel, visually
  below C in normal image coordinates. They are not metal-rim points,
  tire points, or wheel-surface points.
- C (`disc_bottom`) sits on the lower visible metal rim / disc boundary
  where the rim meets the tire. Do not place it on the floor, the tire's
  road-contact point, or the hub centre.
- For chrome / highly reflective rims, use the visible physical rim/disc
  boundary for C, not a reflection inside it.

## 4. Visibility flags

COCO-pose convention. Per keypoint, exactly one of:

| Value | Meaning | When to use |
|---|---|---|
| `2` | Visible, labelled | You can see the point directly. Default for most keypoints. |
| `1` | Labelled but occluded | The point is hidden behind something (fender, another wheel, body trim) but you can confidently infer **where** it would be. xy is the inferred position. |
| `0` | Not in frame / not inferrable | Point is off-image, or fully behind a large opaque object with no inferrable position. xy is ignored — the converter writes `(0, 0, 0)`. |

In the CVAT UI: `visible` = 2, `occluded` = 1, `outside` = 0. See
`docs/ANNOTATION_TOOLING.md` for the project setup.

## 5. Inclusion threshold

| Wheel state | Label? |
|---|---|
| ≥ 50% of the metallic disc visible and recognisable | **Yes** |
| < 50% of the metallic disc visible | **No** — skip the wheel |
| Tire visible but disc almost fully hidden | **No** |
| Disc clearly visible but heavily blurred | See edge cases below |

Use the disc (metal part), not the tire, as the visibility yardstick.

## 6. Edge cases

| Case | Rule |
|---|---|
| Motion-blurred wheel | Skip if you cannot place A/B/C within ~5 px confidence. Otherwise label with best estimate, `visibility = 2`. |
| Heavy occlusion by fender | If you can infer a keypoint's hidden position, label with `visibility = 1` at the inferred xy. If you cannot, set that keypoint's `visibility = 0`. |
| Wheel half-clipped at image edge | Label the wheel. Visible keypoints get `visibility = 2`; off-image keypoints get `visibility = 0` (xy ignored). |
| Reflective / chrome rim | Label normally. Place C on the visible rim/disc boundary, not on a reflection. A/B still go on the floor/ground near the wheel footprint. |
| Two wheels overlap (rear-quarter view, far wheel behind near wheel) | Label each wheel as its **own instance**. For the partially-hidden wheel, set keypoints occluded by the other wheel to `visibility = 1` if you can infer their position, else `0`. |
| Spare wheel on rear door / roof | **Do not** label. |
| Motorcycle / bicycle / mural / toy / detached wheel | **Do not** label. |
| Wheel only seen through a window from inside the cabin | **Do not** label. |
| Trailer wheel attached to a car / truck | Label as a normal wheel if it is driveable, ≥ 50% disc visible, and not a "spare". |
| Painted-over / dirt-covered rim where outline is still readable | Label normally. |

## 7. Consistency checks

These must hold for any reasonable orientation. Use them as a sanity
filter before submitting a wheel:

- `rim_left.x < rim_right.x` in image coordinates.
- `disc_bottom.y < min(rim_left.y, rim_right.y)` — C is visually above
  the floor-ray A/B points because y grows downward.
- `rim_left` and `rim_right` are near the bottom/footprint area, not on
  the metal rim, tire rubber, or wheel surface.
- Keypoints with `visibility = 0` have ignored xy — do not waste time
  placing them precisely.

`src/check_dataset.py` enforces the 14-field YOLO format and value
ranges; per-keypoint geometric checks above are *your* job.

## 8. Common mistakes to avoid

- Placing A/B (`rim_left` / `rim_right`) on the metal rim.
- Placing A/B on the tire rubber.
- Placing A/B on the wheel instead of on the floor/ground near the
  footprint.
- Confusing the legacy internal names `rim_left` / `rim_right` with the
  obsolete edge semantics. They now mean floor-ray A/B points.
- Placing C (`disc_bottom`) on the floor or tire instead of the lower
  metal rim / disc boundary where the rim meets the tire.
- Placing `disc_bottom` at the tire's road-contact point (bottom of
  rubber). It is the lower visible metal rim / disc point, not the tire.
- Annotating spare wheels on the back door / roof — these are explicitly
  out of scope.
- Forgetting to mark off-image keypoints as `visibility = 0` (the
  converter still emits zeros, but consistency matters for inter-rater
  agreement metrics).
- Skipping wheels that are heavily occluded but inferrable — label them
  with `visibility = 1`. Skipping them throws away useful occlusion
  training signal.

## See also

- `docs/KEYPOINT_SPEC.md` — geometric rationale for A / B / C.
- `docs/ANNOTATION_JSON_FORMAT.md` — interim per-image JSON schema the
  converter consumes (annotators do not write this directly; the export
  tool does).
- `docs/DATASET_SPEC.md` — on-disk YOLO-pose label format.
- `docs/AR_ML_CONTRACT.md` — confirmed ML output JSON and AR/ML
  responsibility split.
- `docs/ANNOTATION_TOOLING.md` — project setup in the labelling tool.
