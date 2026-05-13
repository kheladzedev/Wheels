# Annotation Guidelines — wheel + 3 keypoints

Audience: human annotator. You have never seen this project before. Read
this end to end, then start labelling. Expect ~10 minutes to internalise
the rules.

> **Pending confirmation.** Final keypoint names and the exact semantics
> of the bottom point are still open with the AR team
> (`docs/OPEN_QUESTIONS_AR_SPEC.md` §1, §3). Until that is resolved, use
> the names in `docs/ANNOTATION_JSON_FORMAT.md`: `rim_left`, `rim_right`,
> `disc_bottom`. The label *order* and *count* (3 keypoints per wheel) are
> locked — only the names may shift later. If they shift, the converter
> renames; no re-annotation needed for naming.

## 1. What gets labelled

| Object | Label? | Notes |
|---|---|---|
| Full wheel (tire rubber + metallic rim/disc) visible from outside the car | **Yes** | Single class `wheel`. |
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
contract; the names are working names.

| Index | Name | Where to place it |
|---|---|---|
| 0 | `rim_left`    | Left-most visible point of the **metallic rim** (the metal disc, **not** the tire rubber). For 3/4 views the visible rim is an ellipse — place the point at the geometric leftmost point of that ellipse. |
| 1 | `rim_right`   | Right-most visible point of the metallic rim. Ellipse rightmost. |
| 2 | `disc_bottom` | Physical **lowest** point of the metallic disc. For a perfectly straight-on (side) view, `disc_bottom` may coincide with `rim_right`. For 3/4 views, `disc_bottom` sits **below** `rim_right`. |

Definitions are aligned to the AR-team mock pipeline; see
`docs/KEYPOINT_SPEC.md` for the geometric rationale (two rim points
constrain a plane via raycast + RANSAC; one disc-bottom point gives
installation height once the plane is known).

> Earlier iterations used "highest / lowest" of the rim — that wording is
> stale. The current contract is **left / right** of the metallic rim
> (kp0, kp1) plus the disc's **physical bottom** (kp2). If the AR team
> later confirms top/bottom-of-rim semantics, we re-label — see
> `docs/OPEN_QUESTIONS_AR_SPEC.md` §1.

### Concrete placement rules

- A/B (`rim_left` / `rim_right`) sit on the **metal**, not on the tire
  rubber. If in doubt, follow the rim's bright outer edge.
- C (`disc_bottom`) sits on the metal. Do not place it on the tire's
  bottom (road-contact point), and do not place it at the hub centre.
- For chrome / highly reflective rims, use the rim's silhouette, not
  the reflection inside it.

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
| Reflective / chrome rim | Label normally — the rim outline is what matters, not the reflection. Place A/B on the actual rim silhouette. |
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
- `disc_bottom.y >= max(rim_left.y, rim_right.y)` — bottom is below or
  level with the rim left/right points (y grows downward).
- `rim_left` and `rim_right` sit on the **metallic disc**, never on the
  tire rubber.
- Keypoints with `visibility = 0` have ignored xy — do not waste time
  placing them precisely.

`src/check_dataset.py` enforces the 14-field YOLO format and value
ranges; per-keypoint geometric checks above are *your* job.

## 8. Common mistakes to avoid

- Placing `rim_left` / `rim_right` on the tire rubber instead of the
  metal rim.
- Placing `disc_bottom` at the tire's road-contact point (bottom of
  rubber). It is the bottom of the **metal disc**, not the tire.
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
- `docs/OPEN_QUESTIONS_AR_SPEC.md` — items still open with the AR team
  (final names, disc-bottom semantics, error budget).
- `docs/ANNOTATION_TOOLING.md` — project setup in the labelling tool.
