# Open Questions for the Team

Status as of 2026-05-12 after reading the AR mechanics spec ("Примерка колес")
at https://docs.google.com/document/d/1HwMfJYc3eWaovN183370iWYmLjTosF9UMconj-UawFg/

> **See also `docs/OPEN_QUESTIONS_AR_SPEC.md`** — focused follow-up list
> from the AR mock-pipeline clarification round (A/B/C semantics,
> per-wheel vs screen-fixed A/B, final field names, frame_id shape,
> tracking responsibility, error budget, Unreal export). That doc
> supersedes the corresponding items below where they overlap.

The spec closed a lot of earlier questions. What follows is the **live** list
of things ML needs from the AR team before we can train a useful model and
freeze the JSON contract.

## What the spec already locked down

These were open earlier, now closed — listed for the record so we don't
re-litigate:

- **Keypoints per wheel = 3.** Two on the rim (used for raycast → plane via
  RANSAC) + one "disc bottom" point (used for height of the disc above the
  recovered plane). Mock system in the spec confirms three points exactly.
- **Coordinates = pixels** (the spec says "screen coordinates" for raycast).
- **Camera transform passthrough is NOT needed.** AR saves the camera
  transform itself at frame capture time and uses it for raycast. ML only
  needs to return `frame_id` (or timestamp) so AR can match the response
  back to the right saved transform.
- **RANSAC, raycast, plane reconstruction, and K-frame accumulation live on
  the AR side.** ML returns per-frame, per-wheel keypoints only.
- **Multi-wheel per image is required.** Spec: "Нейросеть должна
  поддерживать обнаружение нескольких колес в одном кадре."

## Critical — blocks training

### Q1. What exactly is the "disc bottom" keypoint?

Three plausible interpretations:

- (a) Lowest visible point of the **metallic rim/disc** (the metal part).
- (b) Lowest point of the **wheel hub**.
- (c) Road-contact point of the **tire**.

The spec uses the word "диск" — strongly implying (a) — and uses this
keypoint specifically for "высота установки диска", i.e. where the
**virtual rim** should sit vertically. That's consistent with (a).

We will default to (a) for label instructions unless told otherwise.
Confirm or correct before annotation starts — re-annotation is expensive.

### Q2. Two rim keypoints — confirming left + right

The AR team's mock diagram ("Тестовый AR-пайплайн без нейросети") shows
the two rim keypoints as **точка А (левая)** and **точка В (правая)** —
i.e. left + right of the rim, not top + bottom. This is also the only
choice that's geometrically sound: top + bottom would lie on the wheel's
vertical axis and not constrain the plane's yaw during RANSAC.

We've switched our code defaults from `rim_top` / `rim_bottom` to
`rim_left` / `rim_right`. Confirming this is the final convention so the
annotation guide can be locked.

### Q3. Occluded keypoints

When a wheel is partially behind a fender / bumper / another wheel, do we
annotate the occluded keypoint with:

- (a) `visibility = 0` (COCO-pose convention) — keypoint position is
      labelled, loss skips it, model still learns context for the visible
      points of that wheel.
- (b) Drop the entire wheel instance from that frame.

(a) is the standard choice and gives more training signal. AR's RANSAC
filters bad observations across K frames anyway. Need explicit confirmation.

## Important — affects contract and architecture

### Q4. Per-keypoint confidence in the JSON?

AR uses RANSAC over K frames to filter noisy keypoints. A per-keypoint
confidence (separate from the wheel-level detection confidence) lets AR
weight observations or drop low-confidence keypoints before RANSAC.

The AR developer in the spec explicitly asked for raycast-hit logs to
calibrate RANSAC noise — per-keypoint confidence is the ML-side companion
to that log.

Default plan: include `keypoints[*].confidence` in the JSON. Confirm AR
wants this.

### Q5. Stable per-wheel ID across frames?

The spec describes accumulating observations "for each wheel" over K frames.
Two ways AR can be doing that:

- (a) AR associates wheels across frames itself, by 3D position after
      raycast. ML returns independent per-frame detections without IDs.
- (b) AR expects ML to provide a tracker that emits a stable per-wheel
      `track_id` across frames.

(a) is implied by the spec's "raycast then accumulate" flow and is the
default. (b) requires a different architecture (detector + tracker) and
significantly more work. Need explicit confirmation that (a) is correct.

### Q6. K — number of frames AR needs?

The spec mentions accumulating over "K" frames before RANSAC, but doesn't
fix the number. Not strictly an ML blocker, but it sets the inference FPS
budget: e.g. if K=20 and the user can wait 2 seconds, we need ~10 FPS on
target hardware.

## Important — needed to plan training

### Q7. Acceptable placement error

Numbers we need so "good enough" is defined:

- Max keypoint pixel error before AR placement looks off (per keypoint,
  or aggregated)?
- Max false-negative rate per frame (missed wheels)?
- Max false-positive rate per frame (phantom wheels)?
- Min recall on partially-occluded wheels (yes/no — is this important)?

Without numbers we'll either over- or under-invest in model capacity and
data volume.

### Q8. Available data right now

- How many real images with rims/wheels labelled at the keypoint level?
- How many real unlabelled images / videos we could annotate?
- Is the Unreal pipeline already producing renders? Volume per week?
- Licensing constraints on any of the above?

This determines whether we lead with synthetic-only training, mixed, or
real-only.

### Q9. Unreal export format

If Unreal is in the loop, what does the per-frame export contain?

- bbox JSON, mask, segmentation buffer, custom binary?
- Camera **intrinsics** and **extrinsics**?
- 3D world positions of wheel hubs / rim centers / disc-bottom points?

If 3D world positions are available for free from Unreal, we can synthesize
keypoint labels with zero manual labelling for synthetic data — huge win.

### Q10. Target platforms and order

Server → web → mobile? This determines export format (ONNX vs CoreML vs
TFLite) and our latency budget per inference call.

## Important — needed for production hardening

### Q11. Min ground-plane area `N` (square meters)

The spec's preparation phase says "система ищет пол на площади N м²". The
actual N isn't named. AR-side, not strictly an ML blocker — but useful for
us to know what "useful frame" looks like (e.g. how zoomed-in / far the
camera can be).

---

## Defaults we will use until told otherwise

| Decision                       | Default                                      |
|--------------------------------|----------------------------------------------|
| Keypoint order in label file   | `[rim_left, rim_right, disc_bottom]`         |
| Disc bottom definition         | Lowest visible point of the metal disc       |
| Occlusion convention           | COCO-pose `visibility = 0` for hidden points |
| Per-keypoint confidence in JSON| Included                                     |
| Tracking                       | Not implemented — per-frame detections only  |
| Coordinate frame               | Pixels, top-left origin                      |
| Class set                      | Single class `wheel` (rim is implicit in kps)|

These defaults will be revised once AR-team answers land.
