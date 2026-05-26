# Unreal Export 0003 Feedback For Unreal Developer

Date: 2026-05-21

## Summary

Export `0003` is technically parseable by the current VSBL Unreal intake tools. The current point mapping also looks visually plausible in the reviewed contact sheets:

- `Left -> points.a`
- `Right -> points.b`
- `Center -> points.c_disc_bottom`
- `LeftTop` / `RightTop -> bbox/helper only`

However, production training is still blocked. The reasons are data quality and unconfirmed physical semantics:

- Technical compatibility: **GREEN with caveats**
- Visual semantic verdict: **PASS WITH RISKS**
- Data-quality verdict: **FAIL**
- Production training readiness: **NOT_READY**
- Smoke/plumbing use only: **PROVISIONAL_EXPERIMENT_ONLY**

The main unresolved semantic question is whether `Left` and `Right` are truly floor/ground footprint points, not tire/rim/wheel-surface points. ML/AR requires `points.a` and `points.b` to be 2D screen-space floor-ray pixels near the wheel footprint/base. They must not be points on the wheel mesh.

## Current Mapping

| Raw Unreal point | Current ML/AR target | Use |
|---|---|---|
| `Left` | `points.a` | left floor-ray / footprint point |
| `Right` | `points.b` | right floor-ray / footprint point |
| `Center` | `points.c_disc_bottom` | lower visible metal rim/disc point |
| `LeftTop` | bbox/helper | not part of final ML/AR point contract |
| `RightTop` | bbox/helper | not part of final ML/AR point contract |

This mapping is provisional until Unreal confirms the physical placement of the points in the scene.

## What Visual QA Showed

Reviewed artifacts:

- `outputs/unreal_export_0003_qa/contact_sheets/valid_contact_sheet.jpg`
- `outputs/unreal_export_0003_qa/contact_sheets/rejected_contact_sheet.jpg`
- `outputs/unreal_export_0003_qa/contact_sheets/multi_wheel_contact_sheet.jpg`
- `outputs/unreal_export_0003_qa/summary.json`

Visual observations:

- `Center` appears in the lower visible disc/rim area in sampled valid overlays, not at the wheel hub center.
- `Left` and `Right` appear near the wheel base/footprint in 2D screen space.
- Several A/B points are very close to the tire contact edge or shadow boundary, so visual QA cannot prove whether they are floor/ground points or wheel-surface points.
- Multiple-wheel frames are supported after filtering.
- Bboxes are derived from helper points or floor-ray heuristics; they are not confirmed native full-wheel object boxes.

The visual result is plausible enough for pipeline smoke tests, but not enough for production training approval without Unreal-side confirmation.

## Data Quality Numbers

| Metric | Count | Ratio |
|---|---:|---:|
| Frames/images | 1713 | - |
| Raw object slots | 24005 | 100.00% |
| All-zero objects | 10180 | 42.41% |
| Partial-zero objects | 1600 | 6.67% |
| Out-of-bounds objects | 8905 | 37.10% |
| Usable wheels after filtering | 1722 | 7.17% |
| Frames with 0 usable wheels | 640 | 37.36% |
| Frames with 1+ usable wheels | 1073 | 62.64% |

Usable wheels per frame:

```json
{
  "0": 640,
  "1": 585,
  "2": 353,
  "3": 113,
  "4": 18,
  "5": 4
}
```

Rejected object categories:

```json
{
  "all_zero": 10180,
  "partial_zero": 1600,
  "out_of_bounds": 8905,
  "bad_floorray_geometry": 1595,
  "invalid_bbox_after_clip": 3
}
```

The usable yield is only **7.17%** of raw object slots. That is the main production training blocker.

## Geometry Metrics For Usable Wheels

Computed from the `1722` usable converted wheels in `outputs/unreal_export_acceptance/unreal_0003/incoming/annotations`.

Definitions:

- `Left relative y`: `(points.a.y - bbox_y1) / bbox_h`
- `Right relative y`: `(points.b.y - bbox_y1) / bbox_h`
- `Center relative y`: `(points.c_disc_bottom.y - bbox_y1) / bbox_h`
- `Vertical gap`: `min(points.a.y, points.b.y) - points.c_disc_bottom.y`
- `A/B spread ratio`: `abs(points.b.x - points.a.x) / bbox_w`

| Metric | min | p05 | p25 | median | p75 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Left relative y | 0.800 | 0.815 | 0.894 | 0.990 | 1.000 | 1.000 | 1.000 |
| Right relative y | 0.800 | 0.817 | 0.888 | 0.999 | 1.000 | 1.000 | 1.000 |
| Center relative y | 0.549 | 0.746 | 0.789 | 0.811 | 0.835 | 0.863 | 0.987 |
| Vertical gap px | 0.072 | 4.617 | 11.449 | 18.204 | 28.291 | 52.552 | 134.188 |
| A/B spread ratio | 0.505 | 0.693 | 0.787 | 0.859 | 0.913 | 0.966 | 0.997 |

Additional geometry counts:

| Check | Count |
|---|---:|
| Usable wheels measured | 1722 |
| `Left.x >= Right.x` | 0 |
| `Center.y >= min(Left.y, Right.y)` | 0 |
| `Center.x` between `Left.x` and `Right.x` | 1722 |
| `Center.x` outside `Left.x`/`Right.x` span | 0 |

Interpretation:

- The usable subset is internally consistent after filtering: `Left.x < Right.x`, `Center.x` lies between A/B, and `Center.y` stays above the lower A/B footprint line in all usable cases.
- These metrics support the current 2D mapping, but they do not prove 3D physical placement. Unreal must still confirm whether A/B are floor/ground markers rather than wheel/tire/rim markers.

## Exact Questions For Unreal Developer

1. Where physically in the Unreal scene is `SphereLeft` / raw `Left` placed: floor/ground, tire mesh, rim, or wheel surface?
2. Where physically in the Unreal scene is `SphereRight` / raw `Right` placed: floor/ground, tire mesh, rim, or wheel surface?
3. Is `Center` exactly the lower visible metal rim/disc point where rim meets tire?
4. Are exported `XY` values final image pixels in the `2048x2048` image with top-left origin?
5. What does `(0,0)` mean: invisible, offscreen, missing projection, failed trace, or a real pixel?
6. What do extreme coordinates like `-290472` or `91623` mean?
7. What does `Goal/DeltaZ/Roll/Pitch/FOV` mean, including units and coordinate frame?
8. What does `Depth/*.jpg` encode: scene depth, custom depth, normalized depth, stencil/mask, or something else?
9. Can invisible/offscreen wheels be skipped instead of writing zero-placeholder object `.txt` files?
10. Can invalid/out-of-bounds objects be dropped or marked with an explicit validity flag?
11. Can native `bbox_xyxy` be exported per wheel, around the full visible wheel silhouette including tire?
12. Can the next clean export be named `0004`, or include an explicit build/export version in metadata?

## Requested Fixes For Next Export

For the next clean export:

- Do not write an object `.txt` file if the wheel is invisible/offscreen.
- Do not write an object if `Left`, `Right`, or `Center` is missing.
- Do not write garbage/extreme coordinates.
- Keep `Left` / `Right` naming as in `0003` if the floor/footprint semantics are confirmed.
- Keep `LeftTop` / `RightTop` as bbox/helper points.
- Add an explicit export/build version if possible.
- Add an explicit stable frame ID if possible.
- Prefer exporting native `bbox_xyxy` per wheel in final image pixel coordinates.
- If invalid objects must remain in the export, add an explicit validity/visibility flag and reason.

## Training Decision

Production training: **NOT_READY**.

Smoke/plumbing training: allowed only as a throwaway provisional experiment.

Weights trained on export `0003` must not be shipped to the AR team. They may be used only to validate plumbing, file formats, metrics, and preview tooling.
