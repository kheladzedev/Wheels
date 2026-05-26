# Unreal Export 0003 QA

## Summary

Source: `/Users/edward/Downloads/0003`
Generated artifacts: `outputs/unreal_export_0003_qa`
Mapping used: `Left -> points.a`, `Right -> points.b`, `Center -> points.c_disc_bottom`; `LeftTop` / `RightTop` are bbox helpers.

This QA pass did not train a model and did not change the ML/AR contract or runtime inference code.

## Counts

| Metric | Count | Ratio |
|---|---:|---:|
| Total frames/images | 1713 | - |
| Raw object slots | 24005 | 100.00% |
| All-zero objects | 10180 | 42.41% |
| Partial-zero objects | 1600 | 6.67% |
| Out-of-bounds objects | 8905 | 37.10% |
| Raw objects with required points in image | 3320 | 13.83% |
| Usable wheels after geometry/bbox gates | 1722 | 7.17% |
| Frames with 0 usable wheels | 640 | 37.36% |
| Frames with 1+ usable wheels | 1073 | 62.64% |

Usable wheels per frame distribution: `{'0': 640, '1': 585, '2': 353, '3': 113, '4': 18, '5': 4}`.

Image resolution: `{'2048x2048': 1713}`. Ground metadata parsed for `1713` frames.

## Rejection Reasons

Filtered import rejection categories:

```json
{
  "all_zero": 10180,
  "partial_zero": 1600,
  "out_of_bounds": 8905,
  "bad_floorray_geometry": 1595,
  "invalid_bbox_after_clip": 3
}
```

BBox strategy for usable wheels:

```json
{
  "top_points": 1681,
  "floorray": 41
}
```

## Overlay Samples

Contact sheets:

- Valid usable wheels: `outputs/unreal_export_0003_qa/contact_sheets/valid_contact_sheet.jpg`
- Rejected examples: `outputs/unreal_export_0003_qa/contact_sheets/rejected_contact_sheet.jpg`
- Multiple-wheel frames: `outputs/unreal_export_0003_qa/contact_sheets/multi_wheel_contact_sheet.jpg`

Generated overlay counts: `{'valid': 30, 'rejected': 17, 'multi_wheel': 8}`.

### Valid Sample Paths

- `outputs/unreal_export_0003_qa/overlays/valid/01_frame_510_obj_5_valid.jpg`
- `outputs/unreal_export_0003_qa/overlays/valid/02_frame_527_obj_5_valid.jpg`
- `outputs/unreal_export_0003_qa/overlays/valid/03_frame_428_obj_0_valid.jpg`
- `outputs/unreal_export_0003_qa/overlays/valid/04_frame_523_obj_5_valid.jpg`
- `outputs/unreal_export_0003_qa/overlays/valid/05_frame_8_obj_5_valid.jpg`
- `outputs/unreal_export_0003_qa/overlays/valid/06_frame_15_obj_6_valid.jpg`
- `outputs/unreal_export_0003_qa/overlays/valid/07_frame_2_obj_1_valid.jpg`
- `outputs/unreal_export_0003_qa/overlays/valid/08_frame_7_obj_6_valid.jpg`
- `outputs/unreal_export_0003_qa/overlays/valid/09_frame_0_obj_1_valid.jpg`
- `outputs/unreal_export_0003_qa/overlays/valid/10_frame_60_obj_3_valid.jpg`
- `outputs/unreal_export_0003_qa/overlays/valid/11_frame_117_obj_3_valid.jpg`
- `outputs/unreal_export_0003_qa/overlays/valid/12_frame_173_obj_1_valid.jpg`

### Rejected Sample Paths

- `outputs/unreal_export_0003_qa/overlays/rejected/01_frame_0_obj_5_empty_all_zero.jpg`
- `outputs/unreal_export_0003_qa/overlays/rejected/02_frame_465_obj_6_empty_all_zero.jpg`
- `outputs/unreal_export_0003_qa/overlays/rejected/03_frame_895_obj_0_empty_all_zero.jpg`
- `outputs/unreal_export_0003_qa/overlays/rejected/04_frame_1314_obj_0_empty_all_zero.jpg`
- `outputs/unreal_export_0003_qa/overlays/rejected/05_frame_1712_obj_13_empty_all_zero.jpg`
- `outputs/unreal_export_0003_qa/overlays/rejected/06_frame_8_obj_9_partial_zero.jpg`
- `outputs/unreal_export_0003_qa/overlays/rejected/07_frame_641_obj_5_partial_zero.jpg`
- `outputs/unreal_export_0003_qa/overlays/rejected/08_frame_963_obj_14_partial_zero.jpg`
- `outputs/unreal_export_0003_qa/overlays/rejected/09_frame_1246_obj_6_partial_zero.jpg`
- `outputs/unreal_export_0003_qa/overlays/rejected/10_frame_1712_obj_7_partial_zero.jpg`
- `outputs/unreal_export_0003_qa/overlays/rejected/11_frame_0_obj_0_out_of_bounds.jpg`
- `outputs/unreal_export_0003_qa/overlays/rejected/12_frame_341_obj_11_out_of_bounds.jpg`

### Multiple-Wheel Sample Paths

- `outputs/unreal_export_0003_qa/overlays/multi_wheel/01_frame_510_5_wheels.jpg`
- `outputs/unreal_export_0003_qa/overlays/multi_wheel/02_frame_527_5_wheels.jpg`
- `outputs/unreal_export_0003_qa/overlays/multi_wheel/03_frame_428_4_wheels.jpg`
- `outputs/unreal_export_0003_qa/overlays/multi_wheel/04_frame_523_4_wheels.jpg`
- `outputs/unreal_export_0003_qa/overlays/multi_wheel/05_frame_8_3_wheels.jpg`
- `outputs/unreal_export_0003_qa/overlays/multi_wheel/06_frame_15_3_wheels.jpg`
- `outputs/unreal_export_0003_qa/overlays/multi_wheel/07_frame_2_2_wheels.jpg`
- `outputs/unreal_export_0003_qa/overlays/multi_wheel/08_frame_7_2_wheels.jpg`

## Visual QA Notes

Contact-sheet review notes:

- Sampled accepted overlays cover different cars/views and include sedan, taxi, SUV/pickup, bus/truck, and close wheel views.
- In many accepted samples, `points.a` and `points.b` land on the visible floor/base region near the tire footprint, not on the rim.
- Some accepted samples place A/B very close to the tire contact edge or near shadow boundaries; this remains a semantic risk unless Unreal confirms these are floor markers.
- `Center / points.c_disc_bottom` generally appears on or near the lower visible metal rim/disc area in sampled valid overlays.
- Bboxes are derived from helper points or floor-ray heuristics, not native Unreal object bboxes. Many are reasonable for visual review, but this is still weaker than exported full-wheel boxes.
- Multi-wheel overlays confirm the importer can emit multiple wheels per frame, including 2, 3, 4, and 5 usable-wheel frames.
- Rejected overlays confirm all-zero placeholders, partial-zero objects, and extreme out-of-frame coordinates are common in this batch.

## Compatibility Verdict

**PASS WITH RISKS for conversion only.**

The filtered subset can be converted into the confirmed ML/AR JSON shape, but the raw export is not clean enough to treat as production training data. The current raw data has a high invalid-slot rate and depends on derived bboxes.

Contract compatibility details:

- `frame_id`: derived from image/keyPoint folder stem.
- `wheels[]`: supported after filtering; multiple wheels per frame are present.
- `points.a`: mapped from raw `Left`.
- `points.b`: mapped from raw `Right`.
- `points.c_disc_bottom`: mapped from raw `Center`.
- `bbox_xyxy`: derived from `LeftTop` / `RightTop` plus required points when possible; otherwise from floor-ray heuristic.
- `confidence`: not present in labels, expected to be model/runtime output later.

## Training Readiness Verdict

**Final verdict: NOT_READY**

Recommendation: **do not train now** on export `0003` as production data.

Reasoning:

- Only `1722` of `24005` raw object slots survive as usable wheels.
- `10180` object slots are all-zero.
- `10505` object slots are partial-zero or out-of-bounds.
- `640` of `1713` frames have no usable wheels after filtering.
- Bboxes are not native Unreal object bboxes; they are derived from point geometry.

Allowed use: QA/debug and provisional experiments only, after human acceptance of geometry samples. Not suitable for production training approval.

## Remaining Risks

- A/B semantics are still the largest risk: they must be floor/footprint screen points, not tire/rim points. Visual overlays are encouraging in many samples but not definitive for all views.
- `Center` looks plausible in samples, but needs exporter-side confirmation that it is always the lower visible metal rim/disc point.
- Heavy invalid-slot volume suggests the exporter is writing placeholders for invisible/offscreen wheels instead of omitting them.
- OOB coordinates can be extremely large, which increases converter complexity and QA burden.
- Derived bboxes may bias detector training toward keypoint hulls rather than full wheel silhouettes.
- Some frames contain many raw object slots but few usable wheels, so class balance and empty-label ratios remain poor.

## Questions For Unreal Developer

1. Are raw `Left` and `Right` generated from floor/ground footprint markers, not from tire/rim mesh points?
2. Is raw `Center` always the lower visible metal rim/disc point where rim meets tire?
3. Why are all-zero slots exported instead of omitting invisible wheel objects?
4. Why are many objects partially zero or far out of image bounds?
5. Can the exporter emit one explicit visibility/validity flag per wheel object?
6. Can the exporter emit a native `bbox_xyxy` around the full visible wheel silhouette, including tire?
7. Should OOB objects be clipped, omitted, or marked invisible at export time?
8. Does each object id remain stable enough within a frame to support multi-wheel QA and debugging?

## Exact Next Tasks

1. Send this report plus contact sheets to the Unreal developer for marker semantics and placeholder-export clarification.
2. Ask for a cleaned export that omits invisible/all-zero objects and exports native full-wheel bboxes.
3. Re-run this same QA pass on the cleaned export.
4. Approve training only if usable ratio, invalid-required ratio, empty-frame ratio, and visual geometry all pass review.
