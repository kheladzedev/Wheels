# Unreal Import Provenance

Date: 2026-05-21

## Purpose

This document tracks Unreal export mapping and provenance rules for VSBL wheel-fitting data intake.

It exists to prevent silent drift between:

- raw Unreal point names,
- converted training annotation names,
- the confirmed ML/AR runtime contract,
- and the data-quality/training approval gate.

The confirmed ML/AR contract remains unchanged:

```json
{
  "frame_id": "...",
  "wheels": [
    {
      "bbox_xyxy": [0, 0, 0, 0],
      "confidence": 0.0,
      "points": {
        "a": [0, 0],
        "b": [0, 0],
        "c_disc_bottom": [0, 0]
      }
    }
  ]
}
```

`points.a` and `points.b` are 2D screen-space floor-ray points near the wheel footprint/base. They are not rim, tire, or wheel-surface points. `points.c_disc_bottom` is the lower visible metal rim/disc point.

## Current Export Family

### `0002`

Historical trial export.

Known behavior: older/inverted Right/Left handling was required during intake diagnostics. Treat any `0002` mapping as legacy and verify against its import metadata before reusing it.

### `0003`

Current reviewed export family.

Current provisional mapping:

- `Left -> points.a`
- `Right -> points.b`
- `Center -> points.c_disc_bottom`
- `LeftTop` / `RightTop -> bbox/helper`

Status:

- Technical compatibility: green with caveats.
- Visual semantics: pass with risks.
- Data quality: fail.
- Production training: not ready.
- Smoke/plumbing only: provisional experiment.

## Mapping Convention

For the current `0003` family:

| Raw Unreal point | Converted point | Meaning |
|---|---|---|
| `Left` | `a` | left floor-ray / footprint point |
| `Right` | `b` | right floor-ray / footprint point |
| `Center` | `c_disc_bottom` | lower visible metal rim/disc point |
| `LeftTop` | bbox/helper | helper only |
| `RightTop` | bbox/helper | helper only |

Do not silently change this mapping. If a future export changes raw naming or physical point placement, record the change in the import manifest and create a new QA report.

## Required Provenance Fields

Any future importer/conversion manifest must record:

- `raw_export_path`
- `export_id` / `build_id`
- `mapping_convention`
- `image_size`
- `frame_count`
- `raw_object_slots`
- `usable_wheels`
- `rejection_counts`
- `converter_version`
- `timestamp`
- whether mapping was automatic or manually overridden

Recommended extra fields:

- `scene_id` or map name
- `camera_id` or capture rig name
- export plugin version
- Unreal project version
- Unreal Engine version
- whether native bbox was exported or derived
- whether depth/goal metadata was included
- point-validity policy, for example omit invalid object vs write placeholders

## Open Semantic Questions

These must be answered before production training approval:

- Physical placement of `Left`: floor/ground vs tire/rim/wheel surface.
- Physical placement of `Right`: floor/ground vs tire/rim/wheel surface.
- Physical meaning of `Center`: lower visible metal rim/disc point vs hub/center/other.
- Whether `XY` values are final image pixels in the exported RGB frame.
- Meaning of `(0,0)` placeholder values.
- Meaning of extreme coordinates far outside the image.
- Meaning and units of `Goal/DeltaZ/Roll/Pitch/FOV`.
- Meaning and pixel encoding of `Depth/*.jpg`.
- Whether bbox is derived from point helpers or exported as a native full-wheel box.

## Production Training Gate

An Unreal export is production-trainable only after all of these are satisfied:

- Visual QA is signed off.
- Unreal developer confirms point semantics.
- Usable yield is `>= 50%`, or a lower yield is explicitly explained and accepted.
- Empty-label frame ratio is `<= 15%`, or a higher ratio is explicitly explained and accepted.
- Scene-aware split is implemented, or Unreal provides `scene_id`.
- Mapping/provenance is saved in a manifest.
- Native or otherwise approved `bbox_xyxy` quality is confirmed.
- Invalid/offscreen objects are omitted or explicitly marked invalid.

Until these gates pass, outputs may be used only for QA, plumbing, smoke tests, and provisional experiments. They must not be treated as production training data.
