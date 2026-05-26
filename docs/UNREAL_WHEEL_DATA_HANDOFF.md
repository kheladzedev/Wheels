# Unreal Wheel Data Handoff

## Goal

Build a reproducible Unreal-to-ML data pipeline for wheel detection/pose:

- input: Unreal scene with vehicle meshes, `wheels` annotation actors, and `CameraCaptureWheels`;
- raw export: `Images/`, `keyPoint/`, `Depth/`, `Goal/`;
- ML dataset: YOLO-pose style labels with bbox plus `a`, `b`, `c_disc_bottom`;
- gate: train only after technical validation, bbox audit, and human preview.

The ML contract is 2D screen-space only. It does not include 3D coordinates,
raycast, RANSAC, tracking, depth, or plane recovery.

## Confirmed Output Contract

```json
{
  "frame_id": "...",
  "wheels": [
    {
      "bbox_xyxy": [0.0, 0.0, 0.0, 0.0],
      "confidence": 0.0,
      "points": {
        "a": [0.0, 0.0],
        "b": [0.0, 0.0],
        "c_disc_bottom": [0.0, 0.0]
      }
    }
  ]
}
```

Semantics:

- `points.a` / `points.b`: lower floor/raycast footprint points.
- `points.c_disc_bottom`: lower visible metal rim/disc point.
- `bbox_xyxy`: full visible tire plus rim, not only the lower keypoint area.

## Current Unreal Project

- Project: `/Users/edward/Desktop/VSBL/NeuralData1 2/NeuralData.uproject`
- Engine used: `/Users/Shared/Epic Games/UE_5.7`
- Capture actor: `CameraCaptureWheels`
- Annotation actor: `wheels`
- Main map currently inspected: `/Game/Wheels/maps/standartWheelsRoom`

Unreal MCP tools are discoverable in Codex, but the live MCP transport currently
returns `Transport closed`. The working control path is Unreal Python via
`UnrealEditor-Cmd`.

## What Is Automated

Scripts used for the current pipeline:

- `scripts/unreal_validate_capture_map.py`: read-only Unreal-side validation.
- `scripts/run_unreal_capture.sh`: runs a map and captures raw export files.
- `scripts/accept_neuraldata1_capture.py`: copies raw export and runs acceptance.
- `scripts/import_unreal_export.py`: imports Unreal keypoints into incoming JSON.
- `scripts/audit_unreal_bbox_quality.py`: bbox provenance/quality audit.
- `scripts/unreal_create_auto_wheel_capture_map.py`: creates a controlled
  duplicate-existing capture map for stable smoke batches.

## Verified Batch: Clean v2 Smoke

Map:

`/Game/Wheels/maps/standartWheelsRoom_auto_wheels_duplicate_existing_v2`

This map uses one existing vehicle mesh with two validated wheel actors. It is
good for pipeline validation and limited provisional experiments.

Acceptance artifact:

`outputs/unreal_export_acceptance_neuraldata1/unreal_neuraldata1_auto_duplicate_existing_v2_100f_20260526_0215/`

Results:

- Images: `100`
- Raw wheel objects: `200`
- Valid wheels: `191`
- Dropped all-zero: `0`
- Dropped out-of-bounds: `9`
- Bad floor-ray geometry: `0`
- Empty label images: `4 / 100`
- Technical status: `PASS`
- Data-quality gate: `PASS`
- Training status: `NOT_APPROVED_FOR_TRAINING_UNTIL_HUMAN_PREVIEW_ACCEPTS_GEOMETRY`
- BBox audit: `191` wheels, point-derived bbox signal `0.0%`

Important limitation:

- Native plugin `WheelBBox` / `BBox` is still not present.
- BBox is synthesized by the adapter from available annotation geometry, not
  exported by the plugin.
- Therefore this remains provisional/debug until human preview or native bbox
  export is accepted.

Preview/audit paths:

- `outputs/unreal_export_acceptance_neuraldata1/unreal_neuraldata1_auto_duplicate_existing_v2_100f_20260526_0215/previews/incoming/`
- `outputs/unreal_export_acceptance_neuraldata1/unreal_neuraldata1_auto_duplicate_existing_v2_100f_20260526_0215/previews/pose/train/`
- `outputs/unreal_bbox_audit/unreal_neuraldata1_auto_duplicate_existing_v2_100f_20260526_0215/contact_sheet.jpg`
- `outputs/unreal_bbox_audit/unreal_neuraldata1_auto_duplicate_existing_v2_100f_20260526_0215/report.md`

## Verified Batch: Existing Multi-Vehicle Map

Map:

`/Game/Wheels/maps/standartWheelsRoom`

Scene inventory from read-only dump:

- Vehicle actors detected: `15`
- Wheel actors detected: `6`
- Capture actors detected: `1`

Acceptance artifact:

`outputs/unreal_export_acceptance_neuraldata1/unreal_neuraldata1_standartWheelsRoom_multi_vehicle_100f_20260526_0259/`

Results:

- Images: `100`
- Raw wheel objects: `600`
- Valid wheels: `38`
- Dropped all-zero: `253`
- Dropped out-of-bounds: `257`
- Bad floor-ray geometry: `52`
- Empty label images: `74 / 100`
- Technical status: `PASS`
- Data-quality gate: `FAIL`
- Training status: `NOT_APPROVED_FOR_TRAINING_DATA_QUALITY_GATE_FAILED`

Conclusion:

The scene has multiple 3D vehicle actors, but the wheel annotations are not
clean enough for training. This map is useful for debugging annotation coverage,
not for training.

## Current Blockers

1. Native plugin bbox is missing.
   - Required field: `BBox` or `WheelBBox` per wheel object in image pixels.
   - Without it, importer must keep `training_approved=false`.

2. Multi-vehicle annotation coverage is poor.
   - Many objects export as all-zero or out-of-bounds.
   - Some floor/raycast points fail geometry checks.

3. Direct MCP session is not currently connected.
   - Tool discovery works.
   - Calls fail with `Transport closed`.
   - Commandlet-based Unreal Python remains the reliable automation path.

## Commands To Reproduce

Validate a map:

```bash
UE_CMD="/Users/Shared/Epic Games/UE_5.7/Engine/Binaries/Mac/UnrealEditor-Cmd"
UPROJECT="/Users/edward/Desktop/VSBL/NeuralData1 2/NeuralData.uproject"
VSBL_UNREAL_VALIDATE_MAP="/Game/Wheels/maps/standartWheelsRoom" \
VSBL_UNREAL_VALIDATE_OUT="outputs/unreal_control/validate_standartWheelsRoom.json" \
"$UE_CMD" "$UPROJECT" \
  -run=pythonscript \
  -script="/Users/edward/Desktop/VSBL/scripts/unreal_validate_capture_map.py" \
  -unattended -nop4 -nosplash -NullRHI
```

Capture a map:

```bash
scripts/run_unreal_capture.sh standartWheelsRoom_auto_wheels_duplicate_existing_v2 100
```

Run acceptance:

```bash
./.venv/bin/python scripts/accept_neuraldata1_capture.py \
  --source-name neuraldata1_auto_duplicate_existing_v2_100f_manual \
  --overwrite \
  --preview-count 30 \
  --right-left-mapping auto
```

Run bbox audit:

```bash
./.venv/bin/python scripts/audit_unreal_bbox_quality.py \
  --source-root outputs/unreal_export_acceptance_neuraldata1/<run>/incoming \
  --out-dir outputs/unreal_bbox_audit/<run> \
  --max-samples 30
```

## Recommended Next Work

1. Add native `WheelBBox` / `BBox` export in the plugin.
2. Fix or suppress invalid `wheels` actors before writing object txt files:
   all-zero, out-of-bounds, missing required points, broken A/B/C geometry.
3. Clean multi-vehicle map annotations:
   start from the 15-vehicle `standartWheelsRoom` and validate each wheel actor.
4. Re-run acceptance after every map or plugin change.
5. Train only after acceptance says the batch passed and human preview confirms
   geometry.

Do not train on the current multi-vehicle batch. Do not claim AR-ready from
these exports.
