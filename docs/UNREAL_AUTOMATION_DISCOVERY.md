# Unreal Automation Discovery

Date: 2026-05-22
Workspace: `/Users/edward/Desktop/VSBL`

This is a discovery-only report. No importer, validator, model, dataset, or
Unreal asset behavior was changed.

## Repository State

Commands requested:

```bash
pwd
git status --short
git diff --stat
```

Observed:

```text
/Users/edward/Desktop/VSBL
 M README.md
?? 0001/
?? 0002/
?? 0003/
?? "NeuralData1 2/"
?? docs/NEURALDATA1_CAPTURE_WORKFLOW.md
?? docs/UNREAL_EXPORT_0003_FEEDBACK.md
?? docs/UNREAL_EXPORT_0003_QA.md
?? docs/UNREAL_IMPORT.md
?? scripts/accept_neuraldata1_capture.py
?? scripts/unreal_create_clean_capture_map.py
?? scripts/unreal_validate_capture_map.py
?? tests/test_accept_neuraldata1_capture.py

README.md | 21 +++++++++++++++++++++
1 file changed, 21 insertions(+)
```

Interpretation: the working tree was already dirty before this discovery
report. The active Unreal project and raw export folders are currently
untracked in Git.

## Project Structure Checked

Checked these roots:

- `src/`
- `scripts/`
- `docs/`
- `configs/`
- `tests/`
- `data/incoming/`
- `outputs/full_pipeline_audit/`

Relevant existing surfaces:

- Incoming validators and converters:
  - `src/check_keypoint_incoming.py`
  - `src/preview_keypoint_annotations.py`
  - `src/convert_keypoint_incoming_to_yolo_pose.py`
  - `src/check_yolo_pose_dataset.py`
  - `src/preview_yolo_pose_labels.py`
- Unreal/raw export tools:
  - `scripts/inspect_unreal_export.py`
  - `scripts/import_unreal_export.py`
  - `scripts/accept_unreal_export.py`
  - `scripts/audit_unreal_bbox_quality.py`
  - `scripts/accept_first_plugin_batch.sh`
  - `scripts/accept_neuraldata1_capture.py`
  - `scripts/unreal_validate_capture_map.py`
  - `scripts/unreal_create_clean_capture_map.py`
- Audit artifacts:
  - `outputs/full_pipeline_audit/REPORT.md`
  - `outputs/full_pipeline_audit/REPORT.json`
  - `outputs/full_pipeline_audit/03_unreal_import_bbox.md`
  - `outputs/full_pipeline_audit/03_unreal_import_bbox.json`

## Contract Summary

The current AR/ML contract remains unchanged:

- ML returns only 2D screen-space values.
- No ML-side 3D, raycast, RANSAC, tracking, plane recovery, depth, or world
  coordinates.
- Confirmed JSON shape is:

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

Semantic constraints:

- `points.a` / `points.b` are floor/raycast footprint points.
- `points.c_disc_bottom` is the lower visible metal disc/rim point.
- `bbox_xyxy` must cover the full visible wheel: tire plus rim.

## Current Importer

Current raw Unreal importer:

```text
scripts/import_unreal_export.py
```

It currently parses point records through:

```text
scripts/inspect_unreal_export.py::parse_keypoint_text
```

Supported raw point names observed in the code and fixtures:

- `Right`
- `Left`
- `Center`
- `LeftTop`
- `RightTop`
- aliases: `SphereRight`, `SphereLeft`, `SphereRightTop`, `SphereLeftTop`

Current bbox behavior:

- If `LeftTop` / `RightTop` exist and are valid, importer builds bbox by
  min/max over `Right`, `Left`, `Center`, `LeftTop`, `RightTop`.
- Otherwise it falls back to a floor-ray heuristic.
- Older audit artifacts also record the previous 3-point margin synthesizer
  for `data/incoming/android_plugin_real`.

Important gap: the importer does not currently parse a real plugin-provided
`BBox` / `WheelBBox` field from raw `keyPoint/*.txt`.

## Current Blocker

Authoritative audit:

```text
outputs/full_pipeline_audit/REPORT.json
```

Key fields:

```text
overall_status: BLOCKED_ON_PLUGIN_WHEEL_BBOX
training_allowed: false
ar_ready_claim_allowed: false
unreal_import_status: ACCEPT_ONLY_AS_DEBUG
external_blocker: plugin must export WheelBBox/BBox around full tire/wheel
next_plugin_request: Add BBox XYXY per visible wheel in final image pixel coordinates
```

Current acceptance marker:

```text
data/incoming/android_plugin_real/metadata/acceptance_status.json
```

Key fields:

```json
{
  "status": "ACCEPT_ONLY_AS_DEBUG",
  "training_allowed": false,
  "requires_plugin_bbox": true,
  "requires_human_preview": true
}
```

Audit finding:

```text
point_derived_bbox_fraction = 1.0
median_aspect_w_over_h = 1.577
median_c_disc_bottom_relative_y = 0.397
```

Interpretation: the imported points are schema-valid, but the training bbox
is not confirmed as a full wheel bbox. Training must remain blocked until the
plugin emits a real object-level bbox or a human explicitly approves a
fallback target.

## Raw Export Folders Checked

`~/Downloads/0001`, `~/Downloads/0002`, and `~/Downloads/0003` are not present
now. Their copies are present inside the repo:

```text
/Users/edward/Desktop/VSBL/0001
/Users/edward/Desktop/VSBL/0002
/Users/edward/Desktop/VSBL/0003
```

Counts:

| Root | Images | Ground | keyPoint | Raw point names | BBox-like matches |
| --- | ---: | ---: | ---: | --- | ---: |
| `0001` | 869 | 869 | 4392 | `Right`, `Left`, `Center` | 0 |
| `0002` | 347 | - | 1594 | `Right`, `Left`, `Center`, `LeftTop`, `RightTop` | 0 |
| `0003` | 1713 | 1713 | 24005 | `Right`, `Left`, `Center`, `LeftTop`, `RightTop` | 0 |
| `NeuralData1 2` current export | 100 | 0 | 300 | `Right`, `Left`, `Center`, `LeftTop`, `RightTop` | 0 |

`/Users/edward/Downloads/NeuralData1` exists but is only 124K and currently
contains only `Saved/`; it is not the active project/export root.

## Unreal Project / Plugin Code Found

Found active Unreal project:

```text
/Users/edward/Desktop/VSBL/NeuralData1 2/NeuralData.uproject
```

Size:

```text
33G  NeuralData1 2
```

Found Unreal C++ module files:

```text
NeuralData1 2/Source/NeuralData/NeuralData.Build.cs
NeuralData1 2/Source/NeuralData/NeuralData.cpp
NeuralData1 2/Source/NeuralData/NeuralData.h
NeuralData1 2/Source/NeuralData/TxtWriter.cpp
NeuralData1 2/Source/NeuralData/TxtWriter.h
NeuralData1 2/Source/NeuralData.Target.cs
NeuralData1 2/Source/NeuralDataEditor.Target.cs
```

The C++ source is minimal. `TxtWriter` exposes a Blueprint-callable
`CreateFile` helper. The capture/export logic itself appears to be in Unreal
Blueprint assets, especially:

```text
NeuralData1 2/Content/Wheels/main/CameraCaptureWheels.uasset
NeuralData1 2/Content/Wheels/main/CanvasWheels.uasset
NeuralData1 2/Content/Wheels/main/wheels.uasset
```

No `.uplugin` file was found in the repo discovery output. This looks like a
project with Blueprint assets and a small C++ helper module, not a standalone
source plugin with editable exporter code in plain text.

Maps found:

```text
NeuralData1 2/Content/Wheels/maps/standartWheelsRoom.umap
NeuralData1 2/Content/Wheels/maps/standartWheelsRoom_capture_clean.umap
NeuralData1 2/Content/Wheels/maps/standartWheelsRoom_capture_clean_v2.umap
```

## Unreal Automation Status

Unreal 5.7 command binaries exist on this machine:

```text
/Users/Shared/Epic Games/UE_5.7/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor
/Users/Shared/Epic Games/UE_5.7/Engine/Binaries/Mac/UnrealEditor-Cmd
```

Existing automation that can run through `UnrealEditor-Cmd`:

- `scripts/unreal_validate_capture_map.py`
  - read-only map validation
  - checks `CameraCaptureWheels`, `wheels` actors, components, size/height,
    and best-effort floor traces
- `scripts/unreal_create_clean_capture_map.py`
  - duplicates a map and removes selected bad wheel actors by observed export id

Existing acceptance wrapper:

- `scripts/accept_neuraldata1_capture.py`
  - copies already-generated `Images/`, `keyPoint/`, `Depth/`, `Goal/`
  - runs the official `scripts/accept_unreal_export.py`
  - does not itself start Unreal capture

Conclusion: direct Unreal automation is partially available now. We can
validate maps and create cleaned maps via commandlets. A full one-command
"generate data from Unreal then validate" workflow is not complete yet,
because the current wrapper starts after Unreal has already written export
folders.

## Can We Automate Unreal Directly Now?

Yes, partially:

- Map validation can be automated.
- Clean-map creation can be automated.
- Copying exports and running acceptance can be automated.
- BBox/data-quality validation can be automated.

Not fully yet:

- Starting a controlled capture run from Unreal and waiting for N frames is not
  currently wrapped in one stable script.
- Modifying the exporter to emit real `BBox` / `WheelBBox` is not directly
  available as plain Python/C++ source in the repo; the relevant logic appears
  to live in Blueprint `.uasset` assets.
- No active Unreal MCP/editor remote-control integration was found in this
  repository. The current practical automation path is `UnrealEditor-Cmd` plus
  Unreal Python commandlets.

## What Is Already Good

- The AR/ML JSON contract is documented and test-backed.
- The raw Unreal export scanner exists.
- The importer exists.
- The incoming validator exists.
- The YOLO-pose converter exists.
- The YOLO dataset validator exists.
- Preview tools exist for both incoming annotations and YOLO-pose labels.
- Full audit artifacts already identify the central blocker.
- Unreal commandlet scripts already exist for map validation and clean-map
  creation.

## What Is Missing

Required before production training:

1. Raw plugin export must include a real per-wheel `BBox` / `WheelBBox` around
   the full visible tire plus rim.
2. `scripts/inspect_unreal_export.py` must detect/report that field.
3. `scripts/import_unreal_export.py` must parse and use that field as
   `wheels[].bbox_xyxy`.
4. Synthetic bbox fallback should become explicit debug-only behavior, not the
   default production path.
5. `scripts/audit_unreal_bbox_quality.py` should become bbox-source aware.
6. Acceptance should fail or downgrade to debug if plugin bbox is missing.
7. A single orchestrator should connect:
   - Unreal map validation
   - controlled capture/export
   - raw archive copy
   - raw scan
   - import
   - incoming preview
   - YOLO conversion
   - YOLO preview
   - bbox audit
   - final `REPORT.md` / `REPORT.json`

## What Would Be Needed From The User

Because the Unreal project is now present in the repo, no extra project folder
is needed for discovery.

For full automation or exporter changes, one of these is needed:

- A new export from the current project that already includes `BBox` /
  `WheelBBox`; or
- Permission/time to edit the Blueprint exporter inside Unreal Editor and save
  the asset; or
- A plain-text C++/plugin source implementation of the exporter logic if Igor
  has one outside the `.uasset` Blueprint graph; or
- Exact confirmation that top helper points are acceptable as production bbox
  targets, plus human preview sign-off. Current audit does not grant that.

## Recommended Next Implementation Step

Do not train yet.

First implement contract-safe BBox intake:

1. Add `BBox` / `WheelBBox` parsing to `scripts/inspect_unreal_export.py`.
2. Add plugin-bbox import path to `scripts/import_unreal_export.py`.
3. Add strict acceptance mode that requires plugin-provided bbox for training.
4. Add tests for raw BBox parsing, import metadata, and debug-only fallback.
5. Re-run the acceptance pipeline on the current raw exports. It should still
   report debug-only until a real BBox appears in raw `keyPoint` files.
