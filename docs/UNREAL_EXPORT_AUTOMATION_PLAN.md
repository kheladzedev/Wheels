# Unreal Export Automation Plan

Date: 2026-05-22
Scope: VSBL wheel-fitting ML data intake and Unreal/plugin export automation.

This plan keeps the confirmed ML/AR contract unchanged. ML outputs only 2D
screen-space `bbox_xyxy` plus `points.a`, `points.b`,
`points.c_disc_bottom`. AR owns raycast, RANSAC, plane recovery, tracking,
camera transforms, and object placement.

Current discovery result: the active Unreal project exists at
`/Users/edward/Desktop/VSBL/NeuralData1 2/NeuralData.uproject`, but the export
logic appears to live mostly in Blueprint `.uasset` assets. Plain-text source
only exposes a minimal `TxtWriter` helper. No raw export checked so far emits
`BBox` / `WheelBBox`.

## A. If Unreal / Plugin Source Is Available

Use this path if the exporter logic is editable either as C++ source or inside
the Unreal Blueprint graph.

### Add BBox Export

Add one object-level wheel bbox to every valid exported wheel object:

```text
{name:"BBox",XYXY:x1,y1,x2,y2}
```

or:

```text
{name:"WheelBBox",XYXY:x1,y1,x2,y2}
```

Rules:

- Coordinates are image pixels in the final exported image space.
- Use axis-aligned `x1,y1,x2,y2`.
- The bbox must cover the full visible tire plus rim, not only the lower
  keypoint cluster and not only the metal rim.
- Use the same frame/object file as the existing `Right`, `Left`, `Center`,
  `LeftTop`, `RightTop` records.

Likely edit locations:

- Blueprint exporter around `CameraCaptureWheels` / `GenerateObjectText`.
- Blueprint object geometry around `wheels`, if bbox is derived from
  `SphereLeftTop` / `SphereRightTop` / object size.
- If moved to C++ later, implement in the project module under
  `NeuralData1 2/Source/NeuralData/`, next to or replacing the current
  Blueprint-only text assembly that calls `UTxtWriter::CreateFile`.

### Add Validation Before Writing Object TXT

Before writing each object txt, validate the projected fields:

- Required points exist: `Right`, `Left`, `Center`.
- Preferred bbox exists: `BBox` / `WheelBBox`.
- No required point is `(0,0)` unless the whole object will be omitted.
- All required points are inside the final image bounds.
- BBox is inside or safely clipped to image bounds.
- BBox has positive area.
- BBox covers plausible wheel geometry:
  - width and height are both non-trivial;
  - aspect ratio is plausible for a visible wheel;
  - `Center` / `c_disc_bottom` is inside the bbox;
  - A/B footprint points sit near the lower bbox band;
  - hidden or fully offscreen wheels are not exported.

### Drop Invalid Objects

Do not write object txt files for invalid wheels.

Drop the whole object when:

- wheel is invisible;
- wheel is fully out of frame;
- any required point is missing;
- any required point is all-zero / partial-zero;
- BBox is missing;
- BBox does not cover the visible wheel;
- BBox is degenerate after clipping.

This is better than writing garbage and relying on the ML importer to discard
it, because it keeps raw export counts meaningful and reduces label noise.

### Manual Export Run

Manual workflow inside Unreal:

1. Open `NeuralData1 2/NeuralData.uproject`.
2. Open the target map, for example:
   - `Content/Wheels/maps/standartWheelsRoom`
   - `Content/Wheels/maps/standartWheelsRoom_capture_clean_v2`
3. Confirm there is one `CameraCaptureWheels` actor.
4. Confirm visible wheels have valid `wheels` actors.
5. Press Play for a controlled capture.
6. Stop after the target frame count.
7. Confirm export folders are non-empty:

```bash
find "/Users/edward/Desktop/VSBL/NeuralData1 2/Images" -type f | wc -l
find "/Users/edward/Desktop/VSBL/NeuralData1 2/keyPoint" -type f | wc -l
find "/Users/edward/Desktop/VSBL/NeuralData1 2/Depth" -type f | wc -l
find "/Users/edward/Desktop/VSBL/NeuralData1 2/Goal" -type f | wc -l
```

### Unreal Python / Commandlet / CLI Export

Current commandlet-ready pieces:

- `scripts/unreal_validate_capture_map.py`
- `scripts/unreal_create_clean_capture_map.py`

Potential automated export wrapper:

1. Run `UnrealEditor-Cmd` with `scripts/unreal_validate_capture_map.py`.
2. Optionally duplicate/clean the map with
   `scripts/unreal_create_clean_capture_map.py`.
3. Launch Unreal in game/headless-capable mode for a bounded capture run.
4. Wait until expected file counts appear under `Images/` and `keyPoint/`.
5. Copy only export folders into `outputs/raw_unreal_exports/<name>/`.
6. Run ML-side acceptance.

The installed Unreal CLI exists on this machine:

```text
/Users/Shared/Epic Games/UE_5.7/Engine/Binaries/Mac/UnrealEditor-Cmd
```

The missing piece is a stable commandlet or automation hook that starts the
capture actor, limits frame count, and exits when the export is complete.

## B. If Unreal / Plugin Source Is Not Available

This is the current practical assumption: the exporter is mostly Blueprint
`.uasset`, and no plain-text source for adding `WheelBBox` was found during
discovery.

### What We Can Do On The ML Side

We can make the intake pipeline strict, reproducible, and hard to misuse.

Importer:

- Parse raw `Images/`, `Ground/`, `keyPoint/`.
- Normalize point aliases.
- Resolve Right/Left mapping.
- Use plugin-provided `BBox` / `WheelBBox` when it exists.
- Keep synthetic bbox fallback debug-only and explicit.
- Write source metadata that records bbox provenance.

Sanitizer:

- Drop missing points.
- Drop all-zero and partial-zero objects.
- Drop out-of-bounds objects.
- Drop degenerate bboxes.
- Drop bad floor-ray geometry.
- Keep counters for every drop reason.

Validation:

- Validate incoming confirmed schema.
- Validate all point coordinates and bboxes.
- Validate YOLO-pose conversion.
- Enforce data-quality gates.

Preview:

- Render incoming overlays.
- Render per-status raw previews.
- Render YOLO-pose label previews.
- Provide preview paths for human review.

BBox audit:

- Report whether bbox is plugin-provided or synthesized.
- Detect point-derived bbox signatures.
- Measure bbox aspect ratio.
- Measure `c_disc_bottom` relative y inside bbox.
- Produce contact sheets and sample crops.

Acceptance status:

- Write machine-readable status:
  - `ACCEPT_FOR_TRAINING`
  - `ACCEPT_ONLY_AS_DEBUG`
  - `REJECT_NEEDS_PLUGIN_FIX`
- Keep `training_allowed=false` unless all gates and human preview pass.
- Never silently mark a batch training-ready.

### What To Request From The Plugin Author

Required:

1. `WheelBBox` / `BBox` per valid visible wheel object.
   - Axis-aligned `x1,y1,x2,y2`.
   - Final image pixel coordinates.
   - Covers full visible tire plus rim.

2. Clean export without invalid object files.
   - Do not write fully invisible wheels.
   - Do not write all-zero objects.
   - Do not write partial-zero objects.
   - Do not write fully out-of-bounds objects.

Strongly useful:

3. `scene_id` for every frame.
   - Needed for train/val split without leaking adjacent video frames.

4. `vehicle_id` or stable wheel/object id.
   - Useful for grouping frames and debugging repeated actor failures.

Optional / future:

5. Camera metadata.
   - Intrinsics/extrinsics, depth, or world coordinates can be stored for future
     3D-aware loss/debug experiments.
   - Not required for the first 2D baseline.
   - Must not change the ML inference contract.

Suggested raw object format:

```text
{
  {name:"Right", XY:600.1,1470.2},
  {name:"Left", XY:716.2,1395.1},
  {name:"Center", XY:662.1,1408.6},
  {name:"LeftTop", XY:714.2,1158.2},
  {name:"RightTop", XY:597.2,1198.8},
  {name:"BBox", XYXY:520.0,1080.0,760.0,1510.0}
}
```

## C. Desired Automated Flow

The desired flow is one command from raw export to training decision:

1. **Raw export arrives or is generated**
   - Source: Unreal project export or received folder.
   - Expected dirs: `Images/`, `keyPoint/`, optional `Ground/`, `Depth/`,
     `Goal/`.

2. **Normalize / import**
   - Run raw scan.
   - Parse points and plugin bbox.
   - Normalize aliases.
   - Write `data/incoming/<source_name>/`.
   - Record bbox source and mapping metadata.

3. **Validate**
   - Run `src/check_keypoint_incoming.py`.
   - Fail on schema errors.
   - Count warnings separately.

4. **Preview**
   - Run `src/preview_keypoint_annotations.py`.
   - Generate raw-status previews from `scripts/inspect_unreal_export.py`.
   - Human reviewer checks bbox and A/B/C geometry.

5. **Convert**
   - Run `src/convert_keypoint_incoming_to_yolo_pose.py`.
   - Use `--fail-on-quality-gate` for production candidates.

6. **BBox audit**
   - Run `scripts/audit_unreal_bbox_quality.py`.
   - Require plugin-provided bbox for training.
   - Reject point-derived bbox unless explicitly accepted as debug/provisional.

7. **Human accept / reject**
   - Accept only if previews show:
     - bbox covers full tire plus rim;
     - A/B are lower footprint points;
     - C is lower visible metal disc/rim point;
     - hidden/offscreen wheels are absent.

8. **Only then train**
   - Training starts only when the final report says `ACCEPT_FOR_TRAINING`
     and `training_allowed=true`.
   - Anything else remains debug/provisional.

Target output folders:

```text
outputs/raw_unreal_exports/<source_name>/
outputs/unreal_export_acceptance/<source_name>/
outputs/plugin_export_acceptance/<source_name>/REPORT.md
outputs/plugin_export_acceptance/<source_name>/REPORT.json
```

## D. Why We Cannot Just Use Free 3D Cars And Train

Free 3D cars are not enough by themselves.

Licensing:

- Assets must be legally usable for dataset generation and downstream model
  training.
- Some free assets are view-only, editorial-only, marketplace-limited, or
  incompatible with commercial ML training.

Correct Unreal placement:

- Cars must be placed into scenes with realistic camera paths, scale, lighting,
  occlusions, floor contact, and wheel visibility.
- Wheel actors must be correctly attached or positioned per visible wheel.
- The capture actor must see enough valid wheels from useful angles.

Correct export fields:

- Every valid wheel still needs:
  - full-wheel bbox;
  - A/B footprint points;
  - C disc-bottom point.
- Without `BBox` / `WheelBBox`, training learns the wrong localization target.

Validation:

- Raw exports must be checked for all-zero, partial-zero, out-of-bounds,
  missing points, invalid bbox, and bad A/B/C geometry.
- Preview and bbox audit are required before training.

Synthetic-only limitation:

- Synthetic data can prove plumbing and model/runtime mechanics.
- It does not prove real-world quality.
- A model trained only on synthetic scenes can overfit Unreal textures,
  camera angles, lighting, wheel models, and annotation artifacts.
- Production confidence requires accepted real or sufficiently diverse
  validated render data plus a holdout evaluation set.

## Immediate Recommendation

Do not train yet.

Next safe implementation step:

1. Add strict `BBox` / `WheelBBox` parsing and provenance tracking to the
   importer and inspector.
2. Make synthetic bbox fallback explicit debug-only behavior.
3. Extend acceptance so missing plugin bbox cannot be confused with
   training-ready data.
4. Run the full acceptance chain on current exports to prove they remain
   debug-only until raw `BBox` appears.
