# NeuralData1 Capture Acceptance Workflow

This is the safe path for the Unreal project Igor sent. The active local copy
is now inside the VSBL workspace:

```text
/Users/edward/Desktop/VSBL/NeuralData1 2
```

That folder is a working Unreal project, not a training dataset by itself.
Training data exists only after Unreal writes non-empty `Images/` and
`keyPoint/` export folders.

The legacy quarantine folder is not training input:

Legacy assets are still present in the Unreal project. Do not use the project
folder itself as ML input; only copy generated export folders (`Images/`,
`keyPoint/`, `Depth/`, `Goal/`) into the VSBL acceptance pipeline.

## Unreal-side capture

1. Open:

   ```text
   /Users/edward/Desktop/VSBL/NeuralData1 2/NeuralData.uproject
   ```

2. Open one of the prepared maps: `01`, `02`, `03`, or `standartWheelsRoom`.
3. Verify `CameraCaptureWheels` is present in the scene.
4. Verify its `Floor` points to `Plane` or another real floor object.
5. Verify visible wheels have `wheels` actors:
   - `Center` is the lower visible metal disc point.
   - `SphereLeft` / `SphereRight` are lower floor-ray footprint points.
   - `SphereLeftTop` / `SphereRightTop` are bbox helper points.
6. Press Play for a short smoke capture, then stop.

Check that Unreal wrote files:

```bash
find "/Users/edward/Desktop/VSBL/NeuralData1 2/Images" -type f | wc -l
find "/Users/edward/Desktop/VSBL/NeuralData1 2/keyPoint" -type f | wc -l
find "/Users/edward/Desktop/VSBL/NeuralData1 2/Depth" -type f | wc -l
find "/Users/edward/Desktop/VSBL/NeuralData1 2/Goal" -type f | wc -l
```

## VSBL-side acceptance

Run the wrapper from the VSBL repo:

```bash
cd /Users/edward/Desktop/VSBL

./.venv/bin/python scripts/accept_neuraldata1_capture.py \
  --overwrite
```

The wrapper copies only:

```text
Images/
keyPoint/
Depth/
Goal/
```

into `outputs/raw_unreal_exports/<source_name>/`, then runs the official
`scripts/accept_unreal_export.py` pipeline.

If `Images/` or `keyPoint/` is empty, it stops before acceptance and writes a
blocked report. This is expected before Unreal has captured anything.

## Human preview gate

Open the preview paths printed in `capture_report.md`:

```bash
open outputs/unreal_export_acceptance_neuraldata1/<source_name>/previews/incoming
open outputs/unreal_export_acceptance_neuraldata1/<source_name>/previews/pose/train
open outputs/unreal_export_acceptance_neuraldata1/<source_name>/inspection/previews/by_status
```

Accept only if:

- bbox covers the full visible tire + rim;
- A/B sit near the wheel footprint, not rim edges;
- C sits on the lower visible metal disc/rim point;
- hidden or out-of-frame wheels are not included as training labels.

## Training approval

Default runs never mark training as allowed. After the automated report passes
and a human accepts previews, record the approval explicitly:

```bash
./.venv/bin/python scripts/accept_neuraldata1_capture.py \
  --source-name <same_source_name> \
  --overwrite \
  --human-preview-accepted
```

Only a `capture_report.json` with:

```json
{
  "training_decision": "ACCEPT_FOR_TRAINING",
  "training_allowed": true
}
```

allows a production-candidate training run. Everything else is debug or
provisional only.
