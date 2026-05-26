# Mac Studio Training Handoff

This is the practical handoff for moving VSBL to Mac Studio and running the
next full MobileNetV2 training pass after the Unreal export gate is satisfied.

Current status: **do not train yet** on the raw multi-vehicle capture. The
pipeline works, but the current multi-vehicle export still needs Unreal/plugin
cleanup or explicit human bbox acceptance.

## What Is Ready

- Unreal capture can generate `Images/` and `keyPoint/`.
- Raw exports can be inspected, imported, previewed, converted to YOLO-pose,
  and audited.
- MobileNetV2 real-data trainer is implemented.
- Runtime/export package path is implemented for already trained checkpoints.
- Guarded Mac Studio training entry point exists:

```bash
./.venv/bin/python scripts/train_mobilenetv2_from_accepted_export.py --help
```

That script refuses to train unless the selected acceptance folder passes the
technical gate, data-quality gate, human preview gate, and bbox provenance gate.

## Current Multi-Vehicle Result

Raw capture:

```text
outputs/raw_unreal_exports/unreal_neuraldata1_standartWheelsRoom_multi_vehicle_100f_20260526_0259
```

Raw batch counts:

| Metric | Count |
| --- | ---: |
| Images | 100 |
| Raw wheel objects | 600 |
| Valid imported wheels | 38 |
| All-zero objects | 253 |
| Out-of-bounds / partial-zero objects | 257 |
| Bad geometry drops | 52 |
| Empty label images | 74 |

Verdict: **not training-ready**.

Sanitized debug copy:

```text
outputs/raw_unreal_exports/unreal_neuraldata1_standartWheelsRoom_multi_vehicle_100f_20260526_0259_sanitized_valid_only
```

Sanitized acceptance:

```text
outputs/unreal_export_acceptance_neuraldata1/unreal_neuraldata1_standartWheelsRoom_multi_vehicle_100f_20260526_0259_sanitized_valid_only
```

Sanitized batch counts:

| Metric | Count |
| --- | ---: |
| Images | 26 |
| Wheel labels | 38 |
| Errors | 0 |
| Warnings | 0 |
| Data-quality gate | PASS |
| Plugin-provided bbox | 0 |
| Synthesized bbox | 38 |

Verdict: **debug/provisional only unless human preview explicitly accepts bbox
geometry**. Production-candidate training should prefer native
`BBox` / `WheelBBox` from the plugin.

## Transfer To Mac Studio

Recommended copy shape:

```bash
rsync -a --info=progress2 \
  --exclude .venv \
  --exclude .tflite-venv \
  --exclude __pycache__ \
  /Users/edward/Desktop/VSBL/ \
  <MAC_STUDIO_USER>@<MAC_STUDIO_HOST>:/Users/edward/Desktop/VSBL/
```

If moving through an external disk, copy the full `VSBL` folder, including:

- `NeuralData1 2/`
- `outputs/raw_unreal_exports/`
- `outputs/unreal_export_acceptance_neuraldata1/`
- `outputs/unreal_bbox_audit/`
- `runs/` if you want old checkpoints on the Mac Studio too

Do not copy `.venv` between Macs. Recreate it.

## Mac Studio Setup

From `/Users/edward/Desktop/VSBL`:

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/pytest tests/test_train_mobilenetv2_from_accepted_export.py tests/test_sanitize_unreal_export.py -q
```

If `python3.11` is unavailable, install it first. Keep TensorFlow/TFLite tools
isolated in `.tflite-venv`; they are not needed for training.

## Unreal / MCP Role

MCP is for Unreal-side inspection, scene cleanup, asset listing, and capture
automation. Training does not require MCP.

The reliable non-MCP fallback already works through commandlet scripts:

```bash
scripts/run_unreal_capture.sh <map_name> <frame_count>
./.venv/bin/python scripts/accept_neuraldata1_capture.py --overwrite
```

For MCP on Mac Studio, enable Unreal Python support first:

1. Open `NeuralData1 2/NeuralData.uproject`.
2. Enable `Python Editor Script Plugin`.
3. Enable Python remote execution in Unreal editor settings.
4. Restart Unreal.
5. Connect the MCP server and run a read-only check before edits.

Do not depend on MCP for the final ML gates. Always verify with acceptance
reports and previews.

## Required Flow Before Training

1. Generate or receive a raw Unreal export.
2. Run acceptance.
3. Open previews and confirm bbox/A/B/C geometry.
4. Run bbox audit.
5. Train only if all gates pass.

Native-BBox preferred command:

```bash
./.venv/bin/python scripts/accept_unreal_export.py \
  --source-root <RAW_EXPORT_ROOT> \
  --source-name <SOURCE_NAME> \
  --out-root outputs/unreal_export_acceptance_neuraldata1 \
  --overwrite \
  --right-left-mapping auto \
  --fail-on-data-quality-gate
```

Debug fallback command for exports without native bbox:

```bash
./.venv/bin/python scripts/accept_unreal_export.py \
  --source-root <RAW_EXPORT_ROOT> \
  --source-name <SOURCE_NAME> \
  --out-root outputs/unreal_export_acceptance_neuraldata1 \
  --overwrite \
  --right-left-mapping auto \
  --allow-synthetic-bbox \
  --fail-on-data-quality-gate
```

Use the fallback only to diagnose or after explicit human bbox acceptance.

## Training Command

Dry-run first:

```bash
./.venv/bin/python scripts/train_mobilenetv2_from_accepted_export.py \
  --acceptance-root outputs/unreal_export_acceptance_neuraldata1/<ACCEPTANCE_DIR> \
  --human-preview-accepted \
  --device mps \
  --epochs 50 \
  --batch 16 \
  --num-workers 8 \
  --dry-run
```

If the export has native plugin bbox, remove `--dry-run`:

```bash
./.venv/bin/python scripts/train_mobilenetv2_from_accepted_export.py \
  --acceptance-root outputs/unreal_export_acceptance_neuraldata1/<ACCEPTANCE_DIR> \
  --human-preview-accepted \
  --device mps \
  --epochs 50 \
  --batch 16 \
  --num-workers 8
```

If the export still uses synthesized bbox, training requires this additional
explicit flag:

```bash
--accept-synthetic-bbox-after-review
```

That should be treated as **provisional**, not production approval.

## What Must Still Be Fixed

- The Unreal/plugin exporter should omit invisible, all-zero, and out-of-frame
  wheel objects instead of writing unusable object txt files.
- The plugin should export native `BBox` or `WheelBBox` per wheel:

```text
{name:"WheelBBox",XYXY:x1,y1,x2,y2}
```

- BBox must cover the full visible tire + rim, not the lower keypoint hull.
- Multi-vehicle scene needs more correct `wheels` actors before it becomes a
  useful production-candidate training source.

## Stop Rules

Do not train when:

- `technical_status != PASS`
- `data_quality_gate.passed != true`
- previews were not reviewed
- bbox is synthesized and not explicitly accepted
- raw export has many all-zero/out-of-bounds objects

Do not claim:

- production-ready model
- AR-ready quality
- Igor changed exporter semantics
- native bbox exists

unless the acceptance report and raw files prove it.
