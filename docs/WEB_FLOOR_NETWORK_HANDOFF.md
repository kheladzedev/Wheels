# Web Floor Network Handoff

This handoff is for the web path that predicts wheels plus floor
`pitch/roll/distance` directly from one RGB frame.

The runtime target is deliberately simple:

```text
RGB image -> ONNX model -> decode wheels + floor -> validated web payload
```

It does not require runtime depth, segmentation, RANSAC, multi-frame tracking,
or backend-side geometric reconstruction.

## Files

Source files added for this path:

```text
docs/WEB_FLOOR_NETWORK_CONTRACT.md
docs/WEB_FLOOR_NETWORK_STATUS.md
docs/WEB_FLOOR_NETWORK_HANDOFF.md
docs/WEB_FLOOR_REAL_DATA_INTAKE.md
configs/pose_dataset_web_floor_fixture.yaml
configs/pose_dataset_web_floor_real_template.yaml
src/web_floor_contract.py
src/web_floor_dataset.py
src/web_floor_training.py
src/web_floor_postprocess.py
src/evaluate_web_floor.py
src/web_floor_export.py
src/web_floor_real_data_gate.py
src/web_floor_annotation_import.py
scripts/create_web_floor_fixture.py
scripts/train_web_multitask.py
scripts/eval_web_floor.py
scripts/export_web_floor_onnx.py
scripts/audit_web_floor_real_data.py
scripts/import_web_floor_annotations.py
scripts/import_unreal_web_floor_export.py
scripts/create_web_floor_evidence_request_bundle.py
web_handoff/package.json
web_handoff/smoke_onnxruntime_web.mjs
```

Fixture data:

```text
tests/fixtures/web_floor/
```

Generated handoff output:

```text
outputs/web_floor_network/handoff/
```

The `outputs/` files are generated and ignored by git. Send the folder as an
artifact if Igor needs to inspect the actual ONNX file.

## Regenerate The Fixture

```bash
./.venv/bin/python scripts/create_web_floor_fixture.py \
  --output-root tests/fixtures/web_floor \
  --config-out configs/pose_dataset_web_floor_fixture.yaml \
  --overwrite
```

## Run Fixture Training Dry Run

```bash
./.venv/bin/python scripts/train_web_multitask.py \
  --config configs/pose_dataset_web_floor_fixture.yaml \
  --stage floor \
  --epochs 1 \
  --batch-size 2 \
  --imgsz 128 \
  --out-dir outputs/web_floor_network/train_fixture \
  --device cpu
```

This writes:

```text
outputs/web_floor_network/train_fixture/web_floor_fixture_checkpoint.pt
outputs/web_floor_network/train_fixture/metrics.json
outputs/web_floor_network/train_fixture/config_snapshot.yaml
```

The checkpoint is fixture-only. Do not report it as trained production quality.

## Run Readiness Evaluation

```bash
./.venv/bin/python scripts/eval_web_floor.py \
  --config configs/pose_dataset_web_floor_fixture.yaml \
  --checkpoint outputs/web_floor_network/train_fixture/web_floor_fixture_checkpoint.pt \
  --output-json outputs/web_floor_network/eval_fixture/web_floor_eval.json \
  --device cpu
```

Expected readiness meaning:

- `pipeline_ready=true` means the pipeline runs and validates.
- `trained_model_ready=false` means production training is not done.
- `production_ready=false` means the gate must not be treated as passed.

## Export ONNX Handoff

```bash
./.venv/bin/python scripts/export_web_floor_onnx.py \
  --checkpoint outputs/web_floor_network/train_fixture/web_floor_fixture_checkpoint.pt \
  --config configs/pose_dataset_web_floor_fixture.yaml \
  --out-dir outputs/web_floor_network/handoff \
  --imgsz 512 \
  --device cpu
```

Expected outputs:

```text
outputs/web_floor_network/handoff/web_floor_multitask.onnx
outputs/web_floor_network/handoff/manifest.json
outputs/web_floor_network/handoff/python_onnx_smoke.json
outputs/web_floor_network/handoff/sample_decoded.json
outputs/web_floor_network/handoff/README.md
```

Current IO shape:

```text
input_name: image
input_shape: [1, 3, 512, 512]
output_names: cls, bbox, kpt, vis, floor
floor_shape: [1, 3]
```

Input preprocessing:

- RGB image,
- resized to 512x512,
- float32,
- values in `[0, 1]`,
- NCHW layout `[1, 3, 512, 512]`.

Floor decode:

```text
floor[0] -> pitch
floor[1] -> roll
floor[2] -> distance
```

Use `distance_mode` from the manifest/dataset metadata. The current fixture
uses `scale_relative` for the export handoff.

## Run ONNX Runtime Web Smoke

```bash
cd web_handoff
npm install
npm run smoke
```

This loads:

```text
../outputs/web_floor_network/handoff/web_floor_multitask.onnx
../outputs/web_floor_network/handoff/manifest.json
```

and writes:

```text
../outputs/web_floor_network/handoff/node_smoke_report.json
```

The smoke baseline uses WASM. WebGPU can be evaluated later, but WASM is the
reproducible compatibility path.

## Web Integration Notes

- Validate decoded payloads against `docs/WEB_FLOOR_NETWORK_CONTRACT.md`.
- Keep wheel point names as `a`, `b`, `c_disc_bottom`.
- Do not add runtime depth/segmentation/RANSAC assumptions to the web contract.
- Treat the current ONNX as a fixture artifact only.
- Put the eventual real ONNX behind a feature flag until holdout metrics and
  browser latency are accepted.

## Real Data Required For Production

For each real frame, the production dataset needs:

- image path and provenance,
- wheel `bbox_xyxy`,
- wheel points `a`, `b`, `c_disc_bottom`,
- floor `pitch`,
- floor `roll`,
- floor `distance`,
- `distance_mode`,
- optional `fov_mode`.

Minimum practical next batch:

- at least 50 real frames,
- at least 80 wheels,
- multiple distances and viewing angles,
- explicit held-out split that is not used for training.

## Synthetic Bootstrap From Existing 0003 Export

Mac Studio currently has a raw Unreal/plugin export here:

```text
/Users/codefactory/Downloads/0003
```

It contains 1713 frames plus `Ground` floor metadata and per-wheel `keyPoint`
labels. Import it into the web-floor manifest with:

```bash
./.venv/bin/python scripts/import_unreal_web_floor_export.py \
  --source-root /Users/codefactory/Downloads/0003 \
  --dataset-root data/web_floor_dataset_unreal_0003 \
  --config-out configs/pose_dataset_web_floor_unreal_0003.yaml \
  --source-name unreal_0003_web_floor_source \
  --overwrite
```

By default the manifest references images by absolute path so the 4.6 GB export
is not duplicated. Use `--image-mode copy` only when a self-contained portable
dataset is needed.

The importer converts `Ground/Pitch` and `Ground/Roll` from degrees to radians
and writes `Ground/DeltaZ` into the public web `distance` field with
`distance_mode=scale_relative`.

Important caveat: this is synthetic Unreal/plugin data. It is useful for
bootstrap training and integration pressure, but it must not satisfy the
production real-data gate.

Current imported 0003 outputs:

```text
data/web_floor_dataset_unreal_0003/manifest.json
configs/pose_dataset_web_floor_unreal_0003.yaml
outputs/web_floor_network/unreal_0003_web_floor_gate.json
```

Current counts:

```text
frames: 1713
wheels: 1722
train / holdout: 1456 / 257
```

Run a bounded floor-head smoke train on this synthetic bootstrap set:

```bash
./.venv/bin/python scripts/train_web_multitask.py \
  --config configs/pose_dataset_web_floor_unreal_0003.yaml \
  --stage floor \
  --epochs 1 \
  --batch-size 16 \
  --imgsz 128 \
  --out-dir outputs/web_floor_network/train_unreal_0003_smoke \
  --device auto
```

Then export and web-smoke the resulting ONNX:

```bash
./.venv/bin/python scripts/export_web_floor_onnx.py \
  --checkpoint outputs/web_floor_network/train_unreal_0003_smoke/web_floor_checkpoint.pt \
  --config configs/pose_dataset_web_floor_unreal_0003.yaml \
  --out-dir outputs/web_floor_network/handoff_unreal_0003_smoke \
  --imgsz 512 \
  --device cpu

cd web_handoff
node smoke_onnxruntime_web.mjs \
  ../outputs/web_floor_network/handoff_unreal_0003_smoke/web_floor_multitask.onnx \
  ../outputs/web_floor_network/handoff_unreal_0003_smoke/manifest.json \
  ../outputs/web_floor_network/handoff_unreal_0003_smoke/node_smoke_report.json
```

## Audit Real Data Before Training

If Igor needs a fill-in package, create one first:

```bash
./.venv/bin/python scripts/create_web_floor_evidence_request_bundle.py
```

Send:

```text
outputs/web_floor_network/web_floor_real_data_request_bundle.zip
```

Once a real manifest exists, run:

```bash
./.venv/bin/python scripts/audit_web_floor_real_data.py \
  --config configs/<real_web_floor_config>.yaml \
  --output-json outputs/web_floor_network/real_data_gate.json \
  --fail-on-not-ready
```

Default gate:

```text
min_frames: 50
min_wheels: 80
required_splits: train, holdout
fixture_only: must be false
provenance: required for every frame
distance_mode: must not be unknown
distance_span: >= 0.5
pitch_or_roll_span: >= 0.05 rad
```

Current repo state: the checked-in fixture fails this gate by design, and the
existing `configs/pose_dataset_real_web_*` files are 2D YOLO demo configs, not
web-floor manifests with `pitch/roll/distance/distance_mode`.

## Igor Summary

Current state: the web pipeline and ONNX handoff are ready to inspect, but the
model is not production-ready. Angles plus distance are direct model outputs;
runtime does not need depth, segmentation, RANSAC, or heavy backend
postprocess. The next real task is to collect/label web or phone frames and
train/evaluate against that holdout.
