# Web Floor Network Status

Date: 2026-06-12
Branch: `feature/web-floor-network`

## Summary

The web floor network path is implemented as an end-to-end fixture pipeline:
contract validation, dataset loading, fixture generation, training dry run,
evaluation/readiness reporting, ONNX export, and `onnxruntime-web` smoke test.

The runtime contract is intentionally light:

- one RGB image tensor,
- one model forward,
- direct floor output `[pitch, roll, distance]`,
- wheel output contract with `a`, `b`, `c_disc_bottom`,
- no required depth map,
- no segmentation mask,
- no RANSAC,
- no multi-frame state,
- no heavy backend geometry postprocess.

This is not production-ready model quality yet. The checked-in fixture proves
the pipeline and web handoff mechanics only. Production requires real
web/phone frames with wheel labels and floor angle/distance labels.

## Implemented

- `docs/WEB_FLOOR_NETWORK_CONTRACT.md`
  - public decoded payload shape for web,
  - `floor={pitch, roll, distance, distance_mode, fov_mode}`,
  - forbidden runtime-heavy fields,
  - preserves existing wheel point names.
- `docs/WEB_FLOOR_REAL_DATA_INTAKE.md`
  - exact manifest/config shape for the first real web/phone batch.
- `src/web_floor_contract.py`
  - validates decoded web payloads,
  - bridges the model's internal third scalar to public `distance`.
- `src/web_floor_dataset.py`
  - reads YAML config plus JSON manifest,
  - returns image tensors, wheel targets, and floor tensor targets,
  - handles empty-wheel frames with stable tensor shapes.
- `scripts/create_web_floor_fixture.py`
  - creates a deterministic four-frame synthetic fixture for tests.
- `src/web_floor_training.py` and `scripts/train_web_multitask.py`
  - run a real forward/backward/checkpoint dry run on the fixture,
  - keep reconstruction loss disabled unless explicitly requested.
- `src/evaluate_web_floor.py` and `scripts/eval_web_floor.py`
  - validate decoded outputs and emit readiness JSON,
  - report that runtime depth/segmentation/RANSAC are not required.
- `src/web_floor_export.py` and `scripts/export_web_floor_onnx.py`
  - export ONNX with stable IO names,
  - run Python ONNX Runtime shape smoke,
  - write handoff manifest and decoded sample.
- `web_handoff/`
  - Node/WASM `onnxruntime-web` smoke package for the web side.
- `src/web_floor_real_data_gate.py` and `scripts/audit_web_floor_real_data.py`
  - production-data gate for real web/phone manifests,
  - blocks fixture-only or 2D-only datasets from being treated as production
    training input.
- `src/web_floor_annotation_import.py` and
  `scripts/import_web_floor_annotations.py`
  - CSV-to-manifest importer for the first real web/phone annotation batch.
- `scripts/import_unreal_web_floor_export.py`
  - converts the existing raw Unreal/plugin export layout
    `Images/`, `Ground/`, `keyPoint/` into the web-floor manifest,
  - normalizes `Ground/Pitch` and `Ground/Roll` from degrees to radians,
  - records `Ground/DeltaZ` as the web `distance` target.
- `scripts/create_web_floor_evidence_request_bundle.py`
  - deterministic request zip for Igor/web with CSV template, example row,
    image placeholder, and intake instructions.

## Found Local Dataset

Mac Studio has a raw Unreal/plugin export at:

```text
/Users/codefactory/Downloads/0003
```

Current inventory:

- 1713 JPEG frames in `Images/` at 2048x2048,
- 1713 `Ground/*.txt` files with `DeltaZ`, `Roll`, `Pitch`, and `FOV`,
- 1713 `keyPoint/<frame_id>/` folders with per-wheel
  `Right/Left/Center/LeftTop/RightTop` labels.

This dataset is useful for web-floor bootstrap training because it has floor
angles plus distance labels, but it is synthetic Unreal/plugin data. It is not
a real web/phone production holdout.

## Current Handoff Artifacts

Fixture-generated artifacts live under:

```text
outputs/web_floor_network/handoff/
```

Main files:

- `web_floor_multitask.onnx` - fixture-trained ONNX model, about 11 MB.
- `manifest.json` - schema, SHA256, input/output names, shapes, caveats.
- `python_onnx_smoke.json` - Python ONNX Runtime smoke result.
- `node_smoke_report.json` - `onnxruntime-web` WASM smoke result.
- `sample_decoded.json` - one decoded payload validated by the web contract.

Current ONNX contract:

```text
input:  image float32 [1, 3, 512, 512], RGB, values in [0, 1]
output: cls   float32 [1, 1, 16, 16]
output: bbox  float32 [1, 4, 16, 16]
output: kpt   float32 [1, 6, 16, 16]
output: vis   float32 [1, 3, 16, 16]
output: floor float32 [1, 3]  # pitch, roll, distance
```

Current manifest SHA256:

```text
b1d79a0e60d083a465925d4cb16a0fc8be73a7adb1a7f04f8025f35a1e909634
```

The ONNX file is generated output and is intentionally not committed by the
repo ignore rules. Regenerate it with the command in
`docs/WEB_FLOOR_NETWORK_HANDOFF.md`.

Synthetic 0003 bootstrap artifacts now also exist under:

```text
outputs/web_floor_network/train_unreal_0003_smoke/
outputs/web_floor_network/eval_unreal_0003_smoke/
outputs/web_floor_network/handoff_unreal_0003_smoke/
outputs/web_floor_network/unreal_0003_web_floor_gate.json
```

Key results:

- web-floor manifest: `data/web_floor_dataset_unreal_0003/manifest.json`,
- config: `configs/pose_dataset_web_floor_unreal_0003.yaml`,
- dataset: 1713 frames, 1722 valid wheels, 1456 train / 257 holdout,
- train smoke: 1 epoch, MPS, `imgsz=128`, checkpoint
  `outputs/web_floor_network/train_unreal_0003_smoke/web_floor_checkpoint.pt`,
- eval smoke: `pipeline_ready=true`, `trained_model_ready=false`,
  `production_ready=false`,
- synthetic handoff ONNX SHA256:
  `d289a714ebc9202b0832f3ba851fe9fc89fc505145b7ddbc14d504f6e2223a0c`,
- real-data gate: all size/split/label checks pass, but
  `production_data_ready=false` because `real_source` fails by design.

## Verification

Fresh verification on 2026-06-12:

- Full Python test suite: `1140 passed, 6 warnings`.
- `git diff --check`: passed.
- Python ONNX Runtime smoke: passed with `CPUExecutionProvider`.
- Node `onnxruntime-web` smoke: passed with WASM execution provider.
- Runtime scope in reports: `single_forward_no_depth_no_ransac`.
- Real-data production gate on current fixture: fails by design because the
  repo still has no real web/phone floor-labelled holdout.
- Unreal 0003 web-floor import: 1713 frames / 1722 wheels, split
  1456 train / 257 holdout, `distance_mode=scale_relative`.
- Unreal 0003 training smoke: MPS, one epoch, checkpoint written, still
  `trained_model_ready=false` and `production_ready=false`.
- Unreal 0003 ONNX export: Python and Node/WASM smokes passed.

The warnings are PyTorch ONNX exporter deprecation warnings. They do not block
the current handoff but should be cleaned up later when moving to the newer
PyTorch export path.

## What Is Not Done

- No real web/phone labeled dataset has landed yet.
- The discovered `/Users/codefactory/Downloads/0003` batch is synthetic
  Unreal/plugin data. It can bootstrap the web-floor task, but the production
  real-data gate must still fail on `real_source` until a real capture/holdout
  lands.
- Existing `configs/pose_dataset_real_web_*` files are YOLO A/B/C demos and the
  corresponding `data/wheel_pose_dataset_real_web_*` folders are absent on this
  machine; they do not contain `pitch/roll/distance/distance_mode` labels.
- No production-quality training has happened.
- No production accuracy thresholds are set or passed.
- No browser latency/memory budget has been certified on target devices.
- The fixture ONNX proves integration mechanics, not real model quality.
- The production pose decoder for web wheel candidates still needs to replace
  the fixture/eval proxy.

## What To Give Igor

- Branch: `feature/web-floor-network`.
- Docs:
  - `docs/WEB_FLOOR_NETWORK_STATUS.md`
  - `docs/WEB_FLOOR_NETWORK_HANDOFF.md`
  - `docs/WEB_FLOOR_NETWORK_CONTRACT.md`
  - `docs/WEB_FLOOR_REAL_DATA_INTAKE.md`
- Handoff package:
  - `web_handoff/package.json`
  - `web_handoff/smoke_onnxruntime_web.mjs`
  - generated `outputs/web_floor_network/handoff/` folder if sending artifacts.
- Real-data request package:
  - `outputs/web_floor_network/web_floor_real_data_request_bundle.zip`
  - `outputs/web_floor_network/web_floor_real_data_request_bundle_manifest.json`
- Explicit caveat:
  - angles and distance are direct model outputs,
  - runtime does not need depth/segmentation/RANSAC/heavy backend postprocess,
  - production gate still needs real labeled holdout and browser performance
    evidence.

## Recommended Next Actions

1. Collect real web/phone frames with provenance.
2. Label wheels using the existing `a`, `b`, `c_disc_bottom` points.
3. Add floor labels: `pitch`, `roll`, `distance`, and `distance_mode`.
4. Use `scripts/create_web_floor_evidence_request_bundle.py` if Igor needs a
   ready-to-fill zip template.
5. Convert the real data into the `WEB_FLOOR_NETWORK_CONTRACT.md` manifest
   shape using `docs/WEB_FLOOR_REAL_DATA_INTAKE.md` or the CSV importer.
6. For synthetic bootstrap from the found Unreal export, import `0003`:

   ```bash
   ./.venv/bin/python scripts/import_unreal_web_floor_export.py \
     --source-root /Users/codefactory/Downloads/0003 \
     --dataset-root data/web_floor_dataset_unreal_0003 \
     --config-out configs/pose_dataset_web_floor_unreal_0003.yaml \
     --source-name unreal_0003_web_floor_source \
     --overwrite
   ```

7. Run the real-data gate:

   ```bash
   ./.venv/bin/python scripts/audit_web_floor_real_data.py \
     --config configs/<real_web_floor_config>.yaml \
     --output-json outputs/web_floor_network/real_data_gate.json \
     --fail-on-not-ready
   ```

8. Train `floor` then `joint` stages on real data only after that gate passes.
9. Evaluate on a held-out real set and define production thresholds.
10. Re-export ONNX and rerun Python + Node/WASM smoke.
11. Integrate in the web runtime behind a feature flag.
12. Measure browser latency and memory before production gate.
