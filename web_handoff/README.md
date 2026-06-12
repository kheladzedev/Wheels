# Wheels Web Floor Handoff

This package smoke-tests the ONNX handoff with `onnxruntime-web`.

Runtime scope: one image tensor, one model forward, simple decode/validation.
It does not require depth maps, segmentation masks, RANSAC, multi-frame state,
or backend-side geometric postprocess.

## Files

- `outputs/web_floor_network/handoff/web_floor_multitask.onnx` — exported model.
- `outputs/web_floor_network/handoff/manifest.json` — SHA256, input/output shapes, runtime scope, caveats.
- `outputs/web_floor_network/handoff/sample_decoded.json` — one decoded contract sample.
- `web_handoff/smoke_onnxruntime_web.mjs` — Node/WASM smoke check.

## Install and run

```bash
cd web_handoff
npm install
npm run smoke
```

For browser bundlers, copy ONNX Runtime Web WASM assets from
`node_modules/onnxruntime-web/dist/` and point `ort.env.wasm.wasmPaths` at that
served directory before `InferenceSession.create(...)`.

WebGPU is optional. If you choose it later, import from
`onnxruntime-web/webgpu` and request `executionProviders: ["webgpu", "wasm"]`.
Keep WASM as the reproducible baseline smoke path.

## Caveat

This is a fixture handoff. It proves export/load/output-shape plumbing, not
production model quality. Production needs real web/phone holdout data with
wheel labels and floor angle/distance labels.
