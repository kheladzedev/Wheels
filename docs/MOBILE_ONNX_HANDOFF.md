# Mobile ONNX Handoff

- OK: True
- Scope: mobile_onnx_runtime_handoff_candidate_not_production_promotion
- Runtime: ONNX Runtime Mobile on Android
- Model: `outputs/production_audit/mobile_onnx/best_mobile_384.onnx`
- Size: 10.356 MB
- SHA256: `9b720e5ef88629a265ce2fc1eccaa87f48171376071e8367bafe290feac76fd9`
- Status: **integration_smoke_not_production**
- Failures: none

## Interface

- Input shape: `[1, 3, 384, 384]`
- Output shape: `[1, 14, 3024]`
- ONNX inputs: `[{"name": "images", "shape": [1, 3, 384, 384], "elem_type": 1}]`
- ONNX outputs: `[{"name": "output0", "shape": [1, 14, 3024], "elem_type": 1}]`
- Opsets: `[{"domain": "ai.onnx", "version": 20}]`

## Runtime Smoke

- OK: True
- Provider: CPUExecutionProvider
- Runs: 3
- Latency ms: `{"min": 5.478, "avg": 6.736, "max": 9.186}`
- Output shapes: `{"output0": [1, 14, 3024]}`

## Package

- Zip: `outputs/production_audit/mobile_onnx/mobile_onnx_handoff.zip`
- Zip SHA256: `c4173d69cb85905019ad7d6dc4d927227063c4a932a310dff25310424b12c1c2`

## Rebuild

```bash
./.venv/bin/python scripts/build_mobile_onnx_handoff.py
```

## Notes

- This is the ONNX Runtime Mobile integration candidate, not a production promotion.
- The existing <=6 MB mobile package remains TFLite/CoreML-only.
- Do not claim production readiness until Android-device latency/memory and real AR holdout evidence are attached.
