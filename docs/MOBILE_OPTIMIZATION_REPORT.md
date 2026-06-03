# Mobile Optimization Report

- OK: True
- Scope: mobile_optimization_candidates_not_production_promotion
- Ready candidates: 6
- Failures: none

## Baselines

| ID | Platform | Size MB | Input | Output | Path |
|---|---|---:|---|---|---|
| tflite_float32_640 | android | 37.409 | `[1, 640, 640, 3]` | `[1, 14, 8400]` | `outputs/production_audit/tflite_export/best_float32.tflite` |
| coreml_float32_640 | ios | 37.185 | `[1, 640, 640, 3]` | `[1, 14, 8400]` | `outputs/production_audit/coreml_export/best.mlmodel` |

## Candidates

| ID | Platform | Precision | Status | Size MB | Ratio | Input | Output | Path |
|---|---|---|---|---:|---:|---|---|---|
| tflite_fp16_640 | android | fp16 | ready | 18.770 | 1.99x | `[1, 640, 640, 3]` | `[1, 14, 8400]` | `outputs/production_audit/mobile_optimization/tflite_fp16_640/best_float16.tflite` |
| tflite_fp16_416 | android | fp16 | ready | 18.677 | 2.00x | `[1, 416, 416, 3]` | `[1, 14, 3549]` | `outputs/production_audit/mobile_optimization/tflite_fp16_416/best_float16.tflite` |
| tflite_dynamic_range_int8_640 | android | dynamic_range_int8_weights | ready | 9.879 | 3.79x | `[1, 640, 640, 3]` | `[1, 14, 8400]` | `outputs/production_audit/mobile_optimization/tflite_dynamic_range_int8_640/best_dynamic_range_quant.tflite` |
| tflite_fp16_384 | android | fp16 | ready | 18.667 | 2.00x | `[1, 384, 384, 3]` | `[1, 14, 3024]` | `outputs/production_audit/mobile_optimization/tflite_fp16_384/best_float16.tflite` |
| coreml_linear_int8_640 | ios | int8_weights | ready | 9.434 | 3.94x | `[1, 640, 640, 3]` | `[1, 14, 8400]` | `outputs/production_audit/mobile_optimization/coreml_linear_int8_640/best_int8.mlmodel` |
| coreml_linear_4bit_640 | ios | linear_4bit_weights | ready | 4.789 | 7.77x | `[1, 640, 640, 3]` | `[1, 14, 8400]` | `outputs/production_audit/mobile_optimization/coreml_linear_4bit_640/best_linear4.mlmodel` |

## Decision

Use the float32 artifacts as the current integration baseline.
Use optimized candidates only after device latency and quality validation.
