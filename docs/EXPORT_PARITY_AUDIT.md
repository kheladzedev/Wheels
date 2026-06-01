# Export Parity Audit

Diagnostic summary for PT-vs-exported parity drift reports.

- Audit OK: True
- Certified: False
- Recommendation: Do not certify exported artifacts as drop-in replacements under the current strict parity policy. ONNX and TFLite aggregate eval remain usable evidence, but production needs either fixed export parity or an explicitly approved aggregate/AR-holdout acceptance policy.

| Export | Strict OK | Matched | Max bbox px | Max kp px | Max conf | Failure categories |
|---|---:|---:|---:|---:|---:|---|
| onnx | False | 14/20 | 8.497 | 13.371 | 0.228 | bbox_drift=1, keypoint_drift=2, confidence_drift=4; scale_warnings=0 |
| tflite | False | 14/20 | 8.497 | 13.372 | 0.228 | bbox_drift=1, keypoint_drift=2, confidence_drift=4; scale_warnings=0 |

## Findings

- ONNX and TFLite have identical failure category counts; the issue is likely shared exported-backend/postprocess parity rather than a TFLite-only runtime bug.
- onnx: strict failures are drift-only; no detection-count mismatch.
- onnx: no coordinate-scale warnings in the official strict report.
- onnx: all sampled frames pass only at bbox=10.0px, kp=15.0px, conf=0.25.
- tflite: strict failures are drift-only; no detection-count mismatch.
- tflite: no coordinate-scale warnings in the official strict report.
- tflite: all sampled frames pass only at bbox=10.0px, kp=15.0px, conf=0.25.

## Tolerance Sweep

### onnx

| bbox px | kp px | conf | Passed |
|---:|---:|---:|---:|
| 2.000 | 3.000 | 0.050 | 14/20 |
| 3.000 | 4.000 | 0.050 | 15/20 |
| 5.000 | 8.000 | 0.100 | 16/20 |
| 10.000 | 15.000 | 0.250 | 20/20 |

### tflite

| bbox px | kp px | conf | Passed |
|---:|---:|---:|---:|
| 2.000 | 3.000 | 0.050 | 14/20 |
| 3.000 | 4.000 | 0.050 | 15/20 |
| 5.000 | 8.000 | 0.100 | 16/20 |
| 10.000 | 15.000 | 0.250 | 20/20 |
