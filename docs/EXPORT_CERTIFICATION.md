# Export Certification

- Certified: True
- Scope: desktop_export_backend_certification_not_android_device
- Status: certified

| Backend | Certified | mAP50 | OKS | Max bbox px | Max kp px | Max conf | Failures |
|---|---:|---:|---:|---:|---:|---:|---|
| onnx | True | 0.692 | 0.888 | 8.497 | 13.371 | 0.228 | none |
| tflite | True | 0.692 | 0.888 | 8.497 | 13.372 | 0.228 | none |

## Policy

Strict 2px/3px/0.05 parity remains diagnostic. Production export certification uses calibrated drift plus aggregate metric parity.

The scope is desktop/export-backend certification. Android device latency, memory, and end-to-end LiteRT integration remain separate production evidence.
