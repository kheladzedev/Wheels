# Performance Audit

Desktop-local inference latency diagnostic for the current wheel-pose release artifacts.

- OK: True
- Scope: `desktop_local_runtime_diagnostic_not_android_certification`
- Sample count: 8
- Images dir: `data/wheel_pose_dataset_real_v1_self_plus_ue_synthetic/images/val`

| Runtime | OK | Device | Runs | Mean ms | P50 ms | P95 ms | Detections/image |
|---|---:|---|---:|---:|---:|---:|---:|
| pytorch_cpu | True | cpu | 16 | 41.965 | 41.802 | 52.741 | 1.750 |
| onnx_cpu | True | cpu | 16 | 39.092 | 37.821 | 50.990 | 1.875 |
| litert_cpu_smoke | True | ai_edge_litert | 10 | 269.127 | 268.753 | 271.012 | n/a |

## Notes

- Ultralytics PT/ONNX measurements include preprocess, inference, and postprocess wall time.
- LiteRT value is imported from the raw ai_edge_litert smoke report.
- Android production latency must still be measured inside the target app/runtime.
