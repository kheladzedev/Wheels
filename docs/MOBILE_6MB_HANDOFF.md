# Mobile 6MB Handoff

- OK: True
- Scope: mobile_6mb_handoff_candidates_not_production_promotion
- Max model size MB: 6.0
- Actual input size: `[384, 384]`
- Failures: none

## Artifacts

| Platform | Role | File | Size MB | Limit MB | Path |
|---|---|---|---:|---:|---|
| android | model | best_float16.tflite | 5.226 | 6.0 | `outputs/production_audit/mobile_6mb/tflite_nano_fp16_384/best_float16.tflite` |
| android | validation | litert_smoke_tflite_nano_fp16_384.json | 0.001 |  | `outputs/production_audit/mobile_6mb/litert_smoke_tflite_nano_fp16_384.json` |
| shared | quality_reference | nano_source_eval_self_plus_ue_conf025.json | 0.012 |  | `outputs/production_audit/mobile_6mb/nano_source_eval_self_plus_ue_conf025.json` |
| ios | model | best_int8.mlmodel | 2.632 | 6.0 | `outputs/production_audit/mobile_6mb/coreml_nano_int8_384/best_int8.mlmodel` |
| ios | validation | coreml_certification.json | 0.001 |  | `outputs/production_audit/mobile_6mb/coreml_nano_int8_384/coreml_certification.json` |
| ios | model | best_linear4.mlmodel | 1.359 | 6.0 | `outputs/production_audit/mobile_6mb/coreml_nano_linear4_384/best_linear4.mlmodel` |
| ios | validation | coreml_certification.json | 0.001 |  | `outputs/production_audit/mobile_6mb/coreml_nano_linear4_384/coreml_certification.json` |

## Model Interfaces

- Android TFLite: float32 input `[1, 384, 384, 3]`, float32 output `[1, 14, 3024]`.
- iOS CoreML: image input `image` 384x384, output `var_1344`, logical output `[1, 14, 3024]`.

## Quality Reference

- Scope: source_checkpoint_eval_not_exact_export_runtime_parity
- Model: `runs/pose/runs/pose/wheel_real_v1_soft_n_aug/weights/best.pt`
- Data: `configs/pose_dataset_real_v1_self_plus_ue_synthetic.yaml`
- Images / GT wheels / matched: 58 / 84 / 55
- mAP50 / mAP50-95: 0.671 / 0.57
- OKS mean: 0.866
- FN rate / FP rate: 0.345 / 0.154

## Platform Zips

| Platform | Size MB | Path |
|---|---:|---|
| android | 4.738 | `outputs/production_audit/mobile_6mb/android_6mb_handoff.zip` |
| ios | 3.505 | `outputs/production_audit/mobile_6mb/ios_6mb_handoff.zip` |

## Notes

- Android artifact uses the nano checkpoint, FP16 TFLite, and 384x384 input.
- iOS includes int8 and 4-bit CoreML candidates; int8 is the safer first test candidate.
- These candidates fit the mobile size constraint but are not production-promoted.
- Promote only after app-device latency plus quality validation.
