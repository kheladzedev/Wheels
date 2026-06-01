# Model Inventory

Generated from local `runs/pose/**/args.yaml` and `outputs/eval/*.json`.

## Summary

- Train runs: 13
- Artifacts: 35 (`.pt`=26, `.onnx`=7, `.tflite`=1, `.mlmodel`=1)
- Run artifacts: 33
- Deployment artifacts: 2
- Eval reports: 29
- Runs with eval evidence: 13
- Runs with lineage warnings: 0
- Champion artifact: `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt`
- Champion run: `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s`
- Champion training data: `configs/pose_dataset_real_v1_self_plus_ue_synthetic.yaml`
- Champion source model: `runs/pose/runs/pose/wheel_real_v1_self_s/weights/best.pt`

## Runs

| Run | Data | Source model | Artifacts | Best eval mAP50 | Best eval OKS | FN | FP | Warnings |
|---|---|---|---:|---:|---:|---:|---:|---|
| `wheel_real_v1_combined_s` | `configs/pose_dataset_real_v1_combined.yaml` | `runs/pose/runs/pose/wheel_real_v1_soft_s_aug/weights/best.pt` | 2 | 0.786 | 0.733 | 0.078 | 0.390 | none |
| `wheel_real_v1_self_s` | `configs/pose_dataset_real_v1_self.yaml` | `runs/pose/runs/pose/wheel_real_v1_soft_s_aug/weights/best.pt` | 3 | 0.903 | 0.894 | 0.078 | 0.280 | none |
| `wheel_real_v1_soft_n_aug` | `configs/pose_dataset_real_v1_soft.yaml` | `yolo11n-pose.pt` | 3 | 0.773 | 0.879 | 0.147 | 0.310 | none |
| `wheel_real_v1_soft_s_aug` | `configs/pose_dataset_real_v1_soft.yaml` | `yolo11s-pose.pt` | 3 | 0.814 | 0.864 | 0.147 | 0.310 | none |
| `wheel_baseline_v1` | `configs/pose_dataset.yaml` | `yolo11n-pose.pt` | 3 | 0.593 | 0.730 | 0.159 | 0.580 | none |
| `wheel_real_self_ue_plus_sketchfab_clean_ft20` | `configs/pose_dataset_real_self_ue_plus_sketchfab_clean.yaml` | `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt` | 2 | 0.682 | 0.846 | 0.310 | 0.293 | none |
| `wheel_real_self_ue_plus_sketchfab_clean_ft20_v2` | `configs/pose_dataset_real_self_ue_plus_sketchfab_clean.yaml` | `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt` | 2 | 0.680 | 0.860 | 0.310 | 0.256 | none |
| `wheel_real_v1_clean` | `configs/pose_dataset_real_v1_clean.yaml` | `yolo11n-pose.pt` | 3 | 0.717 | 0.647 | 0.250 | 0.250 | none |
| `wheel_real_v1_self_plus_ue_synthetic_s` | `configs/pose_dataset_real_v1_self_plus_ue_synthetic.yaml` | `runs/pose/runs/pose/wheel_real_v1_self_s/weights/best.pt` | 3 | 0.912 | 0.887 | 0.062 | 0.250 | none |
| `wheel_synthetic_v0_1_baseline` | `configs/pose_dataset_ue_synthetic_v0_1.yaml` | `yolo11n-pose.pt` | 2 | 0.013 | 0.197 | 0.803 | 0.981 | none |
| `wheel_synthetic_v0_2_baseline` | `configs/pose_dataset_ue_synthetic_v0_2.yaml` | `yolo11n-pose.pt` | 2 | 0.944 | 0.886 | 0.058 | 0.033 | none |
| `wheel_ue_sketchfab_geometry_clean_ft20` | `configs/pose_dataset_ue_sketchfab_geometry_clean.yaml` | `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt` | 2 | 0.113 | 0.176 | 0.857 | 0.571 | none |
| `wheel_ue_synthetic_from_self_s` | `configs/pose_dataset_ue_synthetic.yaml` | `runs/pose/runs/pose/wheel_real_v1_self_s/weights/best.pt` | 3 | 0.125 | 0.188 | 0.850 | 0.571 | none |

## Deployment Artifacts

| Artifact | Kind | Size MB |
|---|---:|---:|
| `outputs/production_audit/coreml_export/best.mlmodel` | mlmodel | 37.185 |
| `outputs/production_audit/tflite_export/best_float32.tflite` | tflite | 37.409 |

## Eval Reports

| Report | Model | Data | mAP50 | OKS | FN | FP | GT / pred / matched |
|---|---|---|---:|---:|---:|---:|---:|
| `outputs/eval/wheel_baseline_v1.json` | `runs/pose/wheel_baseline_v1/weights/best.pt` | `configs/pose_dataset.yaml` | 0.593 | 0.730 | 0.159 | 0.580 | 44 / 88 / 37 |
| `outputs/eval/wheel_combined_s.json` | `runs/pose/runs/pose/wheel_real_v1_combined_s/weights/best.pt` | `configs/pose_dataset_real_v1_combined.yaml` | 0.786 | 0.733 | 0.078 | 0.390 | 51 / 77 / 47 |
| `outputs/eval/wheel_combined_s_last.json` | `runs/pose/runs/pose/wheel_real_v1_combined_s/weights/last.pt` | `configs/pose_dataset_real_v1_combined.yaml` | 0.728 | 0.664 | 0.098 | 0.471 | 51 / 87 / 46 |
| `outputs/eval/wheel_real_self_ue_plus_sketchfab_clean_ft20_on_real.json` | `runs/pose/wheel_real_self_ue_plus_sketchfab_clean_ft20/weights/best.pt` | `configs/pose_dataset_real_v1_self_plus_ue_synthetic.yaml` | 0.682 | 0.846 | 0.310 | 0.293 | 84 / 82 / 58 |
| `outputs/eval/wheel_real_self_ue_plus_sketchfab_clean_ft20_v2_on_real.json` | `runs/pose/wheel_real_self_ue_plus_sketchfab_clean_ft20_v2/weights/best.pt` | `configs/pose_dataset_real_v1_self_plus_ue_synthetic.yaml` | 0.680 | 0.860 | 0.310 | 0.256 | 84 / 78 / 58 |
| `outputs/eval/wheel_real_v1_clean.json` | `runs/pose/wheel_real_v1_clean/weights/best.pt` | `configs/pose_dataset_real_v1_clean.yaml` | 0.717 | 0.647 | 0.250 | 0.250 | 16 / 16 / 12 |
| `outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s.json` | `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt` | `configs/pose_dataset_real_v1_self_plus_ue_synthetic.yaml` | 0.697 | 0.887 | 0.286 | 0.259 | 84 / 81 / 60 |
| `outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_on_self_val.json` | `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt` | `configs/pose_dataset_real_v1_self.yaml` | 0.912 | 0.887 | 0.062 | 0.250 | 64 / 80 / 60 |
| `outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_onnx_on_self_plus_ue_val.json` | `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.onnx` | `configs/pose_dataset_real_v1_self_plus_ue_synthetic.yaml` | 0.692 | 0.888 | 0.286 | 0.268 | 84 / 82 / 60 |
| `outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_tflite_on_self_plus_ue_val.json` | `outputs/production_audit/tflite_export/best_float32.tflite` | `configs/pose_dataset_real_v1_self_plus_ue_synthetic.yaml` | 0.692 | 0.888 | 0.286 | 0.268 | 84 / 82 / 60 |
| `outputs/eval/wheel_real_v1_self_s.json` | `runs/pose/runs/pose/wheel_real_v1_self_s/weights/best.pt` | `configs/pose_dataset_real_v1_self.yaml` | 0.903 | 0.894 | 0.078 | 0.280 | 64 / 82 / 59 |
| `outputs/eval/wheel_real_v1_self_s_on_self_plus_ue_val.json` | `runs/pose/runs/pose/wheel_real_v1_self_s/weights/best.pt` | `configs/pose_dataset_real_v1_self_plus_ue_synthetic.yaml` | 0.688 | 0.894 | 0.298 | 0.280 | 84 / 82 / 59 |
| `outputs/eval/wheel_real_v1_soft_n_aug.json` | `runs/pose/runs/pose/wheel_real_v1_soft_n_aug/weights/best.pt` | `configs/pose_dataset_real_v1_soft.yaml` | 0.773 | 0.879 | 0.147 | 0.310 | 34 / 42 / 29 |
| `outputs/eval/wheel_real_v1_soft_s_aug.json` | `runs/pose/runs/pose/wheel_real_v1_soft_s_aug/weights/best.pt` | `configs/pose_dataset_real_v1_soft.yaml` | 0.814 | 0.864 | 0.147 | 0.310 | 34 / 42 / 29 |
| `outputs/eval/wheel_self_s.json` | `runs/pose/runs/pose/wheel_real_v1_self_s/weights/best.pt` | `configs/pose_dataset_real_v1_self.yaml` | 0.903 | 0.894 | 0.078 | 0.280 | 64 / 82 / 59 |
| `outputs/eval/wheel_smoke.json` | `runs/pose/wheel_smoke/weights/best.pt` | `configs/dataset.yaml` | n/a | 0.000 | 0.160 | 0.045 | 25 / 22 / 21 |
| `outputs/eval/wheel_smoke_calibrated.json` | `runs/pose/wheel_smoke/weights/best.pt` | `configs/dataset.yaml` | n/a | 0.019 | 0.160 | 0.045 | 25 / 22 / 21 |
| `outputs/eval/wheel_soft_n_aug.json` | `runs/pose/runs/pose/wheel_real_v1_soft_n_aug/weights/best.pt` | `configs/pose_dataset_real_v1_soft.yaml` | 0.773 | 0.879 | 0.147 | 0.310 | 34 / 42 / 29 |
| `outputs/eval/wheel_soft_s_aug.json` | `runs/pose/runs/pose/wheel_real_v1_soft_s_aug/weights/best.pt` | `configs/pose_dataset_real_v1_soft.yaml` | 0.814 | 0.864 | 0.147 | 0.310 | 34 / 42 / 29 |
| `outputs/eval/wheel_synthetic_v0_1_baseline.json` | `runs/pose/wheel_synthetic_v0_1_baseline/weights/best.pt` | `configs/pose_dataset_ue_synthetic_v0_1.yaml` | 0.000 | n/a | 1.000 | n/a | 391 / 0 / 0 |
| `outputs/eval/wheel_synthetic_v0_1_baseline_conf001.json` | `runs/pose/wheel_synthetic_v0_1_baseline/weights/best.pt` | `configs/pose_dataset_ue_synthetic_v0_1.yaml` | 0.013 | 0.197 | 0.803 | 0.981 | 391 / 4000 / 77 |
| `outputs/eval/wheel_synthetic_v0_2_baseline_on_self_val.json` | `runs/pose/wheel_synthetic_v0_2_baseline/weights/best.pt` | `configs/pose_dataset_real_v1_self.yaml` | 0.000 | n/a | 1.000 | 1.000 | 64 / 32 / 0 |
| `outputs/eval/wheel_synthetic_v0_2_baseline_test.json` | `runs/pose/wheel_synthetic_v0_2_baseline/weights/best.pt` | `configs/pose_dataset_ue_synthetic_v0_2_testswap.yaml` | 0.922 | 0.912 | 0.070 | 0.070 | 185 / 185 / 172 |
| `outputs/eval/wheel_synthetic_v0_2_baseline_val.json` | `runs/pose/wheel_synthetic_v0_2_baseline/weights/best.pt` | `configs/pose_dataset_ue_synthetic_v0_2.yaml` | 0.944 | 0.886 | 0.058 | 0.033 | 189 / 184 / 178 |
| `outputs/eval/wheel_ue_sketchfab_geometry_clean_ft20_on_real.json` | `runs/pose/wheel_ue_sketchfab_geometry_clean_ft20/weights/best.pt` | `configs/pose_dataset_real_v1_self_plus_ue_synthetic.yaml` | 0.113 | 0.176 | 0.857 | 0.571 | 84 / 28 / 12 |
| `outputs/eval/wheel_ue_synthetic_from_self_s.json` | `runs/pose/wheel_ue_synthetic_from_self_s/weights/best.pt` | `configs/pose_dataset_ue_synthetic.yaml` | 0.095 | 0.152 | 0.950 | 0.500 | 20 / 2 / 1 |
| `outputs/eval/wheel_ue_synthetic_from_self_s_conf005.json` | `runs/pose/wheel_ue_synthetic_from_self_s/weights/best.pt` | `configs/pose_dataset_ue_synthetic.yaml` | 0.125 | 0.188 | 0.850 | 0.571 | 20 / 7 / 3 |
| `outputs/eval/wheel_v2.json` | `runs/pose/wheel_v2/weights/best.pt` | `configs/dataset.yaml` | n/a | 0.130 | 0.000 | 0.138 | 25 / 29 / 25 |
| `outputs/eval/wheel_v3.json` | `runs/pose/wheel_v3/weights/best.pt` | `configs/dataset.yaml` | n/a | 0.941 | 0.041 | 0.028 | 220 / 217 / 211 |
