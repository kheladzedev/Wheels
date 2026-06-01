# Dataset Audit

Generated from `configs/pose_dataset*.yaml`.

## Summary

- Overall OK: False
- Configs: 22
- Passed: 2
- Failed: 20
- Total train images across configs: 5705
- Total val images across configs: 1287
- Total wheel label lines across configs: 12148

## Gate

- Gate OK: True
- Gate scope: configured_subset
- Gate configs: 2
- Gate missing configs: 0
- Gate failed configs: 0
- Gate train images: 423
- Gate val images: 106
- Gate wheel label lines: 602

## Configs

| Config | Root | OK | Train img / wheels | Val img / wheels | Leakage stem/hash | Failures |
|---|---|---:|---:|---:|---:|---|
| `configs/pose_dataset.yaml` | `/Users/codefactory/Desktop/ML/VSBL/Wheels/data/wheel_pose_dataset` | False | 319 / 177 | 80 / 44 | 0 / 0 | train_label_errors:17, val_label_errors:5 |
| `configs/pose_dataset_real_self_ue_plus_sketchfab_clean.yaml` | `/Users/codefactory/Desktop/ML/VSBL/Wheels/data/wheel_pose_dataset_real_self_ue_plus_sketchfab_clean` | False | 354 / 830 | 58 / 84 | 0 / 0 | train_label_errors:708, val_label_errors:51 |
| `configs/pose_dataset_real_v1.yaml` | `/Users/codefactory/Desktop/ML/VSBL/Wheels/data/wheel_pose_dataset_real_v1` | False | 319 / 177 | 80 / 44 | 0 / 0 | train_label_errors:17, val_label_errors:5 |
| `configs/pose_dataset_real_v1_clean.yaml` | `/Users/codefactory/Desktop/ML/VSBL/Wheels/data/wheel_pose_dataset_real_v1_clean` | False | 62 / 73 | 15 / 16 | 0 / 0 | train_label_errors:2, val_label_errors:2 |
| `configs/pose_dataset_real_v1_combined.yaml` | `/Users/codefactory/Desktop/ML/VSBL/Wheels/data/wheel_pose_dataset_real_v1_combined` | False | 166 / 204 | 42 / 51 | 0 / 0 | train_label_errors:38, val_label_errors:9 |
| `configs/pose_dataset_real_v1_self.yaml` | `/Users/codefactory/Desktop/ML/VSBL/Wheels/data/wheel_pose_dataset_real_v1_self` | False | 191 / 257 | 48 / 64 | 0 / 0 | train_label_errors:31 |
| `configs/pose_dataset_real_v1_self_plus_ue_synthetic.yaml` | `/Users/codefactory/Desktop/ML/VSBL/Wheels/data/wheel_pose_dataset_real_v1_self_plus_ue_synthetic` | False | 232 / 327 | 58 / 84 | 0 / 0 | train_label_errors:205, val_label_errors:51 |
| `configs/pose_dataset_real_v1_self_plus_ue_synthetic_strict.yaml` | `/Users/codefactory/Desktop/ML/VSBL/Wheels/data/wheel_pose_dataset_real_v1_self_plus_ue_synthetic_strict` | True | 232 / 237 | 58 / 64 | 0 / 0 | none |
| `configs/pose_dataset_real_v1_self_strict.yaml` | `/Users/codefactory/Desktop/ML/VSBL/Wheels/data/wheel_pose_dataset_real_v1_self_strict` | True | 191 / 237 | 48 / 64 | 0 / 0 | none |
| `configs/pose_dataset_real_v1_soft.yaml` | `/Users/codefactory/Desktop/ML/VSBL/Wheels/data/wheel_pose_dataset_real_v1_soft` | False | 118 / 143 | 29 / 34 | 0 / 0 | train_label_errors:9, val_label_errors:1 |
| `configs/pose_dataset_real_web_angle_demo.yaml` | `/Users/codefactory/Desktop/ML/VSBL/Wheels/data/wheel_pose_dataset_real_web_angle_demo` | False | 0 / 0 | 0 / 0 | 0 / 0 | dataset_root_missing:/Users/codefactory/Desktop/ML/VSBL/Wheels/data/wheel_pose_dataset_real_web_angle_demo, train_has_no_images, train_has_no_labels, val_has_no_images, val_has_no_labels |
| `configs/pose_dataset_real_web_combined_demo.yaml` | `/Users/codefactory/Desktop/ML/VSBL/Wheels/data/wheel_pose_dataset_real_web_combined_demo` | False | 0 / 0 | 0 / 0 | 0 / 0 | dataset_root_missing:/Users/codefactory/Desktop/ML/VSBL/Wheels/data/wheel_pose_dataset_real_web_combined_demo, train_has_no_images, train_has_no_labels, val_has_no_images, val_has_no_labels |
| `configs/pose_dataset_real_web_demo.yaml` | `/Users/codefactory/Desktop/ML/VSBL/Wheels/data/wheel_pose_dataset_real_web_demo` | False | 0 / 0 | 0 / 0 | 0 / 0 | dataset_root_missing:/Users/codefactory/Desktop/ML/VSBL/Wheels/data/wheel_pose_dataset_real_web_demo, train_has_no_images, train_has_no_labels, val_has_no_images, val_has_no_labels |
| `configs/pose_dataset_ue_sketchfab_geometry.yaml` | `/Users/codefactory/Desktop/ML/VSBL/Wheels/data/wheel_pose_dataset_ue_sketchfab_geometry` | False | 154 / 560 | 38 / 142 | 0 / 0 | train_label_errors:560, val_label_errors:142 |
| `configs/pose_dataset_ue_sketchfab_geometry_clean.yaml` | `/Users/codefactory/Desktop/ML/VSBL/Wheels/data/wheel_pose_dataset_ue_sketchfab_geometry_clean` | False | 122 / 503 | 30 / 123 | 0 / 0 | train_label_errors:503, val_label_errors:123 |
| `configs/pose_dataset_ue_synthetic.yaml` | `/Users/codefactory/Desktop/ML/VSBL/Wheels/data/wheel_pose_dataset_ue_synthetic` | False | 41 / 70 | 10 / 20 | 0 / 0 | train_label_errors:174, val_label_errors:51 |
| `configs/pose_dataset_ue_synthetic_pseudo_from_champion.yaml` | `/Users/codefactory/Desktop/ML/VSBL/Wheels/data/wheel_pose_dataset_ue_synthetic_pseudo_from_champion` | False | 4 / 5 | 1 / 1 | 0 / 0 | train_label_errors:10, val_label_errors:2 |
| `configs/pose_dataset_ue_synthetic_v0_1.yaml` | `/Users/codefactory/Desktop/ML/VSBL/Wheels/data/wheel_pose_dataset_ue_synthetic_v0_1` | False | 800 / 1572 | 200 / 391 | 0 / 8 | train_label_errors:864, val_label_errors:249, train_val_hash_overlap:8 |
| `configs/pose_dataset_ue_synthetic_v0_1_clean.yaml` | `/Users/codefactory/Desktop/ML/VSBL/Wheels/data/wheel_pose_dataset_ue_synthetic_v0_1_clean` | False | 800 / 1565 | 192 / 382 | 0 / 8 | train_label_errors:889, val_label_errors:208, train_val_hash_overlap:8, conversion_report:conversion_dataset_image_count_mismatch:1000!=992, conversion_report:conversion_wheel_count_mismatch:1963!=1947 |
| `configs/pose_dataset_ue_synthetic_v0_2.yaml` | `/Users/codefactory/Desktop/ML/VSBL/Wheels/data/wheel_pose_dataset_ue_synthetic_v0_2` | False | 800 / 1520 | 100 / 189 | 0 / 0 | train_label_errors:842, val_label_errors:98, conversion_report:conversion_dataset_image_count_mismatch:1000!=900, conversion_report:conversion_wheel_count_mismatch:1894!=1709 |
| `configs/pose_dataset_ue_synthetic_v0_2_test_as_val.yaml` | `/Users/codefactory/Desktop/ML/VSBL/Wheels/data/wheel_pose_dataset_ue_synthetic_v0_2` | False | 800 / 1520 | 100 / 189 | 0 / 0 | train_label_errors:842, val_label_errors:98, conversion_report:conversion_dataset_image_count_mismatch:1000!=900, conversion_report:conversion_wheel_count_mismatch:1894!=1709 |
| `configs/pose_dataset_ue_synthetic_v0_2_testswap.yaml` | `/Users/codefactory/Desktop/ML/VSBL/Wheels/data/wheel_pose_dataset_ue_synthetic_v0_2_testaval` | False | 0 / 0 | 100 / 185 | 0 / 0 | train_has_no_images, train_has_no_labels, val_label_errors:101 |
