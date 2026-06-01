# Model Selection Audit

Machine-readable champion retention and candidate promotion guard.

- OK: True
- Selected champion: `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt`
- Anchor data: `configs/pose_dataset_real_v1_self_plus_ue_synthetic.yaml`
- Real validation data: `configs/pose_dataset_real_v1_self.yaml`
- Promotion required: 0
- Failures: none

## 3D Disc-Height Acceptance

- Status: `insufficient_evidence`
- Detail: no eval3d report supplied — data-blocked: needs a floor-ray correct export + model-predicted A/B/C (see docs/EVAL3D_AND_3D_LOSS_STATUS.md, docs/EXPORT_PARITY_AUDIT.md)
- Promotion blocked on 3D: none

## Champion Evidence

- Anchor eval: `outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s.json`
- Anchor metrics: mAP50=0.697, OKS=0.887, FN=0.286, FP=0.259, matched=60
- Real eval: `outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_on_self_val.json`
- Real metrics: mAP50=0.912, OKS=0.887, FN=0.062, FP=0.250, matched=60

## Candidates

| Status | Run | Model | Anchor mAP50 | Anchor OKS | FN | FP | Matched | Real eval | Reasons |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| not_promoted | `wheel_real_v1_self_s` | `runs/pose/runs/pose/wheel_real_v1_self_s/weights/best.pt` | 0.688 | 0.894 | 0.298 | 0.280 | 59 | `outputs/eval/wheel_real_v1_self_s.json` | bbox_mAP50_below_champion, fn_rate_above_champion, fp_rate_above_champion, matched_below_champion |
| not_promoted | `wheel_real_self_ue_plus_sketchfab_clean_ft20` | `runs/pose/wheel_real_self_ue_plus_sketchfab_clean_ft20/weights/best.pt` | 0.682 | 0.846 | 0.310 | 0.293 | 58 | `missing` | bbox_mAP50_below_champion, oks_mean_below_champion, fn_rate_above_champion, fp_rate_above_champion, matched_below_champion, missing_real_only_eval_for_promotion |
| not_promoted | `wheel_real_self_ue_plus_sketchfab_clean_ft20_v2` | `runs/pose/wheel_real_self_ue_plus_sketchfab_clean_ft20_v2/weights/best.pt` | 0.680 | 0.860 | 0.310 | 0.256 | 58 | `missing` | bbox_mAP50_below_champion, oks_mean_below_champion, fn_rate_above_champion, matched_below_champion, missing_real_only_eval_for_promotion |
| selected_champion | `wheel_real_v1_self_plus_ue_synthetic_s` | `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt` | 0.697 | 0.887 | 0.286 | 0.259 | 60 | `outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_on_self_val.json` | configured_champion |
| not_promoted | `wheel_ue_sketchfab_geometry_clean_ft20` | `runs/pose/wheel_ue_sketchfab_geometry_clean_ft20/weights/best.pt` | 0.113 | 0.176 | 0.857 | 0.571 | 12 | `missing` | bbox_mAP50_below_champion, oks_mean_below_champion, fn_rate_above_champion, fp_rate_above_champion, matched_below_champion, missing_real_only_eval_for_promotion |
