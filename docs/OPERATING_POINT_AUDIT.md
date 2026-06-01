# Operating Point Audit

Real-validation confidence threshold selection for the current champion.

- OK: True
- Policy: lowest_confidence_threshold_that_meets_all_quality_gates
- Selected report: `outputs/production_audit/threshold_conf080_real_val.json`
- Selected conf: 0.800
- Selected metrics: mAP50=0.903, OKS=0.888, FN=0.094, FP=0.147

| Report | Conf | OK | mAP50 | OKS | FN | FP | Failures |
|---|---:|---:|---:|---:|---:|---:|---|
| `outputs/production_audit/threshold_conf015_real_val.json` | 0.150 | False | 0.912 | 0.887 | 0.062 | 0.277 | false_positive_rate_above_maximum |
| `outputs/production_audit/threshold_conf020_real_val.json` | 0.200 | False | 0.912 | 0.887 | 0.062 | 0.268 | false_positive_rate_above_maximum |
| `outputs/production_audit/threshold_conf030_real_val.json` | 0.300 | False | 0.912 | 0.887 | 0.062 | 0.250 | false_positive_rate_above_maximum |
| `outputs/production_audit/threshold_conf040_real_val.json` | 0.400 | False | 0.912 | 0.887 | 0.062 | 0.221 | false_positive_rate_above_maximum |
| `outputs/production_audit/threshold_conf050_real_val.json` | 0.500 | False | 0.912 | 0.887 | 0.062 | 0.211 | false_positive_rate_above_maximum |
| `outputs/production_audit/threshold_conf060_real_val.json` | 0.600 | False | 0.912 | 0.887 | 0.078 | 0.213 | false_positive_rate_above_maximum |
| `outputs/production_audit/threshold_conf070_real_val.json` | 0.700 | False | 0.903 | 0.888 | 0.094 | 0.194 | false_positive_rate_above_maximum |
| `outputs/production_audit/threshold_conf075_real_val.json` | 0.750 | False | 0.903 | 0.888 | 0.094 | 0.183 | false_positive_rate_above_maximum |
| `outputs/production_audit/threshold_conf080_real_val.json` | 0.800 | True | 0.903 | 0.888 | 0.094 | 0.147 | none |
