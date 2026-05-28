#!/usr/bin/env bash
# Run scripts/eval_baseline.sh against every checkpoint we trained this
# session, using the same val set for each (the model's own training val).
# Produces JSON + a per-model summary line so apples-to-apples ranking is
# easy.

set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p outputs/eval

# (model_name, weights_path, data_yaml)
declare -a EXPS=(
  "baseline_v1|runs/pose/wheel_baseline_v1/weights/best.pt|configs/pose_dataset.yaml"
  "real_v1_clean|runs/pose/wheel_real_v1_clean/weights/best.pt|configs/pose_dataset_real_v1_clean.yaml"
  "soft_n_aug|runs/pose/runs/pose/wheel_real_v1_soft_n_aug/weights/best.pt|configs/pose_dataset_real_v1_soft.yaml"
  "soft_s_aug|runs/pose/runs/pose/wheel_real_v1_soft_s_aug/weights/best.pt|configs/pose_dataset_real_v1_soft.yaml"
)

# Combined model may not exist yet — skip cleanly if absent.
if [ -f "runs/pose/runs/pose/wheel_real_v1_combined_s/weights/best.pt" ]; then
  EXPS+=("combined_s|runs/pose/runs/pose/wheel_real_v1_combined_s/weights/best.pt|configs/pose_dataset_real_v1_combined.yaml")
fi

# Self-labeled (final champion)
if [ -f "runs/pose/runs/pose/wheel_real_v1_self_s/weights/best.pt" ]; then
  EXPS+=("self_s|runs/pose/runs/pose/wheel_real_v1_self_s/weights/best.pt|configs/pose_dataset_real_v1_self.yaml")
fi

echo "name,box_mAP50,box_mAP50_95,oks_mean,fn_rate,fp_rate,kp_a_px,kp_b_px,kp_c_px" \
  > outputs/eval/all_models_summary.csv

for entry in "${EXPS[@]}"; do
  IFS='|' read -r name weights data <<< "$entry"
  if [ ! -f "$weights" ]; then
    echo "[skip] $name — no weights at $weights"
    continue
  fi
  echo "[run] $name → eval against $(basename "$data")"
  MODEL="$weights" \
    DATA="$data" \
    OUT="outputs/eval/wheel_${name}.json" \
    bash scripts/eval_baseline.sh > /tmp/eval_$name.log 2>&1 || {
    echo "[err] eval failed for $name; see /tmp/eval_$name.log"
    continue
  }
  ./.venv/bin/python -c "
import json
d = json.load(open('outputs/eval/wheel_${name}.json'))
mb = d['metrics_bbox']
rates = d['rates']
oks = d['oks']
kp = d['per_keypoint_pixel_error']
def med(k):
    e = kp.get(k, {})
    v = e.get('median') if isinstance(e, dict) else None
    return round(v, 2) if v is not None else 'na'
row = ','.join(str(x) for x in (
    '${name}',
    round(mb['mAP50'], 4), round(mb['mAP50_95'], 4),
    round(oks['mean'], 4),
    round(rates['false_negative_rate'], 4),
    round(rates['false_positive_rate'], 4),
    med('point_a'), med('point_b'), med('point_c_disc_bottom'),
))
print(row)
open('outputs/eval/all_models_summary.csv', 'a').write(row + '\n')
"
done

echo ''
echo '=== outputs/eval/all_models_summary.csv ==='
cat outputs/eval/all_models_summary.csv | column -t -s,
