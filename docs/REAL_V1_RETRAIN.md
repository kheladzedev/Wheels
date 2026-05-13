# wheel_real_v1 — retrain on human-verified labels

Step-by-step from the current 396 auto-drafts to a model trained on
QA-passed annotations. Everything before §3 is one-time per QA round;
§3 onwards reruns whenever the dataset changes.

## Starting state

- `data/incoming/real_v1/images/` — 396 photos from Wikimedia Commons.
- `data/incoming/real_v1/annotations/` — 396 auto-drafts produced by
  `src/auto_annotate_wheels.py` (COCO-vehicle + SAM-2 grid prompts),
  each flagged `_draft: true,
  _warning: NOT_GROUND_TRUTH_REQUIRES_HUMAN_REVIEW`.
- `runs/pose/wheel_baseline_v1/weights/best.pt` — the auto-drafts-only
  baseline. Use it as the comparison point for the retrain.

## 1. Manual QA pass

```bash
./scripts/qa_real_v1.sh
```

What happens:

- Opens `src/manual_keypoint_annotator.py` with
  `--prefill-from data/incoming/real_v1/annotations`.
- Each photo opens with its draft wheels drawn. Keys (annotator
  source has the canonical list):
  - `y` / Enter — accept current wheel as-is
  - drag a point — fix one keypoint
  - click in bbox + `d` — drop a wheel
  - `e` — clear and re-click from scratch
  - `n` — next image (drops remaining unconfirmed wheels)
  - `q` — quit (progress is saved per image)
- Re-run the script to pick up where you left off; already-saved
  images are skipped. Pass `--rerun` to redo them.
- QA output:
  - per-image JSONs in `data/incoming/real_v1/annotations_qa/`
  - bundle (images + annotations + metadata) in
    `data/incoming/real_v1_qa/`

Target: aim for ≥200 wheels confirmed across ≥120 images for the v1
retrain. Quality > quantity — drop ambiguous wheels, don't fight them.

## 2. Validate the QA bundle

```bash
./.venv/bin/python src/check_keypoint_incoming.py \
    --source-root data/incoming/real_v1_qa
```

Exit 0 with `Errors: 0` is required. Common issues to fix manually:
- Wheel bbox not enclosing a/b/c — re-open the image, re-position.
- Annotation file referencing a missing image — usually a typo in a
  manual rename; delete or restore.

Optional visual sanity:

```bash
./.venv/bin/python src/preview_keypoint_annotations.py \
    --source-root data/incoming/real_v1_qa --count 10
# inspect outputs/keypoint_preview/*
```

## 3. Convert to YOLO-pose

```bash
./.venv/bin/python src/convert_keypoint_incoming_to_yolo_pose.py \
    --source-root data/incoming/real_v1_qa \
    --dataset-root data/wheel_pose_dataset_real_v1 \
    --overwrite
./.venv/bin/python src/check_yolo_pose_dataset.py \
    --dataset-root data/wheel_pose_dataset_real_v1
```

## 4. Train

```bash
./.venv/bin/python src/train_yolo.py \
    --data configs/pose_dataset_real_v1.yaml \
    --model yolo11n-pose.pt \
    --epochs 50 --device mps \
    --project runs/pose --name wheel_real_v1
```

Expected wall time on M3 Ultra MPS: ~15 min for 50 epochs at ~200
wheels. Watch `runs/pose/wheel_real_v1/results.png` for the usual
overfitting tell (val loss climbs while train loss keeps dropping) —
if it appears before epoch 40, cut epochs.

## 5. Export

```bash
./.venv/bin/python src/export_model.py \
    --model runs/pose/wheel_real_v1/weights/best.pt \
    --format onnx --device cpu --simplify
```

Sanity check should print `matched: True` with bbox drift <2px,
keypoint drift <3px, conf drift <0.05.

## 6. Eval + diff vs baseline

```bash
MODEL=runs/pose/wheel_real_v1/weights/best.pt \
DATA=configs/pose_dataset_real_v1.yaml \
OUT=outputs/eval/wheel_real_v1.json \
    ./scripts/eval_baseline.sh

diff \
    <(./.venv/bin/python -c "import json; d=json.load(open('outputs/eval/wheel_baseline_v1.json')); print(json.dumps({'mAP50': d['metrics_bbox']['mAP50'], 'oks_mean': d['oks']['mean'], 'fn_rate': d['rates']['false_negative_rate'], 'fp_rate': d['rates']['false_positive_rate']}, indent=2))") \
    <(./.venv/bin/python -c "import json; d=json.load(open('outputs/eval/wheel_real_v1.json')); print(json.dumps({'mAP50': d['metrics_bbox']['mAP50'], 'oks_mean': d['oks']['mean'], 'fn_rate': d['rates']['false_negative_rate'], 'fp_rate': d['rates']['false_positive_rate']}, indent=2))") || true
```

Hand-write `outputs/eval/wheel_real_v1_summary.md` (mirror the
baseline summary structure) once the numbers land.

Target per Stage 4 in `docs/TASK_PLAN.md`:

- mAP50 (box) ≥ 0.85
- OKS mean (σ=0.10) ≥ 0.5
- Median per-keypoint pixel error ≤ 5 px

If the numbers regress vs `wheel_baseline_v1`, **do not ship** — most
likely the QA pass dropped too many wheels and the train set is now
too small. Either expand QA coverage or supplement with a second
Wikimedia scrape pass.

## 7. Update presentation

After acceptable numbers:

- Append a "wheel_real_v1" row to the table in `README.md` (alongside
  the existing baseline row).
- Refresh the headline numbers in `docs/DEMO.md` §3.
- Mark Stage 5 done in `docs/TASK_PLAN.md`.

## 8. Optional disk cleanup

Once `wheel_real_v1` ships, you can reclaim:

- `data/incoming/real_v1/annotations/` — the auto-drafts are now
  redundant (the QA outputs are the source of truth).
- `data/wheel_pose_dataset/` — the baseline's converted output. Keep
  if you need to reproduce the baseline metrics; otherwise drop.
- `data/incoming/manual_real_auto/` — superseded by real_v1, can be
  removed.

Do not delete `data/incoming/real_v1/images/` — those are the
source-of-truth photos the QA bundle references.
