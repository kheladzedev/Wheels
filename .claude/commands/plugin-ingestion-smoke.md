---
description: Full Android-plugin ingestion smoke — generator → incoming check → converter → YOLO-pose check → label preview. Validates dataset plumbing end-to-end on synthetic data.
argument-hint: [optional --count N, default 50]
---

# /plugin-ingestion-smoke — full plugin ingestion smoke chain

Run the entire `android_plugin` ingestion pipeline on synthetic data,
end to end. Validates plumbing only — see the `ml-pipeline-review`
skill: synthetic green ≠ production ready.

Argument: `$ARGUMENTS` — optional `--count N` to override the synthetic
batch size (default 50). Anything else is passed through verbatim to
the first script.

If any step exits non-zero, **stop and report the failing command +
output**. Do not proceed to the next step.

## Steps (sequential, foreground)

### 1. Generate a synthetic plugin batch

```bash
./.venv/bin/python src/create_sample_keypoint_incoming.py --count 50 --overwrite $ARGUMENTS
```

Expected: `data/incoming/android_plugin/` is regenerated with N images,
N annotations, and `metadata/source_info.json`. Default N = 50.

### 2. Validate the incoming batch format

```bash
./.venv/bin/python src/check_keypoint_incoming.py \
    --source-root data/incoming/android_plugin
```

Expected: `Errors: 0`. ERROR-level findings fail the smoke;
WARNING-level findings are reported but do not.

### 3. Preview a few incoming samples (pre-conversion)

```bash
./.venv/bin/python src/preview_keypoint_annotations.py \
    --source-root data/incoming/android_plugin --count 10
```

Expected: 10 files under `outputs/keypoint_preview/` showing bbox +
A/B/C overlay on the raw incoming images.

### 4. Convert plugin batch → YOLO-pose dataset

```bash
./.venv/bin/python src/convert_keypoint_incoming_to_yolo_pose.py \
    --source-root data/incoming/android_plugin \
    --dataset-root data/wheel_pose_dataset --overwrite
```

Expected: `converted == source_images`, `skipped == 0`. Drop reasons
all zero. `metadata/split_manifest.json` and
`metadata/conversion_report.json` written.

### 5. Validate the converted YOLO-pose dataset

```bash
./.venv/bin/python src/check_yolo_pose_dataset.py \
    --dataset-root data/wheel_pose_dataset
```

Expected: `OK — dataset layout looks valid.` Exit `0`. No
`FAILED` line.

### 6. Preview YOLO-pose labels

```bash
./.venv/bin/python src/preview_yolo_pose_labels.py \
    --dataset-root data/wheel_pose_dataset --split train --count 10
```

Expected: 10 files under `outputs/pose_label_preview/train/` showing
bbox + A (green) / B (yellow) / C (red) overlay. Each must show the
same wheel count as the source annotation.

### 7. Final report

After all six pass, report:

- Source batch: N images, M wheels (`metadata/source_info.json`).
- Incoming check: clean.
- Incoming preview: 10 files in `outputs/keypoint_preview/`.
- Conversion: K converted, 0 skipped, L wheel lines emitted
  (`metadata/conversion_report.json`).
- Dataset check: OK.
- Label preview: 10 files in `outputs/pose_label_preview/train/`.

End with one of: `Plugin ingestion smoke OK.` or
`Plugin ingestion smoke FAILED at step N` + the exact failing command
+ first line of failure output.

## Notes

- This command **regenerates** `data/incoming/android_plugin/` and
  `data/wheel_pose_dataset/`. If a real batch or hand-curated dataset
  is staged there, back it up first.
- The synthetic generator produces cartoon cars — pure smoke fixture,
  not training data. Detection quality on this data is **not** a
  signal about model quality. See the `ml-pipeline-review` skill, §5.
- For just the fast healthcheck (no conversion), use `/healthcheck`.
- For a contract-level audit before merging, use `/review-ar-contract`.
