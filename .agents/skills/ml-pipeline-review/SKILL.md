---
name: ml-pipeline-review
description: Use before claiming an ML / dataset / inference task is "done" in VSBL. Activates whenever Codex is about to summarise completed work, write the final report for a /goal, or open a PR. Forces a verification pass — running tests, listing changed files, distinguishing synthetic-data plumbing from real quality signal — so we never ship a green-pytest-on-cartoons as "production ready".
---

# VSBL — pipeline review checklist

Run this checklist BEFORE writing the final summary. It is cheap; the
cost of claiming "done" on broken state is high (silent contract
violation, leaked training data, lost weights).

If any step fails or can't be completed, **do not claim done** — report
the gap explicitly in the summary.

## 1. Tests are green

```bash
./.venv/bin/pytest -q
```

- Must exit `0`.
- If anything fails, paste the exact failing test IDs + the first
  failure message into the summary. Do not skip, mark `xfail`, or
  delete tests to make them pass. If a test is wrong, fix the test
  and say so.
- A green run on synthetic data is **plumbing OK**, not **quality OK**.
  See §5.

## 2. Smoke checks for whichever pipeline you touched

Plugin flow (run if you touched any `*keypoint*` script,
`pose_dataset.yaml`, or `convert_keypoint_incoming_to_yolo_pose.py`):

```bash
./.venv/bin/python src/create_sample_keypoint_incoming.py --count 20 --overwrite
./.venv/bin/python src/check_keypoint_incoming.py --source-root data/incoming/android_plugin
./.venv/bin/python src/preview_keypoint_annotations.py --source-root data/incoming/android_plugin --count 5
./.venv/bin/python src/convert_keypoint_incoming_to_yolo_pose.py \
    --source-root data/incoming/android_plugin \
    --dataset-root data/wheel_pose_dataset --overwrite
./.venv/bin/python src/check_yolo_pose_dataset.py --dataset-root data/wheel_pose_dataset
./.venv/bin/python src/preview_yolo_pose_labels.py \
    --dataset-root data/wheel_pose_dataset --split train --count 5
```

Legacy flow (run if you touched any `*incoming*` script — non-keypoint
variant — `dataset.yaml`, `check_dataset.py`, `preview_labels.py`):

```bash
./.venv/bin/python src/create_sample_incoming.py --count 20 --overwrite
./.venv/bin/python src/convert_incoming_to_yolo.py \
    --source-root data/incoming/manual_sample \
    --dataset-root data/wheel_dataset --overwrite
./.venv/bin/python src/check_dataset.py --dataset-root data/wheel_dataset
./.venv/bin/python src/preview_labels.py --dataset-root data/wheel_dataset --split train --count 5
```

Inference / training paths require weights; do not run them as part of
the checklist unless explicitly asked.

## 3. List every changed file

In the final summary, enumerate every new and modified file. Group by
purpose if it helps readability, but never elide a file because it
"feels small". Generated artefacts under `data/`, `runs/`, `outputs/`
are not "changed source" — call them out separately if relevant.

A useful command:

```bash
git status --short
```

## 4. Confirm the contract was not broken

If the task touched any of:

- `src/infer_image.py`, `src/infer_batch.py`,
  `src/postprocess_wheels.py`, `src/visualize_predictions.py`
- `docs/AR_ML_CONTRACT.md`, `docs/KEYPOINT_SPEC.md`, README schema
  sections
- `tests/test_ar_contract.py`

then explicitly state in the summary: "AR JSON contract: unchanged"
(or, if you did change it, point at the AR sign-off entry in
`docs/OPEN_QUESTIONS_AR_SPEC.md`).

Invoke the `vsbl-ar-contract` skill if unsure.

### 4a. Schema field-name invariants (post-2026-05-14)

Verify each of the following is still true before claiming done. If any
fails, **fix the regression and rerun the checklist** — do not paper
over with a comment.

- Confirmed AR JSON top-level keys are **exactly** `{frame_id, wheels}`.
- Each `wheels[]` entry has **exactly** `{bbox_xyxy, confidence, points}`.
- `points` keys are **exactly** `{a, b, c_disc_bottom}`.
- The string `track_id` never appears in any inference-side response
  schema, test fixture, or docs claiming to be the AR contract.
- No 3D / world-space coordinates, plane parameters, or RANSAC
  residuals leak into the ML side. Those are AR responsibilities.
- The legacy literal label strings `rim_left` / `rim_right` may still
  appear in `src/postprocess_wheels.KEYPOINT_NAMES` and the legacy
  converter; they must **never** be used to *describe* A/B in any
  user-facing doc, skill, README, annotator label, or test message.

### 4b. A/B semantics invariants (post-2026-05-14)

Quick grep before summarising:

```bash
grep -nE "rim_left|rim_right|metal rim left|metal rim right|left point of metal rim|right point of metal rim" \
    docs/ README.md AGENTS.md .Codex/skills/ src/manual_keypoint_annotator.py \
    src/preview_keypoint_annotations.py src/preview_yolo_pose_labels.py 2>/dev/null
```

Hits are allowed **only** when the surrounding text marks them as
*legacy* / *obsolete* / *drifted*. A bare `rim_left` describing A is
a bug. Confirm A and B are described as floor / raycast points
everywhere user-facing.

### 4c. Responsibility-split invariants

ML side must do only:
- single-frame 2D wheel detection;
- single-frame 2D keypoint regression (A, B, C in screen space);
- wheel-level `confidence`;
- echo `frame_id`.

ML side must **not** do:
- raycasting (of any kind);
- RANSAC;
- 3D plane recovery or any plane math;
- world / camera-space coordinates;
- cross-frame tracking or `track_id`;
- camera-transform handling;
- K-frame accumulation or averaging.

If a diff added any of the forbidden behaviours into ML code, it is a
contract violation regardless of test status.

## 5. Do not claim production quality on synthetic data

The synthetic generators (`create_sample_incoming.py`,
`create_sample_keypoint_incoming.py`) produce cartoon cars on a grey
background. A model trained on them will not generalise. A green
`check_*` pass on them validates **format**, not **accuracy**.

In the summary, mark every metric or quality claim with its data
source:

- "Plumbing validated on synthetic data" — OK to say.
- "Detection works on synthetic data" — OK to say.
- "Model is production ready" / "Detections look good" — **forbidden**
  without real labelled data and a held-out val score.

If the task is "validate the plumbing", a green synthetic run is
sufficient evidence of done. If the task is "make the model good",
synthetic-only evidence is **never** sufficient — call this out.

## 6. Final summary format

Always include:

1. **Changed files** — every new / modified path.
2. **Commands run** — exact invocations, in the order they were run.
3. **Test result** — pytest count + status, plus any failures.
4. **What is now the source of truth** — which file / config / skill
   should the next contributor read first.
5. **Open items** — what's still blocked, what we'd do next.

Keep it terse (per repo style) but do not skip these five.

## See also

- `docs/CLAUDE_CODE_WORKFLOW.md` — when to use `/goal`, how done is
  defined, how to scope changes.
- Skill `vsbl-ar-contract` — contract-side review hooks.
- Skill `yolo-pose-dataset` — dataset-side commands and invariants.
