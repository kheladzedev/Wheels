# VSBL — AR Wheel Fitting (ML side)

Operating manual for Claude Code in this repo. Read top-to-bottom before
touching anything contract-bearing.

## Project mission

Build the ML side of the AR "Примерка колес" mechanic: given a single
RGB camera frame from the AR client, return every detected wheel as a
bbox + three named 2D keypoints. The AR client uses those keypoints to
raycast + RANSAC into a vertical wheel plane and recover disc-installation
height. **Everything 3D lives on the AR side.** Android is the first
target platform (TFLite / LiteRT).

This repo owns: dataset ingestion, YOLO-pose training, inference,
postprocessing into the AR JSON payload, and export to runtime formats.

This repo does **not** own: 3D reconstruction, RANSAC, plane fitting,
K-frame accumulation, raycasting, or per-frame ↔ camera-transform
association. Those are AR-client responsibilities.

## Current confirmed AR/ML contract

Confirmed by the AR team **2026-05-13**. Authoritative shape for the
response. Per-frame, per-wheel, in pixel coordinates:

```json
{
  "frame_id": "frame_0001",
  "wheels": [
    {
      "bbox_xyxy": [x1, y1, x2, y2],
      "confidence": 0.94,
      "points": {
        "a":             [xa, ya],
        "b":             [xb, yb],
        "c_disc_bottom": [xc, yc]
      }
    }
  ]
}
```

Load-bearing rules from the same confirmation round:

- ML returns **per-frame results only**. No batching across frames.
- `frame_id` is echoed back from AR (string). AR matches it with the
  camera transform it saved at capture time.
- `wheels[].bbox_xyxy` — pixels, top-left + bottom-right corners.
- `wheels[].confidence` — wheel-level detection confidence in `[0, 1]`.
- `wheels[].points.a` — **A: left floor-ray point**. Screen-space
  position that AR raycasts onto the floor plane near the wheel's
  ground footprint to anchor the vertical wheel plane. **Not** a metal
  rim point.
- `wheels[].points.b` — **B: right floor-ray point**. Screen-space
  position that AR raycasts onto the floor; together with A defines
  the base direction of the vertical wheel plane via RANSAC across K
  frames. **Not** a metal rim point.
- `wheels[].points.c_disc_bottom` — **C: lowest visible point of the
  metal rim / disc**. AR raycasts onto the already-recovered vertical
  plane to compute the disc's installation height.
- **No `track_id`.** Tracking belongs to AR.
- **No `timestamp`** unless AR explicitly asks for it; the camera
  transform is matched via `frame_id` only.
- **No per-keypoint confidence.** Wheel-level confidence is sufficient.
- **No `visibility` flag in the response.** Partially occluded wheels
  are dropped at annotation time and never emitted at inference.
- **No 3D coordinates from ML.** Pixel space only.

AR-side responsibilities (do not duplicate here):

- Raycasting `a`, `b` onto the floor plane to anchor the wheel plane.
- RANSAC across K frames to recover the plane.
- Raycasting `c_disc_bottom` onto the recovered plane → disc height.
- Track / scene association across frames.

Full contract narrative: `docs/AR_ML_CONTRACT.md`. Open confirmation
items still on the AR team: `docs/OPEN_QUESTIONS_AR_SPEC.md`.

## Current target output

Production response from `src/infer_image.py` and `src/postprocess_wheels.py`
must match the confirmed shape above exactly. The transitional output
some scripts emit today (`wheel_bbox` xyxy, named keypoints with
`visibility` / `keypoints_confidence`) is being aligned to the confirmed
schema as a code-side follow-up — **target = confirmed schema**, not the
transitional one.

Dataset side: YOLO-pose, one class (`wheel`), three keypoints per wheel.
Internal training labels still carry the literal strings `rim_left` /
`rim_right` / `disc_bottom` (legacy converter) or `a` / `b` /
`c_disc_bottom` (plugin converter) — but **A/B semantics shifted on
2026-05-14**: under the current contract, A and B are screen-space
floor-ray points, **not** rim points. Bundles annotated before that
date (including legacy `manual_sample` and the synthetic
`create_sample_*` fixtures) carry the old "rim edge" geometry and
must be re-annotated before they can be used to train against the
new contract. The literal label strings remain for backward
compatibility with `postprocess_wheels.py` and the converters; only
the *content* of A/B has changed.

## Folder structure

```
VSBL/
  CLAUDE.md                            # this file
  README.md                            # user-facing docs / commands
  requirements.txt
  configs/
    dataset.yaml                       # legacy YOLO-pose config (manual_sample flow)
    pose_dataset.yaml                  # plugin YOLO-pose config (android_plugin flow)
  src/
    create_sample_incoming.py          # legacy synthetic batch generator
    create_sample_keypoint_incoming.py # plugin synthetic batch generator
    convert_incoming_to_yolo.py        # legacy: manual_sample → wheel_dataset
    convert_keypoint_incoming_to_yolo_pose.py  # plugin: android_plugin → wheel_pose_dataset
    check_dataset.py                   # legacy dataset validator
    check_yolo_pose_dataset.py         # plugin dataset validator
    check_keypoint_incoming.py         # incoming-batch validator
    preview_labels.py                  # legacy label preview
    preview_yolo_pose_labels.py        # plugin label preview
    preview_keypoint_annotations.py    # incoming-batch preview
    train_yolo.py                      # YOLO-pose trainer (used by both flows)
    infer_image.py                     # single-image inference → AR JSON
    infer_batch.py                     # batch inference
    postprocess_wheels.py              # YOLO detections → AR payload
    visualize_predictions.py           # render a saved AR JSON onto an image
    export_model.py                    # PT → ONNX / CoreML / TFLite
    eval_keypoints.py                  # keypoint metrics
  data/
    incoming/                          # raw batches (do not commit)
      android_plugin/                  # plugin contract
      manual_sample/                   # legacy contract
    wheel_dataset/                     # legacy converter output
    wheel_pose_dataset/                # plugin converter output
    raw/, processed/, samples/
  docs/
    AR_ML_CONTRACT.md                  # responsibility split + target JSON
    KEYPOINT_SPEC.md                   # A/B/C definitions
    KEYPOINT_DATASET_FORMAT.md         # plugin incoming format
    ANNOTATION_JSON_FORMAT.md          # legacy incoming format
    DATASET_SPEC.md                    # YOLO-pose dataset layout
    OPEN_QUESTIONS_AR_SPEC.md          # AR-team confirmation queue
    PLUGIN_DATA_EXPECTATION.md         # what we expect from the plugin
    REAL_DATA_INGESTION.md             # how real batches are ingested
    CLAUDE_CODE_WORKFLOW.md            # how Claude works in this repo
    agents/, settings.json             # under .claude/
  .claude/
    settings.json
    agents/                            # project-local agent definitions
    skills/                            # project skills (vsbl-ar-contract, …)
  tests/                               # pytest suite
  runs/                                # Ultralytics outputs (do not commit)
  outputs/                             # inference / preview artefacts
  yolo11n.pt, yolo11n-pose.pt          # baseline weights
```

## Commands to run

```bash
# Healthcheck (fast — no training, no GPU needed)
./scripts/healthcheck.sh

# Plugin flow (preferred for new work)
./.venv/bin/python src/create_sample_keypoint_incoming.py --count 50 --overwrite
./.venv/bin/python src/convert_keypoint_incoming_to_yolo_pose.py \
    --source-root data/incoming/android_plugin \
    --dataset-root data/wheel_pose_dataset --overwrite
./.venv/bin/python src/check_yolo_pose_dataset.py \
    --dataset-root data/wheel_pose_dataset
./.venv/bin/python src/preview_yolo_pose_labels.py \
    --dataset-root data/wheel_pose_dataset --split train --count 10

# Legacy flow (still supported, do not remove)
./.venv/bin/python src/create_sample_incoming.py --count 20 --overwrite
./.venv/bin/python src/convert_incoming_to_yolo.py \
    --source-root data/incoming/manual_sample \
    --dataset-root data/wheel_dataset --overwrite
./.venv/bin/python src/check_dataset.py --dataset-root data/wheel_dataset
./.venv/bin/python src/preview_labels.py \
    --dataset-root data/wheel_dataset --split train --count 10

# Tests
./.venv/bin/pytest -q

# Inference smoke
./.venv/bin/python src/infer_image.py --image data/samples/car.jpg --device cpu
```

## Do not break

- **Confirmed AR JSON schema** — `frame_id`, `wheels[]`, `bbox_xyxy`,
  `confidence`, `points.{a, b, c_disc_bottom}`. No additions, no
  renames, no extra fields without AR-team sign-off in
  `docs/OPEN_QUESTIONS_AR_SPEC.md`.
- **Legacy pipeline** (`manual_sample` → `wheel_dataset` via
  `convert_incoming_to_yolo.py`, validated by `check_dataset.py`,
  previewed by `preview_labels.py`). Real annotated data may still
  arrive in that format — keep it working.
- **Plugin pipeline** (`android_plugin` → `wheel_pose_dataset` via
  `convert_keypoint_incoming_to_yolo_pose.py`, validated by
  `check_yolo_pose_dataset.py`, previewed by
  `preview_yolo_pose_labels.py`).
- **Training / inference code** — do not touch `train_yolo.py`,
  `infer_image.py`, `infer_batch.py`, `postprocess_wheels.py`,
  `export_model.py` unless the task explicitly says so.
- **Keypoint count and order**: `[a, b, c_disc_bottom]` (or the legacy
  literal strings `[rim_left, rim_right, disc_bottom]`, whose A/B
  *names* now drift from the confirmed floor-ray semantics — see the
  "Current confirmed AR/ML contract" section above) — 3 keypoints,
  fixed order, `flip_idx: [1, 0, 2]`. Adding a 4th point is an
  AR-team decision (open in `docs/OPEN_QUESTIONS_AR_SPEC.md`).
- **Dependencies**: stdlib + `opencv-python` + `numpy` + `pytest` +
  `ultralytics` are the only deps allowed without sign-off. No torch
  outside ultralytics. No tracking libs (DeepSORT etc.) — tracking is
  not ours.
- **Git-ignored paths** — never commit `data/`, `runs/`, `outputs/*.jpg`,
  or model weights other than the baseline `yolo11n*.pt`.

## Testing policy

- Every new pure function gets at least a smoke test under `tests/`.
  Validators, converters, label-line formatters, split assigners — all
  testable without GPU.
- End-to-end tests must run on synthetic data (`tmp_path` fixture + the
  `create_sample_*` generators). Real data never goes into the test
  suite.
- `pytest -q` must stay green before claiming done. Mark a task done
  only after the full suite passes — never after partial runs.
- Do not mock the file system at the boundary `cv2.imwrite` /
  `cv2.imread` — use `tmp_path` and actual files. The harness is fast
  enough.
- Inference / training paths are not unit-tested (require weights and
  GPU). The smoke check there is "import succeeds and CLI `--help` works".
- A green pytest run on synthetic data is **not** a production quality
  signal. Synthetic cartoons don't generalise; they validate plumbing.

## Current blockers

Tracked in `docs/OPEN_QUESTIONS_AR_SPEC.md` and
`docs/QUESTIONS_FOR_TEAM.md`. Live items as of 2026-05-13:

- **No real labelled batch yet.** Everything in the pipeline runs on
  cartoon synthetic data from `create_sample_*`. Detection quality is
  unproven until the plugin sends real frames.
- **Train/val split for video frames.** The plugin converter does a
  random per-image split — unsafe when consecutive frames belong to one
  scene. Needs `scene_id` (or similar group key) from the plugin, or a
  `--scene-regex` extension.
- **Occlusion handling.** Plugin contract has no `visibility` field.
  Wheels with partially hidden A/B/C are dropped upstream. If real data
  contains such cases, we need a decision: drop ML-side too, or extend
  the format.
- **Inference response alignment.** `src/infer_image.py` still emits
  the transitional schema. Aligning it to the confirmed contract is
  pending — schedule before first AR integration.
- **Production export format.** TFLite is the first target, but we have
  no quantization tolerances from AR yet. See Q10 in
  `docs/QUESTIONS_FOR_TEAM.md`.

## Useful agents for this repo

- `ar-spec-checker` — sanity-check diffs against the confirmed contract
  and open questions before merging.
- `cv-debugger` — when training diverges or detections look wrong.
- `python-reviewer` — generic Python/ML review on new code.
- `test-writer` — when adding logic that lacks coverage.

## Project skills

Loaded automatically by Claude Code from `.claude/skills/`:

- `vsbl-ar-contract` — the AR/ML contract, forbidden fields,
  responsibility split. Invoked whenever a change might touch the JSON.
- `yolo-pose-dataset` — incoming → YOLO-pose conversion, label format,
  validation, preview rules.
- `ml-pipeline-review` — the "done" checklist. Invoked before claiming
  a feature is complete.

Workflow rules (when to use `/goal`, how to define done, what to report)
live in `docs/CLAUDE_CODE_WORKFLOW.md`.
