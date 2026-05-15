# Real-Data Retrain Checklist — Floor-Ray A/B Semantics

Authoritative runbook for the next training run on **real** labelled
data, under the **2026-05-13 floor-ray contract**. Every step below
must be completed in order; skipping any step invalidates the run.

This file is **process-only**. It does not change code, schemas, or
weights. It is the gate document for `wheel_pose_real_floorray_v1`.

## 0. TL;DR — the only allowed entry point

1. Drop a real batch under `data/incoming/android_plugin_real/` or
   `data/incoming/manual_real/`.
2. Visually verify every frame against §2.
3. Pass all four acceptance commands in §3.
4. Mark `ACCEPT_FOR_TRAINING` (write the flag file in §3.6).
5. Run the single training command in §4 — and *only* that command.
6. Run the post-training audit in §5. If geometry FAILs, the model
   is **not** AR-mock-ready, regardless of mAP.
7. The stale models in §6 are forbidden from any AR-bound claim
   regardless of any geometry numbers they happen to produce.

## 1. Required input

Pick exactly one source directory; do not mix:

| Path | When to use |
|---|---|
| `data/incoming/android_plugin_real/` | Real frames captured by the Android collection plugin (plugin contract — see `docs/KEYPOINT_DATASET_FORMAT.md`). |
| `data/incoming/manual_real/` | Manually-curated real photos labelled by hand under the same plugin contract. |

Layout in both cases:

```
<source_root>/
  images/<stem>.{jpg,jpeg,png,bmp,webp}
  annotations/<stem>.json
  metadata/source_info.json
```

Pixel coordinates, top-left origin, native image resolution. No
normalisation, no `[0, 1]` scaling, no nested 3D fields. Stems on
images and annotations must match.

Frame-id rule: `frame_id` inside each annotation JSON must equal the
image stem; the validator (§3.1) fails otherwise.

## 2. Mandatory visual checks (per frame)

Use `preview_keypoint_annotations.py` (§3.2). Look at every preview.
**A run is invalid if any of these is wrong on any frame:**

### 2.1 Bbox covers the full wheel including tire

- Top edge at the upper tire rim.
- Bottom edge at the bottom of the tire contact patch.
- Left/right edges just outside the tire sidewall.
- **NOT** "just the rim" — include the rubber.
- **NOT** clipped by the car body in a 3/4 view; if the wheel is
  partially occluded, drop the whole wheel (§2.4).

### 2.2 A and B are floor-ray points near the wheel's footprint

- A is **on the ground** to the left of where the tire meets the floor.
- B is **on the ground** to the right of where the tire meets the floor.
- Both should sit in the **lower 20%** of the bbox (`rel_y >= 0.80`).
- A and B must be horizontally separated by at least 50% of the bbox
  width (`|B_x - A_x| / bbox_w >= 0.50`) — they anchor the wheel
  plane's base direction.
- **NOT** on the rim. **NOT** on the tire sidewall. **Not** inside the
  hub. Floor points only.

### 2.3 C is the lowest visible point of the metal rim / disc

- On the metal of the rim/disc, at the **6 o'clock position** of the
  rim circle.
- **Above** A and B in image coordinates (`C_y < min(A_y, B_y)`) — C
  is on the rim, A/B are on the floor.
- **NOT** on the tire contact patch.
- **NOT** the hub centre.
- If the bottom of the rim is hidden (rim guard / mud flap / shadow
  ambiguity), drop the whole wheel (§2.4).

### 2.4 No occluded wheels

- If **any** of A / B / C is not visible (blocked by car body,
  bumper, another wheel, scene element, debris), the **whole wheel**
  is omitted from the annotation. Confirmed AR decision 2026-05-13.
- Empty-wheels frames are allowed (`"wheels": []`) — the image stays
  in the batch so frame pairing by stem works for AR.

If you cannot decide whether a point is visible, drop it. Do not guess.

## 3. Acceptance gate (commands must all pass)

The gate runs four programmatic checks plus two manual previews.
Every command below is fail-fast: non-zero exit code aborts the gate.

### 3.1 Incoming-batch validation

```bash
./.venv/bin/python src/check_keypoint_incoming.py \
    --source-root data/incoming/android_plugin_real
```
(replace `android_plugin_real` with `manual_real` if that's the source.)

Acceptance:
- `Errors: 0`
- `Warnings: 0` (warnings indicate keypoint-outside-bbox slop > 5 px
  or out-of-image points — investigate before continuing)

### 3.2 Incoming-batch preview (manual inspection)

```bash
./.venv/bin/python src/preview_keypoint_annotations.py \
    --source-root data/incoming/android_plugin_real \
    --count 20 \
    --output-root outputs/retrain_preview_incoming
```

Acceptance:
- Open every image in `outputs/retrain_preview_incoming/`.
- Confirm §2.1 – §2.4 on each.
- If any frame fails: **do not proceed**. Re-label, re-run §3.1 + §3.2.

### 3.3 Conversion to YOLO-pose with quality gate

```bash
./.venv/bin/python src/convert_keypoint_incoming_to_yolo_pose.py \
    --source-root data/incoming/android_plugin_real \
    --dataset-root data/wheel_pose_dataset_real_floorray \
    --overwrite
```

Acceptance:
- `Quality gate: passed: True`
- `Skipped: 0` ideally; > 0 only acceptable if `skipped_ratio` is
  below `0.0500` (5%) — anything skipped is investigated against §2
  before continuing.

### 3.4 YOLO-pose dataset validation

```bash
./.venv/bin/python src/check_yolo_pose_dataset.py \
    --dataset-root data/wheel_pose_dataset_real_floorray
```

Acceptance:
- `OK — dataset layout looks valid.` printed
- Per-split image / label counts match (no orphan labels or missing
  labels)

### 3.5 YOLO-pose preview (manual inspection)

```bash
./.venv/bin/python src/preview_yolo_pose_labels.py \
    --dataset-root data/wheel_pose_dataset_real_floorray \
    --split train --count 20
./.venv/bin/python src/preview_yolo_pose_labels.py \
    --dataset-root data/wheel_pose_dataset_real_floorray \
    --split val --count 10
```

Acceptance:
- Open every preview. The bbox + A/B/C overlays must still match
  §2.1 – §2.4 after the pixel → normalized → pixel round trip.
- If any preview shows a mis-rendered keypoint that wasn't broken in
  §3.2: the converter is buggy — **stop** and file an issue against
  `src/convert_keypoint_incoming_to_yolo_pose.py`. Do not proceed.

### 3.6 ACCEPT_FOR_TRAINING marker

Only after §3.1 – §3.5 pass, write the gate file:

```bash
date -u +%Y-%m-%dT%H:%M:%SZ > \
    data/wheel_pose_dataset_real_floorray/ACCEPT_FOR_TRAINING
```

The marker is the only authorisation for §4. If it is missing or
older than the dataset's `metadata/split_manifest.json`, re-run the
gate.

## 4. Training command (run only after §3 passed)

**Exactly this command, with no edits, no extra flags:**

```bash
./.venv/bin/python src/train_yolo.py \
    --data configs/pose_dataset.yaml \
    --model yolo11n-pose.pt \
    --epochs 50 \
    --device mps \
    --project runs/pose \
    --name wheel_pose_real_floorray_v1
```

Notes:
- `configs/pose_dataset.yaml` points at `data/wheel_pose_dataset`
  by default. Before invoking, either (a) update its `path:` to
  `data/wheel_pose_dataset_real_floorray`, (b) symlink
  `data/wheel_pose_dataset_real_floorray` → `data/wheel_pose_dataset`,
  or (c) write a sibling `configs/pose_dataset_real_floorray.yaml`
  and pass it instead. The decision is one-line; **do not** start
  training while the config points at the synthetic dataset.
- `--model yolo11n-pose.pt` initialises from the ImageNet/COCO-pretrained
  weights, **not** from `wheel_v3` or `wheel_v4_real`. Those are
  semantically poisoned for A/B (§6).
- `--epochs 50` is the floor. If val loss has not plateaued, extend
  to 100 by re-running with `--epochs 100`; do not reduce below 50.
- `--device mps` for Apple Silicon. On CUDA, pass `--device 0`.
- The run name `wheel_pose_real_floorray_v1` encodes the
  contract version. **Do not rename** — downstream audit + AR
  replays key off this name.

## 5. Post-training audit (also gating)

The model is not AR-mock-ready until this audit passes.

### 5.1 Hold-out inference

Hold out at least 20 real frames **before** §3 (set aside, not
labelled, not in train or val). Run inference:

```bash
./.venv/bin/python src/infer_batch.py \
    --source <held_out_dir_or_video> \
    --model runs/pose/wheel_pose_real_floorray_v1/weights/best.pt \
    --out-dir outputs/wheel_pose_real_floorray_v1_holdout \
    --device cpu
```

`infer_batch.py` now writes the confirmed AR schema as primary
(2026-05-14 schema-drift fix). Do not pass `--emit-legacy` unless
you are also debugging the legacy intermediate.

### 5.2 Geometry audit

```bash
./.venv/bin/python scripts/audit_geometry.py
```

The script reads `outputs/test_infer_single/` and
`outputs/test_batch_ar/`. Point it at the new holdout folder if
needed (it's a one-line edit at the top of the script — see
`SINGLE_DIR` / `BATCH_DIR` constants). Output:
`outputs/real_infer_geometry_audit.json` and `_audit.md`.

Acceptance:

| Check | Threshold | Notes |
|---|---|---|
| A in lower band | `rel_y_a >= 0.80` on every wheel | Floor-ray rule |
| B in lower band | `rel_y_b >= 0.80` on every wheel | Floor-ray rule |
| A/B horizontal separation | `\|B_x - A_x\| / bbox_w >= 0.50` | Plane base width |
| C above A/B in image | `C_y < min(A_y, B_y)` on every wheel | C on rim, A/B on floor |
| Wheels PASS / total | **100%** | Any FAIL → not AR-mock-ready |

Per-wheel WARN status is informational only; **only PASS is
acceptable** to claim AR-mock readiness on a wheel.

### 5.3 No AR-ready claim unless geometry passes

If §5.2 shows even one wheel with verdict `WARN` or `FAIL`:
- The status update to AR team is **"schema-compatible real-image
  smoke"**.
- **NOT** "AR-ready", **NOT** "production", **NOT** "quality
  validated", **NOT** "AR can integrate".
- The exact wording corrections are documented in
  `outputs/real_infer_geometry_audit.md` §"Corrected status update".

Only when 100% of holdout wheels PASS may the wording change to
"AR-mock-ready on a 20-frame holdout under the floor-ray contract"
— and only with the audit JSON cited.

## 6. Stale-model warning — DO NOT USE FOR AR

The following checkpoints are **semantically invalid for A/B** under
the 2026-05-13 floor-ray contract. They emit A/B at the rim edges,
not the floor footprint:

| Path | Reason it's stale | Allowed use |
|---|---|---|
| `runs/pose/wheel_v4_real/weights/best.pt` | Fine-tuned on a seed deliberately labelled under legacy rim semantics (see `scripts/label_manual_real.py`). Geometry audit confirms `rel_y_{a,b} ≈ 0.47` and `C below A/B`. | Schema-compatible smoke only. **NOT** AR. |
| `runs/pose/wheel_v3/weights/best.pt` | Trained 2026-05-13 on cartoon synthetic with legacy rim labels. | Cartoon synthetic only. **NOT** AR. |
| `runs/pose/wheel_v2/weights/best.pt` | Predecessor of `wheel_v3`. | Archive only. **NOT** AR. |
| `runs/pose/wheel_pose_lr/weights/best.pt` | Pre-contract LR-sweep run. | Archive only. **NOT** AR. |
| `runs/pose/wheel_smoke/weights/best.pt` | Earliest pose smoke. | Archive only. **NOT** AR. |
| `runs/detect/wheel_baseline/`, `runs/detect/wheel_synthetic_600_e30/` | Detect-only runs without pose head. | Not relevant to AR (no keypoints). |
| Any `wheel_demo` checkpoint | Demo run on a pre-floor-ray distribution. | **NOT** AR. |
| **Any run trained before 2026-05-14** | Predates the floor-ray contract confirmation. By construction A/B were rim points, not floor-ray points. | Schema-compatible smoke only. **NOT** AR. |

Rule of thumb: if the run name does **not** contain `floorray` *and*
the run was created before 2026-05-14, treat A/B as legacy rim. Do
not ship to AR. Do not claim AR-ready in any status update.

`wheel_pose_real_floorray_v1` (this checklist's output) is the first
run intended to be A/B-correct. It is still subject to §5 before any
AR-bound claim.

## 7. The one next command, when the real batch arrives

The moment the batch lands at
`data/incoming/android_plugin_real/`:

```bash
./.venv/bin/python src/check_keypoint_incoming.py \
    --source-root data/incoming/android_plugin_real
```

That single command starts the gate. Everything downstream follows
this document in order.

## See also

- `docs/AR_ML_CONTRACT.md` — confirmed AR runtime schema.
- `docs/KEYPOINT_SPEC.md` — A/B/C geometric definitions under the
  floor-ray contract.
- `docs/KEYPOINT_DATASET_FORMAT.md` — on-disk plugin contract.
- `docs/MODEL_ARCHITECTURE_PROPOSAL.md` — forward architecture
  proposal (MobileNetV2-skipless).
- `outputs/real_infer_geometry_audit.md` — geometry audit of the
  current (stale) `wheel_v4_real` outputs; reference for what
  "FAIL" looks like.
- `scripts/audit_geometry.py` — the auditor itself.
