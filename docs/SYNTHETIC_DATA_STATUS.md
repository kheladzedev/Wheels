# Synthetic data status

Date: 2026-05-27.

## What exists now

### UE synthetic batch

There is a valid Unreal-generated incoming batch:

| Item | Value |
|---|---:|
| Incoming root | `data/incoming/ue_synthetic` |
| Images | 51 |
| Annotations | 51 |
| Wheels | 90 |
| Validation errors | 0 |
| Converted YOLO root | `data/wheel_pose_dataset_ue_synthetic` |
| Train / val images | 41 / 10 |
| Dataset config | `configs/pose_dataset_ue_synthetic.yaml` |
| Preview | `outputs/ue_synthetic_preview_check/` |

Validation commands:

```bash
./.venv/bin/python src/check_keypoint_incoming.py \
  --source-root data/incoming/ue_synthetic

./.venv/bin/python src/convert_keypoint_incoming_to_yolo_pose.py \
  --source-root data/incoming/ue_synthetic \
  --dataset-root data/wheel_pose_dataset_ue_synthetic \
  --source-name ue_synthetic_v1 \
  --overwrite

./.venv/bin/python src/check_yolo_pose_dataset.py \
  --dataset-root data/wheel_pose_dataset_ue_synthetic
```

Caveat: this batch came from UE rim keypoints mapped with a 5-to-3
heuristic. It is good enough for pipeline smoke tests and possible
synthetic pretraining, but every annotation should be treated as draft
for the AR-confirmed A/B floor-ray semantics.

### Car-body model pool

The local car-body GLB pool has reached the current 300-model target.
It is a mixed-provenance pool: Sketchfab API downloads are kept as-is,
and Objaverse LVIS is used as a fallback while Sketchfab's live download
endpoint is rate-limited. Objaverse files use an `ov_` filename prefix
and JSON manifests keep `source_platform`, `source_dataset`, and
`source_category` explicit.

| Item | Value |
|---|---:|
| Clean car-body GLB models | 300 |
| Sketchfab-origin GLBs | 234 |
| Objaverse fallback GLBs | 66 |
| Rejected obvious non-car-body / non-vehicle GLBs | 19 |
| Candidate manifests discovered | 2719 |
| Objaverse manifests | 66 |
| Current target | 300 car-body GLBs |
| Local root | `data/sketchfab_cars/` |

The downloader now supports multiple queries, UID dedupe, vehicle-name
filtering, stricter whole-car/body filtering, candidate offset/limit,
deterministic candidate shuffling, small-first sorting by manifest
complexity, size/time limits, temporary download-failure caching,
consecutive-failure auto-stop, rejected-model cleanup, resumable runs,
and continuation from already-saved JSON manifests. It stops instead of
hammering the API when Sketchfab temporarily returns HTTP 405 / 429.

Continue / expand Sketchfab-origin downloads after cooldown:

```bash
FROM_EXISTING_MANIFESTS=1 TARGET_TOTAL=300 MAX_MODELS=200 MAX_MB=45 \
  WORKERS=1 RETRY_SLEEP=120 DOWNLOAD_RETRIES=0 DOWNLOAD_DELAY=2 \
  MAX_DOWNLOAD_SECONDS=30 CAR_BODY_ONLY=1 SORT_CANDIDATES=small-first \
  MAX_FACE_COUNT=250000 SKIP_FAILED_WITHIN_HOURS=24 \
  MAX_CONSECUTIVE_FAILURES=5 \
  ./scripts/prepare_synthetic_data.sh
```

For unattended cooldown-aware completion to 300:

```bash
TARGET_TOTAL=300 RATE_LIMIT_SLEEP=900 \
./scripts/fetch_sketchfab_until_target.sh
```

Fallback when the live Sketchfab endpoint returns HTTP 429:

```bash
./.venv/bin/python src/fetch_objaverse_cars.py \
  --output-dir data/sketchfab_cars \
  --target-total 300 \
  --max-mb 45
```

The fetcher now returns `75` on Sketchfab temporary blocks / HTTP 429 and
`76` when a local candidate segment has too many consecutive download
failures; the loop script uses those codes for backoff and offset
rotation.

Current handoff readiness gate:

```bash
./.venv/bin/python src/project_readiness.py
```

Latest result: the 300-model render/import pool is present, UnrealMCP is
reachable, champion PT/ONNX/eval are present, the full NeuralData
engine-keypoint batch validates, and the Sketchfab/Objaverse geometry
label batch now validates with nonblack RGB PNGs. The champion
pseudo-label path remains too sparse for training and is tracked as a
diagnostic only.

Single-command orchestrator:

```bash
./scripts/finish_project_today.sh
```

Safe default runs readiness/checks only. Full remaining path after
Sketchfab cooldown and UnrealMCP startup:

```bash
RUN_FETCH=1 RUN_OBJAVERSE=1 RUN_UE=1 \
./scripts/finish_project_today.sh
```

If the Unreal Editor/plugin may start later, let the script wait for MCP:

```bash
RUN_FETCH=1 RUN_OBJAVERSE=1 RUN_UE=1 WAIT_FOR_MCP=1 \
MCP_WAIT_TIMEOUT=1800 MCP_WAIT_INTERVAL=10 \
./scripts/finish_project_today.sh
```

Current note: after reaching 234 clean Sketchfab-origin GLBs, the live
Sketchfab download endpoint returned HTTP 429. The Objaverse fallback
then added 66 clean vehicle/body GLBs and brought the local pool to
300/300. Cleanup moved obvious non-car-body assets such as bus stops,
nameplates, hand trucks, standalone wheel/tire/rim assets, sirens,
lightbars, drones, and driver/person assets into `rejected/`.

## Unreal status

The repo has a TCP helper for the UnrealMCP plugin:

```bash
./.venv/bin/python scripts/ue/_send.py exec_code 'print("ue_mcp_ping")'
```

Current local check now reaches UnrealMCP on `127.0.0.1:55557` with
`/Users/codefactory/Downloads/NeuralData1 2/NeuralData.uproject` opened
in Unreal Editor 5.7. The project already referenced `UnrealMCP`; its
plugin folder is a symlink to the checked-out UnrealMCP plugin under
`/Users/codefactory/Desktop/unreal-mcp/MCPGameProject/Plugins/UnrealMCP`.

MCP-run outputs from this pass:

| Item | Value |
|---|---:|
| Imported local GLB tasks | 300 |
| UE import status | `outputs/ue_tasks/import_sketchfab_glbs_status.json` |
| Rendered model groups | 300 |
| Rendered PNGs | 1200 |
| Render status | `outputs/ue_tasks/render_sketchfab_cars_status.json` |
| Engine-keypoint incoming | `data/incoming/ue_neuraldata_keypoint_full` |
| Engine-keypoint frames / wheels | 51 / 90 |
| Engine-keypoint YOLO train / val | 41 / 10 |
| Geometry-label incoming | `data/incoming/ue_sketchfab_geometry` |
| Geometry-label frames / wheels | 172 / 622 |
| Geometry-label YOLO train / val | 138 / 34 |
| Geometry-label dataset config | `configs/pose_dataset_ue_sketchfab_geometry.yaml` |
| Geometry-label status | `outputs/ue_tasks/render_sketchfab_geometry_labels_status.json` |
| Geometry-label RGB content gate | passing: sampled PNGs are nonblack |
| Clean geometry incoming | `data/incoming/ue_sketchfab_geometry_clean` |
| Clean geometry frames / wheels | 132 / 548 |
| Clean geometry YOLO train / val | 106 / 26 |
| Clean geometry dataset config | `configs/pose_dataset_ue_sketchfab_geometry_clean.yaml` |
| Clean geometry QA report | `data/incoming/ue_sketchfab_geometry_clean/metadata/qa_report.json` |
| Mixed real+self+UE+Sketchfab clean YOLO train / val | 338 / 58 |
| Mixed clean dataset config | `configs/pose_dataset_real_self_ue_plus_sketchfab_clean.yaml` |
| Mixed clean checkpoint | `runs/pose/wheel_real_self_ue_plus_sketchfab_clean_ft20/weights/best.pt` |
| Mixed clean real eval | `outputs/eval/wheel_real_self_ue_plus_sketchfab_clean_ft20_on_real.json` |

Important QA result: the 300-model Sketchfab/Objaverse render pool is
present, but pseudo-label yield from the champion model is too low for a
training set (`data/incoming/ue_sketchfab_pseudo_conf005`: 2 images / 2
wheels even at conf 0.05). These render labels are review candidates
only. The stronger path now uses UE geometry labels from wheel/tire/rim
mesh parts: `data/incoming/ue_sketchfab_geometry` validates with 172
frames and 622 wheel annotations, then converts to
`data/wheel_pose_dataset_ue_sketchfab_geometry` with 138 train and 34
val images. The production-facing training candidate is the QA-filtered
subset `data/incoming/ue_sketchfab_geometry_clean`: 132 frames, 548
wheels, converted to `data/wheel_pose_dataset_ue_sketchfab_geometry_clean`
with 106 train and 26 val images. The QA filter drops black/empty images
and oversized projected wheel boxes, and writes exact drop counts to
`metadata/qa_report.json`. The labels are marked draft/review-needed
because they are projected mesh-part geometry proxies rather than
manually confirmed AR labels.

Training result as of this handoff: the clean UE geometry dataset is
valid and usable for experiments, but it does not yet improve production
quality. The UE-only fine-tune
`runs/pose/wheel_ue_sketchfab_geometry_clean_ft20/weights/best.pt`
regressed badly on the real validation split: OKS mean `0.176`, FN
`0.857`, FP `0.571`, bbox mAP50 `0.113`. The mixed fine-tune
`runs/pose/wheel_real_self_ue_plus_sketchfab_clean_ft20/weights/best.pt`
is much closer but still below the champion: OKS mean `0.846`, FN
`0.310`, FP `0.293`, bbox mAP50 `0.682`. The current production
checkpoint remains
`runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt`
with OKS mean `0.887`, FN `0.286`, FP `0.259`, bbox mAP50 `0.697`.

The import side is now scripted:

```bash
./.venv/bin/python scripts/ue/_send.py exec_file scripts/ue/import_sketchfab_glbs.py
```

By default it imports up to 300 local `data/sketchfab_cars/*.glb`
models into `/Game/SketchfabCars`, skipping already-imported destination
folders. Override with `VSBL_SKETCHFAB_GLB_ROOT`, `VSBL_UE_IMPORT_LIMIT`,
or `VSBL_UE_IMPORT_DEST` if needed.

After import, this existing smoke script can spawn a few imported static
meshes and export a capture preview:

```bash
./.venv/bin/python scripts/ue/_send.py exec_file scripts/ue/spawn_one_car.py
```

The production-oriented batch render script is:

```bash
./.venv/bin/python scripts/ue/_send.py exec_file scripts/ue/render_sketchfab_cars.py
```

It renders StaticMeshes under `/Game/SketchfabCars` into
`outputs/ue_sketchfab_renders/images/` using the existing CameraCapture
rig. Override with `VSBL_UE_RENDER_SOURCE`, `VSBL_UE_RENDER_OUT`,
`VSBL_UE_RENDER_LIMIT`, or `VSBL_UE_RENDER_VIEWS`.

The geometry-label exporter for the imported model pool is:

```bash
./.venv/bin/python scripts/ue/_send.py exec_file scripts/ue/render_sketchfab_geometry_labels.py
```

It groups imported meshes by model, detects wheel/tire/rim mesh parts by
asset/path names, renders the group with a temporary SceneCapture2D and
fresh render target, projects part bounds into image space, and writes
keypoint-incoming annotations under
`data/incoming/ue_sketchfab_geometry`. The temporary SceneCapture2D path
is intentional: the existing `CameraCapture_C` had a post-depth capture
configuration that produced black RGB PNGs for this batch.

The post-render pseudo-label bridge is also scripted:

```bash
./.venv/bin/python src/pseudo_label_images_to_incoming.py \
  --images-dir outputs/ue_sketchfab_renders/images \
  --output-root data/incoming/ue_sketchfab_pseudo \
  --model runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt \
  --source-name ue_sketchfab_pseudo \
  --conf 0.25 \
  --device mps \
  --overwrite
```

For the whole import/render/pseudo-label/convert/check flow:

```bash
RUN_MCP_IMPORT=1 \
UE_RENDER_SCRIPT=scripts/ue/render_sketchfab_cars.py \
UE_RENDER_IMAGES_DIR=outputs/ue_sketchfab_renders/images \
./scripts/prepare_ue_sketchfab_pseudo_data.sh
```

Smoke check on the existing `data/incoming/ue_synthetic/images` renders:

| Item | Value |
|---|---:|
| Pseudo incoming root | `data/incoming/ue_synthetic_pseudo_from_champion` |
| Images scanned | 51 |
| Frames written | 5 |
| Wheels written | 6 |
| Incoming validation errors / warnings | 0 / 0 |
| YOLO-pose root | `data/wheel_pose_dataset_ue_synthetic_pseudo_from_champion` |
| Train / val images | 4 / 1 |
| Dataset config | `configs/pose_dataset_ue_synthetic_pseudo_from_champion.yaml` |

This is a bridge/QA path, not engine ground truth. The output annotations
are explicitly marked as pseudo labels requiring review.

## Training runs

### Synthetic-only fine-tune

Trained from the current self-labeled real checkpoint into the UE
synthetic-only dataset:

```bash
./.venv/bin/python src/train_yolo.py \
  --data configs/pose_dataset_ue_synthetic.yaml \
  --model runs/pose/runs/pose/wheel_real_v1_self_s/weights/best.pt \
  --epochs 120 --imgsz 640 --batch 8 --device mps \
  --project runs/pose --name wheel_ue_synthetic_from_self_s
```

Artifacts:

| Item | Path |
|---|---|
| Best checkpoint | `runs/pose/wheel_ue_synthetic_from_self_s/weights/best.pt` |
| Last checkpoint | `runs/pose/wheel_ue_synthetic_from_self_s/weights/last.pt` |
| ONNX export | `runs/pose/wheel_ue_synthetic_from_self_s/weights/best.onnx` |
| Eval JSON, conf 0.25 | `outputs/eval/wheel_ue_synthetic_from_self_s.json` |
| Eval JSON, conf 0.05 | `outputs/eval/wheel_ue_synthetic_from_self_s_conf005.json` |

Best checkpoint final Ultralytics validation on the UE synthetic val
split:

| Metric | Value |
|---|---:|
| Box mAP50 / mAP50-95 | 0.123 / 0.0645 |
| Pose mAP50 / mAP50-95 | 0.147 / 0.106 |

Independent eval at `conf=0.25` found only 1 matched wheel out of 20
GT wheels, with OKS mean 0.152 and FN rate 0.95. Lowering to
`conf=0.05` matched 3 / 20 GT wheels, with OKS mean 0.188 and FN rate
0.85. This run proves the training/export path, but it is not a useful
production candidate by itself.

### Mixed real+self plus UE synthetic fine-tune

Built a mixed YOLO-pose dataset from real self-labeled data plus the UE
synthetic batch:

| Item | Value |
|---|---:|
| Dataset root | `data/wheel_pose_dataset_real_v1_self_plus_ue_synthetic` |
| Train / val images | 232 / 58 |
| Total labeled wheels | 411 |
| Dataset config | `configs/pose_dataset_real_v1_self_plus_ue_synthetic.yaml` |

Training command:

```bash
./.venv/bin/python src/train_yolo.py \
  --data configs/pose_dataset_real_v1_self_plus_ue_synthetic.yaml \
  --model runs/pose/runs/pose/wheel_real_v1_self_s/weights/best.pt \
  --epochs 100 --imgsz 640 --batch 8 --device mps \
  --project runs/pose --name wheel_real_v1_self_plus_ue_synthetic_s
```

Artifacts:

| Item | Path |
|---|---|
| Best checkpoint | `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt` |
| Last checkpoint | `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/last.pt` |
| ONNX export | `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.onnx` |
| Eval JSON | `outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s.json` |
| Training plots/logs | `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/` |

Best checkpoint final Ultralytics validation on the mixed val split:

| Metric | Value |
|---|---:|
| Box precision / recall | 0.865 / 0.702 |
| Box mAP50 / mAP50-95 | 0.744 / 0.640 |
| Pose precision / recall | 0.865 / 0.702 |
| Pose mAP50 / mAP50-95 | 0.775 / 0.747 |

Independent `eval_keypoints.py` at `conf=0.25`:

| Metric | Value |
|---|---:|
| GT / predicted / matched wheels | 84 / 81 / 60 |
| False negatives / false positives | 24 / 21 |
| FN rate / FP rate | 0.286 / 0.259 |
| OKS mean | 0.887 |
| Median error point_a | 7.54 px |
| Median error point_b | 7.73 px |
| Median error point_c_disc_bottom | 7.71 px |
| BBox mAP50 / mAP50-95 in eval script | 0.697 / 0.621 |

ONNX export sanity check passed on a real val sample:

| Check | Value |
|---|---:|
| PT detections / ONNX detections | 1 / 1 |
| Max bbox drift | 0.851 px |
| Max keypoint drift | 0.260 px |
| Max confidence drift | 0.000 |

Conclusion: use the mixed checkpoint as the practical synthetic-assisted
candidate. The synthetic-only run should stay as a pipeline smoke test
until Unreal produces direct floor-ray A/B/C annotations from imported
models.

## Recommended next step

Use the existing `ue_synthetic` batch only as auxiliary data, not as the
production validation set. The highest-value synthetic follow-up is:

1. Run UnrealMCP.
2. Import `data/sketchfab_cars/*.glb` into UE.
3. Render orbit captures with direct projected `a`, `b`,
   `c_disc_bottom` points, not the old rim-to-floor heuristic.
4. Validate with `check_keypoint_incoming.py`.
5. Convert to YOLO-pose and train a synthetic-pretrain + real-finetune
   experiment against the existing champion.
