# Session log 2026-05-23

## What worked

### Built reusable UE automation stack
- `scripts/run_unreal_capture.sh` — launches UE 5.7 in `-game` mode against a given `umap`, polls `Images/` and kills UE when target reached or no growth for 60 s.
- `scripts/run_unreal_capture_loop.sh` — wraps the above so the CameraCaptureWheels spline can be replayed N times; each replay yields ~100 frames before the spline ends. Uses SIGTERM grace then SIGKILL, with 10 s cooldown between iterations so UE can release shared memory and lock files.
- `scripts/unreal_validate_capture_map.py` — headless commandlet (`UnrealEditor-Cmd -run=pythonscript`) that enumerates `wheels` actors, components, and floor traces. Confirmed working against `standartWheelsRoom` (6 wheels, 1 CameraCaptureWheels) and `standartWheelsRoom_capture_clean_v2` (3 wheels).
- `scripts/unreal_dump_map_contents.py` — read-only Python that dumps all actors, vehicles, and the CameraCaptureWheels Blueprint property values. This is how we discovered the `ToRotate=source` lock and the lack of `RotateObjects` entries on `standartWheelsRoom`.
- `scripts/unreal_diversify_capture_map.py` — writes `ToRotate=None` and fills `RotateObjects` with all in-scene vehicle StaticMeshActors, then saves the map. Used in-place on `standartWheelsRoom` with a `.umap.bak` backup beside it.
- `scripts/retrain_on_map03_capture.sh` — atomic accept → combine → train MN2 (mps, e20, batch 8) → ONNX export with parity → TFLite export with parity → smoke on `0003/Images`.

### Confirmed the export contract still holds
- New `NeuralData1 2/` exports parse cleanly through `scripts/accept_neuraldata1_capture.py`. Quality gate passes (zero `skipped`/`warnings` ratio) with 100% of frames containing 2-3 valid wheels each.
- Geometry verified visually: `Left` / `Right` sit at the footprint, `Center` at the disc bottom, bbox covers tire + rim. Same mapping as `0003`.
- Acceptance reports run with `--human-preview-accepted` flag for the 100-frame, 277-frame, and 1000-frame captures all produced `ACCEPT_FOR_TRAINING`.

### Validated map 01/02/03 failure mode
- `unreal_validate_capture_map.py` against `01`, `02`, `03` reports zero `wheel_actors` and zero `CameraCaptureWheels` actors. Same when launched in `-game` mode — `Images/` count stays at zero indefinitely.
- These maps rely on World Partition / streaming sublevels that are not auto-loaded by `EditorLoadingAndSavingUtils.load_map()` or by a bare `-game` launch. They need explicit data layer activation before they become capture-capable.

## What did not work

### Three retrain attempts all regressed against the prev v3 baseline

`prev v3` = `mn2_combined_0003_neuraldata1_review_patch_v3_e20_provisional`, the previously shipped provisional handoff. On the same 8 frames of `0003/Images`, with conf=0.30, NMS iou=0.5:

| Run | Source added on top of `0003` | Train loss | Val loss | Confirmed wheels on 0003 smoke (8 frames) |
|---|---|---:|---:|---:|
| prev v3 | `neural_map01_folderfix_smoke` | n/a (shipped) | n/a | 6 |
| v3_277 | 277 frames `clean_v2` (3 wheels, ToRotate locked) | 0.218 | 0.640 | **2** |
| diversified_v1 | 1000 frames `standartWheelsRoom` diversified (ToRotate=None, RotateObjects=[demolition_derby_car]) | 0.256 | 0.567 | **2** |

Val loss on the combined val split improved, but the combined val split is *also dominated by the new synthetic*, so it does not measure generalization to `0003`-style city scenes. The smoke run on `0003` is the only meaningful generalization probe we have.

### Root cause of regression: dataset bias toward a single scene

`standartWheelsRoom` contains exactly one vehicle (`demolition_derby_car`) inside one styled room with a fixed lighting rig. Adding hundreds-to-thousands of frames of this single scene makes the gradient updates overwhelmingly favour that single car under that single lighting, so the network stops generalizing to the multi-vehicle, multi-lighting `0003` city scenes that match the AR target domain.

Even with the diversification (random camera rotation + per-frame random object rotation), the underlying content of every frame is "the one car in the one room". Per-frame randomness changes the camera angle but not the visual content distribution.

### Map 03 inaccessible in `-game` mode

Map 03 is the Tokyo-style city scene with many vehicles that *would* match the `0003` distribution, but it relies on UE 5 World Partition streaming. In a bare `-game` launch the streaming sublevels do not activate, so the capture actor sees an empty scene and writes zero frames. We confirmed this with a 30-minute UE run that produced zero new frames.

## Production model decision

**Production handoff stays on `prev v3`.** It already shipped to AR.

```
outputs/model_packages/mn2_combined_0003_neuraldata1_review_patch_v3_e20_provisional/
```

We did not ship the worse v3_277 or diversified_v1 models. They remain in `runs/pose_mn2/` only as evidence of the regression for the next session.

## What Igor (Unreal side) needs to do for the next iteration

Already documented in `docs/IGOR_FEEDBACK_MESSAGE_RU.md`. The new ask after tonight:

1. Either re-save `01` / `02` / `03` so their data layers load on a bare `-game` launch, or hand us a Python snippet that activates the required streaming sublevels before `unreal.EditorLoadingAndSavingUtils.load_map()`.
2. Stand up a `standartWheelsRoom_multivehicle` map with 5–10 different `StaticMeshActor` vehicles plus matching `wheels` markers, so a single capture run already has variety, not the same-car-from-many-angles bias we hit tonight.
3. Native `BBox` / `WheelBBox` per visible wheel in raw `keyPoint/*.txt` (covers full tire + rim), instead of the current point-derived bbox.

Until at least 1 and 2 land, every UE-side retrain we attempt will reproduce tonight's regression.

## Engineering hygiene notes for the next session

- `pkill -f` uses ERE not BRE: `pkill -f "A|B"` works, `pkill -f "A\|B"` silently misses. Memory entry written: `memory/feedback_pkill_regex.md`.
- `UnrealEditor-Cmd -script=path/to/script.py` resolves `script=` relative to the binary's CWD (`/Users/Shared/Epic Games/UE_5.7/Engine/Binaries/Mac/`), not the project root. Always pass the absolute path.
- Standalone `-game` UE on macOS does not respect `Saved/Logs/<Project>.log` — the wrapper output goes to whatever `>` redirection the launcher used, and no crash dump appears in `Saved/Crashes/` for clean shutdowns; this can look like a crash but is just our own SIGTERM.
- The CameraCaptureWheels spline is finite at ~100 frames per replay. Multi-replay loops are the documented way to accumulate; this matches the pattern Igor used to produce `0003` (1713 frames over many replays).
