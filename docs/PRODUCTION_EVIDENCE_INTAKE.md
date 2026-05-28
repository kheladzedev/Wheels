# Production Evidence Intake

After Android/AR teams provide real device evidence, run the intake
command first:

If the team returns a zip or folder, import it first:

```bash
./.venv/bin/python src/import_external_evidence_drop.py path/to/evidence_drop.zip --dry-run
./.venv/bin/python src/import_external_evidence_drop.py path/to/evidence_drop.zip --overwrite
```

Or use the one-command intake path, which imports the drop and then runs
the validators/gates:

```bash
./.venv/bin/python src/run_production_evidence_intake.py --evidence-drop path/to/evidence_drop.zip --evidence-drop-overwrite
```

For a full handoff-to-final-audit run in one command, add `--finalize`.
This runs the validators/gates first and, only if they are green, runs
the final production audit suite with pytest:

```bash
./.venv/bin/python src/run_production_evidence_intake.py --evidence-drop path/to/evidence_drop.zip --evidence-drop-overwrite --finalize
```

`--finalize` is intentionally restricted to the canonical production
paths under `data/incoming/...` and `outputs/production_audit/...`,
because the final audit suite rereads those paths. Custom source/eval
paths are supported for validation/debug runs, but they must be copied
into the canonical layout before final production certification.

The importer copies only the expected files into `data/incoming/...` and
rejects unsafe archive paths, directory or zip symlinks, oversized zip
entries, oversized total zip payloads, empty required files, duplicate
destinations, leftover placeholder files, bad AR holdout file extensions,
malformed AR holdout annotation JSON contracts, incomplete Android LiteRT
runtime reports, Android reports with missing or wrong input tensor
metadata (`[1, 640, 640, 3]`, `float32`, `zero_float32_smoke`), Android
reports from emulator-like devices (`device.is_emulator != false`), Android
reports whose artifact hash does not match the current
`outputs/production_audit/tflite_export/best_float32.tflite`,
incomplete AR holdout provenance, missing or unsupported `schema_version`
values, too-small production holdout/replay drops, and malformed AR replay
JSONL observations. The
minimum accepted production drop has 50 matched AR holdout images and
annotation JSON files, 80 total annotated wheel instances, 30 AR replay
observations, `schema_version=1` on the Android report, every AR holdout
annotation JSON file, AR holdout provenance, and every AR replay row, non-future UTC evidence dates,
per-observation camera pose evidence
(`camera_transform` with finite numeric `R` 3x3 and `t` vec3 fields, or `camera_pose_ref`),
RANSAC labels/residuals, per-observation recovered plane evidence,
`c_plane_hit`, `c_height_value`, and at least one final disc bottom
position.
Its report includes per-file SHA-256 hashes plus an
`evidence_manifest_sha256` for custody of the accepted Android/AR
evidence files.

Keep the generated import report:

```text
outputs/production_audit/external_evidence_drop_import.json
```

The production evidence audit requires this non-dry-run report once the
Android, AR holdout, and AR replay validation reports are otherwise
green. It verifies that every canonical input under `data/incoming/...`
appears in `copied_artifacts` and that its current SHA-256 still matches
the accepted external drop.

The validation reports are also bound to the exact current evidence
content:

- Android LiteRT validation writes `source_sha256` for
  `data/incoming/android_litert_device_report.json`.
- AR replay validation writes `source_sha256` for
  `data/incoming/ar_3d_replay/ar_replay.jsonl`.
- AR holdout evaluation writes `source_manifest_sha256` over
  `images/`, `annotations/`, and `metadata/provenance.json` or
  `metadata/source_info.json`.

The production evidence audit recomputes these hashes. A report with
`ok=true` is rejected if it was generated for a different file, an older
file version, or a changed AR holdout directory.

To give Android/AR an empty return zip shape:

```bash
./.venv/bin/python scripts/create_external_evidence_return_template.py
```

Template output:

```text
outputs/production_audit/external_evidence_return_template.zip
outputs/production_audit/external_evidence_return_template_manifest.json
```

The return zip includes `EXPECTED_ANDROID_ARTIFACT.json` and repeats the
same path/SHA in `README_RETURN_EVIDENCE.md`. Android LiteRT evidence
must report that exact artifact SHA for the current
`outputs/production_audit/tflite_export/best_float32.tflite`; otherwise
the importer rejects the drop before copying it into `data/incoming`.

```bash
./.venv/bin/python src/run_production_evidence_intake.py
```

Preflight without changing gate/report artifacts:

```bash
./.venv/bin/python src/run_production_evidence_intake.py --dry-run
./.venv/bin/python src/run_production_evidence_intake.py --dry-run --evidence-drop path/to/evidence_drop.zip
```

Default inputs:

- `data/incoming/android_litert_device_report.json`
- `data/incoming/ar_device_holdout/`
- `data/incoming/ar_3d_replay/ar_replay.jsonl`

Useful templates:

- `outputs/production_audit/android_litert_device_report.template.json`
- `outputs/production_audit/ar_device_holdout_provenance.template.json`
- `outputs/production_audit/ar_3d_replay.template.jsonl`

Android-side evidence producer:

- `android_litert_harness/README.md`
- `android_litert_harness/AndroidLiteRtDeviceValidationTest.kt`

AR replay evidence producer:

- `ar_replay_harness/README.md`
- `ar_replay_harness/ArReplayLogger.kt`

AR holdout evidence producer:

- `ar_holdout_harness/README.md`
- `ar_holdout_harness/ArHoldoutAnnotationWriter.kt`

Handoff bundle for Android/AR:

- `outputs/production_audit/external_evidence_handoff_bundle.zip`
- `outputs/production_audit/external_evidence_handoff_bundle_manifest.json`
- `outputs/production_audit/external_evidence_handoff_bundle_verification.json`
- `outputs/production_audit/external_evidence_return_template.zip`

The runner executes:

- Optional evidence drop import via `src/import_external_evidence_drop.py`
  when `--evidence-drop` is passed.
- Android LiteRT report validation.
- Human-reviewed AR-device holdout conversion/evaluation.
- AR replay/RANSAC validation.
- Consolidated production evidence audit.
- Integration and production gates.
- Senior ML audit, requirements traceability, Russian executive report,
  objective completion audit, and release integrity refresh.

Required output:

```text
outputs/production_audit/production_evidence_intake_status.json
```

The full audit suite writes its own non-destructive dry-run snapshot to:

```text
outputs/production_audit/production_evidence_intake_preflight_status.json
```

This keeps routine suite preflight checks from overwriting the real
intake status after Android/AR evidence has been accepted.

The command exits `0` only when the production evidence audit,
production gate, and objective completion audit are all green.

After a green intake, run the final production audit suite:

```bash
./.venv/bin/python src/production_audit_suite.py --with-pytest
```

Without `--finalize`, the intake status includes
`finalization_required=true` and this finalization command. With
`--finalize`, the status records the final suite result and sets
`finalization_required=false` only when the suite passes. The final suite
refreshes the release package, Russian executive report, handoff report,
report consistency audit, and full pytest evidence from the accepted
external inputs. The runner writes the real intake status before the
final suite so release integrity can include it in production-ready
packages, then rewrites the final status once the suite completes and
refreshes the production report, handoff report, release manifest, and
report consistency audit against that final status. If the
post-finalization report refresh fails, the intake status records
`post_finalization_refresh` and the command exits non-zero.

`--dry-run` exits `0` only when all required input paths are present,
non-empty, and free of leftover placeholder files; it does not validate
production gate outputs. It still rejects malformed Android report JSON,
unsupported Android `schema_version`, wrong Android production
`source_type`, incomplete Android LiteRT device metadata, emulator
reports, wrong TFLite artifact hash, wrong input/output tensor
contracts, invalid latency/memory metrics, malformed AR replay JSONL
rows, unsupported AR replay row `schema_version`, and wrong AR replay
production `source_type`.
Each AR replay row must also include production capture
metadata, a non-negative `capture_index`, camera pose evidence, finite
screen points and floor raycast hits, RANSAC `inlier`/`residual`, a valid
recovered plane, `c_plane_hit`, and non-negative `c_height_value`. Replay
rows must be non-decreasing by `capture_index` inside each session; repeated
rows for the same session/frame/capture index are allowed only when every row
has a unique `wheel_index` or `wheel_track_id`.
For the AR holdout it checks `images/`, `annotations/`,
`metadata/provenance.json`, production human-reviewed provenance with an
accepted independent review, and matching image/annotation stems, not just
the root directory. Image files must use `.jpg`, `.jpeg`, `.png`, `.bmp`,
or `.webp`; annotation files must use `.json`. Each annotation JSON must
be an object with `frame_id` equal to the annotation/image stem, `image`
equal to an actual filename in `images/`, and `wheels` as an array. Each
wheel must use the production annotation schema: finite
`bbox_xyxy: [x1, y1, x2, y2]` with positive area and
`points.{a,b,c_disc_bottom}` as finite `[x, y]` pixel points that lie
inside that bbox.
