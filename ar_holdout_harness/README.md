# AR Holdout Annotation Harness

This harness is the runnable AR/annotation-side evidence producer for the
`human_labelled_ar_device_holdout` production blocker.

It writes the exact incoming keypoint dataset layout consumed by
`src/evaluate_ar_holdout.py`:

```text
data/incoming/ar_device_holdout/
  images/<frame_id>.jpg
  annotations/<frame_id>.json
  metadata/provenance.json
```

The AR app or review tool still owns the UI for human labelling and
review. This harness only makes the saved evidence format deterministic.

## Files To Copy

Copy or adapt:

```text
ar_holdout_harness/ArHoldoutAnnotationWriter.kt
```

Recommended Android target path:

```text
app/src/main/java/com/vsbl/wheels/ar/ArHoldoutAnnotationWriter.kt
```

Change only the package line if the Android app uses a different package.

## Integration Shape

Create one writer per holdout batch:

```kotlin
val writer = ArHoldoutAnnotationWriter(context)
writer.reset()
writer.writeProvenance(
    captureDevice = "${Build.MANUFACTURER} ${Build.MODEL}",
    captureAppVersion = BuildConfig.VERSION_NAME,
    captureDateUtc = "2026-05-27",
    annotator = "human_labeler",
    reviewer = "qa_reviewer",
    notes = "production AR holdout"
)
```

For every reviewed frame, save the image bytes and matching annotation:

```kotlin
writer.writeFrame(
    frameId = "frame_0001",
    imageFileName = "frame_0001.jpg",
    imageBytes = jpegBytes,
    wheels = listOf(
        ArHoldoutAnnotationWriter.Wheel(
            bboxXyxy = ArHoldoutAnnotationWriter.BBox(120.0, 280.0, 220.0, 380.0),
            points = ArHoldoutAnnotationWriter.Points(
                a = ArHoldoutAnnotationWriter.Vec2(125.0, 320.0),
                b = ArHoldoutAnnotationWriter.Vec2(215.0, 320.0),
                cDiscBottom = ArHoldoutAnnotationWriter.Vec2(170.0, 370.0)
            )
        )
    )
)
```

Frames with no fully visible wheels are valid and should be written with
`wheels = emptyList()` so frame pairing remains deterministic.

The writer outputs:

```text
<app external files dir>/ar_device_holdout/
```

Copy that directory into this ML repo:

```text
data/incoming/ar_device_holdout/
```

Then validate and evaluate:

```bash
./.venv/bin/python src/evaluate_ar_holdout.py \
  --source-root data/incoming/ar_device_holdout \
  --eval-out outputs/production_audit/ar_device_holdout_eval.json
```

## Acceptance

The production evaluator requires:

- `metadata/provenance.json` with `source_type=android_ar_device_human_labelled`.
- `schema_version=1`.
- every `annotations/<frame_id>.json` file also carries `schema_version=1`.
- `label_type=human_reviewed`.
- non-placeholder `capture_device`.
- `review_status=accepted`.
- non-placeholder `capture_app_version`, real `capture_date_utc` in
  `YYYY-MM-DD` format that is not in the future, `annotator`, and `reviewer`.
- independent review: `annotator` and `reviewer` must be different.
- image filenames are plain filenames, not paths, and use jpg/jpeg/png/bmp/webp extensions.
- at least 50 evaluated AR-device frames and 80 labelled wheels.
- bbox mAP50 `>= 0.85`, OKS `>= 0.80`, false negative rate `<= 0.10`.

See `docs/KEYPOINT_DATASET_FORMAT.md` and
`docs/PRODUCTION_EVIDENCE_CHECKLIST.md`.
