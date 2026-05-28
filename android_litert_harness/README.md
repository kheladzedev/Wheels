# Android LiteRT Validation Harness

This harness is the runnable Android-side evidence producer for the
`android_litert_device_validation` production blocker.

It is intentionally small: copy the production TFLite model into the
target app test assets, add one instrumentation test, run it on a real
device, and place the generated JSON report at the ML repo intake path.

## Dependency

Use the current LiteRT Maven package in the Android app or test module:

```kotlin
dependencies {
    implementation("com.google.ai.edge.litert:litert:2.1.0")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
}
```

The harness uses the LiteRT `Interpreter` API because it exposes tensor
shape and raw output checks needed by the production validator.

## Files To Copy

Copy the exact production artifact into Android test assets:

```text
outputs/production_audit/tflite_export/best_float32.tflite
```

Copy the expected-artifact manifest from the ML return template as well:

```text
outputs/production_audit/external_evidence_return_template.zip:EXPECTED_ANDROID_ARTIFACT.json
```

Recommended target paths inside the Android project:

```text
app/src/androidTest/assets/best_float32.tflite
app/src/androidTest/assets/EXPECTED_ANDROID_ARTIFACT.json
```

Copy or adapt:

```text
android_litert_harness/AndroidLiteRtDeviceValidationTest.kt
```

Recommended target path:

```text
app/src/androidTest/java/com/vsbl/wheels/ml/AndroidLiteRtDeviceValidationTest.kt
```

Change only the package line if the Android app uses a different test
package. Do not rename the model asset unless you also update
`MODEL_ASSET_NAME` in the test. Do not edit
`EXPECTED_ANDROID_ARTIFACT.json`; the instrumented test reads
`expected_android_artifact.sha256` from it and fails before measurement
if the Android asset is not the current ML-provided TFLite artifact.

## Run

Run on a connected physical Android device:

```bash
./gradlew connectedAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=com.vsbl.wheels.ml.AndroidLiteRtDeviceValidationTest
```

The test prints and writes:

```text
<app external files dir>/android_litert_device_report.json
```

Pull or copy that file into this ML repo:

```text
data/incoming/android_litert_device_report.json
```

Then validate and refresh production evidence:

```bash
./.venv/bin/python src/run_production_evidence_intake.py --dry-run
./.venv/bin/python src/run_production_evidence_intake.py
```

## Acceptance

The JSON must pass:

```bash
./.venv/bin/python src/validate_android_litert_report.py \
  --source data/incoming/android_litert_device_report.json \
  --out outputs/production_audit/android_litert_device_eval.json
```

Default gate thresholds are documented in
`docs/ANDROID_LITERT_DEVICE_REPORT.md`.

The production validator requires the harness-generated
`schema_version=1`, `source_type=android_litert_device_validation`, a non-placeholder
`test_session_id`, non-placeholder `test_app_version`, real
`test_date_utc` in `YYYY-MM-DD` format that is not in the future, physical-device identity fields
with `device.is_emulator=false`, finite non-degenerate output statistics
with `min < max` and `min <= mean <= max`, expected input
tensor metadata (`[1, 640, 640, 3]`, `float32`,
`zero_float32_smoke`), and a SHA256 match against the exact packaged
TFLite file.
The Android test also checks that match on-device before writing the
report. The harness also fails before writing when it detects an
emulator-like environment, a blank Android `versionName`, non-positive
peak memory, or invalid output statistics, because those cases would
produce evidence rejected by the production validator.
