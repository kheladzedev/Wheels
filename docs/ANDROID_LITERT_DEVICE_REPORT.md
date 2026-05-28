# Android LiteRT Device Report

This is the required evidence for the production blocker
`android_litert_device_validation`.

The desktop package is already certified by
`outputs/production_audit/tflite_certification.json`. Production still
needs the exact `.tflite` artifact measured inside the target Android app
or device test.

## Input File

Save the Android measurement as:

```text
data/incoming/android_litert_device_report.json
```

Generate a fill-in template:

```bash
./.venv/bin/python scripts/create_android_litert_report_template.py
```

Template output:

```text
outputs/production_audit/android_litert_device_report.template.json
```

Android-side evidence producer:

```text
android_litert_harness/README.md
android_litert_harness/AndroidLiteRtDeviceValidationTest.kt
```

The harness runs the exact TFLite artifact on a physical Android device,
computes the artifact SHA256, measures latency/memory/output statistics,
and writes the JSON shape below.
`test_date_utc` must be a real UTC date in `YYYY-MM-DD` format and must
not be in the future.

## Required Shape

```json
{
  "schema_version": 1,
  "source_type": "android_litert_device_validation",
  "test_session_id": "android_litert_2026_05_27_001",
  "test_app_version": "1.2.3",
  "test_date_utc": "2026-05-27",
  "device": {
    "model": "Pixel 8 Pro",
    "manufacturer": "Google",
    "android_version": "15",
    "soc": "Tensor G3",
    "is_emulator": false
  },
  "runtime": "LiteRT",
  "artifact": {
    "path": "outputs/production_audit/tflite_export/best_float32.tflite",
    "sha256": "sha256 of the exact file installed in the app",
    "format": "tflite_float32"
  },
  "input": {
    "shape": [1, 640, 640, 3],
    "dtype": "float32",
    "profile": "zero_float32_smoke"
  },
  "latency_ms": {
    "runs": 30,
    "mean": 42.0,
    "p50": 40.0,
    "p95": 70.0
  },
  "memory_mb": {
    "peak": 200.0
  },
  "output": {
    "shape": [1, 14, 8400],
    "finite": true,
    "min": -0.05,
    "max": 1.04,
    "mean": 0.41
  }
}
```

## Validation

```bash
./.venv/bin/python src/validate_android_litert_report.py \
  --source data/incoming/android_litert_device_report.json \
  --out outputs/production_audit/android_litert_device_eval.json
```

Default production thresholds:

| Check | Required |
|---|---:|
| Runs | `>= 20` |
| Mean latency | `<= 120 ms` |
| P95 latency | `<= 180 ms` |
| Peak memory | present, `> 0 MB`, and `<= 512 MB` |
| Output shape | `[1, 14, 8400]` |
| Output finite | `true` |
| Output stats | finite numeric `min`, `max`, and `mean` with `min < max` and `min <= mean <= max` |
| Source type | `android_litert_device_validation` |
| Test session id | present, not `FILL_ME`/`TODO`/`TBD`/`unknown` |
| Test app version | present, not `FILL_ME`/`TODO`/`TBD`/`unknown` |
| Test date UTC | real `YYYY-MM-DD` date; impossible dates such as `2026-99-99` are rejected |
| Device model/manufacturer/Android version/SoC | present, not `FILL_ME`/`TODO`/`TBD`/`unknown` |
| Physical device | `device.is_emulator` must be `false` |
| Artifact format | `tflite_float32` |
| Artifact SHA256 | exact SHA256 of `outputs/production_audit/tflite_export/best_float32.tflite` |
| Input shape / dtype / profile | `[1, 640, 640, 3]`, `float32`, `zero_float32_smoke` |

The validator rejects placeholder values, wrong source types, impossible
test dates, missing peak-memory measurements, incomplete or degenerate
output statistics, latency/memory threshold violations, and hash mismatches.
This keeps the Android report tied to a concrete physical-device test
session, app build, UTC test date, and the exact TFLite artifact shipped
in the ML package, rather than any locally rebuilt or stale model file.

The production gate consumes
`outputs/production_audit/android_litert_device_eval.json`.
