from __future__ import annotations

from pathlib import Path

from scripts.build_external_evidence_handoff_bundle import DEFAULT_BUNDLE_ARTIFACTS
from src.release_integrity import DEFAULT_REQUIRED_ARTIFACTS


HARNESS_README = Path("android_litert_harness/README.md")
HARNESS_TEST = Path("android_litert_harness/AndroidLiteRtDeviceValidationTest.kt")


def test_android_litert_harness_documents_device_evidence_flow():
    text = HARNESS_README.read_text(encoding="utf-8")

    assert "com.google.ai.edge.litert:litert:2.1.0" in text
    assert "app/src/androidTest/assets/best_float32.tflite" in text
    assert "app/src/androidTest/assets/EXPECTED_ANDROID_ARTIFACT.json" in text
    assert "expected_android_artifact.sha256" in text
    assert "data/incoming/android_litert_device_report.json" in text
    assert "src/run_production_evidence_intake.py" in text


def test_android_litert_harness_writes_validator_compatible_report():
    text = HARNESS_TEST.read_text(encoding="utf-8")

    assert "class AndroidLiteRtDeviceValidationTest" in text
    assert 'MODEL_ASSET_NAME = "best_float32.tflite"' in text
    assert 'EXPECTED_ARTIFACT_ASSET_NAME = "EXPECTED_ANDROID_ARTIFACT.json"' in text
    assert 'REPORT_FILE_NAME = "android_litert_device_report.json"' in text
    assert 'SOURCE_TYPE_ANDROID_LITERT_DEVICE_VALIDATION =' in text
    assert '"android_litert_device_validation"' in text
    assert '"test_session_id"' in text
    assert '"test_app_version"' in text
    assert '"test_date_utc"' in text
    assert "Android package versionName must be non-blank" in text
    assert "Android LiteRT production evidence must run on a physical device" in text
    assert "val likelyEmulator = isLikelyEmulator()" in text
    assert '"is_emulator", likelyEmulator' in text
    assert "LocalDate.now(ZoneOffset.UTC)" in text
    assert "LiteRT input dtype must be" in text
    assert 'EXPECTED_INPUT_DTYPE = "float32"' in text
    assert "EXPECTED_OUTPUT_SHAPE = intArrayOf(1, 14, 8400)" in text
    assert "LiteRT output range must be non-degenerate" in text
    assert "LiteRT output mean must lie within [min, max]" in text
    assert "kotlin.math.floor(pos)" in text
    assert "kotlin.math.ceil(pos)" in text
    assert "peak memory must be positive for production evidence" in text
    assert '"outputs/production_audit/tflite_export/best_float32.tflite"' in text
    assert '"LiteRT"' in text
    assert "sha256(modelBytes)" in text
    assert "expectedArtifactSha256(context)" in text
    assert '"expected_sha256"' in text
    assert "assertEquals(" in text
    assert "interpreter.run(inputBuffer, outputBuffer)" in text


def test_android_litert_harness_is_in_handoff_and_release_sets():
    assert str(HARNESS_README) in DEFAULT_BUNDLE_ARTIFACTS
    assert str(HARNESS_TEST) in DEFAULT_BUNDLE_ARTIFACTS
    assert str(HARNESS_README) in DEFAULT_REQUIRED_ARTIFACTS
    assert str(HARNESS_TEST) in DEFAULT_REQUIRED_ARTIFACTS
