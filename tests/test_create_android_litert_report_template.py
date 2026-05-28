from __future__ import annotations

from scripts.create_android_litert_report_template import build_template


def test_android_litert_template_uses_artifact_sha(tmp_path):
    artifact = tmp_path / "model.tflite"
    artifact.write_bytes(b"model")

    template = build_template(artifact)

    assert template["source_type"] == "android_litert_device_validation"
    assert template["test_session_id"] == "FILL_ME"
    assert template["test_app_version"] == "FILL_ME"
    assert template["test_date_utc"] == "FILL_ME_YYYY-MM-DD"
    assert template["runtime"] == "LiteRT"
    assert template["device"]["is_emulator"] is False
    assert template["artifact"]["path"] == str(artifact)
    assert template["artifact"]["sha256"] != "FILL_ME"
    assert template["input"] == {
        "shape": [1, 640, 640, 3],
        "dtype": "float32",
        "profile": "zero_float32_smoke",
    }
    assert template["output"]["shape"] == [1, 14, 8400]
    assert template["latency_ms"]["runs"] == 30


def test_android_litert_template_is_not_gate_input(tmp_path):
    template = build_template(tmp_path / "missing.tflite")

    assert template["artifact"]["sha256"] == "FILL_ME"
    assert "android_litert_device_report.json" in template["notes"]
    assert "template" not in "data/incoming/android_litert_device_report.json"
