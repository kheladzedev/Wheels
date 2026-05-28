from __future__ import annotations

import argparse
import hashlib
import math

from src.validate_android_litert_report import build_report


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        min_runs=20,
        max_mean_latency_ms=120.0,
        max_p95_latency_ms=180.0,
        max_peak_memory_mb=512.0,
    )


VALID_INPUT = {"shape": [1, 640, 640, 3], "dtype": "float32", "profile": "zero_float32_smoke"}


def test_android_litert_report_passes_valid_device_payload(tmp_path):
    source = tmp_path / "android.json"
    source.write_text("{}", encoding="utf-8")
    payload = {
        "schema_version": 1,
        "source_type": "android_litert_device_validation",
        "test_session_id": "android-litert-001",
        "test_app_version": "1.2.3",
        "test_date_utc": "2026-05-27",
        "device": {
            "model": "Pixel test",
            "manufacturer": "Google",
            "android_version": "15",
            "soc": "Tensor test",
            "is_emulator": False,
        },
        "runtime": "LiteRT",
        "artifact": {"sha256": "abc", "format": "tflite_float32"},
        "input": VALID_INPUT,
        "latency_ms": {"runs": 30, "mean": 40.0, "p95": 70.0},
        "memory_mb": {"peak": 200.0},
        "output": {"shape": [1, 14, 8400], "finite": True, "min": 0.0, "max": 1.0, "mean": 0.5},
    }

    report = build_report(source, payload, _args())

    assert report["ok"] is True
    assert report["failures"] == []
    assert report["source_schema_version"] == 1
    assert report["source_sha256"] == _sha(source)
    assert report["output"] == {
        "shape": [1, 14, 8400],
        "finite": True,
        "min": 0.0,
        "max": 1.0,
        "mean": 0.5,
    }


def test_android_litert_report_rejects_bad_shape_and_latency(tmp_path):
    payload = {
        "source_type": "android_litert_device_validation",
        "test_session_id": "android-litert-001",
        "device": {
            "model": "Pixel test",
            "manufacturer": "Google",
            "android_version": "15",
            "soc": "Tensor test",
            "is_emulator": False,
        },
        "runtime": "LiteRT",
        "artifact": {"sha256": "abc", "format": "tflite_float32"},
        "input": VALID_INPUT,
        "latency_ms": {"runs": 1, "mean": 400.0, "p95": 500.0},
        "output": {"shape": [1, 10, 8400], "finite": False, "min": 0.0, "max": 1.0, "mean": 0.5},
    }

    report = build_report(tmp_path / "android.json", payload, _args())

    assert report["ok"] is False
    assert any("unexpected_output_shape" in failure for failure in report["failures"])
    assert any("too_few_runs" in failure for failure in report["failures"])
    assert any("mean_latency" in failure for failure in report["failures"])


def test_android_litert_report_allows_p95_latency_below_mean_for_skewed_samples(tmp_path):
    payload = {
        "schema_version": 1,
        "source_type": "android_litert_device_validation",
        "test_session_id": "android-litert-001",
        "test_app_version": "1.2.3",
        "test_date_utc": "2026-05-27",
        "device": {
            "model": "Pixel test",
            "manufacturer": "Google",
            "android_version": "15",
            "soc": "Tensor test",
            "is_emulator": False,
        },
        "runtime": "LiteRT",
        "artifact": {"sha256": "abc", "format": "tflite_float32"},
        "input": VALID_INPUT,
        "latency_ms": {"runs": 30, "mean": 70.0, "p95": 40.0},
        "memory_mb": {"peak": 200.0},
        "output": {"shape": [1, 14, 8400], "finite": True, "min": 0.0, "max": 1.0, "mean": 0.5},
    }

    report = build_report(tmp_path / "android.json", payload, _args())

    assert report["ok"] is True
    assert "p95_latency_less_than_mean:40.000<70.000" not in report["failures"]


def test_android_litert_report_rejects_non_finite_latency_and_memory(tmp_path):
    payload = {
        "schema_version": 1,
        "source_type": "android_litert_device_validation",
        "test_session_id": "android-litert-001",
        "test_app_version": "1.2.3",
        "test_date_utc": "2026-05-27",
        "device": {
            "model": "Pixel test",
            "manufacturer": "Google",
            "android_version": "15",
            "soc": "Tensor test",
            "is_emulator": False,
        },
        "runtime": "LiteRT",
        "artifact": {"sha256": "abc", "format": "tflite_float32"},
        "input": VALID_INPUT,
        "latency_ms": {"runs": math.nan, "mean": math.nan, "p95": math.inf},
        "memory_mb": {"peak": math.nan},
        "output": {"shape": [1, 14, 8400], "finite": True, "min": 0.0, "max": 1.0, "mean": 0.5},
    }

    report = build_report(tmp_path / "android.json", payload, _args())

    assert report["ok"] is False
    assert "missing_latency_runs" in report["failures"]
    assert "missing_mean_latency" in report["failures"]
    assert "missing_p95_latency" in report["failures"]
    assert "missing_peak_memory" in report["failures"]
    assert report["metrics"]["runs"] == 0
    assert report["metrics"]["mean_latency_ms"] is None
    assert report["metrics"]["p95_latency_ms"] is None
    assert report["metrics"]["peak_memory_mb"] is None


def test_android_litert_report_rejects_fractional_latency_runs(tmp_path):
    payload = {
        "schema_version": 1,
        "source_type": "android_litert_device_validation",
        "test_session_id": "android-litert-001",
        "test_app_version": "1.2.3",
        "test_date_utc": "2026-05-27",
        "device": {
            "model": "Pixel test",
            "manufacturer": "Google",
            "android_version": "15",
            "soc": "Tensor test",
            "is_emulator": False,
        },
        "runtime": "LiteRT",
        "artifact": {"sha256": "abc", "format": "tflite_float32"},
        "input": VALID_INPUT,
        "latency_ms": {"runs": 30.5, "mean": 40.0, "p95": 70.0},
        "memory_mb": {"peak": 200.0},
        "output": {"shape": [1, 14, 8400], "finite": True, "min": 0.0, "max": 1.0, "mean": 0.5},
    }

    report = build_report(tmp_path / "android.json", payload, _args())

    assert report["ok"] is False
    assert "invalid_latency_runs" in report["failures"]
    assert report["metrics"]["runs"] == 0


def test_android_litert_report_rejects_missing_schema_version(tmp_path):
    payload = {
        "source_type": "android_litert_device_validation",
        "test_session_id": "android-litert-001",
        "test_app_version": "1.2.3",
        "test_date_utc": "2026-05-27",
        "device": {
            "model": "Pixel test",
            "manufacturer": "Google",
            "android_version": "15",
            "soc": "Tensor test",
            "is_emulator": False,
        },
        "runtime": "LiteRT",
        "artifact": {"sha256": "abc", "format": "tflite_float32"},
        "input": VALID_INPUT,
        "latency_ms": {"runs": 30, "mean": 40.0, "p95": 70.0},
        "memory_mb": {"peak": 200.0},
        "output": {"shape": [1, 14, 8400], "finite": True, "min": 0.0, "max": 1.0, "mean": 0.5},
    }

    report = build_report(tmp_path / "android.json", payload, _args())

    assert report["ok"] is False
    assert "unsupported_schema_version:missing" in report["failures"]


def test_android_litert_report_rejects_boolean_schema_version(tmp_path):
    payload = {
        "schema_version": True,
        "source_type": "android_litert_device_validation",
        "test_session_id": "android-litert-001",
        "test_app_version": "1.2.3",
        "test_date_utc": "2026-05-27",
        "device": {
            "model": "Pixel test",
            "manufacturer": "Google",
            "android_version": "15",
            "soc": "Tensor test",
            "is_emulator": False,
        },
        "runtime": "LiteRT",
        "artifact": {"sha256": "abc", "format": "tflite_float32"},
        "input": VALID_INPUT,
        "latency_ms": {"runs": 30, "mean": 40.0, "p95": 70.0},
        "memory_mb": {"peak": 200.0},
        "output": {"shape": [1, 14, 8400], "finite": True, "min": 0.0, "max": 1.0, "mean": 0.5},
    }

    report = build_report(tmp_path / "android.json", payload, _args())

    assert report["ok"] is False
    assert "unsupported_schema_version:True" in report["failures"]


def test_android_litert_report_rejects_placeholders(tmp_path):
    payload = {
        "source_type": "FILL_ME",
        "test_session_id": "FILL_ME",
        "test_app_version": "FILL_ME",
        "test_date_utc": "FILL_ME_YYYY-MM-DD",
        "device": {
            "model": "FILL_ME",
            "manufacturer": "FILL_ME",
            "android_version": "FILL_ME",
            "soc": "FILL_ME",
            "is_emulator": True,
        },
        "runtime": "LiteRT",
        "artifact": {"sha256": "FILL_ME", "format": "FILL_ME"},
        "input": {"shape": [], "dtype": "FILL_ME", "profile": "FILL_ME"},
        "latency_ms": {"runs": 30, "mean": 40.0, "p95": 70.0},
        "memory_mb": {"peak": 200.0},
        "output": {"shape": [1, 14, 8400], "finite": True, "min": 0.0, "max": 1.0, "mean": 0.5},
    }

    report = build_report(tmp_path / "android.json", payload, _args())

    assert report["ok"] is False
    assert any("source_type" in failure for failure in report["failures"])
    assert "missing_test_session_id" in report["failures"]
    assert "missing_test_app_version" in report["failures"]
    assert "invalid_test_date_utc" in report["failures"]
    assert "missing_device_model" in report["failures"]
    assert "missing_device_manufacturer" in report["failures"]
    assert "missing_android_version" in report["failures"]
    assert "missing_device_soc" in report["failures"]
    assert "missing_artifact_sha256" in report["failures"]
    assert any("unexpected_artifact_format" in failure for failure in report["failures"])


def test_android_litert_report_rejects_impossible_test_date(tmp_path):
    payload = {
        "source_type": "android_litert_device_validation",
        "test_session_id": "android-litert-001",
        "test_app_version": "1.2.3",
        "test_date_utc": "2026-99-99",
        "device": {
            "model": "Pixel test",
            "manufacturer": "Google",
            "android_version": "15",
            "soc": "Tensor test",
            "is_emulator": False,
        },
        "runtime": "LiteRT",
        "artifact": {"sha256": "abc", "format": "tflite_float32"},
        "input": VALID_INPUT,
        "latency_ms": {"runs": 30, "mean": 40.0, "p95": 70.0},
        "memory_mb": {"peak": 200.0},
        "output": {"shape": [1, 14, 8400], "finite": True, "min": 0.0, "max": 1.0, "mean": 0.5},
    }

    report = build_report(tmp_path / "android.json", payload, _args())

    assert report["ok"] is False
    assert "invalid_test_date_utc" in report["failures"]


def test_android_litert_report_rejects_future_test_date(tmp_path):
    payload = {
        "schema_version": 1,
        "source_type": "android_litert_device_validation",
        "test_session_id": "android-litert-001",
        "test_app_version": "1.2.3",
        "test_date_utc": "2999-01-01",
        "device": {
            "model": "Pixel test",
            "manufacturer": "Google",
            "android_version": "15",
            "soc": "Tensor test",
            "is_emulator": False,
        },
        "runtime": "LiteRT",
        "artifact": {"sha256": "abc", "format": "tflite_float32"},
        "input": VALID_INPUT,
        "latency_ms": {"runs": 30, "mean": 40.0, "p95": 70.0},
        "memory_mb": {"peak": 200.0},
        "output": {"shape": [1, 14, 8400], "finite": True, "min": 0.0, "max": 1.0, "mean": 0.5},
    }

    report = build_report(tmp_path / "android.json", payload, _args())

    assert report["ok"] is False
    assert "invalid_test_date_utc" in report["failures"]


def test_android_litert_report_rejects_wrong_artifact_hash(tmp_path):
    expected = tmp_path / "best.tflite"
    expected.write_bytes(b"expected")
    args = _args()
    args.expected_artifact = expected
    payload = {
        "source_type": "android_litert_device_validation",
        "test_session_id": "android-litert-001",
        "device": {
            "model": "Pixel test",
            "manufacturer": "Google",
            "android_version": "15",
            "soc": "Tensor test",
            "is_emulator": False,
        },
        "runtime": "LiteRT",
        "artifact": {"sha256": "not-the-real-sha", "format": "tflite_float32"},
        "input": VALID_INPUT,
        "latency_ms": {"runs": 30, "mean": 40.0, "p95": 70.0},
        "memory_mb": {"peak": 200.0},
        "output": {"shape": [1, 14, 8400], "finite": True, "min": 0.0, "max": 1.0, "mean": 0.5},
    }

    report = build_report(tmp_path / "android.json", payload, args)

    assert report["ok"] is False
    assert "artifact_sha256_mismatch" in report["failures"]


def test_android_litert_report_rejects_missing_output_stats(tmp_path):
    payload = {
        "source_type": "android_litert_device_validation",
        "test_session_id": "android-litert-001",
        "device": {
            "model": "Pixel test",
            "manufacturer": "Google",
            "android_version": "15",
            "soc": "Tensor test",
            "is_emulator": False,
        },
        "runtime": "LiteRT",
        "artifact": {"sha256": "abc", "format": "tflite_float32"},
        "input": VALID_INPUT,
        "latency_ms": {"runs": 30, "mean": 40.0, "p95": 70.0},
        "memory_mb": {"peak": 200.0},
        "output": {"shape": [1, 14, 8400], "finite": True},
    }

    report = build_report(tmp_path / "android.json", payload, _args())

    assert report["ok"] is False
    assert "missing_or_non_finite_output_stats" in report["failures"]


def test_android_litert_report_rejects_degenerate_output_range(tmp_path):
    payload = {
        "schema_version": 1,
        "source_type": "android_litert_device_validation",
        "test_session_id": "android-litert-001",
        "test_app_version": "1.2.3",
        "test_date_utc": "2026-05-27",
        "device": {
            "model": "Pixel test",
            "manufacturer": "Google",
            "android_version": "15",
            "soc": "Tensor test",
            "is_emulator": False,
        },
        "runtime": "LiteRT",
        "artifact": {"sha256": "abc", "format": "tflite_float32"},
        "input": VALID_INPUT,
        "latency_ms": {"runs": 30, "mean": 40.0, "p95": 70.0},
        "memory_mb": {"peak": 200.0},
        "output": {"shape": [1, 14, 8400], "finite": True, "min": 0.0, "max": 0.0, "mean": 0.0},
    }

    report = build_report(tmp_path / "android.json", payload, _args())

    assert report["ok"] is False
    assert "degenerate_output_range" in report["failures"]


def test_android_litert_report_rejects_output_mean_outside_range(tmp_path):
    payload = {
        "schema_version": 1,
        "source_type": "android_litert_device_validation",
        "test_session_id": "android-litert-001",
        "test_app_version": "1.2.3",
        "test_date_utc": "2026-05-27",
        "device": {
            "model": "Pixel test",
            "manufacturer": "Google",
            "android_version": "15",
            "soc": "Tensor test",
            "is_emulator": False,
        },
        "runtime": "LiteRT",
        "artifact": {"sha256": "abc", "format": "tflite_float32"},
        "input": VALID_INPUT,
        "latency_ms": {"runs": 30, "mean": 40.0, "p95": 70.0},
        "memory_mb": {"peak": 200.0},
        "output": {"shape": [1, 14, 8400], "finite": True, "min": -0.1, "max": 1.0, "mean": 1.5},
    }

    report = build_report(tmp_path / "android.json", payload, _args())

    assert report["ok"] is False
    assert "output_mean_outside_range" in report["failures"]


def test_android_litert_report_rejects_missing_peak_memory(tmp_path):
    payload = {
        "source_type": "android_litert_device_validation",
        "test_session_id": "android-litert-001",
        "device": {
            "model": "Pixel test",
            "manufacturer": "Google",
            "android_version": "15",
            "soc": "Tensor test",
            "is_emulator": False,
        },
        "runtime": "LiteRT",
        "artifact": {"sha256": "abc", "format": "tflite_float32"},
        "input": VALID_INPUT,
        "latency_ms": {"runs": 30, "mean": 40.0, "p95": 70.0},
        "output": {"shape": [1, 14, 8400], "finite": True, "min": 0.0, "max": 1.0, "mean": 0.5},
    }

    report = build_report(tmp_path / "android.json", payload, _args())

    assert report["ok"] is False
    assert "missing_peak_memory" in report["failures"]


def test_android_litert_report_rejects_wrong_input_contract(tmp_path):
    payload = {
        "source_type": "android_litert_device_validation",
        "test_session_id": "android-litert-001",
        "device": {
            "model": "Pixel test",
            "manufacturer": "Google",
            "android_version": "15",
            "soc": "Tensor test",
            "is_emulator": False,
        },
        "runtime": "LiteRT",
        "artifact": {"sha256": "abc", "format": "tflite_float32"},
        "input": {"shape": [1, 3, 640, 640], "dtype": "uint8", "profile": "unknown"},
        "latency_ms": {"runs": 30, "mean": 40.0, "p95": 70.0},
        "memory_mb": {"peak": 200.0},
        "output": {"shape": [1, 14, 8400], "finite": True, "min": 0.0, "max": 1.0, "mean": 0.5},
    }

    report = build_report(tmp_path / "android.json", payload, _args())

    assert report["ok"] is False
    assert "unexpected_input_shape:[1, 3, 640, 640]" in report["failures"]
    assert "unexpected_input_dtype:uint8" in report["failures"]
    assert "unexpected_input_profile:unknown" in report["failures"]


def test_android_litert_report_rejects_non_integer_shape_values(tmp_path):
    payload = {
        "schema_version": 1,
        "source_type": "android_litert_device_validation",
        "test_session_id": "android-litert-001",
        "test_app_version": "1.2.3",
        "test_date_utc": "2026-05-27",
        "device": {
            "model": "Pixel test",
            "manufacturer": "Google",
            "android_version": "15",
            "soc": "Tensor test",
            "is_emulator": False,
        },
        "runtime": "LiteRT",
        "artifact": {"sha256": "abc", "format": "tflite_float32"},
        "input": {"shape": [True, 640, 640, 3], "dtype": "float32", "profile": "zero_float32_smoke"},
        "latency_ms": {"runs": 30, "mean": 40.0, "p95": 70.0},
        "memory_mb": {"peak": 200.0},
        "output": {"shape": [1.0, 14, 8400], "finite": True, "min": 0.0, "max": 1.0, "mean": 0.5},
    }

    report = build_report(tmp_path / "android.json", payload, _args())

    assert report["ok"] is False
    assert "unexpected_input_shape:[]" in report["failures"]
    assert "unexpected_output_shape:[]" in report["failures"]


def test_android_litert_report_rejects_emulator_device(tmp_path):
    payload = {
        "source_type": "android_litert_device_validation",
        "test_session_id": "android-litert-001",
        "device": {
            "model": "Pixel test",
            "manufacturer": "Google",
            "android_version": "15",
            "soc": "Tensor test",
            "is_emulator": True,
        },
        "runtime": "LiteRT",
        "artifact": {"sha256": "abc", "format": "tflite_float32"},
        "input": VALID_INPUT,
        "latency_ms": {"runs": 30, "mean": 40.0, "p95": 70.0},
        "memory_mb": {"peak": 200.0},
        "output": {"shape": [1, 14, 8400], "finite": True, "min": 0.0, "max": 1.0, "mean": 0.5},
    }

    report = build_report(tmp_path / "android.json", payload, _args())

    assert report["ok"] is False
    assert "device_must_be_physical:is_emulator=True" in report["failures"]
