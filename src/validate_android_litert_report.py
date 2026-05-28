"""Validate target-device Android LiteRT runtime evidence.

Expected input is a JSON report exported by the Android integration test.
The validator writes a normalized production-gate report with `ok`,
`failures`, thresholds, and key runtime metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import date
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path("data/incoming/android_litert_device_report.json")
DEFAULT_OUT = Path("outputs/production_audit/android_litert_device_eval.json")
DEFAULT_EXPECTED_ARTIFACT = Path("outputs/production_audit/tflite_export/best_float32.tflite")
EXPECTED_SCHEMA_VERSION = 1
EXPECTED_OUTPUT_SHAPE = [1, 14, 8400]
PRODUCTION_SOURCE_TYPES = {"android_litert_device_validation"}
DEFAULT_MIN_RUNS = 20
DEFAULT_MAX_MEAN_LATENCY_MS = 120.0
DEFAULT_MAX_P95_LATENCY_MS = 180.0
DEFAULT_MAX_PEAK_MEMORY_MB = 512.0
EXPECTED_INPUT_SHAPE = [1, 640, 640, 3]
EXPECTED_INPUT_DTYPE = "float32"
EXPECTED_INPUT_PROFILE = "zero_float32_smoke"
UTC_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def metric(report: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    cur: Any = report
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default


def string_value(report: dict[str, Any], *keys: str) -> str:
    cur: Any = report
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return ""
        cur = cur[key]
    return cur if isinstance(cur, str) else ""


def list_value(report: dict[str, Any], *keys: str) -> list[Any]:
    cur: Any = report
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return []
        cur = cur[key]
    return cur if isinstance(cur, list) else []


def integer_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def int_list_value(report: dict[str, Any], *keys: str) -> list[int]:
    values = list_value(report, *keys)
    if not all(integer_count(value) for value in values):
        return []
    return values


def normalized_dtype(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"float", "float32", "float_32"}:
        return "float32"
    if lowered.endswith(".float32") or lowered == "float32_t":
        return "float32"
    return lowered


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_placeholder(value: str) -> bool:
    normalized = value.strip().lower()
    return not normalized or "fill_me" in normalized or normalized in {"todo", "tbd", "unknown"}


def valid_utc_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip()
    if not UTC_DATE_RE.match(normalized):
        return False
    try:
        parsed = date.fromisoformat(normalized)
    except ValueError:
        return False
    return parsed <= date.today()


def finite_metric(value: float) -> bool:
    return math.isfinite(value)


def build_report(source: Path, payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    source_type = string_value(payload, "source_type")
    test_session_id = string_value(payload, "test_session_id")
    test_app_version = string_value(payload, "test_app_version")
    test_date_utc = string_value(payload, "test_date_utc")
    latency_mean = metric(payload, "latency_ms", "mean")
    latency_p95 = metric(payload, "latency_ms", "p95")
    memory_peak = metric(payload, "memory_mb", "peak", default=-1.0)
    latency = payload.get("latency_ms", {}) if isinstance(payload.get("latency_ms"), dict) else {}
    runs_raw = latency.get("runs")
    runs_metric = metric(payload, "latency_ms", "runs")
    runs = runs_raw if integer_count(runs_raw) else 0
    input_shape = int_list_value(payload, "input", "shape")
    input_dtype = normalized_dtype(string_value(payload, "input", "dtype"))
    input_profile = string_value(payload, "input", "profile")
    output_shape = int_list_value(payload, "output", "shape")
    output_min = metric(payload, "output", "min", default=float("nan"))
    output_max = metric(payload, "output", "max", default=float("nan"))
    output_mean = metric(payload, "output", "mean", default=float("nan"))
    device_model = string_value(payload, "device", "model")
    manufacturer = string_value(payload, "device", "manufacturer")
    android_version = string_value(payload, "device", "android_version")
    soc = string_value(payload, "device", "soc")
    is_emulator = payload.get("device", {}).get("is_emulator")
    artifact_sha = string_value(payload, "artifact", "sha256")
    artifact_format = string_value(payload, "artifact", "format")
    expected_artifact = getattr(args, "expected_artifact", None)
    expected_sha = sha256_file(expected_artifact) if isinstance(expected_artifact, Path) else None
    failures: list[str] = []

    if not integer_count(payload.get("schema_version")) or payload.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        failures.append(
            f"unsupported_schema_version:{payload.get('schema_version', 'missing')}"
        )
    if source_type not in PRODUCTION_SOURCE_TYPES:
        failures.append(
            "source_type must be one of "
            f"{sorted(PRODUCTION_SOURCE_TYPES)}, got {source_type or 'missing'}"
        )
    if is_placeholder(test_session_id):
        failures.append("missing_test_session_id")
    if is_placeholder(test_app_version):
        failures.append("missing_test_app_version")
    if not valid_utc_date(test_date_utc):
        failures.append("invalid_test_date_utc")
    if is_placeholder(device_model):
        failures.append("missing_device_model")
    if is_placeholder(manufacturer):
        failures.append("missing_device_manufacturer")
    if is_placeholder(android_version):
        failures.append("missing_android_version")
    if is_placeholder(soc):
        failures.append("missing_device_soc")
    if is_emulator is not False:
        failures.append(f"device_must_be_physical:is_emulator={is_emulator}")
    if string_value(payload, "runtime").lower() not in {"litert", "ai_edge_litert", "tensorflow_lite"}:
        failures.append(f"unsupported_runtime:{string_value(payload, 'runtime') or 'missing'}")
    if artifact_format != "tflite_float32":
        failures.append(f"unexpected_artifact_format:{artifact_format or 'missing'}")
    if input_shape != EXPECTED_INPUT_SHAPE:
        failures.append(f"unexpected_input_shape:{input_shape}")
    if input_dtype != EXPECTED_INPUT_DTYPE:
        failures.append(f"unexpected_input_dtype:{input_dtype or 'missing'}")
    if input_profile != EXPECTED_INPUT_PROFILE:
        failures.append(f"unexpected_input_profile:{input_profile or 'missing'}")
    if output_shape != EXPECTED_OUTPUT_SHAPE:
        failures.append(f"unexpected_output_shape:{output_shape}")
    if payload.get("output", {}).get("finite") is not True:
        failures.append("non_finite_or_missing_output")
    if not all(math.isfinite(v) for v in (output_min, output_max, output_mean)):
        failures.append("missing_or_non_finite_output_stats")
    elif output_min > output_max:
        failures.append("invalid_output_range")
    else:
        if output_min == output_max:
            failures.append("degenerate_output_range")
        if output_mean < output_min or output_mean > output_max:
            failures.append("output_mean_outside_range")
    if not finite_metric(runs_metric):
        failures.append("missing_latency_runs")
    elif not integer_count(runs_raw):
        failures.append("invalid_latency_runs")
    elif runs < args.min_runs:
        failures.append(f"too_few_runs:{runs}<{args.min_runs}")
    if not finite_metric(latency_mean) or latency_mean <= 0:
        failures.append("missing_mean_latency")
    elif latency_mean > args.max_mean_latency_ms:
        failures.append(f"mean_latency:{latency_mean:.3f}>{args.max_mean_latency_ms:.3f}")
    if not finite_metric(latency_p95) or latency_p95 <= 0:
        failures.append("missing_p95_latency")
    elif latency_p95 > args.max_p95_latency_ms:
        failures.append(f"p95_latency:{latency_p95:.3f}>{args.max_p95_latency_ms:.3f}")
    if not finite_metric(memory_peak) or memory_peak <= 0:
        failures.append("missing_peak_memory")
    elif memory_peak > args.max_peak_memory_mb:
        failures.append(f"peak_memory:{memory_peak:.3f}>{args.max_peak_memory_mb:.3f}")
    if is_placeholder(artifact_sha):
        failures.append("missing_artifact_sha256")
    if isinstance(expected_artifact, Path):
        if expected_sha is None:
            failures.append(f"missing_expected_artifact:{expected_artifact}")
        elif artifact_sha and artifact_sha != expected_sha:
            failures.append("artifact_sha256_mismatch")

    return {
        "schema_version": 1,
        "ok": not failures,
        "source": str(source),
        "source_schema_version": payload.get("schema_version"),
        "source_type": source_type,
        "test_session_id": test_session_id,
        "test_app_version": test_app_version,
        "test_date_utc": test_date_utc,
        "source_sha256": sha256_file(source),
        "failures": failures,
        "thresholds": {
            "min_runs": args.min_runs,
            "max_mean_latency_ms": args.max_mean_latency_ms,
            "max_p95_latency_ms": args.max_p95_latency_ms,
            "max_peak_memory_mb": args.max_peak_memory_mb,
            "expected_input_shape": EXPECTED_INPUT_SHAPE,
            "expected_input_dtype": EXPECTED_INPUT_DTYPE,
            "expected_input_profile": EXPECTED_INPUT_PROFILE,
            "expected_output_shape": EXPECTED_OUTPUT_SHAPE,
            "expected_artifact": str(expected_artifact) if isinstance(expected_artifact, Path) else None,
            "expected_artifact_sha256": expected_sha,
        },
        "device": payload.get("device", {}),
        "runtime": payload.get("runtime"),
        "artifact": payload.get("artifact", {}),
        "input": {
            "shape": input_shape,
            "dtype": input_dtype,
            "profile": input_profile,
        },
        "metrics": {
            "runs": runs,
            "mean_latency_ms": latency_mean if finite_metric(latency_mean) else None,
            "p95_latency_ms": latency_p95 if finite_metric(latency_p95) else None,
            "peak_memory_mb": memory_peak if finite_metric(memory_peak) and memory_peak >= 0 else None,
        },
        "output": {
            "shape": output_shape,
            "finite": payload.get("output", {}).get("finite"),
            "min": output_min if math.isfinite(output_min) else None,
            "max": output_max if math.isfinite(output_max) else None,
            "mean": output_mean if math.isfinite(output_mean) else None,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--expected-artifact", type=Path, default=DEFAULT_EXPECTED_ARTIFACT)
    parser.add_argument("--min-runs", type=int, default=DEFAULT_MIN_RUNS)
    parser.add_argument("--max-mean-latency-ms", type=float, default=DEFAULT_MAX_MEAN_LATENCY_MS)
    parser.add_argument("--max-p95-latency-ms", type=float, default=DEFAULT_MAX_P95_LATENCY_MS)
    parser.add_argument("--max-peak-memory-mb", type=float, default=DEFAULT_MAX_PEAK_MEMORY_MB)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.source.is_file():
        report = {
            "schema_version": 1,
            "ok": False,
            "source": str(args.source),
            "failures": [f"missing_source:{args.source}"],
        }
    else:
        report = build_report(args.source, read_json(args.source), args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"ok={report['ok']} failures={report.get('failures', [])}")
    print(f"report={args.out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
