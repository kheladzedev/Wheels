"""Audit external Android/AR production evidence inputs and outputs.

This is an intake/status audit, not a substitute for the production gate.
It checks whether the required external artifacts are present, whether
their generated validation reports pass, and whether those reports are
fresh relative to the source inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from evaluate_ar_holdout import validate_production_annotations, validate_production_provenance
from validate_ar_replay import ReplayThresholds, build_report as build_ar_replay_report, load_jsonl
from validate_android_litert_report import (
    DEFAULT_MAX_MEAN_LATENCY_MS,
    DEFAULT_MAX_P95_LATENCY_MS,
    DEFAULT_MAX_PEAK_MEMORY_MB,
    DEFAULT_MIN_RUNS,
    DEFAULT_EXPECTED_ARTIFACT,
    EXPECTED_INPUT_DTYPE,
    EXPECTED_INPUT_PROFILE,
    EXPECTED_INPUT_SHAPE,
    EXPECTED_OUTPUT_SHAPE,
    PRODUCTION_SOURCE_TYPES as ANDROID_PRODUCTION_SOURCE_TYPES,
    build_report as build_android_litert_report,
    is_placeholder,
    valid_utc_date,
)


DEFAULT_JSON_OUT = Path("outputs/production_audit/production_evidence_audit.json")
DEFAULT_MD_OUT = Path("docs/PRODUCTION_EVIDENCE_AUDIT.md")
DEFAULT_AR_REPLAY_JSONL = Path("data/incoming/ar_3d_replay/ar_replay.jsonl")
DEFAULT_IMPORT_REPORT = Path("outputs/production_audit/external_evidence_drop_import.json")
SUPPORTED_ANDROID_RUNTIMES = {"litert", "ai_edge_litert", "tensorflow_lite"}
DEFAULT_MIN_AR_HOLDOUT_IMAGES = 50
DEFAULT_MIN_AR_HOLDOUT_GT_WHEELS = 80
DEFAULT_AR_REPLAY_THRESHOLDS = ReplayThresholds()
AR_HOLDOUT_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
AR_HOLDOUT_REQUIRED_POINTS = ("a", "b", "c_disc_bottom")
AR_HOLDOUT_COUNT_KEYS = ("images", "gt_wheels")
AR_HOLDOUT_INTEGER_THRESHOLD_KEYS = ("min_images", "min_gt_wheels")


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


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


def integer_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def schema_version_matches(value: Any, expected: int) -> bool:
    return integer_count(value) and value == expected


def shape_matches(value: Any, expected: list[int]) -> bool:
    return (
        isinstance(value, list)
        and len(value) == len(expected)
        and all(integer_count(item) for item in value)
        and value == expected
    )


def strict_integer_value(
    value: Any,
    *,
    key: str,
    failures: list[str],
    failure_prefix: str,
) -> int:
    if not integer_count(value):
        failures.append(f"{failure_prefix}:{key}:{value if value is not None else 'missing'}")
        return 0
    return value


def iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    elif path.is_dir():
        for child in path.rglob("*"):
            if child.is_file():
                yield child


def latest_mtime(paths: Iterable[Path]) -> float | None:
    mtimes = [path.stat().st_mtime for path in paths if path.exists()]
    return max(mtimes) if mtimes else None


def is_stale(output: Path, source_paths: Iterable[Path]) -> bool:
    if not output.is_file():
        return False
    latest_source = latest_mtime(path for source in source_paths for path in iter_files(source))
    if latest_source is None:
        return False
    return output.stat().st_mtime < latest_source


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_sha256(entries: list[dict[str, Any]]) -> str:
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ar_holdout_manifest_entries(source_root: Path) -> list[dict[str, Any]]:
    paths: list[Path] = []
    for relative in [
        Path("images"),
        Path("annotations"),
        Path("metadata/provenance.json"),
        Path("metadata/source_info.json"),
    ]:
        candidate = source_root / relative
        if candidate.is_file():
            paths.append(candidate)
        elif candidate.is_dir():
            paths.extend(sorted(path for path in candidate.rglob("*") if path.is_file()))
    entries: list[dict[str, Any]] = []
    for path in sorted(set(paths)):
        entries.append(
            {
                "path": path.relative_to(source_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return entries


def ar_holdout_source_manifest_sha256(source_root: Path) -> str:
    return manifest_sha256(ar_holdout_manifest_entries(source_root))


def point2(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(finite_number(item) for item in value)


def ar_holdout_wheel_failures(wheel: Any, *, relative: str, index: int) -> list[str]:
    prefix = f"{relative}:wheel[{index}]"
    if not isinstance(wheel, dict):
        return [f"wheel_not_object:{prefix}"]
    failures: list[str] = []
    bbox = wheel.get("bbox_xyxy")
    bbox_xyxy: tuple[float, float, float, float] | None = None
    if not isinstance(bbox, list) or len(bbox) != 4 or not all(finite_number(item) for item in bbox):
        failures.append(f"wheel_invalid_bbox_xyxy:{prefix}")
    else:
        x1, y1, x2, y2 = [float(item) for item in bbox]
        if x2 <= x1 or y2 <= y1:
            failures.append(f"wheel_invalid_bbox_order:{prefix}")
        else:
            bbox_xyxy = (x1, y1, x2, y2)
    points = wheel.get("points")
    if not isinstance(points, dict):
        failures.append(f"wheel_missing_points:{prefix}")
    else:
        for name in AR_HOLDOUT_REQUIRED_POINTS:
            point = points.get(name)
            if not point2(point):
                failures.append(f"wheel_invalid_point_{name}:{prefix}")
            elif bbox_xyxy is not None:
                x1, y1, x2, y2 = bbox_xyxy
                x, y = [float(item) for item in point]
                if x < x1 or x > x2 or y < y1 or y > y2:
                    failures.append(f"wheel_point_{name}_outside_bbox:{prefix}")
    return failures


def ar_holdout_source_stats(source_root: Path) -> dict[str, Any]:
    images = source_root / "images"
    annotations = source_root / "annotations"
    image_files = (
        sorted(
            path
            for path in images.rglob("*")
            if path.is_file() and path.suffix.lower() in AR_HOLDOUT_IMAGE_EXTENSIONS
        )
        if images.is_dir()
        else []
    )
    annotation_files = (
        sorted(path for path in annotations.rglob("*.json") if path.is_file())
        if annotations.is_dir()
        else []
    )
    image_stems = {path.relative_to(images).with_suffix("").as_posix(): path for path in image_files}
    annotation_stems = {
        path.relative_to(annotations).with_suffix("").as_posix(): path for path in annotation_files
    }
    missing_annotations = sorted(set(image_stems) - set(annotation_stems))
    missing_images = sorted(set(annotation_stems) - set(image_stems))
    failures: list[str] = []
    gt_wheels = 0
    invalid_annotation_count = 0
    invalid_wheel_count = 0
    annotation_image_mismatches: list[str] = []

    if missing_annotations:
        failures.append(f"missing_annotations_for_images:{missing_annotations[:20]}")
    if missing_images:
        failures.append(f"missing_images_for_annotations:{missing_images[:20]}")

    for stem, path in annotation_stems.items():
        payload = read_json(path)
        relative = path.relative_to(annotations).as_posix()
        if not payload:
            invalid_annotation_count += 1
            failures.append(f"invalid_annotation_object:{relative}")
            continue
        frame_id = payload.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            failures.append(f"annotation_missing_frame_id:{relative}")
        elif frame_id != Path(relative).stem:
            failures.append(f"annotation_frame_id_mismatch:{relative}:{frame_id}")
        image_value = payload.get("image")
        if not isinstance(image_value, str) or not image_value.strip():
            annotation_image_mismatches.append(relative)
        elif Path(image_value).name != image_value:
            annotation_image_mismatches.append(relative)
        else:
            image_stem = Path(image_value).with_suffix("").as_posix()
            if image_stem != stem or image_stem not in image_stems:
                annotation_image_mismatches.append(relative)
        wheels = payload.get("wheels")
        if not isinstance(wheels, list):
            invalid_annotation_count += 1
            failures.append(f"invalid_annotation_wheels:{relative}")
            continue
        for index, wheel in enumerate(wheels):
            wheel_failures = ar_holdout_wheel_failures(wheel, relative=relative, index=index)
            if wheel_failures:
                invalid_wheel_count += 1
                failures.extend(wheel_failures)
            else:
                gt_wheels += 1

    if annotation_image_mismatches:
        failures.append(f"annotation_image_field_mismatch:{annotation_image_mismatches[:20]}")

    return {
        "image_count": len(image_files),
        "annotation_count": len(annotation_files),
        "gt_wheels": gt_wheels,
        "missing_annotations": missing_annotations,
        "missing_images": missing_images,
        "invalid_annotation_count": invalid_annotation_count,
        "invalid_wheel_count": invalid_wheel_count,
        "failures": failures,
    }


def path_matches(value: Any, expected: Path) -> bool:
    if not isinstance(value, str) or not value:
        return False
    base_dirs = [Path.cwd(), expected.parent]
    return bool(_path_keys(Path(value), base_dirs=base_dirs) & _path_keys(expected, base_dirs=base_dirs))


def quality_failures(
    report: dict[str, Any],
    *,
    min_map50: float,
    min_oks: float,
    max_fn: float,
    min_images: int,
    min_gt_wheels: int,
) -> list[str]:
    map50 = metric(report, "metrics_bbox", "mAP50")
    oks = metric(report, "oks", "mean")
    fn = metric(report, "rates", "false_negative_rate", default=1.0)
    failures: list[str] = []
    counts = report.get("counts", {}) if isinstance(report.get("counts"), dict) else {}
    images = strict_integer_value(
        counts.get("images"),
        key="images",
        failures=failures,
        failure_prefix="ar_holdout_count_not_integer",
    )
    gt_wheels = strict_integer_value(
        counts.get("gt_wheels"),
        key="gt_wheels",
        failures=failures,
        failure_prefix="ar_holdout_count_not_integer",
    )
    if images < min_images:
        failures.append(f"images:{images}<{min_images}")
    if gt_wheels < min_gt_wheels:
        failures.append(f"gt_wheels:{gt_wheels}<{min_gt_wheels}")
    if map50 < min_map50:
        failures.append(f"bbox_mAP50:{map50:.3f}<{min_map50:.3f}")
    if oks < min_oks:
        failures.append(f"OKS:{oks:.3f}<{min_oks:.3f}")
    if fn > max_fn:
        failures.append(f"FN:{fn:.3f}>{max_fn:.3f}")
    return failures


def quality_ok(
    report: dict[str, Any],
    *,
    min_map50: float,
    min_oks: float,
    max_fn: float,
    min_images: int = DEFAULT_MIN_AR_HOLDOUT_IMAGES,
    min_gt_wheels: int = DEFAULT_MIN_AR_HOLDOUT_GT_WHEELS,
) -> bool:
    return not quality_failures(
        report,
        min_map50=min_map50,
        min_oks=min_oks,
        max_fn=max_fn,
        min_images=min_images,
        min_gt_wheels=min_gt_wheels,
    )


def android_report_completeness_failures(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    device = report.get("device", {}) if isinstance(report.get("device"), dict) else {}
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    input_meta = report.get("input", {}) if isinstance(report.get("input"), dict) else {}
    output = report.get("output", {}) if isinstance(report.get("output"), dict) else {}
    artifact = report.get("artifact", {}) if isinstance(report.get("artifact"), dict) else {}
    runtime = str(report.get("runtime", "")).strip().lower()
    source_type = str(report.get("source_type", "")).strip()
    test_session_id = str(report.get("test_session_id", "")).strip()
    test_app_version = str(report.get("test_app_version", "")).strip()
    test_date_utc = str(report.get("test_date_utc", "")).strip()

    if not schema_version_matches(report.get("source_schema_version"), 1):
        failures.append(
            f"unsupported_source_schema_version:{report.get('source_schema_version', 'missing')}"
        )
    if source_type not in ANDROID_PRODUCTION_SOURCE_TYPES:
        failures.append(
            "source_type must be one of "
            f"{sorted(ANDROID_PRODUCTION_SOURCE_TYPES)}, got {source_type or 'missing'}"
        )
    if is_placeholder(test_session_id):
        failures.append("missing_test_session_id")
    if is_placeholder(test_app_version):
        failures.append("missing_test_app_version")
    if not valid_utc_date(test_date_utc):
        failures.append("invalid_test_date_utc")
    for key in ("model", "manufacturer", "android_version", "soc"):
        if is_placeholder(str(device.get(key, ""))):
            failures.append(f"missing_device_{key}")
    if device.get("is_emulator") is not False:
        failures.append(f"device_must_be_physical:is_emulator={device.get('is_emulator', 'missing')}")
    if runtime not in SUPPORTED_ANDROID_RUNTIMES:
        failures.append(f"unsupported_runtime:{runtime or 'missing'}")
    if artifact.get("format") != "tflite_float32":
        failures.append(f"unexpected_artifact_format:{artifact.get('format', 'missing')}")
    if not shape_matches(input_meta.get("shape"), EXPECTED_INPUT_SHAPE):
        failures.append(f"unexpected_input_shape:{input_meta.get('shape', 'missing')}")
    if str(input_meta.get("dtype", "")).strip().lower() != EXPECTED_INPUT_DTYPE:
        failures.append(f"unexpected_input_dtype:{input_meta.get('dtype', 'missing')}")
    if input_meta.get("profile") != EXPECTED_INPUT_PROFILE:
        failures.append(f"unexpected_input_profile:{input_meta.get('profile', 'missing')}")
    raw_runs = metrics.get("runs")
    runs_metric = metric(report, "metrics", "runs")
    runs = raw_runs if integer_count(raw_runs) else 0
    mean_latency = metric(report, "metrics", "mean_latency_ms")
    p95_latency = metric(report, "metrics", "p95_latency_ms")
    peak_memory = metric(report, "metrics", "peak_memory_mb", default=-1.0)
    if not math.isfinite(runs_metric):
        failures.append("missing_latency_runs")
    elif not integer_count(raw_runs):
        failures.append("invalid_latency_runs")
    elif runs < DEFAULT_MIN_RUNS:
        failures.append(f"too_few_runs:{runs}<{DEFAULT_MIN_RUNS}")
    if not math.isfinite(mean_latency) or mean_latency <= 0:
        failures.append("missing_mean_latency")
    elif mean_latency > DEFAULT_MAX_MEAN_LATENCY_MS:
        failures.append(f"mean_latency:{mean_latency:.3f}>{DEFAULT_MAX_MEAN_LATENCY_MS:.3f}")
    if not math.isfinite(p95_latency) or p95_latency <= 0:
        failures.append("missing_p95_latency")
    elif p95_latency > DEFAULT_MAX_P95_LATENCY_MS:
        failures.append(f"p95_latency:{p95_latency:.3f}>{DEFAULT_MAX_P95_LATENCY_MS:.3f}")
    if not math.isfinite(peak_memory) or peak_memory <= 0:
        failures.append("missing_peak_memory")
    elif peak_memory > DEFAULT_MAX_PEAK_MEMORY_MB:
        failures.append(f"peak_memory:{peak_memory:.3f}>{DEFAULT_MAX_PEAK_MEMORY_MB:.3f}")
    if not shape_matches(output.get("shape"), EXPECTED_OUTPUT_SHAPE):
        failures.append(f"unexpected_output_shape:{output.get('shape', 'missing')}")
    if output.get("finite") is not True:
        failures.append("non_finite_or_missing_output")
    output_values = []
    for key in ("min", "max", "mean"):
        try:
            output_values.append(float(output.get(key)))
        except (TypeError, ValueError):
            output_values.append(float("nan"))
    if not all(math.isfinite(value) for value in output_values):
        failures.append("missing_or_non_finite_output_stats")
    elif output_values[0] > output_values[1]:
        failures.append("invalid_output_range")
    else:
        if output_values[0] == output_values[1]:
            failures.append("degenerate_output_range")
        if output_values[2] < output_values[0] or output_values[2] > output_values[1]:
            failures.append("output_mean_outside_range")
    return failures


def android_source_revalidation_failures(
    source: Path,
    report: dict[str, Any],
    *,
    expected_artifact: Path,
) -> list[str]:
    source_report = build_android_litert_report(
        source,
        read_json(source),
        argparse.Namespace(
            expected_artifact=expected_artifact,
            min_runs=DEFAULT_MIN_RUNS,
            max_mean_latency_ms=DEFAULT_MAX_MEAN_LATENCY_MS,
            max_p95_latency_ms=DEFAULT_MAX_P95_LATENCY_MS,
            max_peak_memory_mb=DEFAULT_MAX_PEAK_MEMORY_MB,
        ),
    )
    failures: list[str] = []
    if source_report.get("ok") is not True:
        failures.append(f"source_revalidation_failed:{source_report.get('failures', [])}")
    for key in (
        "schema_version",
        "ok",
        "source_schema_version",
        "source_type",
        "test_session_id",
        "test_app_version",
        "test_date_utc",
        "runtime",
    ):
        if not _same_report_value(report.get(key), source_report.get(key)):
            failures.append(f"android_report_field_mismatch:{key}:{report.get(key)}!={source_report.get(key)}")
    if report.get("failures") != source_report.get("failures"):
        failures.append("android_report_failures_mismatch")
    thresholds = report.get("thresholds", {}) if isinstance(report.get("thresholds"), dict) else {}
    source_thresholds = (
        source_report.get("thresholds", {})
        if isinstance(source_report.get("thresholds"), dict)
        else {}
    )
    if set(thresholds) != set(source_thresholds):
        failures.append("android_report_threshold_keys_mismatch")
    for key, expected in source_thresholds.items():
        observed = thresholds.get(key)
        if not _same_report_value(observed, expected):
            failures.append(f"android_report_threshold_mismatch:{key}:{observed}!={expected}")
    for section in ("device", "artifact", "input", "output"):
        if not _same_report_value(report.get(section), source_report.get(section)):
            failures.append(f"android_report_section_mismatch:{section}")
    for key in ("runs", "mean_latency_ms", "p95_latency_ms", "peak_memory_mb"):
        observed = metric(report, "metrics", key, default=float("nan"))
        expected = metric(source_report, "metrics", key, default=float("nan"))
        if not _same_number(observed, expected):
            failures.append(f"android_report_metric_mismatch:{key}:{observed}!={expected}")
    return failures


def ar_holdout_pipeline_completeness_failures(
    pipeline: dict[str, Any],
    *,
    eval_report_path: Path | None = None,
    eval_report: dict[str, Any] | None = None,
    min_map50: float = 0.85,
    min_oks: float = 0.80,
    max_fn: float = 0.10,
    min_images: int = DEFAULT_MIN_AR_HOLDOUT_IMAGES,
    min_gt_wheels: int = DEFAULT_MIN_AR_HOLDOUT_GT_WHEELS,
) -> list[str]:
    failures: list[str] = []
    if pipeline.get("stage") != "done":
        failures.append(f"pipeline_stage_not_done:{pipeline.get('stage', 'missing')}")
    eval_returncode = pipeline.get("eval_returncode")
    if not integer_count(eval_returncode) or eval_returncode != 0:
        failures.append(f"pipeline_eval_returncode_not_zero:{pipeline.get('eval_returncode', 'missing')}")
    evaluation = pipeline.get("evaluation", {}) if isinstance(pipeline.get("evaluation"), dict) else {}
    if evaluation.get("ok") is not True:
        failures.append(f"pipeline_evaluation_not_ok:{evaluation.get('failures', [])}")
    if eval_report is not None:
        expected_failures = quality_failures(
            eval_report,
            min_map50=min_map50,
            min_oks=min_oks,
            max_fn=max_fn,
            min_images=min_images,
            min_gt_wheels=min_gt_wheels,
        )
        expected_metrics = {
            "bbox_mAP50": metric(eval_report, "metrics_bbox", "mAP50"),
            "oks_mean": metric(eval_report, "oks", "mean"),
            "false_negative_rate": metric(eval_report, "rates", "false_negative_rate", default=1.0),
            "images": strict_integer_value(
                eval_report.get("counts", {}).get("images")
                if isinstance(eval_report.get("counts"), dict)
                else None,
                key="images",
                failures=[],
                failure_prefix="ar_holdout_count_not_integer",
            ),
            "gt_wheels": strict_integer_value(
                eval_report.get("counts", {}).get("gt_wheels")
                if isinstance(eval_report.get("counts"), dict)
                else None,
                key="gt_wheels",
                failures=[],
                failure_prefix="ar_holdout_count_not_integer",
            ),
        }
        expected_thresholds = {
            "min_map50": min_map50,
            "min_oks": min_oks,
            "max_fn": max_fn,
            "min_images": min_images,
            "min_gt_wheels": min_gt_wheels,
        }
        if evaluation.get("ok") != (not expected_failures):
            failures.append("pipeline_evaluation_ok_mismatch")
        if evaluation.get("failures") != expected_failures:
            failures.append("pipeline_evaluation_failures_mismatch")
        thresholds = evaluation.get("thresholds", {}) if isinstance(evaluation.get("thresholds"), dict) else {}
        for key in AR_HOLDOUT_INTEGER_THRESHOLD_KEYS:
            if key in thresholds and not integer_count(thresholds.get(key)):
                failures.append(f"pipeline_evaluation_threshold_not_integer:{key}:{thresholds.get(key)}")
        for key, expected in expected_thresholds.items():
            observed = thresholds.get(key)
            if isinstance(expected, float):
                if not _same_number(observed, expected):
                    failures.append(f"pipeline_evaluation_threshold_mismatch:{key}:{observed}!={expected}")
            elif observed != expected:
                failures.append(f"pipeline_evaluation_threshold_mismatch:{key}:{observed}!={expected}")
        metrics = evaluation.get("metrics", {}) if isinstance(evaluation.get("metrics"), dict) else {}
        for key in AR_HOLDOUT_COUNT_KEYS:
            if key in metrics and not integer_count(metrics.get(key)):
                failures.append(f"pipeline_evaluation_metric_count_not_integer:{key}:{metrics.get(key)}")
        for key, expected in expected_metrics.items():
            observed = metrics.get(key)
            if isinstance(expected, float):
                if not _same_number(observed, expected):
                    failures.append(f"pipeline_evaluation_metric_mismatch:{key}:{observed}!={expected}")
            elif observed != expected:
                failures.append(f"pipeline_evaluation_metric_mismatch:{key}:{observed}!={expected}")
    conversion = pipeline.get("conversion", {}) if isinstance(pipeline.get("conversion"), dict) else {}
    if conversion and conversion.get("ok") is False:
        failures.append("pipeline_conversion_not_ok")
    if eval_report_path is not None:
        if not path_matches(pipeline.get("eval_report"), eval_report_path):
            failures.append(f"pipeline_eval_report_mismatch:{pipeline.get('eval_report', 'missing')}")
        pipeline_eval_sha = str(pipeline.get("eval_report_sha256", "")).strip()
        actual_eval_sha = sha256_file(eval_report_path)
        if not pipeline_eval_sha:
            failures.append("missing_pipeline_eval_report_sha256")
        elif pipeline_eval_sha != actual_eval_sha:
            failures.append("pipeline_eval_report_sha256_mismatch")
    return failures


def _threshold_number(
    thresholds: dict[str, Any],
    key: str,
    default: float,
) -> float:
    if key not in thresholds:
        return default
    value = thresholds.get(key)
    if not finite_number(value):
        return default
    return float(value)


def _threshold_int(thresholds: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(thresholds.get(key, default))
    except (TypeError, ValueError):
        return default


AR_REPLAY_INTEGER_THRESHOLD_KEYS = (
    "min_observations",
    "min_sessions",
    "min_final_positions",
)
AR_REPLAY_FLOAT_THRESHOLD_KEYS = (
    "min_floor_hit_rate",
    "min_inlier_rate",
    "max_median_residual",
    "max_p95_residual",
)
AR_REPLAY_COUNT_KEYS = (
    "observations_total",
    "observations_valid",
    "schema_errors",
    "sessions",
    "floor_hits_complete",
    "production_source_observations",
    "ransac_labelled",
    "inliers",
    "outliers",
    "residuals",
    "recovered_planes",
    "c_plane_hits",
    "c_height_values",
    "final_disc_bottom_positions",
)
AR_REPLAY_METRIC_KEYS = (
    "floor_hit_rate",
    "inlier_rate",
    "median_residual",
    "p95_residual",
)


def _strict_count(value: Any, *, key: str, failures: list[str]) -> int:
    if not integer_count(value):
        failures.append(f"replay_count_not_integer:{key}:{value if value is not None else 'missing'}")
        return 0
    return value


def ar_replay_report_completeness_failures(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    thresholds = report.get("thresholds", {}) if isinstance(report.get("thresholds"), dict) else {}
    counts = report.get("counts", {}) if isinstance(report.get("counts"), dict) else {}
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    for key in AR_REPLAY_INTEGER_THRESHOLD_KEYS:
        if key in thresholds and not integer_count(thresholds.get(key)):
            failures.append(f"replay_threshold_not_integer:{key}:{thresholds.get(key)}")
    for key in AR_REPLAY_FLOAT_THRESHOLD_KEYS:
        if key in thresholds and not finite_number(thresholds.get(key)):
            failures.append(f"replay_threshold_not_number:{key}:{thresholds.get(key)}")
    for key in AR_REPLAY_COUNT_KEYS:
        _strict_count(counts.get(key), key=key, failures=failures)
    for key in AR_REPLAY_METRIC_KEYS:
        if key in metrics and not finite_number(metrics.get(key)):
            failures.append(f"replay_metric_not_number:{key}:{metrics.get(key)}")

    min_observations = _threshold_int(
        thresholds,
        "min_observations",
        DEFAULT_AR_REPLAY_THRESHOLDS.min_observations,
    )
    min_sessions = _threshold_int(
        thresholds,
        "min_sessions",
        DEFAULT_AR_REPLAY_THRESHOLDS.min_sessions,
    )
    min_floor_hit_rate = _threshold_number(
        thresholds,
        "min_floor_hit_rate",
        DEFAULT_AR_REPLAY_THRESHOLDS.min_floor_hit_rate,
    )
    min_inlier_rate = _threshold_number(
        thresholds,
        "min_inlier_rate",
        DEFAULT_AR_REPLAY_THRESHOLDS.min_inlier_rate,
    )
    max_median_residual = _threshold_number(
        thresholds,
        "max_median_residual",
        DEFAULT_AR_REPLAY_THRESHOLDS.max_median_residual,
    )
    max_p95_residual = _threshold_number(
        thresholds,
        "max_p95_residual",
        DEFAULT_AR_REPLAY_THRESHOLDS.max_p95_residual,
    )
    min_final_positions = _threshold_int(
        thresholds,
        "min_final_positions",
        DEFAULT_AR_REPLAY_THRESHOLDS.min_final_positions,
    )

    observations_valid = _strict_count(counts.get("observations_valid"), key="observations_valid", failures=[])
    sessions = _strict_count(counts.get("sessions"), key="sessions", failures=[])
    schema_errors = _strict_count(counts.get("schema_errors"), key="schema_errors", failures=[])
    floor_hits_complete = _strict_count(counts.get("floor_hits_complete"), key="floor_hits_complete", failures=[])
    production_source_observations = _strict_count(
        counts.get("production_source_observations"),
        key="production_source_observations",
        failures=[],
    )
    ransac_labelled = _strict_count(counts.get("ransac_labelled"), key="ransac_labelled", failures=[])
    residuals = _strict_count(counts.get("residuals"), key="residuals", failures=[])
    recovered_planes = _strict_count(counts.get("recovered_planes"), key="recovered_planes", failures=[])
    c_plane_hits = _strict_count(counts.get("c_plane_hits"), key="c_plane_hits", failures=[])
    c_height_values = _strict_count(counts.get("c_height_values"), key="c_height_values", failures=[])
    final_positions = _strict_count(
        counts.get("final_disc_bottom_positions"),
        key="final_disc_bottom_positions",
        failures=[],
    )
    floor_hit_rate = metric(report, "metrics", "floor_hit_rate")
    inlier_rate = metric(report, "metrics", "inlier_rate", default=-1.0)
    median_residual = metric(report, "metrics", "median_residual", default=float("inf"))
    p95_residual = metric(report, "metrics", "p95_residual", default=float("inf"))

    if thresholds.get("require_production_source") is not True:
        failures.append("production_source_not_required_in_report")
    if "min_observations" not in thresholds or min_observations < DEFAULT_AR_REPLAY_THRESHOLDS.min_observations:
        failures.append(f"min_observations_too_low:{thresholds.get('min_observations', 'missing')}")
    if "min_sessions" not in thresholds or min_sessions < DEFAULT_AR_REPLAY_THRESHOLDS.min_sessions:
        failures.append(f"min_sessions_too_low:{thresholds.get('min_sessions', 'missing')}")
    if "min_floor_hit_rate" not in thresholds or min_floor_hit_rate < DEFAULT_AR_REPLAY_THRESHOLDS.min_floor_hit_rate:
        failures.append(f"min_floor_hit_rate_too_low:{thresholds.get('min_floor_hit_rate', 'missing')}")
    if thresholds.get("require_ransac") is not True:
        failures.append("ransac_not_required_in_report")
    if "min_inlier_rate" not in thresholds or min_inlier_rate < DEFAULT_AR_REPLAY_THRESHOLDS.min_inlier_rate:
        failures.append(f"min_inlier_rate_too_low:{thresholds.get('min_inlier_rate', 'missing')}")
    if "max_median_residual" not in thresholds or max_median_residual > DEFAULT_AR_REPLAY_THRESHOLDS.max_median_residual:
        failures.append(f"max_median_residual_too_high:{thresholds.get('max_median_residual', 'missing')}")
    if "max_p95_residual" not in thresholds or max_p95_residual > DEFAULT_AR_REPLAY_THRESHOLDS.max_p95_residual:
        failures.append(f"max_p95_residual_too_high:{thresholds.get('max_p95_residual', 'missing')}")
    if "min_final_positions" not in thresholds or min_final_positions < DEFAULT_AR_REPLAY_THRESHOLDS.min_final_positions:
        failures.append(f"min_final_positions_too_low:{thresholds.get('min_final_positions', 'missing')}")

    if schema_errors != 0:
        failures.append(f"schema_errors_present:{schema_errors}")
    if observations_valid < min_observations:
        failures.append(f"observations_valid:{observations_valid}<{min_observations}")
    if sessions < min_sessions:
        failures.append(f"sessions:{sessions}<{min_sessions}")
    if production_source_observations != observations_valid:
        failures.append(
            f"production_source_count_mismatch:{production_source_observations}!={observations_valid}"
        )
    if floor_hits_complete < observations_valid:
        failures.append(f"incomplete_floor_hits:{floor_hits_complete}!={observations_valid}")
    if floor_hit_rate < min_floor_hit_rate:
        failures.append(f"floor_hit_rate:{floor_hit_rate:.3f}<{min_floor_hit_rate:.3f}")
    if thresholds.get("require_ransac") is True:
        if ransac_labelled < observations_valid:
            failures.append(f"incomplete_ransac_labels:{ransac_labelled}!={observations_valid}")
        if residuals < observations_valid:
            failures.append(f"incomplete_residuals:{residuals}!={observations_valid}")
        if recovered_planes < observations_valid:
            failures.append(f"incomplete_recovered_planes:{recovered_planes}!={observations_valid}")
        if c_plane_hits < observations_valid:
            failures.append(f"incomplete_c_plane_hits:{c_plane_hits}!={observations_valid}")
        if c_height_values < observations_valid:
            failures.append(f"incomplete_c_height_values:{c_height_values}!={observations_valid}")
        if inlier_rate < min_inlier_rate:
            failures.append(f"inlier_rate:{inlier_rate:.3f}<{min_inlier_rate:.3f}")
        if median_residual > max_median_residual:
            failures.append(f"median_residual:{median_residual:.6f}>{max_median_residual:.6f}")
        if p95_residual > max_p95_residual:
            failures.append(f"p95_residual:{p95_residual:.6f}>{max_p95_residual:.6f}")
        if median_residual < 0:
            failures.append(f"negative_median_residual:{median_residual:.6f}")
        if p95_residual < 0:
            failures.append(f"negative_p95_residual:{p95_residual:.6f}")
        if p95_residual < median_residual:
            failures.append(f"p95_residual_less_than_median:{p95_residual:.6f}<{median_residual:.6f}")
        if final_positions < min_final_positions:
            failures.append(f"final_positions:{final_positions}<{min_final_positions}")
    return failures


def _same_number(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    try:
        left_float = float(left)
        right_float = float(right)
    except (TypeError, ValueError):
        return False
    return math.isfinite(left_float) and math.isfinite(right_float) and math.isclose(
        left_float,
        right_float,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


def _same_report_value(observed: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return observed is expected
    if isinstance(expected, int):
        return integer_count(observed) and observed == expected
    if isinstance(expected, float):
        return _same_number(observed, expected)
    if isinstance(expected, list):
        return (
            isinstance(observed, list)
            and len(observed) == len(expected)
            and all(_same_report_value(left, right) for left, right in zip(observed, expected))
        )
    if isinstance(expected, dict):
        return (
            isinstance(observed, dict)
            and set(observed) == set(expected)
            and all(_same_report_value(observed[key], expected[key]) for key in expected)
        )
    return observed == expected


def ar_replay_source_revalidation_failures(source: Path, report: dict[str, Any]) -> list[str]:
    source_report = build_ar_replay_report(
        load_jsonl(source),
        DEFAULT_AR_REPLAY_THRESHOLDS,
        source=source,
    )
    failures: list[str] = []
    if source_report.get("ok") is not True:
        failures.append(f"source_revalidation_failed:{source_report.get('failures', [])}")
    if not _same_report_value(report.get("ok"), source_report.get("ok")):
        failures.append(f"replay_report_field_mismatch:ok:{report.get('ok')}!={source_report.get('ok')}")
    if report.get("failures") != source_report.get("failures"):
        failures.append("replay_report_failures_mismatch")
    thresholds = report.get("thresholds", {}) if isinstance(report.get("thresholds"), dict) else {}
    source_thresholds = (
        source_report.get("thresholds", {})
        if isinstance(source_report.get("thresholds"), dict)
        else {}
    )
    if set(thresholds) != set(source_thresholds):
        failures.append("replay_report_threshold_keys_mismatch")
    for key, expected in source_thresholds.items():
        observed = thresholds.get(key)
        if not _same_report_value(observed, expected):
            failures.append(f"replay_report_threshold_mismatch:{key}:{observed}!={expected}")
    report_counts = report.get("counts", {}) if isinstance(report.get("counts"), dict) else {}
    source_counts = source_report.get("counts", {}) if isinstance(source_report.get("counts"), dict) else {}
    for key in AR_REPLAY_COUNT_KEYS:
        observed = report_counts.get(key)
        expected = source_counts.get(key)
        if not integer_count(observed):
            failures.append(f"replay_report_count_not_integer:{key}:{observed if observed is not None else 'missing'}")
            continue
        if observed != expected:
            failures.append(f"replay_report_count_mismatch:{key}:{observed}!={expected}")
    for key in ("floor_hit_rate", "inlier_rate", "median_residual", "p95_residual"):
        observed_value = report.get("metrics", {}).get(key) if isinstance(report.get("metrics"), dict) else None
        expected_value = (
            source_report.get("metrics", {}).get(key)
            if isinstance(source_report.get("metrics"), dict)
            else None
        )
        if observed_value is None or expected_value is None:
            if observed_value != expected_value:
                failures.append(f"replay_report_metric_mismatch:{key}:{observed_value}!={expected_value}")
        elif not _same_number(observed_value, expected_value):
            failures.append(f"replay_report_metric_mismatch:{key}:{observed_value}!={expected_value}")
    return failures


def cmd(script: str, *args: str | Path) -> list[str]:
    return ["./.venv/bin/python", script, *(str(arg) for arg in args)]


def build_required_evidence(args: argparse.Namespace) -> list[dict[str, Any]]:
    ar_replay_jsonl = getattr(args, "ar_replay_jsonl", DEFAULT_AR_REPLAY_JSONL)
    expected_android_artifact = getattr(args, "expected_android_artifact", DEFAULT_EXPECTED_ARTIFACT)
    return [
        {
            "name": "android_litert_device_validation",
            "owner": "android",
            "required_inputs": [str(args.android_litert_source)],
            "evidence_producer": [
                "android_litert_harness/README.md",
                "android_litert_harness/AndroidLiteRtDeviceValidationTest.kt",
            ],
            "validation_command": cmd(
                "src/validate_android_litert_report.py",
                "--source",
                args.android_litert_source,
                "--out",
                args.android_litert_eval,
                "--expected-artifact",
                expected_android_artifact,
                "--min-runs",
                DEFAULT_MIN_RUNS,
                "--max-mean-latency-ms",
                DEFAULT_MAX_MEAN_LATENCY_MS,
                "--max-p95-latency-ms",
                DEFAULT_MAX_P95_LATENCY_MS,
                "--max-peak-memory-mb",
                DEFAULT_MAX_PEAK_MEMORY_MB,
            ),
            "required_outputs": [str(args.android_litert_eval)],
            "production_gate_requirement": {"ok": True},
            "thresholds": {
                "min_runs": 20,
                "max_mean_latency_ms": 120.0,
                "max_p95_latency_ms": 180.0,
                "max_peak_memory_mb": 512.0,
                "expected_input_shape": EXPECTED_INPUT_SHAPE,
                "expected_input_dtype": EXPECTED_INPUT_DTYPE,
                "expected_input_profile": EXPECTED_INPUT_PROFILE,
                "expected_output_shape": EXPECTED_OUTPUT_SHAPE,
                "expected_artifact": str(expected_android_artifact),
                "expected_artifact_sha256": sha256_file(expected_android_artifact),
            },
        },
        {
            "name": "human_labelled_ar_device_holdout",
            "owner": "ar_data_collection",
            "required_inputs": [
                str(args.ar_holdout_source / "images"),
                str(args.ar_holdout_source / "annotations"),
                str(args.ar_holdout_source / "metadata" / "provenance.json"),
            ],
            "evidence_producer": [
                "ar_holdout_harness/README.md",
                "ar_holdout_harness/ArHoldoutAnnotationWriter.kt",
            ],
            "validation_command": cmd(
                "src/evaluate_ar_holdout.py",
                "--source-root",
                args.ar_holdout_source,
                "--eval-out",
                args.ar_holdout_eval,
                "--status-out",
                args.ar_holdout_pipeline,
                "--min-map50",
                args.min_ar_holdout_map50,
                "--min-oks",
                args.min_ar_holdout_oks,
                "--max-fn",
                args.max_ar_holdout_fn,
                "--min-images",
                getattr(args, "min_ar_holdout_images", DEFAULT_MIN_AR_HOLDOUT_IMAGES),
                "--min-gt-wheels",
                getattr(args, "min_ar_holdout_gt_wheels", DEFAULT_MIN_AR_HOLDOUT_GT_WHEELS),
            ),
            "required_outputs": [str(args.ar_holdout_eval), str(args.ar_holdout_pipeline)],
            "production_gate_requirement": {
                "images": f">={getattr(args, 'min_ar_holdout_images', DEFAULT_MIN_AR_HOLDOUT_IMAGES)}",
                "gt_wheels": f">={getattr(args, 'min_ar_holdout_gt_wheels', DEFAULT_MIN_AR_HOLDOUT_GT_WHEELS)}",
                "bbox_mAP50": f">={args.min_ar_holdout_map50}",
                "oks_mean": f">={args.min_ar_holdout_oks}",
                "false_negative_rate": f"<={args.max_ar_holdout_fn}",
            },
            "thresholds": {
                "min_images": getattr(args, "min_ar_holdout_images", DEFAULT_MIN_AR_HOLDOUT_IMAGES),
                "min_gt_wheels": getattr(args, "min_ar_holdout_gt_wheels", DEFAULT_MIN_AR_HOLDOUT_GT_WHEELS),
                "min_bbox_map50": args.min_ar_holdout_map50,
                "min_oks": args.min_ar_holdout_oks,
                "max_false_negative_rate": args.max_ar_holdout_fn,
                "provenance_source_type": "android_ar_device_human_labelled",
                "provenance_label_type": "human_reviewed",
                "provenance_review_status": "accepted",
            },
        },
        {
            "name": "ar_3d_replay_validation",
            "owner": "ar_runtime",
            "required_inputs": [str(ar_replay_jsonl)],
            "evidence_producer": [
                "ar_replay_harness/README.md",
                "ar_replay_harness/ArReplayLogger.kt",
            ],
            "validation_command": cmd(
                "src/validate_ar_replay.py",
                "--jsonl",
                ar_replay_jsonl,
                "--out",
                args.ar_replay_eval,
                "--min-observations",
                DEFAULT_AR_REPLAY_THRESHOLDS.min_observations,
                "--min-sessions",
                DEFAULT_AR_REPLAY_THRESHOLDS.min_sessions,
                "--min-floor-hit-rate",
                DEFAULT_AR_REPLAY_THRESHOLDS.min_floor_hit_rate,
                "--min-inlier-rate",
                DEFAULT_AR_REPLAY_THRESHOLDS.min_inlier_rate,
                "--max-median-residual",
                DEFAULT_AR_REPLAY_THRESHOLDS.max_median_residual,
                "--max-p95-residual",
                DEFAULT_AR_REPLAY_THRESHOLDS.max_p95_residual,
                "--min-final-positions",
                DEFAULT_AR_REPLAY_THRESHOLDS.min_final_positions,
            ),
            "required_outputs": [str(args.ar_replay_eval)],
            "production_gate_requirement": {"ok": True},
            "thresholds": {
                "require_production_source": True,
                "required_source_type": "android_ar_device_replay",
                "template_sources_allowed": False,
                "min_observations": DEFAULT_AR_REPLAY_THRESHOLDS.min_observations,
                "min_sessions": DEFAULT_AR_REPLAY_THRESHOLDS.min_sessions,
                "min_floor_hit_rate": DEFAULT_AR_REPLAY_THRESHOLDS.min_floor_hit_rate,
                "require_ransac": DEFAULT_AR_REPLAY_THRESHOLDS.require_ransac,
                "min_inlier_rate": DEFAULT_AR_REPLAY_THRESHOLDS.min_inlier_rate,
                "max_median_residual": DEFAULT_AR_REPLAY_THRESHOLDS.max_median_residual,
                "max_p95_residual": DEFAULT_AR_REPLAY_THRESHOLDS.max_p95_residual,
                "min_final_positions": DEFAULT_AR_REPLAY_THRESHOLDS.min_final_positions,
            },
        },
    ]


def check_android_litert(args: argparse.Namespace) -> dict[str, Any]:
    source = args.android_litert_source
    report_path = args.android_litert_eval
    report = read_json(report_path)
    expected_artifact = getattr(args, "expected_android_artifact", DEFAULT_EXPECTED_ARTIFACT)
    actual_expected_artifact_sha = sha256_file(expected_artifact)
    failures: list[str] = []
    if not source.is_file():
        failures.append(f"missing_source:{source}")
    if not report_path.is_file():
        failures.append(f"missing_report:{report_path}")
    if report_path.is_file() and not bool(report.get("ok", False)):
        failures.append(f"report_not_ok:{report.get('failures', [])}")
    if report_path.is_file():
        failures.extend(android_report_completeness_failures(report))
    if report_path.is_file() and not path_matches(report.get("source"), source):
        failures.append(f"report_source_mismatch:{report.get('source', 'missing')}")
    if report_path.is_file():
        report_source_sha = str(report.get("source_sha256", "")).strip()
        actual_source_sha = sha256_file(source)
        if not report_source_sha:
            failures.append("missing_report_source_sha256")
        elif actual_source_sha != report_source_sha:
            failures.append("report_source_sha256_mismatch")
        if source.is_file():
            failures.extend(
                android_source_revalidation_failures(
                    source,
                    report,
                    expected_artifact=expected_artifact,
                )
            )
    expected_sha = str(report.get("thresholds", {}).get("expected_artifact_sha256", "")).strip()
    expected_path = report.get("thresholds", {}).get("expected_artifact")
    artifact_sha = str(report.get("artifact", {}).get("sha256", "")).strip()
    if report_path.is_file() and actual_expected_artifact_sha is None:
        failures.append(f"missing_expected_artifact:{expected_artifact}")
    if report_path.is_file() and expected_path and not path_matches(expected_path, expected_artifact):
        failures.append(f"expected_artifact_path_mismatch:{expected_path}!={expected_artifact}")
    if report_path.is_file() and not expected_sha:
        failures.append("missing_expected_artifact_sha256")
    if (
        report_path.is_file()
        and expected_sha
        and actual_expected_artifact_sha is not None
        and expected_sha != actual_expected_artifact_sha
    ):
        failures.append("expected_artifact_sha256_mismatch")
    if report_path.is_file() and expected_sha and artifact_sha != expected_sha:
        failures.append("artifact_sha256_mismatch")
    stale = is_stale(report_path, [source])
    if stale:
        failures.append("report_stale")
    return {
        "name": "android_litert_device_validation",
        "ready": not failures,
        "source": str(source),
        "report": str(report_path),
        "source_present": source.is_file(),
        "report_present": report_path.is_file(),
        "report_ok": bool(report.get("ok", False)),
        "expected_artifact": str(expected_artifact),
        "expected_artifact_sha256": actual_expected_artifact_sha,
        "stale": stale,
        "failures": failures,
    }


def check_ar_holdout(args: argparse.Namespace) -> dict[str, Any]:
    root = args.ar_holdout_source
    images = root / "images"
    annotations = root / "annotations"
    provenance = root / "metadata" / "provenance.json"
    report_path = args.ar_holdout_eval
    pipeline_path = args.ar_holdout_pipeline
    report = read_json(report_path)
    pipeline = read_json(pipeline_path)
    provenance_payload = read_json(provenance)
    source_stats = ar_holdout_source_stats(root)
    image_count = int(source_stats["image_count"])
    annotation_count = int(source_stats["annotation_count"])
    source_gt_wheels = int(source_stats["gt_wheels"])
    min_images = getattr(args, "min_ar_holdout_images", DEFAULT_MIN_AR_HOLDOUT_IMAGES)
    min_gt_wheels = getattr(args, "min_ar_holdout_gt_wheels", DEFAULT_MIN_AR_HOLDOUT_GT_WHEELS)
    failures: list[str] = []
    if not images.is_dir() or not annotations.is_dir():
        failures.append(f"missing_source_dirs:{root}")
    elif image_count <= 0 or annotation_count <= 0:
        failures.append("empty_source_dirs")
    elif image_count != annotation_count:
        failures.append(f"source_count_mismatch:{image_count}!={annotation_count}")
    if not provenance.is_file():
        failures.append(f"missing_provenance:{provenance}")
    else:
        provenance_failures = validate_production_provenance(provenance_payload)
        if provenance_failures:
            failures.append(f"invalid_provenance:{provenance_failures}")
    if annotations.is_dir():
        annotation_failures = validate_production_annotations(root)
        if annotation_failures:
            failures.append(f"invalid_annotations:{annotation_failures[:20]}")
        if source_stats["failures"]:
            failures.append(f"invalid_annotation_source:{source_stats['failures'][:20]}")
        if source_gt_wheels < min_gt_wheels:
            failures.append(f"source_gt_wheels:{source_gt_wheels}<{min_gt_wheels}")
    if not report_path.is_file():
        failures.append(f"missing_report:{report_path}")
    else:
        report_counts = report.get("counts", {}) if isinstance(report.get("counts"), dict) else {}
        count_failures: list[str] = []
        report_image_count = strict_integer_value(
            report_counts.get("images"),
            key="images",
            failures=count_failures,
            failure_prefix="ar_holdout_count_not_integer",
        )
        report_gt_wheels = strict_integer_value(
            report_counts.get("gt_wheels"),
            key="gt_wheels",
            failures=count_failures,
            failure_prefix="ar_holdout_count_not_integer",
        )
        failures.extend(count_failures)
        if report_image_count != image_count:
            failures.append(f"eval_image_count_source_mismatch:{report_image_count}!={image_count}")
        if report_gt_wheels != source_gt_wheels:
            failures.append(f"eval_gt_wheels_source_mismatch:{report_gt_wheels}!={source_gt_wheels}")
        quality = quality_failures(
            report,
            min_map50=args.min_ar_holdout_map50,
            min_oks=args.min_ar_holdout_oks,
            max_fn=args.max_ar_holdout_fn,
            min_images=min_images,
            min_gt_wheels=min_gt_wheels,
        )
        if quality:
            failures.append("quality_gate_failed:" + str(quality))
    if not pipeline_path.is_file():
        failures.append(f"missing_pipeline:{pipeline_path}")
    elif not bool(pipeline.get("ok", False)):
        failures.append(f"pipeline_not_ok:{pipeline.get('failure_reason', 'unknown')}")
    elif not path_matches(pipeline.get("source_root"), root):
        failures.append(f"pipeline_source_mismatch:{pipeline.get('source_root', 'missing')}")
    elif pipeline_path.is_file():
        failures.extend(
            ar_holdout_pipeline_completeness_failures(
                pipeline,
                eval_report_path=report_path,
                eval_report=report if report_path.is_file() else None,
                min_map50=args.min_ar_holdout_map50,
                min_oks=args.min_ar_holdout_oks,
                max_fn=args.max_ar_holdout_fn,
                min_images=min_images,
                min_gt_wheels=min_gt_wheels,
            )
        )
        pipeline_manifest_sha = str(pipeline.get("source_manifest_sha256", "")).strip()
        actual_manifest_sha = ar_holdout_source_manifest_sha256(root)
        if not pipeline_manifest_sha:
            failures.append("missing_pipeline_source_manifest_sha256")
        elif pipeline_manifest_sha != actual_manifest_sha:
            failures.append("pipeline_source_manifest_sha256_mismatch")
    stale = is_stale(report_path, [images, annotations, provenance])
    pipeline_stale = is_stale(pipeline_path, [images, annotations, provenance])
    if stale:
        failures.append("report_stale")
    if pipeline_stale:
        failures.append("pipeline_stale")
    return {
        "name": "human_labelled_ar_device_holdout",
        "ready": not failures,
        "source": str(root),
        "report": str(report_path),
        "pipeline": str(pipeline_path),
        "image_count": image_count,
        "annotation_count": annotation_count,
        "source_gt_wheels": source_gt_wheels,
        "provenance_present": provenance.is_file(),
        "provenance_valid": provenance.is_file() and not validate_production_provenance(provenance_payload),
        "report_present": report_path.is_file(),
        "pipeline_present": pipeline_path.is_file(),
        "pipeline_ok": bool(pipeline.get("ok", False)),
        "report_quality_ok": report_path.is_file()
        and quality_ok(
            report,
            min_map50=args.min_ar_holdout_map50,
            min_oks=args.min_ar_holdout_oks,
            max_fn=args.max_ar_holdout_fn,
            min_images=min_images,
            min_gt_wheels=min_gt_wheels,
        ),
        "stale": stale,
        "pipeline_stale": pipeline_stale,
        "failures": failures,
    }


def check_ar_replay(args: argparse.Namespace) -> dict[str, Any]:
    report_path = args.ar_replay_eval
    report = read_json(report_path)
    expected_source = getattr(args, "ar_replay_jsonl", DEFAULT_AR_REPLAY_JSONL)
    source = Path(str(report.get("source", ""))) if report.get("source") else None
    failures: list[str] = []
    if not expected_source.is_file():
        failures.append(f"missing_source:{expected_source}")
    if not report_path.is_file():
        failures.append(f"missing_report:{report_path}")
    if report_path.is_file() and not bool(report.get("ok", False)):
        failures.append(f"report_not_ok:{report.get('failures', [])}")
    if report_path.is_file():
        failures.extend(ar_replay_report_completeness_failures(report))
    if report_path.is_file() and source is None:
        failures.append("missing_report_source")
    if report_path.is_file() and not path_matches(report.get("source"), expected_source):
        failures.append(f"report_source_mismatch:{report.get('source', 'missing')}")
    if report_path.is_file():
        report_source_sha = str(report.get("source_sha256", "")).strip()
        actual_source_sha = sha256_file(expected_source)
        if not report_source_sha:
            failures.append("missing_report_source_sha256")
        elif actual_source_sha != report_source_sha:
            failures.append("report_source_sha256_mismatch")
        if expected_source.is_file():
            failures.extend(ar_replay_source_revalidation_failures(expected_source, report))
    if source is not None and not source.is_file():
        failures.append(f"missing_source:{source}")
    if source is not None and "template" in source.name.lower():
        failures.append(f"template_source_not_allowed:{source}")
    counts = report.get("counts", {}) if isinstance(report.get("counts"), dict) else {}
    observations_valid = int(counts.get("observations_valid", 0) or 0)
    production_source_observations = int(counts.get("production_source_observations", 0) or 0)
    sessions = int(counts.get("sessions", 0) or 0)
    thresholds = report.get("thresholds", {}) if isinstance(report.get("thresholds"), dict) else {}
    stale_sources = [expected_source] if expected_source.is_file() else [source] if source is not None else []
    stale = is_stale(report_path, stale_sources)
    if stale:
        failures.append("report_stale")
    return {
        "name": "ar_3d_replay_validation",
        "ready": not failures,
        "source": str(source) if source is not None else None,
        "report": str(report_path),
        "source_present": bool(source and source.is_file()),
        "report_present": report_path.is_file(),
        "report_ok": bool(report.get("ok", False)),
        "require_production_source": thresholds.get("require_production_source"),
        "observations_valid": observations_valid,
        "sessions": sessions,
        "production_source_observations": production_source_observations,
        "stale": stale,
        "failures": failures,
    }


def canonical_external_input_files(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = [
        args.android_litert_source,
        args.ar_holdout_source / "metadata" / "provenance.json",
        getattr(args, "ar_replay_jsonl", DEFAULT_AR_REPLAY_JSONL),
    ]
    for source_dir in [
        args.ar_holdout_source / "images",
        args.ar_holdout_source / "annotations",
    ]:
        if source_dir.is_dir():
            paths.extend(sorted(path for path in source_dir.rglob("*") if path.is_file()))
    return [path for path in paths if path.is_file()]


def _path_keys(path: Path, *, base_dirs: Iterable[Path] = ()) -> set[str]:
    keys = {str(path), path.as_posix()}
    candidates = [path]
    if not path.is_absolute():
        candidates.extend(base / path for base in base_dirs)
    for candidate in candidates:
        keys.add(str(candidate))
        keys.add(candidate.as_posix())
        try:
            resolved = candidate.resolve(strict=False)
        except OSError:
            continue
        keys.add(str(resolved))
        keys.add(resolved.as_posix())
    return keys


def check_external_evidence_custody(args: argparse.Namespace, *, required: bool) -> dict[str, Any]:
    report_path = getattr(args, "external_evidence_import_report", DEFAULT_IMPORT_REPORT)
    expected_android_artifact = getattr(args, "expected_android_artifact", DEFAULT_EXPECTED_ARTIFACT)
    report = read_json(report_path)
    failures: list[str] = []
    if required and not report_path.is_file():
        failures.append(f"missing_import_report:{report_path}")
    if report_path.is_file():
        if not schema_version_matches(report.get("schema_version"), 1):
            failures.append(f"unsupported_import_report_schema:{report.get('schema_version', 'missing')}")
        if report.get("ok") is not True:
            failures.append(f"import_report_not_ok:{report.get('failures', [])}")
        if report.get("dry_run") is True:
            failures.append("import_report_is_dry_run")
        source = report.get("source")
        if not isinstance(source, str) or not source:
            failures.append("missing_import_source")
        source_kind = report.get("source_kind")
        if source_kind not in {"directory", "zip"}:
            failures.append(f"unsupported_import_source_kind:{source_kind or 'missing'}")
        if source_kind == "zip":
            source_sha = str(report.get("source_sha256", "")).strip()
            if not source_sha:
                failures.append("missing_zip_source_sha256")
            elif isinstance(source, str) and source:
                source_path = Path(source)
                if source_path.is_file():
                    actual_source_sha = sha256_file(source_path)
                    if actual_source_sha != source_sha:
                        failures.append("zip_source_sha256_mismatch")
        if not isinstance(report.get("dest_root"), str) or not report.get("dest_root"):
            failures.append("missing_import_dest_root")
        actual_expected_artifact_sha = sha256_file(expected_android_artifact)
        report_expected_artifact = report.get("expected_android_artifact")
        report_expected_artifact_sha = str(report.get("expected_android_artifact_sha256", "")).strip()
        if actual_expected_artifact_sha is None:
            failures.append(f"missing_expected_android_artifact:{expected_android_artifact}")
        if not report_expected_artifact:
            failures.append("missing_import_expected_android_artifact")
        elif not path_matches(report_expected_artifact, expected_android_artifact):
            failures.append(
                "import_expected_android_artifact_path_mismatch:"
                f"{report_expected_artifact}!={expected_android_artifact}"
            )
        if not report_expected_artifact_sha:
            failures.append("missing_import_expected_android_artifact_sha256")
        elif actual_expected_artifact_sha is not None and report_expected_artifact_sha != actual_expected_artifact_sha:
            failures.append("import_expected_android_artifact_sha256_mismatch")
        if not report.get("evidence_manifest_sha256"):
            failures.append("missing_evidence_manifest_sha256")
        copied = report.get("copied_artifacts", [])
        if not isinstance(copied, list) or not copied:
            failures.append("missing_copied_artifacts")
            copied = []
        planned = report.get("planned", [])
        if not isinstance(planned, list) or not planned:
            failures.append("missing_planned_artifacts")
            planned = []
        elif report.get("evidence_manifest_sha256") != manifest_sha256(planned):
            failures.append("evidence_manifest_sha256_mismatch")
        file_count = report.get("file_count")
        if not integer_count(file_count) or file_count < len(copied):
            failures.append(f"invalid_import_file_count:{file_count if file_count is not None else 'missing'}")
        elif file_count != len(copied):
            failures.append(f"copied_artifact_count_mismatch:{len(copied)}!={file_count}")
        if integer_count(file_count) and planned and file_count != len(planned):
            failures.append(f"planned_artifact_count_mismatch:{len(planned)}!={file_count}")
        base_dirs = [Path.cwd()]
        dest_root = report.get("dest_root")
        if isinstance(dest_root, str) and dest_root:
            dest_root_path = Path(dest_root)
            base_dirs.append(dest_root_path)
            if not dest_root_path.is_absolute():
                base_dirs.append(Path.cwd() / dest_root_path)
        if report_path.parent:
            base_dirs.append(report_path.parent)
        copied_by_dest: dict[str, dict[str, Any]] = {}
        for item in copied:
            if not isinstance(item, dict):
                continue
            dest = item.get("dest")
            if not isinstance(dest, str):
                continue
            size_bytes = item.get("size_bytes")
            if not integer_count(size_bytes) or size_bytes < 0:
                failures.append(
                    "copied_artifact_invalid_size_bytes:"
                    f"{dest}:{size_bytes if size_bytes is not None else 'missing'}"
                )
            for key in _path_keys(Path(dest), base_dirs=base_dirs):
                copied_by_dest[key] = item
        planned_by_dest: dict[str, dict[str, Any]] = {}
        for item in planned:
            if not isinstance(item, dict):
                continue
            dest = item.get("dest")
            if not isinstance(dest, str):
                continue
            size_bytes = item.get("size_bytes")
            if not integer_count(size_bytes) or size_bytes < 0:
                failures.append(
                    "planned_artifact_invalid_size_bytes:"
                    f"{dest}:{size_bytes if size_bytes is not None else 'missing'}"
                )
            for key in _path_keys(Path(dest), base_dirs=base_dirs):
                planned_by_dest[key] = item
        for copied_item in copied:
            if not isinstance(copied_item, dict):
                continue
            copied_dest = copied_item.get("dest")
            if not isinstance(copied_dest, str):
                continue
            planned_item = None
            for key in _path_keys(Path(copied_dest), base_dirs=base_dirs):
                planned_item = planned_by_dest.get(key)
                if planned_item:
                    break
            if planned_item is None:
                failures.append(f"copied_artifact_not_planned:{copied_dest}")
                continue
            for key in ("size_bytes", "sha256"):
                if copied_item.get(key) != planned_item.get(key):
                    failures.append(f"copied_artifact_{key}_mismatch:{copied_dest}")
        for path in canonical_external_input_files(args):
            artifact = None
            for key in _path_keys(path, base_dirs=base_dirs):
                artifact = copied_by_dest.get(key)
                if artifact:
                    break
            if artifact is None:
                failures.append(f"input_not_in_import_report:{path}")
                continue
            expected_sha = artifact.get("sha256")
            actual_sha = sha256_file(path)
            if not expected_sha:
                failures.append(f"input_missing_import_sha256:{path}")
            elif actual_sha != expected_sha:
                failures.append(f"input_sha256_mismatch:{path}")
            expected_size = artifact.get("size_bytes")
            actual_size = path.stat().st_size
            if not integer_count(expected_size) or expected_size < 0:
                failures.append(
                    f"input_invalid_import_size_bytes:{path}:"
                    f"{expected_size if expected_size is not None else 'missing'}"
                )
            elif actual_size != expected_size:
                failures.append(f"input_size_mismatch:{path}:{expected_size}!={actual_size}")
    ready = not failures if required else True
    return {
        "name": "external_evidence_custody",
        "ready": ready,
        "required": required,
        "source": str(report_path),
        "report": str(report_path),
        "report_present": report_path.is_file(),
        "report_ok": bool(report.get("ok", False)),
        "dry_run": report.get("dry_run"),
        "expected_android_artifact": str(expected_android_artifact),
        "expected_android_artifact_sha256": sha256_file(expected_android_artifact),
        "canonical_input_count": len(canonical_external_input_files(args)),
        "failures": failures if required else [],
        "warnings": failures if not required else [],
    }


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    core_checks = [
        check_android_litert(args),
        check_ar_holdout(args),
        check_ar_replay(args),
    ]
    custody_required = all(check["ready"] for check in core_checks)
    checks = [
        *core_checks,
        check_external_evidence_custody(args, required=custody_required),
    ]
    blockers = [check["name"] for check in checks if not check["ready"]]
    required_evidence = build_required_evidence(args)
    return {
        "schema_version": 1,
        "ok": True,
        "production_evidence_ready": not blockers,
        "blockers": blockers,
        "required_evidence": required_evidence,
        "next_actions": [
            {
                "name": item["name"],
                "owner": item["owner"],
                "missing_or_failing": item["name"] in blockers,
                "evidence_producer": item.get("evidence_producer", []),
                "required_inputs": item["required_inputs"],
                "validation_command": item["validation_command"],
            }
            for item in required_evidence
            if item["name"] in blockers
        ],
        "checks": checks,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Production Evidence Audit",
        "",
        f"- Audit OK: {audit.get('ok')}",
        f"- Production evidence ready: {audit.get('production_evidence_ready')}",
        f"- Blockers: {', '.join(audit.get('blockers', [])) or 'none'}",
        "",
        "## Required Evidence",
        "",
        "| Evidence | Owner | Producer | Inputs | Command | Gate |",
        "|---|---|---|---|---|---|",
    ]
    for item in audit.get("required_evidence", []):
        producers = "<br>".join(f"`{path}`" for path in item.get("evidence_producer", [])) or "n/a"
        inputs = "<br>".join(f"`{path}`" for path in item.get("required_inputs", []))
        command = " ".join(item.get("validation_command", []))
        gate = json.dumps(item.get("production_gate_requirement", {}), ensure_ascii=False)
        lines.append(
            "| "
            f"{item.get('name')} | "
            f"{item.get('owner')} | "
            f"{producers} | "
            f"{inputs} | "
            f"`{command}` | "
            f"`{gate}` |"
        )
    lines.extend(
        [
            "",
            "## Current Checks",
            "",
            "| Evidence | Ready | Source | Report | Failures |",
            "|---|---:|---|---|---|",
        ]
    )
    for check in audit.get("checks", []):
        failures = ", ".join(check.get("failures", [])) or "none"
        lines.append(
            "| "
            f"{check.get('name')} | "
            f"{check.get('ready')} | "
            f"`{check.get('source')}` | "
            f"`{check.get('report')}` | "
            f"{failures} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    parser.add_argument(
        "--android-litert-source",
        type=Path,
        default=Path("data/incoming/android_litert_device_report.json"),
    )
    parser.add_argument(
        "--android-litert-eval",
        type=Path,
        default=Path("outputs/production_audit/android_litert_device_eval.json"),
    )
    parser.add_argument(
        "--ar-holdout-source",
        type=Path,
        default=Path("data/incoming/ar_device_holdout"),
    )
    parser.add_argument(
        "--ar-holdout-eval",
        type=Path,
        default=Path("outputs/production_audit/ar_device_holdout_eval.json"),
    )
    parser.add_argument(
        "--ar-holdout-pipeline",
        type=Path,
        default=Path("outputs/production_audit/ar_device_holdout_pipeline.json"),
    )
    parser.add_argument(
        "--ar-replay-eval",
        type=Path,
        default=Path("outputs/production_audit/ar_3d_replay_eval.json"),
    )
    parser.add_argument(
        "--ar-replay-jsonl",
        type=Path,
        default=DEFAULT_AR_REPLAY_JSONL,
    )
    parser.add_argument(
        "--external-evidence-import-report",
        type=Path,
        default=DEFAULT_IMPORT_REPORT,
    )
    parser.add_argument(
        "--expected-android-artifact",
        type=Path,
        default=DEFAULT_EXPECTED_ARTIFACT,
    )
    parser.add_argument("--min-ar-holdout-map50", type=float, default=0.85)
    parser.add_argument("--min-ar-holdout-oks", type=float, default=0.80)
    parser.add_argument("--max-ar-holdout-fn", type=float, default=0.10)
    parser.add_argument("--min-ar-holdout-images", type=int, default=DEFAULT_MIN_AR_HOLDOUT_IMAGES)
    parser.add_argument("--min-ar-holdout-gt-wheels", type=int, default=DEFAULT_MIN_AR_HOLDOUT_GT_WHEELS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = build_audit(args)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(audit), encoding="utf-8")
    print(
        f"ok={audit['ok']} production_evidence_ready={audit['production_evidence_ready']} "
        f"blockers={audit['blockers']}"
    )
    print(f"json={args.json_out}")
    print(f"markdown={args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
