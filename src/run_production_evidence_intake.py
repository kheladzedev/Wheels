"""Run the external Android/AR production evidence intake flow.

This runner is the one command to use after Android/AR teams provide
real production evidence. It validates each external input, rebuilds the
consolidated evidence audit, gates, and final reports, then writes a
machine-readable intake status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

try:
    from .import_external_evidence_drop import (
        DEFAULT_EXPECTED_ANDROID_ARTIFACT,
        DEFAULT_REPORT_OUT as DEFAULT_IMPORT_REPORT_OUT,
        build_import_report,
    )
except ImportError:  # pragma: no cover - used when executed as a script
    from import_external_evidence_drop import (
        DEFAULT_EXPECTED_ANDROID_ARTIFACT,
        DEFAULT_REPORT_OUT as DEFAULT_IMPORT_REPORT_OUT,
        build_import_report,
    )


DEFAULT_ANDROID_LITERT_SOURCE = Path("data/incoming/android_litert_device_report.json")
DEFAULT_ANDROID_LITERT_EVAL = Path("outputs/production_audit/android_litert_device_eval.json")
DEFAULT_AR_HOLDOUT_SOURCE = Path("data/incoming/ar_device_holdout")
DEFAULT_AR_HOLDOUT_EVAL = Path("outputs/production_audit/ar_device_holdout_eval.json")
DEFAULT_AR_HOLDOUT_PIPELINE = Path("outputs/production_audit/ar_device_holdout_pipeline.json")
DEFAULT_AR_REPLAY_JSONL = Path("data/incoming/ar_3d_replay/ar_replay.jsonl")
DEFAULT_AR_REPLAY_EVAL = Path("outputs/production_audit/ar_3d_replay_eval.json")
DEFAULT_EVIDENCE_DROP_DEST_ROOT = Path("data/incoming")
DEFAULT_STATUS_OUT = Path("outputs/production_audit/production_evidence_intake_status.json")
DEFAULT_MIN_ANDROID_RUNS = 20
DEFAULT_MAX_ANDROID_MEAN_LATENCY_MS = 120.0
DEFAULT_MAX_ANDROID_P95_LATENCY_MS = 180.0
DEFAULT_MAX_ANDROID_PEAK_MEMORY_MB = 512.0
DEFAULT_MIN_AR_HOLDOUT_IMAGES = 50
DEFAULT_MIN_AR_HOLDOUT_GT_WHEELS = 80
DEFAULT_MIN_AR_HOLDOUT_MAP50 = 0.85
DEFAULT_MIN_AR_HOLDOUT_OKS = 0.80
DEFAULT_MAX_AR_HOLDOUT_FN = 0.10
DEFAULT_MIN_AR_REPLAY_OBSERVATIONS = 30
DEFAULT_MIN_AR_REPLAY_SESSIONS = 1
DEFAULT_MIN_AR_REPLAY_FLOOR_HIT_RATE = 0.90
DEFAULT_MIN_AR_REPLAY_INLIER_RATE = 0.70
DEFAULT_MAX_AR_REPLAY_MEDIAN_RESIDUAL = 0.02
DEFAULT_MAX_AR_REPLAY_P95_RESIDUAL = 0.05
DEFAULT_MIN_AR_REPLAY_FINAL_POSITIONS = 1
FINALIZATION_COMMAND = [
    "./.venv/bin/python",
    "src/production_audit_suite.py",
    "--with-pytest",
]
POST_FINALIZATION_REPORT_REFRESH_COMMANDS = [
    [sys.executable, "scripts/write_production_audit_report.py"],
    [sys.executable, "scripts/write_handoff_report.py"],
]
POST_FINALIZATION_RELEASE_REFRESH_COMMANDS = [
    [sys.executable, "src/release_integrity.py"],
    [sys.executable, "src/report_consistency_audit.py"],
]
POST_FINALIZATION_REFRESH_COMMANDS = (
    POST_FINALIZATION_REPORT_REFRESH_COMMANDS
    + POST_FINALIZATION_RELEASE_REFRESH_COMMANDS
)
PLACEHOLDER_NAME_MARKERS = (
    ".PLACEHOLDER",
    "PLACE_FRAMES_HERE",
    "PLACE_ANNOTATIONS_HERE",
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ANNOTATION_EXT = ".json"
AR_HOLDOUT_REQUIRED_POINTS = ("a", "b", "c_disc_bottom")
AR_HOLDOUT_EXPECTED_SCHEMA_VERSION = 1
AR_HOLDOUT_PRODUCTION_SOURCE_TYPE = "android_ar_device_human_labelled"
AR_HOLDOUT_LABEL_TYPE = "human_reviewed"
AR_HOLDOUT_ACCEPTED_REVIEW_STATUSES = {"accepted"}
ANDROID_EXPECTED_SCHEMA_VERSION = 1
ANDROID_EXPECTED_SOURCE_TYPE = "android_litert_device_validation"
ANDROID_EXPECTED_INPUT_SHAPE = [1, 640, 640, 3]
ANDROID_EXPECTED_INPUT_DTYPE = "float32"
ANDROID_EXPECTED_INPUT_PROFILE = "zero_float32_smoke"
ANDROID_EXPECTED_OUTPUT_SHAPE = [1, 14, 8400]
AR_REPLAY_EXPECTED_SCHEMA_VERSION = 1
AR_REPLAY_EXPECTED_SOURCE_TYPES = {
    "android_ar_device_replay",
    "ios_ar_device_replay",
    "ar_device_replay",
}
AR_REPLAY_REQUIRED_SCREEN_POINTS = ("a", "b", "c_disc_bottom")
AR_REPLAY_REQUIRED_FLOOR_HITS = ("a", "b")
AR_REPLAY_UNIT_NORMAL_TOLERANCE = 0.05
UTC_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class IntakeStep:
    name: str
    cmd: list[str]
    input_path: Path | None = None
    preflight_paths: list[Path] | None = None
    required_input: bool = False
    production_required: bool = True
    expected_artifact: Path | None = None


@dataclass
class IntakeResult:
    name: str
    returncode: int | None
    ok: bool
    skipped: bool
    missing_input: bool
    input_path: str | None
    cmd: list[str]


@dataclass
class EvidenceDropImportResult:
    source: str
    dest_root: str
    report_out: str
    ok: bool
    dry_run: bool
    file_count: int
    failures: list[str]


@dataclass
class FinalizationResult:
    command: list[str]
    returncode: int | None
    ok: bool
    skipped: bool
    reason: str | None = None


@dataclass
class RefreshResult:
    name: str
    command: list[str]
    returncode: int
    ok: bool


def py_cmd(script: str, *args: str) -> list[str]:
    return [sys.executable, script, *args]


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_status(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def iter_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(child for child in path.rglob("*") if child.is_file())
    return []


def preflight_invalid_reason(path: Path) -> str | None:
    if not path.exists():
        return None
    if path.is_file() and path.stat().st_size <= 0:
        return "empty_file"
    if path.is_dir():
        files = iter_files(path)
        if not files:
            return "empty_directory"
        placeholders = [
            file.relative_to(path).as_posix()
            for file in files
            if any(marker in file.name for marker in PLACEHOLDER_NAME_MARKERS)
        ]
        if placeholders:
            return f"placeholder_files:{','.join(placeholders[:20])}"
    return None


def ar_holdout_pair_failures(source_root: Path) -> list[str]:
    images = source_root / "images"
    annotations = source_root / "annotations"
    if not images.is_dir() or not annotations.is_dir():
        return []
    image_files = [
        path
        for path in iter_files(images)
        if not any(marker in path.name for marker in PLACEHOLDER_NAME_MARKERS)
    ]
    annotation_files = [
        path
        for path in iter_files(annotations)
        if not any(marker in path.name for marker in PLACEHOLDER_NAME_MARKERS)
    ]
    image_stems = {
        path.stem
        for path in image_files
        if path.suffix.lower() in IMAGE_EXTS
    }
    annotation_stems = {
        path.stem
        for path in annotation_files
        if path.suffix.lower() == ANNOTATION_EXT
    }
    failures: list[str] = []
    bad_image_exts = sorted(
        path.relative_to(images).as_posix()
        for path in image_files
        if path.suffix.lower() not in IMAGE_EXTS
    )
    bad_annotation_exts = sorted(
        path.relative_to(annotations).as_posix()
        for path in annotation_files
        if path.suffix.lower() != ANNOTATION_EXT
    )
    if bad_image_exts:
        failures.append(f"ar_holdout_bad_image_extensions:{','.join(bad_image_exts[:20])}")
    if bad_annotation_exts:
        failures.append(f"ar_holdout_bad_annotation_extensions:{','.join(bad_annotation_exts[:20])}")
    missing_annotations = sorted(image_stems - annotation_stems)
    missing_images = sorted(annotation_stems - image_stems)
    if missing_annotations:
        failures.append(f"ar_holdout_missing_annotations:{','.join(missing_annotations[:20])}")
    if missing_images:
        failures.append(f"ar_holdout_missing_images:{','.join(missing_images[:20])}")
    return failures


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def integer_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def shape_matches(value: Any, expected: list[int]) -> bool:
    return (
        isinstance(value, list)
        and len(value) == len(expected)
        and all(integer_count(item) for item in value)
        and value == expected
    )


def schema_version_matches(value: Any, expected: int) -> bool:
    return integer_count(value) and value == expected


def placeholder(value: Any) -> bool:
    if not isinstance(value, str):
        return True
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


def point2(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(finite_number(item) for item in value)


def point(value: Any, dims: int) -> bool:
    return isinstance(value, list) and len(value) == dims and all(finite_number(item) for item in value)


def unit_vector3(value: Any) -> bool:
    if not point(value, 3):
        return False
    norm = math.sqrt(sum(float(item) * float(item) for item in value))
    return abs(norm - 1.0) <= AR_REPLAY_UNIT_NORMAL_TOLERANCE


def camera_transform(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    rotation = value.get("R")
    translation = value.get("t")
    return (
        isinstance(rotation, list)
        and len(rotation) == 3
        and all(isinstance(row, list) and len(row) == 3 and all(finite_number(item) for item in row) for row in rotation)
        and point(translation, 3)
    )


def recovered_plane(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    support = value.get("support")
    return (
        unit_vector3(value.get("normal"))
        and point(value.get("point"), 3)
        and isinstance(support, int)
        and not isinstance(support, bool)
        and support > 0
    )


def replay_wheel_identity(payload: dict[str, Any]) -> str | None:
    wheel_index = payload.get("wheel_index")
    if isinstance(wheel_index, int) and not isinstance(wheel_index, bool) and wheel_index >= 0:
        return f"wheel_index:{wheel_index}"
    wheel_track_id = payload.get("wheel_track_id")
    if isinstance(wheel_track_id, str) and not placeholder(wheel_track_id):
        return f"wheel_track_id:{wheel_track_id.strip()}"
    return None


def ar_holdout_wheel_failures(wheel: Any, *, relative: str, index: int) -> list[str]:
    prefix = f"{relative}:wheel[{index}]"
    if not isinstance(wheel, dict):
        return [f"ar_holdout_annotation_wheel_not_object:{prefix}"]
    failures: list[str] = []
    bbox = wheel.get("bbox_xyxy")
    bbox_xyxy: tuple[float, float, float, float] | None = None
    if not isinstance(bbox, list) or len(bbox) != 4 or not all(finite_number(item) for item in bbox):
        failures.append(f"ar_holdout_annotation_wheel_invalid_bbox:{prefix}")
    else:
        x1, y1, x2, y2 = [float(item) for item in bbox]
        if x2 <= x1 or y2 <= y1:
            failures.append(f"ar_holdout_annotation_wheel_nonpositive_bbox:{prefix}")
        else:
            bbox_xyxy = (x1, y1, x2, y2)
    points = wheel.get("points")
    if not isinstance(points, dict):
        failures.append(f"ar_holdout_annotation_wheel_missing_points:{prefix}")
    else:
        for point_name in AR_HOLDOUT_REQUIRED_POINTS:
            point = points.get(point_name)
            if not point2(point):
                failures.append(f"ar_holdout_annotation_wheel_invalid_point_{point_name}:{prefix}")
            elif bbox_xyxy is not None:
                x1, y1, x2, y2 = bbox_xyxy
                x, y = [float(item) for item in point]
                if x < x1 or x > x2 or y < y1 or y > y2:
                    failures.append(f"ar_holdout_annotation_wheel_point_{point_name}_outside_bbox:{prefix}")
    return failures


def ar_holdout_annotation_failures(source_root: Path) -> list[str]:
    images = source_root / "images"
    annotations = source_root / "annotations"
    if not images.is_dir() or not annotations.is_dir():
        return []
    image_names = {
        path.name
        for path in iter_files(images)
        if path.suffix.lower() in IMAGE_EXTS
        and not any(marker in path.name for marker in PLACEHOLDER_NAME_MARKERS)
    }
    annotation_files = [
        path
        for path in iter_files(annotations)
        if path.suffix.lower() == ANNOTATION_EXT
        and not any(marker in path.name for marker in PLACEHOLDER_NAME_MARKERS)
    ]
    failures: list[str] = []
    for path in annotation_files:
        relative = path.relative_to(annotations).as_posix()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failures.append(f"ar_holdout_invalid_annotation_json:{relative}")
            continue
        if not isinstance(payload, dict):
            failures.append(f"ar_holdout_annotation_not_object:{relative}")
            continue
        if not schema_version_matches(payload.get("schema_version"), 1):
            failures.append(
                "ar_holdout_annotation_unsupported_schema_version:"
                f"{relative}:{payload.get('schema_version', 'missing')}"
            )
        frame_id = payload.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            failures.append(f"ar_holdout_annotation_missing_frame_id:{relative}")
        elif frame_id != path.stem:
            failures.append(f"ar_holdout_annotation_frame_id_mismatch:{relative}:{frame_id}")
        image_name = payload.get("image")
        if not isinstance(image_name, str) or not image_name:
            failures.append(f"ar_holdout_annotation_missing_image:{relative}")
        elif Path(image_name).name != image_name:
            failures.append(f"ar_holdout_annotation_image_not_filename:{relative}:{image_name}")
        else:
            if image_name not in image_names:
                failures.append(f"ar_holdout_annotation_image_missing:{relative}:{image_name}")
            if Path(image_name).stem != path.stem:
                failures.append(f"ar_holdout_annotation_image_stem_mismatch:{relative}:{image_name}")
        if not isinstance(payload.get("wheels"), list):
            failures.append(f"ar_holdout_annotation_wheels_not_array:{relative}")
            continue
        for index, wheel in enumerate(payload["wheels"]):
            failures.extend(ar_holdout_wheel_failures(wheel, relative=relative, index=index))
    return failures


def ar_holdout_provenance_failures(source_root: Path) -> list[str]:
    path = source_root / "metadata" / "provenance.json"
    if not path.is_file() or path.stat().st_size <= 0:
        return []
    payload = read_json(path)
    if not payload:
        return ["ar_holdout_invalid_provenance_json"]
    failures: list[str] = []
    source_type = payload.get("source_type")
    label_type = payload.get("label_type")
    review_status = payload.get("review_status")
    capture_device = payload.get("capture_device")
    capture_app_version = payload.get("capture_app_version")
    capture_date_utc = payload.get("capture_date_utc")
    annotator = payload.get("annotator")
    reviewer = payload.get("reviewer")
    if not schema_version_matches(payload.get("schema_version"), AR_HOLDOUT_EXPECTED_SCHEMA_VERSION):
        failures.append(
            "ar_holdout_unsupported_schema_version:"
            f"{payload.get('schema_version', 'missing')}"
        )
    if source_type != AR_HOLDOUT_PRODUCTION_SOURCE_TYPE:
        failures.append(f"ar_holdout_invalid_source_type:{source_type or 'missing'}")
    if label_type != AR_HOLDOUT_LABEL_TYPE:
        failures.append(f"ar_holdout_invalid_label_type:{label_type or 'missing'}")
    if review_status not in AR_HOLDOUT_ACCEPTED_REVIEW_STATUSES:
        failures.append(f"ar_holdout_invalid_review_status:{review_status or 'missing'}")
    for key, value in (
        ("capture_device", capture_device),
        ("capture_app_version", capture_app_version),
        ("annotator", annotator),
        ("reviewer", reviewer),
    ):
        if placeholder(value):
            failures.append(f"ar_holdout_missing_{key}")
    if placeholder(capture_date_utc) or not valid_utc_date(capture_date_utc):
        failures.append("ar_holdout_invalid_capture_date_utc")
    if (
        isinstance(annotator, str)
        and isinstance(reviewer, str)
        and annotator.strip()
        and annotator.strip() == reviewer.strip()
    ):
        failures.append("ar_holdout_annotator_reviewer_not_independent")
    return failures


def android_report_preflight_failures(path: Path, *, expected_artifact: Path | None = None) -> list[str]:
    if not path.is_file() or path.stat().st_size <= 0:
        return []
    payload = read_json(path)
    if not payload:
        return ["android_report_invalid_json_object"]
    failures: list[str] = []
    if not schema_version_matches(payload.get("schema_version"), ANDROID_EXPECTED_SCHEMA_VERSION):
        failures.append(
            "android_report_unsupported_schema_version:"
            f"{payload.get('schema_version', 'missing')}"
        )
    if payload.get("source_type") != ANDROID_EXPECTED_SOURCE_TYPE:
        failures.append(
            "android_report_invalid_source_type:"
            f"{payload.get('source_type') or 'missing'}"
        )
    if placeholder(payload.get("test_session_id")):
        failures.append("android_report_missing_test_session_id")
    if placeholder(payload.get("test_app_version")):
        failures.append("android_report_missing_test_app_version")
    if not valid_utc_date(payload.get("test_date_utc")):
        failures.append("android_report_invalid_test_date_utc")
    device = payload.get("device")
    if not isinstance(device, dict):
        failures.append("android_report_missing_device_object")
        device = {}
    for key in ("model", "manufacturer", "android_version", "soc"):
        if placeholder(device.get(key)):
            failures.append(f"android_report_missing_device_{key}")
    if device.get("is_emulator") is not False:
        failures.append(f"android_report_device_must_be_physical:{device.get('is_emulator', 'missing')}")
    runtime = str(payload.get("runtime", "")).strip().lower()
    if runtime not in {"litert", "ai_edge_litert", "tensorflow_lite"}:
        failures.append(f"android_report_unsupported_runtime:{payload.get('runtime') or 'missing'}")
    artifact = payload.get("artifact")
    if not isinstance(artifact, dict):
        failures.append("android_report_missing_artifact_object")
        artifact = {}
    if placeholder(artifact.get("sha256")):
        failures.append("android_report_missing_artifact_sha256")
    if artifact.get("format") != "tflite_float32":
        failures.append(f"android_report_unexpected_artifact_format:{artifact.get('format') or 'missing'}")
    if expected_artifact is not None:
        expected_sha = sha256_file(expected_artifact)
        if expected_sha is None:
            failures.append(f"android_report_missing_expected_artifact:{expected_artifact}")
        elif isinstance(artifact.get("sha256"), str) and artifact.get("sha256") != expected_sha:
            failures.append("android_report_artifact_sha256_mismatch")
    input_meta = payload.get("input")
    if not isinstance(input_meta, dict):
        failures.append("android_report_missing_input_object")
        input_meta = {}
    if not shape_matches(input_meta.get("shape"), ANDROID_EXPECTED_INPUT_SHAPE):
        failures.append(f"android_report_unexpected_input_shape:{input_meta.get('shape') or 'missing'}")
    if str(input_meta.get("dtype", "")).strip().lower() != ANDROID_EXPECTED_INPUT_DTYPE:
        failures.append(f"android_report_unexpected_input_dtype:{input_meta.get('dtype') or 'missing'}")
    if input_meta.get("profile") != ANDROID_EXPECTED_INPUT_PROFILE:
        failures.append(f"android_report_unexpected_input_profile:{input_meta.get('profile') or 'missing'}")
    latency = payload.get("latency_ms")
    if not isinstance(latency, dict):
        failures.append("android_report_missing_latency_ms_object")
        latency = {}
    for key in ("mean", "p95"):
        if not finite_number(latency.get(key)):
            failures.append(f"android_report_missing_latency_{key}")
    runs = latency.get("runs")
    if runs is None or (not isinstance(runs, bool) and not finite_number(runs)):
        failures.append("android_report_missing_latency_runs")
    elif not integer_count(runs):
        failures.append("android_report_invalid_latency_runs")
    elif runs < DEFAULT_MIN_ANDROID_RUNS:
        failures.append(f"android_report_too_few_runs:{runs}<{DEFAULT_MIN_ANDROID_RUNS}")
    if finite_number(latency.get("mean")):
        mean_latency = float(latency["mean"])
        if mean_latency <= 0:
            failures.append(f"android_report_invalid_mean_latency:{mean_latency:.3f}")
        elif mean_latency > DEFAULT_MAX_ANDROID_MEAN_LATENCY_MS:
            failures.append(
                "android_report_mean_latency_high:"
                f"{mean_latency:.3f}>{DEFAULT_MAX_ANDROID_MEAN_LATENCY_MS:.3f}"
            )
    if finite_number(latency.get("p95")):
        p95_latency = float(latency["p95"])
        if p95_latency <= 0:
            failures.append(f"android_report_invalid_p95_latency:{p95_latency:.3f}")
        elif p95_latency > DEFAULT_MAX_ANDROID_P95_LATENCY_MS:
            failures.append(
                "android_report_p95_latency_high:"
                f"{p95_latency:.3f}>{DEFAULT_MAX_ANDROID_P95_LATENCY_MS:.3f}"
            )
    output = payload.get("output")
    if not isinstance(output, dict):
        failures.append("android_report_missing_output_object")
        output = {}
    if not shape_matches(output.get("shape"), ANDROID_EXPECTED_OUTPUT_SHAPE):
        failures.append(f"android_report_unexpected_output_shape:{output.get('shape') or 'missing'}")
    if not isinstance(output.get("finite"), bool):
        failures.append("android_report_missing_output_finite")
    elif output.get("finite") is not True:
        failures.append("android_report_output_not_finite")
    output_stats = [output.get("min"), output.get("max"), output.get("mean")]
    if not all(finite_number(value) for value in output_stats):
        failures.append("android_report_missing_output_stats")
    else:
        output_min = float(output["min"])
        output_max = float(output["max"])
        output_mean = float(output["mean"])
        if output_min > output_max:
            failures.append("android_report_invalid_output_range")
        else:
            if output_min == output_max:
                failures.append("android_report_degenerate_output_range")
            if output_mean < output_min or output_mean > output_max:
                failures.append("android_report_output_mean_outside_range")
    memory = payload.get("memory_mb")
    if not isinstance(memory, dict) or not finite_number(memory.get("peak")) or float(memory["peak"]) <= 0:
        failures.append("android_report_missing_peak_memory")
    elif float(memory["peak"]) > DEFAULT_MAX_ANDROID_PEAK_MEMORY_MB:
        failures.append(
            "android_report_peak_memory_high:"
            f"{float(memory['peak']):.3f}>{DEFAULT_MAX_ANDROID_PEAK_MEMORY_MB:.3f}"
        )
    return failures


def ar_replay_preflight_failures(path: Path) -> list[str]:
    if not path.is_file() or path.stat().st_size <= 0:
        return []
    failures: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return ["ar_replay_not_utf8"]
    observations = 0
    last_capture_index_by_session: dict[str, tuple[int, str, int]] = {}
    observations_by_frame: dict[tuple[str, int, str], list[tuple[int, dict[str, Any]]]] = {}
    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        observations += 1
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            failures.append(f"ar_replay_invalid_json_line:{line_no}")
            continue
        if not isinstance(payload, dict):
            failures.append(f"ar_replay_line_not_object:{line_no}")
            continue
        if not schema_version_matches(payload.get("schema_version"), AR_REPLAY_EXPECTED_SCHEMA_VERSION):
            failures.append(
                "ar_replay_line_unsupported_schema_version:"
                f"{line_no}:{payload.get('schema_version', 'missing')}"
            )
        for key in ("capture_device", "capture_app_version", "session_id", "frame_id"):
            if placeholder(payload.get(key)):
                failures.append(f"ar_replay_line_missing_{key}:{line_no}")
        if not valid_utc_date(payload.get("capture_date_utc")):
            failures.append(f"ar_replay_line_invalid_capture_date_utc:{line_no}")
        capture_index = payload.get("capture_index")
        if not isinstance(capture_index, int) or isinstance(capture_index, bool):
            failures.append(f"ar_replay_line_missing_capture_index:{line_no}")
        elif capture_index < 0:
            failures.append(f"ar_replay_line_negative_capture_index:{line_no}:{capture_index}")
        else:
            session_id = payload.get("session_id")
            frame_id = payload.get("frame_id")
            if isinstance(session_id, str) and not placeholder(session_id) and isinstance(frame_id, str) and frame_id:
                previous = last_capture_index_by_session.get(session_id)
                if previous is not None:
                    previous_index, previous_frame_id, previous_line = previous
                    if capture_index < previous_index:
                        failures.append(
                            "ar_replay_line_decreasing_capture_index:"
                            f"{line_no}:{capture_index}<{previous_index}:previous_line={previous_line}"
                        )
                    elif capture_index == previous_index and frame_id != previous_frame_id:
                        failures.append(
                            "ar_replay_line_repeated_capture_index_frame_mismatch:"
                            f"{line_no}:{capture_index}:{frame_id}!={previous_frame_id}"
                        )
                last_capture_index_by_session[session_id] = (capture_index, frame_id, line_no)
                observations_by_frame.setdefault((session_id, capture_index, frame_id), []).append((line_no, payload))
        if "wheel_index" in payload and payload["wheel_index"] is not None:
            if not isinstance(payload["wheel_index"], int) or isinstance(payload["wheel_index"], bool) or payload["wheel_index"] < 0:
                failures.append(f"ar_replay_line_invalid_wheel_index:{line_no}")
        if "wheel_track_id" in payload and payload["wheel_track_id"] is not None:
            if not isinstance(payload["wheel_track_id"], str) or placeholder(payload["wheel_track_id"]):
                failures.append(f"ar_replay_line_invalid_wheel_track_id:{line_no}")
        source_type = payload.get("source_type")
        if source_type not in AR_REPLAY_EXPECTED_SOURCE_TYPES:
            failures.append(f"ar_replay_line_invalid_source_type:{line_no}:{source_type or 'missing'}")
        camera_pose_ref = payload.get("camera_pose_ref")
        inline_camera_transform = payload.get("camera_transform")
        has_camera_pose_ref = isinstance(camera_pose_ref, str) and not placeholder(camera_pose_ref)
        has_camera_transform = camera_transform(inline_camera_transform)
        if inline_camera_transform is not None and not has_camera_transform:
            failures.append(f"ar_replay_line_invalid_camera_transform:{line_no}")
        if camera_pose_ref is not None and not isinstance(camera_pose_ref, str):
            failures.append(f"ar_replay_line_invalid_camera_pose_ref:{line_no}")
        if isinstance(camera_pose_ref, str) and placeholder(camera_pose_ref):
            failures.append(f"ar_replay_line_placeholder_camera_pose_ref:{line_no}")
        if has_camera_transform and has_camera_pose_ref:
            failures.append(f"ar_replay_line_camera_pose_conflict:{line_no}")
        if not has_camera_transform and not has_camera_pose_ref:
            failures.append(f"ar_replay_line_missing_camera_pose_evidence:{line_no}")
        screen_points = payload.get("screen_points")
        if not isinstance(screen_points, dict):
            failures.append(f"ar_replay_line_missing_screen_points:{line_no}")
        else:
            for point_name in AR_REPLAY_REQUIRED_SCREEN_POINTS:
                if not point(screen_points.get(point_name), 2):
                    failures.append(f"ar_replay_line_invalid_screen_point_{point_name}:{line_no}")
        floor_hits = payload.get("floor_raycast_hits")
        if not isinstance(floor_hits, dict):
            failures.append(f"ar_replay_line_missing_floor_raycast_hits:{line_no}")
        else:
            for hit_name in AR_REPLAY_REQUIRED_FLOOR_HITS:
                if not point(floor_hits.get(hit_name), 3):
                    failures.append(f"ar_replay_line_invalid_floor_hit_{hit_name}:{line_no}")
        if not isinstance(payload.get("inlier"), bool):
            failures.append(f"ar_replay_line_missing_inlier:{line_no}")
        residual = payload.get("residual")
        if not finite_number(residual):
            failures.append(f"ar_replay_line_missing_residual:{line_no}")
        elif float(residual) < 0:
            failures.append(f"ar_replay_line_negative_residual:{line_no}")
        replay_plane = payload.get("recovered_plane")
        if replay_plane is None:
            failures.append(f"ar_replay_line_missing_recovered_plane:{line_no}")
        elif not recovered_plane(replay_plane):
            failures.append(f"ar_replay_line_invalid_recovered_plane:{line_no}")
        if not point(payload.get("c_plane_hit"), 3):
            failures.append(f"ar_replay_line_missing_c_plane_hit:{line_no}")
        c_height = payload.get("c_height_value")
        if not finite_number(c_height):
            failures.append(f"ar_replay_line_missing_c_height_value:{line_no}")
        elif float(c_height) < 0:
            failures.append(f"ar_replay_line_negative_c_height_value:{line_no}")
        final_position = payload.get("final_disc_bottom_position")
        if final_position is not None and not point(final_position, 3):
            failures.append(f"ar_replay_line_invalid_final_disc_bottom_position:{line_no}")
    for (session_id, capture_index, frame_id), frame_observations in observations_by_frame.items():
        if len(frame_observations) <= 1:
            continue
        identities = [replay_wheel_identity(payload) for _, payload in frame_observations]
        lines = ",".join(str(line_no) for line_no, _ in frame_observations)
        if any(identity is None for identity in identities):
            failures.append(
                "ar_replay_repeated_frame_missing_wheel_identity:"
                f"{session_id}:{frame_id}:{capture_index}:lines={lines}"
            )
            continue
        duplicates = sorted(
            identity for identity in set(identities) if identities.count(identity) > 1
        )
        if duplicates:
            failures.append(
                "ar_replay_repeated_frame_duplicate_wheel_identity:"
                f"{session_id}:{frame_id}:{capture_index}:{','.join(duplicates)}"
            )
    if observations == 0:
        failures.append("ar_replay_no_observations")
    return failures


def build_steps(args: argparse.Namespace) -> list[IntakeStep]:
    return [
        IntakeStep(
            "android_litert_device_validation",
            py_cmd(
                "src/validate_android_litert_report.py",
                "--source",
                str(args.android_litert_source),
                "--out",
                str(args.android_litert_eval),
                "--expected-artifact",
                str(args.expected_android_artifact),
                "--min-runs",
                str(DEFAULT_MIN_ANDROID_RUNS),
                "--max-mean-latency-ms",
                str(DEFAULT_MAX_ANDROID_MEAN_LATENCY_MS),
                "--max-p95-latency-ms",
                str(DEFAULT_MAX_ANDROID_P95_LATENCY_MS),
                "--max-peak-memory-mb",
                str(DEFAULT_MAX_ANDROID_PEAK_MEMORY_MB),
            ),
            input_path=args.android_litert_source,
            preflight_paths=[args.android_litert_source],
            required_input=True,
            expected_artifact=args.expected_android_artifact,
        ),
        IntakeStep(
            "human_labelled_ar_device_holdout",
            py_cmd(
                "src/evaluate_ar_holdout.py",
                "--source-root",
                str(args.ar_holdout_source),
                "--eval-out",
                str(args.ar_holdout_eval),
                "--status-out",
                str(args.ar_holdout_pipeline),
                "--min-map50",
                str(DEFAULT_MIN_AR_HOLDOUT_MAP50),
                "--min-oks",
                str(DEFAULT_MIN_AR_HOLDOUT_OKS),
                "--max-fn",
                str(DEFAULT_MAX_AR_HOLDOUT_FN),
                "--min-images",
                str(DEFAULT_MIN_AR_HOLDOUT_IMAGES),
                "--min-gt-wheels",
                str(DEFAULT_MIN_AR_HOLDOUT_GT_WHEELS),
            ),
            input_path=args.ar_holdout_source,
            preflight_paths=[
                args.ar_holdout_source / "images",
                args.ar_holdout_source / "annotations",
                args.ar_holdout_source / "metadata" / "provenance.json",
            ],
            required_input=True,
        ),
        IntakeStep(
            "ar_3d_replay_validation",
            py_cmd(
                "src/validate_ar_replay.py",
                "--jsonl",
                str(args.ar_replay_jsonl),
                "--out",
                str(args.ar_replay_eval),
                "--min-observations",
                str(DEFAULT_MIN_AR_REPLAY_OBSERVATIONS),
                "--min-sessions",
                str(DEFAULT_MIN_AR_REPLAY_SESSIONS),
                "--min-floor-hit-rate",
                str(DEFAULT_MIN_AR_REPLAY_FLOOR_HIT_RATE),
                "--min-inlier-rate",
                str(DEFAULT_MIN_AR_REPLAY_INLIER_RATE),
                "--max-median-residual",
                str(DEFAULT_MAX_AR_REPLAY_MEDIAN_RESIDUAL),
                "--max-p95-residual",
                str(DEFAULT_MAX_AR_REPLAY_P95_RESIDUAL),
                "--min-final-positions",
                str(DEFAULT_MIN_AR_REPLAY_FINAL_POSITIONS),
            ),
            input_path=args.ar_replay_jsonl,
            preflight_paths=[args.ar_replay_jsonl],
            required_input=True,
        ),
        IntakeStep(
            "production_evidence_audit",
            py_cmd(
                "src/production_evidence_audit.py",
                "--android-litert-source",
                str(args.android_litert_source),
                "--android-litert-eval",
                str(args.android_litert_eval),
                "--ar-holdout-source",
                str(args.ar_holdout_source),
                "--ar-holdout-eval",
                str(args.ar_holdout_eval),
                "--ar-holdout-pipeline",
                str(args.ar_holdout_pipeline),
                "--ar-replay-jsonl",
                str(args.ar_replay_jsonl),
                "--ar-replay-eval",
                str(args.ar_replay_eval),
                "--external-evidence-import-report",
                str(args.evidence_drop_report_out),
                "--expected-android-artifact",
                str(args.expected_android_artifact),
            ),
            production_required=True,
        ),
        IntakeStep(
            "integration_gate",
            py_cmd(
                "src/production_gate.py",
                "--mode",
                "integration",
                "--android-litert-eval",
                str(args.android_litert_eval),
                "--ar-holdout-eval",
                str(args.ar_holdout_eval),
                "--ar-3d-eval",
                str(args.ar_replay_eval),
                "--json-out",
                "outputs/production_audit/integration_gate.json",
            ),
            production_required=False,
        ),
        IntakeStep(
            "production_gate",
            py_cmd(
                "src/production_gate.py",
                "--mode",
                "production",
                "--android-litert-eval",
                str(args.android_litert_eval),
                "--ar-holdout-eval",
                str(args.ar_holdout_eval),
                "--ar-3d-eval",
                str(args.ar_replay_eval),
                "--json-out",
                "outputs/production_audit/production_gate.json",
            ),
            production_required=True,
        ),
        IntakeStep(
            "senior_ml_audit",
            py_cmd(
                "src/senior_ml_audit.py",
                "--android-litert-eval",
                str(args.android_litert_eval),
                "--ar-holdout-eval",
                str(args.ar_holdout_eval),
                "--ar-replay-eval",
                str(args.ar_replay_eval),
                "--production-evidence-audit",
                "outputs/production_audit/production_evidence_audit.json",
                "--integration-gate",
                "outputs/production_audit/integration_gate.json",
                "--production-gate",
                "outputs/production_audit/production_gate.json",
            ),
            production_required=False,
        ),
        IntakeStep(
            "requirements_traceability",
            py_cmd("src/requirements_traceability.py"),
            production_required=False,
        ),
        IntakeStep("executive_report_ru", py_cmd("src/executive_report_ru.py"), production_required=False),
        IntakeStep(
            "objective_completion_audit",
            py_cmd("src/objective_completion_audit.py"),
            production_required=True,
        ),
        IntakeStep("release_integrity", py_cmd("src/release_integrity.py"), production_required=False),
    ]


def run_step(step: IntakeStep) -> IntakeResult:
    missing_input = bool(step.required_input and step.input_path and not step.input_path.exists())
    if missing_input:
        return IntakeResult(
            name=step.name,
            returncode=None,
            ok=False,
            skipped=True,
            missing_input=True,
            input_path=str(step.input_path) if step.input_path else None,
            cmd=step.cmd,
        )
    completed = subprocess.run(step.cmd, check=False)
    return IntakeResult(
        name=step.name,
        returncode=completed.returncode,
        ok=completed.returncode == 0,
        skipped=False,
        missing_input=False,
        input_path=str(step.input_path) if step.input_path else None,
        cmd=step.cmd,
    )


def run_finalization(command: list[str]) -> FinalizationResult:
    completed = subprocess.run(command, check=False)
    return FinalizationResult(
        command=command,
        returncode=completed.returncode,
        ok=completed.returncode == 0,
        skipped=False,
    )


def run_post_finalization_refresh(commands: list[list[str]] | None = None) -> list[RefreshResult]:
    results: list[RefreshResult] = []
    for command in commands or POST_FINALIZATION_REFRESH_COMMANDS:
        completed = subprocess.run(command, check=False)
        results.append(
            RefreshResult(
                name=Path(command[1]).stem if len(command) > 1 else command[0],
                command=command,
                returncode=completed.returncode,
                ok=completed.returncode == 0,
            )
        )
        if completed.returncode != 0:
            break
    return results


def run_evidence_drop_import(args: argparse.Namespace, *, dry_run: bool) -> EvidenceDropImportResult | None:
    if args.evidence_drop is None:
        return None
    report = build_import_report(
        args.evidence_drop,
        dest_root=args.evidence_drop_dest_root,
        dry_run=dry_run,
        overwrite=args.evidence_drop_overwrite,
        expected_android_artifact=args.expected_android_artifact,
    )
    args.evidence_drop_report_out.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_drop_report_out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    file_count = report.get("file_count")
    failures = list(report.get("failures", []))
    if not integer_count(file_count):
        failures.append(f"invalid_import_file_count:{file_count if file_count is not None else 'missing'}")
        file_count = 0
    return EvidenceDropImportResult(
        source=str(args.evidence_drop),
        dest_root=str(args.evidence_drop_dest_root),
        report_out=str(args.evidence_drop_report_out),
        ok=bool(report.get("ok", False)) and not failures,
        dry_run=dry_run,
        file_count=file_count,
        failures=failures,
    )


def evidence_drop_expected_input_paths(dest_root: Path) -> set[str]:
    return {
        path_key(dest_root / "android_litert_device_report.json"),
        path_key(dest_root / "ar_device_holdout" / "images"),
        path_key(dest_root / "ar_device_holdout" / "annotations"),
        path_key(dest_root / "ar_device_holdout" / "metadata" / "provenance.json"),
        path_key(dest_root / "ar_3d_replay" / "ar_replay.jsonl"),
    }


def path_key(path: Path) -> str:
    return path.resolve(strict=False).as_posix()


def evidence_drop_destination_mismatches(
    steps: list[IntakeStep],
    evidence_drop_import: EvidenceDropImportResult | None,
) -> list[dict[str, str]]:
    if evidence_drop_import is None or not evidence_drop_import.ok:
        return []
    expected_paths = evidence_drop_expected_input_paths(Path(evidence_drop_import.dest_root))
    mismatches: list[dict[str, str]] = []
    for step in steps:
        if not step.required_input:
            continue
        for path in step.preflight_paths or ([step.input_path] if step.input_path is not None else []):
            if path_key(path) not in expected_paths:
                mismatches.append(
                    {
                        "path": str(path),
                        "reason": f"evidence_drop_destination_mismatch:{evidence_drop_import.dest_root}",
                        "step": step.name,
                    }
                )
    return mismatches


def fail_import_for_destination_mismatch(
    steps: list[IntakeStep],
    evidence_drop_import: EvidenceDropImportResult | None,
) -> EvidenceDropImportResult | None:
    mismatches = evidence_drop_destination_mismatches(steps, evidence_drop_import)
    if evidence_drop_import is None or not mismatches:
        return evidence_drop_import
    return EvidenceDropImportResult(
        source=evidence_drop_import.source,
        dest_root=evidence_drop_import.dest_root,
        report_out=evidence_drop_import.report_out,
        ok=False,
        dry_run=evidence_drop_import.dry_run,
        file_count=evidence_drop_import.file_count,
        failures=[
            *evidence_drop_import.failures,
            *[f"{item['reason']}:{item['path']}" for item in mismatches],
        ],
    )


def finalization_canonical_path_failures(args: argparse.Namespace) -> list[str]:
    """Return path overrides that would make the final audit suite reread different inputs."""

    expected = {
        "android_litert_source": DEFAULT_ANDROID_LITERT_SOURCE,
        "android_litert_eval": DEFAULT_ANDROID_LITERT_EVAL,
        "ar_holdout_source": DEFAULT_AR_HOLDOUT_SOURCE,
        "ar_holdout_eval": DEFAULT_AR_HOLDOUT_EVAL,
        "ar_holdout_pipeline": DEFAULT_AR_HOLDOUT_PIPELINE,
        "ar_replay_jsonl": DEFAULT_AR_REPLAY_JSONL,
        "ar_replay_eval": DEFAULT_AR_REPLAY_EVAL,
        "evidence_drop_dest_root": DEFAULT_EVIDENCE_DROP_DEST_ROOT,
        "evidence_drop_report_out": DEFAULT_IMPORT_REPORT_OUT,
        "expected_android_artifact": DEFAULT_EXPECTED_ANDROID_ARTIFACT,
    }
    failures: list[str] = []
    for attr, default_path in expected.items():
        actual = getattr(args, attr)
        if path_key(actual) != path_key(default_path):
            failures.append(f"non_canonical_{attr}:{actual}!={default_path}")
    return failures


def build_status(
    results: list[IntakeResult],
    evidence_drop_import: EvidenceDropImportResult | None = None,
    finalization: FinalizationResult | None = None,
    post_finalization_refresh: list[RefreshResult] | None = None,
) -> dict[str, Any]:
    evidence = read_json(Path("outputs/production_audit/production_evidence_audit.json"))
    integration_gate = read_json(Path("outputs/production_audit/integration_gate.json"))
    production_gate = read_json(Path("outputs/production_audit/production_gate.json"))
    objective_audit = read_json(Path("outputs/production_audit/objective_completion_audit.json"))
    production_required_failures = [
        result.name for result in results if not result.ok and result.name != "integration_gate"
    ]
    if evidence_drop_import is not None and not evidence_drop_import.ok:
        production_required_failures.insert(0, "external_evidence_drop_import")
    if finalization is not None and not finalization.ok:
        production_required_failures.append("finalization")
    refresh_results = post_finalization_refresh or []
    if any(not result.ok for result in refresh_results):
        production_required_failures.append("post_finalization_refresh")
    production_ready = bool(evidence.get("production_evidence_ready")) and bool(production_gate.get("ok"))
    objective_complete = bool(objective_audit.get("objective_complete", False))
    finalization_required = not (finalization is not None and finalization.ok)
    return {
        "schema_version": 1,
        "ok": (
            not production_required_failures
            and production_ready
            and objective_complete
            and not finalization_required
        )
        if finalization is not None
        else not production_required_failures and production_ready and objective_complete,
        "production_ready": production_ready,
        "objective_complete": objective_complete,
        "finalization_required": finalization_required,
        "finalization_command": FINALIZATION_COMMAND,
        "finalization_reason": (
            "Run the full production audit suite after intake so final reports, "
            "release integrity, report consistency, and pytest evidence are "
            "refreshed from the accepted external evidence."
        ),
        "finalization": asdict(finalization) if finalization else None,
        "post_finalization_refresh": [asdict(result) for result in refresh_results],
        "objective_failed_requirements": objective_audit.get("failed_requirements", []),
        "production_evidence_ready": bool(evidence.get("production_evidence_ready", False)),
        "integration_gate_ok": bool(integration_gate.get("ok", False)),
        "production_gate_ok": bool(production_gate.get("ok", False)),
        "production_blockers": evidence.get("blockers", []),
        "production_required_failures": production_required_failures,
        "evidence_drop_import": asdict(evidence_drop_import) if evidence_drop_import else None,
        "steps": [asdict(result) for result in results],
    }


def build_preflight_status(
    steps: list[IntakeStep],
    evidence_drop_import: EvidenceDropImportResult | None = None,
) -> dict[str, Any]:
    required_inputs = [
        {
            "step": step.name,
            "path": str(path),
            "present": path.exists(),
            "invalid_reason": preflight_invalid_reason(path),
            "cmd": step.cmd,
        }
        for step in steps
        if step.required_input
        for path in (step.preflight_paths or ([step.input_path] if step.input_path is not None else []))
    ]
    evidence_drop_will_provide_inputs = bool(evidence_drop_import and evidence_drop_import.ok)
    missing = [
        item["path"]
        for item in required_inputs
        if not item["present"] and not evidence_drop_will_provide_inputs
    ]
    invalid = [
        {
            "path": item["path"],
            "reason": item["invalid_reason"],
            "step": item["step"],
        }
        for item in required_inputs
        if item["present"] and item["invalid_reason"] and not evidence_drop_will_provide_inputs
    ]
    invalid.extend(evidence_drop_destination_mismatches(steps, evidence_drop_import))
    if not evidence_drop_will_provide_inputs:
        for step in steps:
            if step.input_path is None:
                continue
            if step.name == "android_litert_device_validation":
                failures = android_report_preflight_failures(
                    step.input_path,
                    expected_artifact=step.expected_artifact,
                )
            elif step.name == "human_labelled_ar_device_holdout":
                failures = [
                    *ar_holdout_pair_failures(step.input_path),
                    *ar_holdout_annotation_failures(step.input_path),
                    *ar_holdout_provenance_failures(step.input_path),
                ]
            elif step.name == "ar_3d_replay_validation":
                failures = ar_replay_preflight_failures(step.input_path)
            else:
                failures = []
            for failure in failures:
                invalid.append(
                    {
                        "path": str(step.input_path),
                        "reason": failure,
                        "step": step.name,
                    }
                )
    return {
        "schema_version": 1,
        "dry_run": True,
        "ok": not missing and not invalid and (evidence_drop_import is None or evidence_drop_import.ok),
        "production_ready": False,
        "finalization_required": True,
        "finalization_command": FINALIZATION_COMMAND,
        "missing_inputs": missing,
        "invalid_inputs": invalid,
        "evidence_drop_import": asdict(evidence_drop_import) if evidence_drop_import else None,
        "required_inputs": required_inputs,
        "planned_steps": [
            {
                "name": step.name,
                "cmd": step.cmd,
                "required_input": step.required_input,
                "input_path": str(step.input_path) if step.input_path else None,
            }
            for step in steps
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--android-litert-source",
        type=Path,
        default=DEFAULT_ANDROID_LITERT_SOURCE,
    )
    parser.add_argument(
        "--android-litert-eval",
        type=Path,
        default=DEFAULT_ANDROID_LITERT_EVAL,
    )
    parser.add_argument(
        "--ar-holdout-source",
        type=Path,
        default=DEFAULT_AR_HOLDOUT_SOURCE,
    )
    parser.add_argument(
        "--ar-holdout-eval",
        type=Path,
        default=DEFAULT_AR_HOLDOUT_EVAL,
    )
    parser.add_argument(
        "--ar-holdout-pipeline",
        type=Path,
        default=DEFAULT_AR_HOLDOUT_PIPELINE,
    )
    parser.add_argument(
        "--ar-replay-jsonl",
        type=Path,
        default=DEFAULT_AR_REPLAY_JSONL,
    )
    parser.add_argument(
        "--ar-replay-eval",
        type=Path,
        default=DEFAULT_AR_REPLAY_EVAL,
    )
    parser.add_argument(
        "--evidence-drop",
        type=Path,
        default=None,
        help="Optional external evidence drop directory or zip. Imported before validation.",
    )
    parser.add_argument(
        "--evidence-drop-dest-root",
        type=Path,
        default=DEFAULT_EVIDENCE_DROP_DEST_ROOT,
        help="Destination root for --evidence-drop import.",
    )
    parser.add_argument(
        "--evidence-drop-report-out",
        type=Path,
        default=DEFAULT_IMPORT_REPORT_OUT,
        help="Import report path used by production evidence custody checks.",
    )
    parser.add_argument(
        "--evidence-drop-overwrite",
        action="store_true",
        help="Allow --evidence-drop import to overwrite existing canonical incoming files.",
    )
    parser.add_argument(
        "--expected-android-artifact",
        type=Path,
        default=DEFAULT_EXPECTED_ANDROID_ARTIFACT,
        help="TFLite artifact whose SHA-256 must match the Android LiteRT evidence report.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only check required input presence and show planned commands; do not run validators or gates.",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help=(
            "After a green intake, run the full production audit suite with pytest "
            "so final reports and release evidence are refreshed in the same command."
        ),
    )
    parser.add_argument("--status-out", type=Path, default=DEFAULT_STATUS_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    steps = build_steps(args)
    evidence_drop_import = run_evidence_drop_import(args, dry_run=args.dry_run)
    evidence_drop_import = fail_import_for_destination_mismatch(steps, evidence_drop_import)
    if args.dry_run:
        status = build_preflight_status(steps, evidence_drop_import=evidence_drop_import)
        write_status(args.status_out, status)
        if evidence_drop_import is not None:
            state = "ok" if evidence_drop_import.ok else f"invalid:{evidence_drop_import.failures}"
            print(
                f"[evidence-intake] evidence_drop_import {state} "
                f"source={evidence_drop_import.source} report={evidence_drop_import.report_out}"
            )
        for item in status["required_inputs"]:
            state = "present" if item["present"] else "missing"
            if item["invalid_reason"]:
                state = f"invalid:{item['invalid_reason']}"
            print(f"[evidence-intake] {state} {item['path']}")
        print(f"[evidence-intake] dry_run=True ok={status['ok']} status={args.status_out}")
        return 0 if status["ok"] else 1

    results: list[IntakeResult] = []
    if evidence_drop_import is not None:
        print(
            f"[evidence-intake] evidence_drop_import ok={evidence_drop_import.ok} "
            f"files={evidence_drop_import.file_count} report={evidence_drop_import.report_out}"
        )
        if not evidence_drop_import.ok:
            status = build_status(results, evidence_drop_import=evidence_drop_import)
            write_status(args.status_out, status)
            print(f"[evidence-intake] status={args.status_out}")
            return 1
    for step in steps:
        print(f"[evidence-intake] step={step.name}")
        if step.input_path is not None:
            print(f"[evidence-intake] input={step.input_path}")
        result = run_step(step)
        results.append(result)
        if result.skipped:
            print(f"[evidence-intake] skipped missing_input={result.input_path}")
        else:
            print(f"[evidence-intake] returncode={result.returncode} ok={result.ok}")

    status = build_status(results, evidence_drop_import=evidence_drop_import)
    finalization: FinalizationResult | None = None
    post_finalization_refresh: list[RefreshResult] = []
    if args.finalize:
        finalization_path_failures = finalization_canonical_path_failures(args)
        if status["ok"]:
            if finalization_path_failures:
                finalization = FinalizationResult(
                    command=FINALIZATION_COMMAND,
                    returncode=None,
                    ok=False,
                    skipped=True,
                    reason=";".join(finalization_path_failures),
                )
                print(
                    "[evidence-intake] finalization skipped: "
                    f"non_canonical_paths={finalization_path_failures}"
                )
            else:
                write_status(args.status_out, status)
                print("[evidence-intake] finalization=" + " ".join(FINALIZATION_COMMAND))
                finalization = run_finalization(FINALIZATION_COMMAND)
                print(
                    "[evidence-intake] "
                    f"finalization_returncode={finalization.returncode} ok={finalization.ok}"
                )
        else:
            finalization = FinalizationResult(
                command=FINALIZATION_COMMAND,
                returncode=None,
                ok=False,
                skipped=True,
                reason="intake_not_green",
            )
            print("[evidence-intake] finalization skipped: intake_not_green")
        status = build_status(
            results,
            evidence_drop_import=evidence_drop_import,
            finalization=finalization,
        )
        if finalization is not None and finalization.ok:
            print("[evidence-intake] post_finalization_refresh")
            post_finalization_refresh = run_post_finalization_refresh(
                POST_FINALIZATION_REPORT_REFRESH_COMMANDS
            )
            for result in post_finalization_refresh:
                print(
                    "[evidence-intake] "
                    f"refresh_step={result.name} returncode={result.returncode} ok={result.ok}"
                )
            status = build_status(
                results,
                evidence_drop_import=evidence_drop_import,
                finalization=finalization,
                post_finalization_refresh=post_finalization_refresh,
            )
            write_status(args.status_out, status)
            if all(result.ok for result in post_finalization_refresh):
                release_refresh = run_post_finalization_refresh(
                    POST_FINALIZATION_RELEASE_REFRESH_COMMANDS
                )
                post_finalization_refresh.extend(release_refresh)
                for result in release_refresh:
                    print(
                        "[evidence-intake] "
                        f"refresh_step={result.name} returncode={result.returncode} ok={result.ok}"
                    )
                status = build_status(
                    results,
                    evidence_drop_import=evidence_drop_import,
                    finalization=finalization,
                    post_finalization_refresh=post_finalization_refresh,
                )
                write_status(args.status_out, status)
            else:
                write_status(args.status_out, status)
        else:
            write_status(args.status_out, status)
    else:
        write_status(args.status_out, status)
    print(
        "[evidence-intake] "
        f"ok={status['ok']} production_ready={status['production_ready']} "
        f"objective_complete={status['objective_complete']} "
        f"production_evidence_ready={status['production_evidence_ready']} "
        f"production_gate_ok={status['production_gate_ok']}"
    )
    print(f"[evidence-intake] status={args.status_out}")
    refresh_ok = all(result.ok for result in post_finalization_refresh)
    return 0 if status["ok"] and refresh_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
