"""Import an Android/AR external evidence drop into data/incoming safely.

Android/AR teams may return either a directory or a zip with the evidence
files produced by the handoff harnesses. This tool accepts both forms,
copies only the expected production-evidence inputs, rejects unsafe archive
paths, and writes a machine-readable import report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_DEST_ROOT = Path("data/incoming")
DEFAULT_REPORT_OUT = Path("outputs/production_audit/external_evidence_drop_import.json")
DEFAULT_EXPECTED_ANDROID_ARTIFACT = Path("outputs/production_audit/tflite_export/best_float32.tflite")
EXPECTED_ANDROID_ARTIFACT_ENTRY = "EXPECTED_ANDROID_ARTIFACT.json"
EXPECTED_ANDROID_ARTIFACT_FORMAT = "tflite_float32"

ANDROID_REPORT = Path("android_litert_device_report.json")
AR_HOLDOUT = Path("ar_device_holdout")
AR_REPLAY = Path("ar_3d_replay/ar_replay.jsonl")
PLACEHOLDER_NAME_MARKERS = (
    ".PLACEHOLDER",
    "PLACE_FRAMES_HERE",
    "PLACE_ANNOTATIONS_HERE",
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ANNOTATION_EXT = ".json"
ZIP_SYMLINK_MODE = 0o120000
ZIP_FILE_TYPE_MASK = 0o170000
MAX_DROP_FILE_BYTES = 64 * 1024 * 1024
MAX_DROP_TOTAL_BYTES = 512 * 1024 * 1024
REPLAY_REQUIRED_SCREEN_POINTS = ("a", "b", "c_disc_bottom")
REPLAY_REQUIRED_FLOOR_HITS = ("a", "b")
ANDROID_PRODUCTION_SOURCE_TYPE = "android_litert_device_validation"
ANDROID_EXPECTED_SCHEMA_VERSION = 1
ANDROID_EXPECTED_INPUT_SHAPE = [1, 640, 640, 3]
ANDROID_EXPECTED_INPUT_DTYPE = "float32"
ANDROID_EXPECTED_INPUT_PROFILE = "zero_float32_smoke"
ANDROID_EXPECTED_OUTPUT_SHAPE = [1, 14, 8400]
ANDROID_MIN_RUNS = 20
ANDROID_MAX_MEAN_LATENCY_MS = 120.0
ANDROID_MAX_P95_LATENCY_MS = 180.0
ANDROID_MAX_PEAK_MEMORY_MB = 512.0
AR_HOLDOUT_PRODUCTION_SOURCE_TYPE = "android_ar_device_human_labelled"
AR_HOLDOUT_EXPECTED_SCHEMA_VERSION = 1
AR_HOLDOUT_LABEL_TYPE = "human_reviewed"
AR_HOLDOUT_ACCEPTED_REVIEW_STATUSES = {"accepted"}
AR_HOLDOUT_MIN_IMAGES = 50
AR_HOLDOUT_MIN_GT_WHEELS = 80
AR_REPLAY_PRODUCTION_SOURCE_TYPE = "android_ar_device_replay"
AR_REPLAY_EXPECTED_SCHEMA_VERSION = 1
AR_REPLAY_MIN_OBSERVATIONS = 30
UTC_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
AR_HOLDOUT_REQUIRED_POINTS = ("a", "b", "c_disc_bottom")
UNIT_NORMAL_TOLERANCE = 0.05


@dataclass(frozen=True)
class DropFile:
    source_name: str
    dest_relative: Path
    data: bytes | None = None
    source_path: Path | None = None

    def read_bytes(self) -> bytes:
        if self.data is not None:
            return self.data
        if self.source_path is None:
            raise ValueError(f"no data source for {self.source_name}")
        return self.source_path.read_bytes()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def schema_version_matches(value: Any, expected: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def ar_holdout_wheel_failures(wheel: Any, *, relative: str, index: int) -> list[str]:
    failures: list[str] = []
    prefix = f"{relative}:wheel[{index}]"
    if not isinstance(wheel, dict):
        return [f"ar_holdout_wheel_not_object:{prefix}"]
    bbox = wheel.get("bbox_xyxy")
    bbox_xyxy: tuple[float, float, float, float] | None = None
    if not _point(bbox, 4):
        failures.append(f"ar_holdout_wheel_invalid_bbox_xyxy:{prefix}")
    else:
        x1, y1, x2, y2 = (float(value) for value in bbox)
        if x2 <= x1 or y2 <= y1:
            failures.append(f"ar_holdout_wheel_invalid_bbox_order:{prefix}")
        else:
            bbox_xyxy = (x1, y1, x2, y2)
    points = wheel.get("points")
    if not isinstance(points, dict):
        failures.append(f"ar_holdout_wheel_missing_points:{prefix}")
    else:
        for name in AR_HOLDOUT_REQUIRED_POINTS:
            point = points.get(name)
            if not _point(point, 2):
                failures.append(f"ar_holdout_wheel_invalid_point_{name}:{prefix}")
            elif bbox_xyxy is not None:
                x1, y1, x2, y2 = bbox_xyxy
                x, y = (float(value) for value in point)
                if x < x1 or x > x2 or y < y1 or y > y2:
                    failures.append(f"ar_holdout_wheel_point_{name}_outside_bbox:{prefix}")
    return failures


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
    return sha256_bytes(canonical.encode("utf-8"))


def safe_posix_path(name: str) -> PurePosixPath | None:
    normalized = name.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    path = PurePosixPath(normalized)
    if path.is_absolute():
        return None
    if any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path


def strip_known_prefix(path: PurePosixPath) -> PurePosixPath:
    parts = path.parts
    prefixes = (
        ("data", "incoming"),
        ("incoming",),
    )
    for prefix in prefixes:
        if parts[: len(prefix)] == prefix:
            return PurePosixPath(*parts[len(prefix) :])
    return path


def destination_for(path: PurePosixPath) -> Path | None:
    stripped = strip_known_prefix(path)
    parts = stripped.parts
    if parts == ("android_litert_device_report.json",):
        return ANDROID_REPORT
    if len(parts) >= 3 and parts[0] == "ar_device_holdout" and parts[1] in {
        "images",
        "annotations",
    }:
        return Path(*parts)
    if parts == ("ar_device_holdout", "metadata", "provenance.json"):
        return Path(*parts)
    if parts == ("ar_3d_replay", "ar_replay.jsonl"):
        return AR_REPLAY
    return None


def destination_candidates(path: PurePosixPath) -> list[PurePosixPath]:
    candidates = [path]
    stripped = strip_known_prefix(path)
    if stripped != path:
        candidates.append(stripped)
    parts = path.parts
    if len(parts) > 1:
        # Common zip shape: evidence_drop/<expected files...>.
        without_root = PurePosixPath(*parts[1:])
        candidates.append(without_root)
        stripped_without_root = strip_known_prefix(without_root)
        if stripped_without_root != without_root:
            candidates.append(stripped_without_root)
    unique: list[PurePosixPath] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.as_posix()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def destination_for_drop_path(path: PurePosixPath) -> Path | None:
    for candidate in destination_candidates(path):
        dest = destination_for(candidate)
        if dest is not None:
            return dest
    return None


def is_expected_android_artifact_metadata_path(path: PurePosixPath) -> bool:
    return any(
        candidate.as_posix() == EXPECTED_ANDROID_ARTIFACT_ENTRY
        for candidate in destination_candidates(path)
    )


def collect_from_dir(source: Path) -> tuple[list[DropFile], list[str], list[str]]:
    files: list[DropFile] = []
    ignored: list[str] = []
    failures: list[str] = []
    if not source.is_dir():
        return files, ignored, [f"source_not_directory:{source}"]
    for path in sorted(p for p in source.rglob("*") if p.is_file() or p.is_symlink()):
        relative = path.relative_to(source)
        if path.is_symlink():
            failures.append(f"symlink_not_allowed:{relative.as_posix()}")
            continue
        safe = safe_posix_path(relative.as_posix())
        if safe is None:
            failures.append(f"unsafe_path:{relative.as_posix()}")
            continue
        dest = destination_for_drop_path(safe)
        if dest is None:
            ignored.append(relative.as_posix())
            continue
        files.append(DropFile(source_name=relative.as_posix(), dest_relative=dest, source_path=path))
    return files, ignored, failures


def zip_entry_guard_failures(info: zipfile.ZipInfo, total_uncompressed: int) -> list[str]:
    failures: list[str] = []
    mode = (info.external_attr >> 16) & ZIP_FILE_TYPE_MASK
    if mode == ZIP_SYMLINK_MODE:
        failures.append(f"zip_symlink_not_allowed:{info.filename}")
    if info.file_size > MAX_DROP_FILE_BYTES:
        failures.append(
            f"zip_entry_too_large:{info.filename}:{info.file_size}>{MAX_DROP_FILE_BYTES}"
        )
    if total_uncompressed + info.file_size > MAX_DROP_TOTAL_BYTES:
        failures.append(
            "zip_total_uncompressed_too_large:"
            f"{total_uncompressed + info.file_size}>{MAX_DROP_TOTAL_BYTES}"
        )
    return failures


def collect_from_zip(source: Path) -> tuple[list[DropFile], list[str], list[str]]:
    files: list[DropFile] = []
    ignored: list[str] = []
    failures: list[str] = []
    try:
        with zipfile.ZipFile(source) as zf:
            total_uncompressed = 0
            for info in sorted(zf.infolist(), key=lambda item: item.filename):
                if info.is_dir():
                    continue
                safe = safe_posix_path(info.filename)
                if safe is None:
                    failures.append(f"unsafe_zip_entry:{info.filename}")
                    continue
                guard_failures = zip_entry_guard_failures(info, total_uncompressed)
                if guard_failures:
                    failures.extend(guard_failures)
                    continue
                total_uncompressed += info.file_size
                dest = destination_for_drop_path(safe)
                if dest is None:
                    ignored.append(info.filename)
                    continue
                files.append(
                    DropFile(
                        source_name=info.filename,
                        dest_relative=dest,
                        data=zf.read(info),
                    )
                )
    except zipfile.BadZipFile:
        failures.append(f"bad_zip_file:{source}")
    return files, ignored, failures


def collect_drop_files(source: Path) -> tuple[list[DropFile], list[str], list[str]]:
    if source.is_dir():
        return collect_from_dir(source)
    if source.is_file():
        return collect_from_zip(source)
    return [], [], [f"missing_source:{source}"]


def required_failures(files: list[DropFile]) -> list[str]:
    dests = {file.dest_relative.as_posix() for file in files}
    image_stems = {
        Path(dest).stem for dest in dests if dest.startswith("ar_device_holdout/images/")
    }
    annotation_stems = {
        Path(dest).stem for dest in dests if dest.startswith("ar_device_holdout/annotations/")
    }
    image_count = len(image_stems)
    annotation_count = len(annotation_stems)
    checks = [
        ("android_litert_device_report", ANDROID_REPORT.as_posix() in dests),
        ("ar_holdout_images", image_count > 0),
        ("ar_holdout_annotations", annotation_count > 0),
        (
            "ar_holdout_provenance",
            "ar_device_holdout/metadata/provenance.json" in dests,
        ),
        ("ar_replay_jsonl", AR_REPLAY.as_posix() in dests),
    ]
    failures = [f"missing_required:{name}" for name, ok in checks if not ok]
    if image_count != annotation_count:
        failures.append(f"ar_holdout_count_mismatch:{image_count}!={annotation_count}")
    if image_count < AR_HOLDOUT_MIN_IMAGES:
        failures.append(f"ar_holdout_too_few_images:{image_count}<{AR_HOLDOUT_MIN_IMAGES}")
    if annotation_count < AR_HOLDOUT_MIN_IMAGES:
        failures.append(f"ar_holdout_too_few_annotations:{annotation_count}<{AR_HOLDOUT_MIN_IMAGES}")
    missing_annotations = sorted(image_stems - annotation_stems)
    missing_images = sorted(annotation_stems - image_stems)
    if missing_annotations:
        failures.append(f"ar_holdout_missing_annotations:{','.join(missing_annotations[:20])}")
    if missing_images:
        failures.append(f"ar_holdout_missing_images:{','.join(missing_images[:20])}")
    return failures


def ar_holdout_contract_failures(files: list[DropFile]) -> list[str]:
    image_files = [
        file
        for file in files
        if file.dest_relative.as_posix().startswith("ar_device_holdout/images/")
    ]
    annotation_files = [
        file
        for file in files
        if file.dest_relative.as_posix().startswith("ar_device_holdout/annotations/")
    ]
    valid_image_names = {
        file.dest_relative.name
        for file in image_files
        if file.dest_relative.suffix.lower() in IMAGE_EXTS
    }
    failures: list[str] = []
    gt_wheels = 0
    bad_image_exts = sorted(
        file.dest_relative.as_posix()
        for file in image_files
        if file.dest_relative.suffix.lower() not in IMAGE_EXTS
    )
    bad_annotation_exts = sorted(
        file.dest_relative.as_posix()
        for file in annotation_files
        if file.dest_relative.suffix.lower() != ANNOTATION_EXT
    )
    if bad_image_exts:
        failures.append(f"ar_holdout_bad_image_extensions:{','.join(bad_image_exts[:20])}")
    if bad_annotation_exts:
        failures.append(f"ar_holdout_bad_annotation_extensions:{','.join(bad_annotation_exts[:20])}")

    for file in annotation_files:
        if file.dest_relative.suffix.lower() != ANNOTATION_EXT:
            continue
        relative = file.dest_relative.relative_to(AR_HOLDOUT / "annotations").as_posix()
        try:
            payload = json.loads(file.read_bytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            failures.append(f"ar_holdout_invalid_annotation_json:{relative}")
            continue
        if not isinstance(payload, dict):
            failures.append(f"ar_holdout_annotation_not_object:{relative}")
            continue
        if not schema_version_matches(payload.get("schema_version"), AR_HOLDOUT_EXPECTED_SCHEMA_VERSION):
            failures.append(
                "ar_holdout_annotation_unsupported_schema_version:"
                f"{relative}:{payload.get('schema_version', 'missing')}"
            )
        frame_id = payload.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id:
            failures.append(f"ar_holdout_annotation_missing_frame_id:{relative}")
        elif frame_id != file.dest_relative.stem:
            failures.append(f"ar_holdout_annotation_frame_id_mismatch:{relative}:{frame_id}")
        image_name = payload.get("image")
        if not isinstance(image_name, str) or not image_name:
            failures.append(f"ar_holdout_annotation_missing_image:{relative}")
        elif Path(image_name).name != image_name:
            failures.append(f"ar_holdout_annotation_image_not_filename:{relative}:{image_name}")
        else:
            if image_name not in valid_image_names:
                failures.append(f"ar_holdout_annotation_image_missing:{relative}:{image_name}")
            if Path(image_name).stem != file.dest_relative.stem:
                failures.append(f"ar_holdout_annotation_image_stem_mismatch:{relative}:{image_name}")
        wheels = payload.get("wheels")
        if not isinstance(wheels, list):
            failures.append(f"ar_holdout_annotation_wheels_not_array:{relative}")
        else:
            for index, wheel in enumerate(wheels):
                wheel_failures = ar_holdout_wheel_failures(wheel, relative=relative, index=index)
                if wheel_failures:
                    failures.extend(wheel_failures)
                else:
                    gt_wheels += 1
    if gt_wheels < AR_HOLDOUT_MIN_GT_WHEELS:
        failures.append(f"ar_holdout_too_few_gt_wheels:{gt_wheels}<{AR_HOLDOUT_MIN_GT_WHEELS}")

    provenance = next(
        (
            file
            for file in files
            if file.dest_relative.as_posix() == "ar_device_holdout/metadata/provenance.json"
        ),
        None,
    )
    if provenance is not None:
        payload = _json_object(provenance.read_bytes())
        if payload is None:
            failures.append("ar_holdout_invalid_provenance_json")
        else:
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
                if _placeholder(value):
                    failures.append(f"ar_holdout_missing_{key}")
            if _placeholder(capture_date_utc) or not valid_utc_date(capture_date_utc):
                failures.append("ar_holdout_invalid_capture_date_utc")
            if (
                isinstance(annotator, str)
                and isinstance(reviewer, str)
                and annotator.strip()
                and annotator.strip() == reviewer.strip()
            ):
                failures.append("ar_holdout_annotator_reviewer_not_independent")
    return failures


def placeholder_failures(ignored: list[str]) -> list[str]:
    failures: list[str] = []
    for name in ignored:
        if any(marker in Path(name).name for marker in PLACEHOLDER_NAME_MARKERS):
            failures.append(f"placeholder_file_not_allowed:{name}")
    return failures


def empty_file_failures(files: list[DropFile]) -> list[str]:
    failures: list[str] = []
    for file in files:
        size = len(file.read_bytes()) if file.data is not None else file.source_path.stat().st_size
        if size <= 0:
            failures.append(f"empty_file:{file.dest_relative.as_posix()}")
    return failures


def duplicate_failures(files: list[DropFile]) -> list[str]:
    counts: dict[str, int] = {}
    for file in files:
        key = file.dest_relative.as_posix()
        counts[key] = counts.get(key, 0) + 1
    return [f"duplicate_destination:{path}" for path, count in sorted(counts.items()) if count > 1]


def _json_object(data: bytes) -> dict[str, Any] | None:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _expected_android_artifact_metadata_records(
    source: Path,
) -> tuple[list[tuple[str, dict[str, Any]]], list[str]]:
    records: list[tuple[str, dict[str, Any]]] = []
    failures: list[str] = []
    if source.is_dir():
        paths = sorted(path for path in source.rglob("*") if path.is_file() and not path.is_symlink())
        for path in paths:
            relative = path.relative_to(source)
            safe = safe_posix_path(relative.as_posix())
            if safe is None or not is_expected_android_artifact_metadata_path(safe):
                continue
            if path.stat().st_size > MAX_DROP_FILE_BYTES:
                failures.append(
                    "expected_android_artifact_metadata_too_large:"
                    f"{relative.as_posix()}:{path.stat().st_size}>{MAX_DROP_FILE_BYTES}"
                )
                continue
            payload = _json_object(path.read_bytes())
            if payload is None:
                failures.append(f"expected_android_artifact_metadata_invalid_json:{relative.as_posix()}")
            else:
                records.append((relative.as_posix(), payload))
        return records, failures
    if not source.is_file():
        return records, failures
    try:
        with zipfile.ZipFile(source) as zf:
            for info in sorted(zf.infolist(), key=lambda item: item.filename):
                if info.is_dir():
                    continue
                safe = safe_posix_path(info.filename)
                if safe is None or not is_expected_android_artifact_metadata_path(safe):
                    continue
                if info.file_size > MAX_DROP_FILE_BYTES:
                    failures.append(
                        "expected_android_artifact_metadata_too_large:"
                        f"{info.filename}:{info.file_size}>{MAX_DROP_FILE_BYTES}"
                    )
                    continue
                payload = _json_object(zf.read(info))
                if payload is None:
                    failures.append(f"expected_android_artifact_metadata_invalid_json:{info.filename}")
                else:
                    records.append((info.filename, payload))
    except zipfile.BadZipFile:
        pass
    return records, failures


def expected_android_artifact_metadata_failures(
    source: Path,
    *,
    expected_artifact: Path | None,
    records: list[tuple[str, dict[str, Any]]] | None = None,
    read_failures: list[str] | None = None,
) -> list[str]:
    if records is None:
        records, discovered_failures = _expected_android_artifact_metadata_records(source)
        failures = discovered_failures
    else:
        failures = list(read_failures or [])
    if not records:
        return failures
    if len(records) > 1:
        failures.append(
            "expected_android_artifact_metadata_duplicate:"
            + ",".join(name for name, _ in records[:20])
        )
    expected_path = str(expected_artifact).replace("\\", "/") if expected_artifact is not None else None
    expected_sha = sha256_file(expected_artifact) if expected_artifact is not None else None
    for name, payload in records:
        if not schema_version_matches(payload.get("schema_version"), 1):
            failures.append(
                "expected_android_artifact_metadata_unsupported_schema_version:"
                f"{name}:{payload.get('schema_version', 'missing')}"
            )
        artifact = payload.get("expected_android_artifact")
        if not isinstance(artifact, dict):
            failures.append(f"expected_android_artifact_metadata_missing_expected_android_artifact:{name}")
            continue
        artifact_path = artifact.get("path")
        artifact_sha = artifact.get("sha256")
        artifact_format = artifact.get("format")
        if expected_path is not None and artifact_path != expected_path:
            failures.append(
                "expected_android_artifact_metadata_path_mismatch:"
                f"{name}:{artifact_path or 'missing'}"
            )
        if artifact_format != EXPECTED_ANDROID_ARTIFACT_FORMAT:
            failures.append(
                "expected_android_artifact_metadata_format_mismatch:"
                f"{name}:{artifact_format or 'missing'}"
            )
        if expected_artifact is not None and expected_sha is None:
            failures.append(f"expected_android_artifact_metadata_missing_expected_artifact:{expected_artifact}")
        elif expected_sha is not None and artifact_sha != expected_sha:
            failures.append(
                "expected_android_artifact_metadata_sha256_mismatch:"
                f"{name}:{artifact_sha or 'missing'}"
            )
    return failures


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_number(value: Any) -> bool:
    return _number(value) and math.isfinite(float(value))


def _integer_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _shape_matches(value: Any, expected: list[int]) -> bool:
    return (
        isinstance(value, list)
        and len(value) == len(expected)
        and all(_integer_count(item) for item in value)
        and value == expected
    )


def _placeholder(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    normalized = value.strip().lower()
    return not normalized or "fill_me" in normalized or normalized in {"todo", "tbd", "unknown"}


def _point(value: Any, dims: int) -> bool:
    return isinstance(value, list) and len(value) == dims and all(_finite_number(item) for item in value)


def _unit_vector3(value: Any) -> bool:
    if not _point(value, 3):
        return False
    norm = math.sqrt(sum(float(item) * float(item) for item in value))
    return abs(norm - 1.0) <= UNIT_NORMAL_TOLERANCE


def _camera_transform(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    rotation = value.get("R")
    translation = value.get("t")
    return (
        isinstance(rotation, list)
        and len(rotation) == 3
        and all(
            isinstance(row, list)
            and len(row) == 3
            and all(_finite_number(item) for item in row)
            for row in rotation
        )
        and _point(translation, 3)
    )


def _recovered_plane(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    support = value.get("support")
    return (
        _unit_vector3(value.get("normal"))
        and _point(value.get("point"), 3)
        and isinstance(support, int)
        and not isinstance(support, bool)
        and support > 0
    )


def _replay_wheel_identity(payload: dict[str, Any]) -> str | None:
    wheel_index = payload.get("wheel_index")
    if isinstance(wheel_index, int) and not isinstance(wheel_index, bool) and wheel_index >= 0:
        return f"wheel_index:{wheel_index}"
    wheel_track_id = payload.get("wheel_track_id")
    if isinstance(wheel_track_id, str) and not _placeholder(wheel_track_id):
        return f"wheel_track_id:{wheel_track_id.strip()}"
    return None


def android_report_contract_failures(
    files: list[DropFile],
    *,
    expected_artifact: Path | None = None,
) -> list[str]:
    report = next((file for file in files if file.dest_relative == ANDROID_REPORT), None)
    if report is None:
        return []
    payload = _json_object(report.read_bytes())
    if payload is None:
        return ["android_report_invalid_json_object"]

    failures: list[str] = []
    if not schema_version_matches(payload.get("schema_version"), ANDROID_EXPECTED_SCHEMA_VERSION):
        failures.append(
            "android_report_unsupported_schema_version:"
            f"{payload.get('schema_version', 'missing')}"
        )
    if payload.get("source_type") != ANDROID_PRODUCTION_SOURCE_TYPE:
        failures.append(
            "android_report_invalid_source_type:"
            f"{payload.get('source_type') or 'missing'}"
        )
    if _placeholder(payload.get("test_session_id")):
        failures.append("android_report_missing_test_session_id")
    if _placeholder(payload.get("test_app_version")):
        failures.append("android_report_missing_test_app_version")
    if not valid_utc_date(payload.get("test_date_utc")):
        failures.append("android_report_invalid_test_date_utc")
    device = payload.get("device")
    if not isinstance(device, dict):
        failures.append("android_report_missing_device_object")
        device = {}
    for key in ("model", "manufacturer", "android_version", "soc"):
        if _placeholder(device.get(key)):
            failures.append(f"android_report_missing_device_{key}")
    if device.get("is_emulator") is not False:
        failures.append(f"android_report_device_must_be_physical:{device.get('is_emulator', 'missing')}")
    runtime = str(payload.get("runtime", "")).lower()
    if runtime not in {"litert", "ai_edge_litert", "tensorflow_lite"}:
        failures.append(f"android_report_unsupported_runtime:{payload.get('runtime') or 'missing'}")

    artifact = payload.get("artifact")
    if not isinstance(artifact, dict):
        failures.append("android_report_missing_artifact_object")
        artifact = {}
    if _placeholder(artifact.get("sha256")):
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
    if not _shape_matches(input_meta.get("shape"), ANDROID_EXPECTED_INPUT_SHAPE):
        failures.append(f"android_report_unexpected_input_shape:{input_meta.get('shape') or 'missing'}")
    if str(input_meta.get("dtype", "")).strip().lower() != ANDROID_EXPECTED_INPUT_DTYPE:
        failures.append(f"android_report_unexpected_input_dtype:{input_meta.get('dtype') or 'missing'}")
    if input_meta.get("profile") != ANDROID_EXPECTED_INPUT_PROFILE:
        failures.append(f"android_report_unexpected_input_profile:{input_meta.get('profile') or 'missing'}")

    latency = payload.get("latency_ms")
    if not isinstance(latency, dict):
        failures.append("android_report_missing_latency_ms_object")
        latency = {}
    for key in ("runs", "mean", "p95"):
        if not _finite_number(latency.get(key)):
            failures.append(f"android_report_missing_latency_{key}")
    if _finite_number(latency.get("runs")) and not _integer_count(latency.get("runs")):
        failures.append("android_report_invalid_latency_runs")
    elif _integer_count(latency.get("runs")) and latency["runs"] < ANDROID_MIN_RUNS:
        failures.append(f"android_report_too_few_runs:{int(latency['runs'])}<{ANDROID_MIN_RUNS}")
    if _finite_number(latency.get("mean")):
        mean_latency = float(latency["mean"])
        if mean_latency <= 0:
            failures.append(f"android_report_invalid_mean_latency:{mean_latency:.3f}")
        elif mean_latency > ANDROID_MAX_MEAN_LATENCY_MS:
            failures.append(
                "android_report_mean_latency_high:"
                f"{mean_latency:.3f}>{ANDROID_MAX_MEAN_LATENCY_MS:.3f}"
            )
    if _finite_number(latency.get("p95")):
        p95_latency = float(latency["p95"])
        if p95_latency <= 0:
            failures.append(f"android_report_invalid_p95_latency:{p95_latency:.3f}")
        elif p95_latency > ANDROID_MAX_P95_LATENCY_MS:
            failures.append(
                "android_report_p95_latency_high:"
                f"{p95_latency:.3f}>{ANDROID_MAX_P95_LATENCY_MS:.3f}"
            )
    output = payload.get("output")
    if not isinstance(output, dict):
        failures.append("android_report_missing_output_object")
        output = {}
    if not _shape_matches(output.get("shape"), ANDROID_EXPECTED_OUTPUT_SHAPE):
        failures.append(f"android_report_unexpected_output_shape:{output.get('shape') or 'missing'}")
    if not isinstance(output.get("finite"), bool):
        failures.append("android_report_missing_output_finite")
    elif output.get("finite") is not True:
        failures.append("android_report_output_not_finite")
    output_stats = [output.get("min"), output.get("max"), output.get("mean")]
    if not all(_finite_number(value) for value in output_stats):
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
    if not isinstance(memory, dict) or not _finite_number(memory.get("peak")) or float(memory["peak"]) <= 0:
        failures.append("android_report_missing_peak_memory")
    elif float(memory["peak"]) > ANDROID_MAX_PEAK_MEMORY_MB:
        failures.append(f"android_report_peak_memory_high:{float(memory['peak']):.3f}>{ANDROID_MAX_PEAK_MEMORY_MB:.3f}")
    return failures


def ar_replay_contract_failures(files: list[DropFile]) -> list[str]:
    replay = next((file for file in files if file.dest_relative == AR_REPLAY), None)
    if replay is None:
        return []
    failures: list[str] = []
    try:
        lines = replay.read_bytes().decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return ["ar_replay_not_utf8"]

    non_empty_lines = 0
    complete_floor_hits = 0
    ransac_labelled = 0
    residuals = 0
    recovered_planes = 0
    c_plane_hits = 0
    c_height_values = 0
    final_positions = 0
    last_capture_index_by_session: dict[str, tuple[int, str, int]] = {}
    observations_by_frame: dict[tuple[str, int, str], list[tuple[int, dict[str, Any]]]] = {}
    for line_no, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        non_empty_lines += 1
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
        for key in ("source_type", "capture_device", "capture_app_version", "session_id", "frame_id"):
            if not isinstance(payload.get(key), str) or not payload.get(key):
                failures.append(f"ar_replay_line_missing_{key}:{line_no}")
        if payload.get("source_type") != AR_REPLAY_PRODUCTION_SOURCE_TYPE:
            failures.append(f"ar_replay_line_invalid_source_type:{line_no}:{payload.get('source_type') or 'missing'}")
        if _placeholder(payload.get("capture_device")):
            failures.append(f"ar_replay_line_missing_capture_device:{line_no}")
        if _placeholder(payload.get("capture_app_version")):
            failures.append(f"ar_replay_line_missing_capture_app_version:{line_no}")
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
            if isinstance(session_id, str) and not _placeholder(session_id) and isinstance(frame_id, str) and frame_id:
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
            if not isinstance(payload["wheel_track_id"], str) or _placeholder(payload["wheel_track_id"]):
                failures.append(f"ar_replay_line_invalid_wheel_track_id:{line_no}")
        camera_transform = payload.get("camera_transform")
        camera_pose_ref = payload.get("camera_pose_ref")
        has_camera_transform = _camera_transform(camera_transform)
        has_camera_pose_ref = isinstance(camera_pose_ref, str) and not _placeholder(camera_pose_ref)
        if camera_transform is not None and not isinstance(camera_transform, dict):
            failures.append(f"ar_replay_line_invalid_camera_transform:{line_no}")
        elif isinstance(camera_transform, dict) and not _camera_transform(camera_transform):
            failures.append(f"ar_replay_line_invalid_camera_transform:{line_no}")
        if camera_pose_ref is not None and not isinstance(camera_pose_ref, str):
            failures.append(f"ar_replay_line_invalid_camera_pose_ref:{line_no}")
        if isinstance(camera_pose_ref, str) and _placeholder(camera_pose_ref):
            failures.append(f"ar_replay_line_placeholder_camera_pose_ref:{line_no}")
        if has_camera_transform and has_camera_pose_ref:
            failures.append(f"ar_replay_line_camera_pose_conflict:{line_no}")
        if not has_camera_transform and not has_camera_pose_ref:
            failures.append(f"ar_replay_line_missing_camera_pose_evidence:{line_no}")
        screen_points = payload.get("screen_points")
        if not isinstance(screen_points, dict):
            failures.append(f"ar_replay_line_missing_screen_points:{line_no}")
        else:
            for name in REPLAY_REQUIRED_SCREEN_POINTS:
                if not _point(screen_points.get(name), 2):
                    failures.append(f"ar_replay_line_invalid_screen_point_{name}:{line_no}")
        floor_hits = payload.get("floor_raycast_hits")
        if not isinstance(floor_hits, dict):
            failures.append(f"ar_replay_line_missing_floor_raycast_hits:{line_no}")
        else:
            line_complete_floor_hits = True
            for name in REPLAY_REQUIRED_FLOOR_HITS:
                if not _point(floor_hits.get(name), 3):
                    failures.append(f"ar_replay_line_invalid_floor_hit_{name}:{line_no}")
                    line_complete_floor_hits = False
            if line_complete_floor_hits:
                complete_floor_hits += 1
        if not isinstance(payload.get("inlier"), bool):
            failures.append(f"ar_replay_line_missing_inlier:{line_no}")
        else:
            ransac_labelled += 1
        if not _finite_number(payload.get("residual")):
            failures.append(f"ar_replay_line_missing_residual:{line_no}")
        elif float(payload["residual"]) < 0:
            failures.append(f"ar_replay_line_negative_residual:{line_no}")
        else:
            residuals += 1
        recovered_plane = payload.get("recovered_plane")
        if recovered_plane is None:
            failures.append(f"ar_replay_line_missing_recovered_plane:{line_no}")
        elif not _recovered_plane(recovered_plane):
            failures.append(f"ar_replay_line_invalid_recovered_plane:{line_no}")
        else:
            recovered_planes += 1
        if not _point(payload.get("c_plane_hit"), 3):
            failures.append(f"ar_replay_line_missing_c_plane_hit:{line_no}")
        else:
            c_plane_hits += 1
        if not _finite_number(payload.get("c_height_value")):
            failures.append(f"ar_replay_line_missing_c_height_value:{line_no}")
        elif float(payload["c_height_value"]) < 0:
            failures.append(f"ar_replay_line_negative_c_height_value:{line_no}")
        else:
            c_height_values += 1
        if payload.get("final_disc_bottom_position") is not None:
            if _point(payload.get("final_disc_bottom_position"), 3):
                final_positions += 1
            else:
                failures.append(f"ar_replay_line_invalid_final_disc_bottom_position:{line_no}")
    for (session_id, capture_index, frame_id), frame_observations in observations_by_frame.items():
        if len(frame_observations) <= 1:
            continue
        identities = [_replay_wheel_identity(payload) for _, payload in frame_observations]
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
    if non_empty_lines == 0:
        failures.append("ar_replay_no_observations")
    if non_empty_lines < AR_REPLAY_MIN_OBSERVATIONS:
        failures.append(f"ar_replay_too_few_observations:{non_empty_lines}<{AR_REPLAY_MIN_OBSERVATIONS}")
    if complete_floor_hits < non_empty_lines:
        failures.append(f"ar_replay_incomplete_floor_hits:{complete_floor_hits}!={non_empty_lines}")
    if ransac_labelled < non_empty_lines:
        failures.append(f"ar_replay_missing_ransac_labels:{ransac_labelled}!={non_empty_lines}")
    if residuals < non_empty_lines:
        failures.append(f"ar_replay_missing_residuals:{residuals}!={non_empty_lines}")
    if recovered_planes < non_empty_lines:
        failures.append(f"ar_replay_missing_recovered_planes:{recovered_planes}!={non_empty_lines}")
    if c_plane_hits < non_empty_lines:
        failures.append(f"ar_replay_missing_c_plane_hits:{c_plane_hits}!={non_empty_lines}")
    if c_height_values < non_empty_lines:
        failures.append(f"ar_replay_missing_c_height_values:{c_height_values}!={non_empty_lines}")
    if final_positions < 1:
        failures.append("ar_replay_missing_final_disc_bottom_position")
    return failures


def build_import_report(
    source: Path,
    *,
    dest_root: Path,
    dry_run: bool,
    overwrite: bool,
    expected_android_artifact: Path | None = None,
) -> dict[str, Any]:
    files, ignored, failures = collect_drop_files(source)
    failures.extend(duplicate_failures(files))
    failures.extend(required_failures(files))
    failures.extend(empty_file_failures(files))
    failures.extend(android_report_contract_failures(files, expected_artifact=expected_android_artifact))
    failures.extend(ar_holdout_contract_failures(files))
    failures.extend(ar_replay_contract_failures(files))
    failures.extend(placeholder_failures(ignored))
    metadata_records, metadata_failures = _expected_android_artifact_metadata_records(source)
    failures.extend(
        expected_android_artifact_metadata_failures(
            source,
            expected_artifact=expected_android_artifact,
            records=metadata_records,
            read_failures=metadata_failures,
        )
    )

    copied: list[str] = []
    copied_artifacts: list[dict[str, Any]] = []
    planned = [
        {
            "source": file.source_name,
            "dest": str(dest_root / file.dest_relative),
            "size_bytes": len(file.read_bytes()) if file.data is not None else file.source_path.stat().st_size,
            "sha256": sha256_bytes(file.read_bytes()),
        }
        for file in files
    ]

    for file in files:
        dest = dest_root / file.dest_relative
        if dest.exists() and not overwrite:
            failures.append(f"would_overwrite:{dest}")

    ok = not failures
    if ok and not dry_run:
        for file in files:
            dest = dest_root / file.dest_relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(file.read_bytes())
            copied.append(str(dest))
            copied_artifacts.append(
                {
                    "dest": str(dest),
                    "size_bytes": dest.stat().st_size,
                    "sha256": sha256_file(dest),
                }
            )

    return {
        "schema_version": 1,
        "ok": ok,
        "dry_run": dry_run,
        "source": str(source),
        "source_kind": "directory" if source.is_dir() else "zip" if source.is_file() else "missing",
        "source_sha256": sha256_file(source) if source.is_file() else None,
        "dest_root": str(dest_root),
        "overwrite": overwrite,
        "expected_android_artifact": str(expected_android_artifact) if expected_android_artifact else None,
        "expected_android_artifact_sha256": (
            sha256_file(expected_android_artifact) if expected_android_artifact else None
        ),
        "expected_android_artifact_metadata_count": len(metadata_records),
        "file_count": len(files),
        "evidence_manifest_sha256": manifest_sha256(planned),
        "ignored_count": len(ignored),
        "failures": failures,
        "ignored": ignored[:200],
        "planned": planned,
        "copied": copied,
        "copied_artifacts": copied_artifacts,
        "next_commands": [
            "./.venv/bin/python src/run_production_evidence_intake.py --dry-run",
            "./.venv/bin/python src/run_production_evidence_intake.py",
            "./.venv/bin/python src/run_production_evidence_intake.py --finalize",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Evidence drop directory or zip file.")
    parser.add_argument("--dest-root", type=Path, default=DEFAULT_DEST_ROOT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--expected-android-artifact", type=Path, default=DEFAULT_EXPECTED_ANDROID_ARTIFACT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_import_report(
        args.source,
        dest_root=args.dest_root,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        expected_android_artifact=args.expected_android_artifact,
    )
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if report["ok"] and not args.dry_run:
        print(f"ok=True copied={len(report['copied'])} report={args.report_out}")
    else:
        print(
            f"ok={report['ok']} dry_run={args.dry_run} "
            f"files={report['file_count']} failures={report['failures']} report={args.report_out}"
        )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
