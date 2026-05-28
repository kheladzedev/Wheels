"""Validate AR-side raycast/RANSAC replay logs for production gating.

The ML model only emits 2D keypoints. Production readiness still needs an
AR-device replay report proving those 2D points survive the downstream
floor-raycast, RANSAC plane fit, and disc-bottom reconstruction flow.

Input is JSONL following docs/AR_MOCK_LOG_CONTRACT.md: one wheel
observation per line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import median
from typing import Any


SCREEN_POINT_NAMES = ("a", "b", "c_disc_bottom")
FLOOR_HIT_NAMES = ("a", "b")
PRODUCTION_SOURCE_TYPES = {
    "android_ar_device_replay",
    "ios_ar_device_replay",
    "ar_device_replay",
}
EXPECTED_OBSERVATION_SCHEMA_VERSION = 1
UTC_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
UNIT_NORMAL_TOLERANCE = 0.05


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ReplayThresholds:
    min_observations: int = 30
    min_sessions: int = 1
    require_production_source: bool = True
    min_floor_hit_rate: float = 0.90
    require_ransac: bool = True
    min_inlier_rate: float = 0.70
    max_median_residual: float = 0.02
    max_p95_residual: float = 0.05
    min_final_positions: int = 1


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _valid_capture_index(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _valid_schema_version(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value == EXPECTED_OBSERVATION_SCHEMA_VERSION
    )


def _wheel_identity(obs: dict[str, Any]) -> str | None:
    wheel_index = obs.get("wheel_index")
    if isinstance(wheel_index, int) and not isinstance(wheel_index, bool) and wheel_index >= 0:
        return f"wheel_index:{wheel_index}"
    wheel_track_id = obs.get("wheel_track_id")
    if isinstance(wheel_track_id, str) and not _is_placeholder(wheel_track_id):
        return f"wheel_track_id:{wheel_track_id.strip()}"
    return None


def _valid_point(value: Any, dims: int) -> bool:
    return isinstance(value, list) and len(value) == dims and all(_is_number(v) for v in value)


def _valid_unit_vector3(value: Any) -> bool:
    if not _valid_point(value, 3):
        return False
    norm = math.sqrt(sum(float(v) * float(v) for v in value))
    return abs(norm - 1.0) <= UNIT_NORMAL_TOLERANCE


def _valid_camera_transform(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    rotation = value.get("R")
    translation = value.get("t")
    return (
        isinstance(rotation, list)
        and len(rotation) == 3
        and all(isinstance(row, list) and len(row) == 3 and all(_is_number(v) for v in row) for row in rotation)
        and _valid_point(translation, 3)
    )


def _valid_recovered_plane(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    support = value.get("support")
    return (
        _valid_unit_vector3(value.get("normal"))
        and _valid_point(value.get("point"), 3)
        and isinstance(support, int)
        and not isinstance(support, bool)
        and support > 0
    )


def _is_placeholder(value: str) -> bool:
    normalized = value.strip().lower()
    return not normalized or "fill_me" in normalized or normalized in {"todo", "tbd", "unknown"}


def _valid_utc_date(value: Any) -> bool:
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


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            observations.append(
                {
                    "_line_no": line_no,
                    "_load_error": f"invalid JSON: {exc.msg}",
                }
            )
            continue
        if isinstance(payload, dict):
            payload["_line_no"] = line_no
            observations.append(payload)
        else:
            observations.append(
                {
                    "_line_no": line_no,
                    "_load_error": f"expected object, got {type(payload).__name__}",
                }
            )
    return observations


def validate_observation(obs: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    line = obs.get("_line_no", "?")
    if "_load_error" in obs:
        return [f"line {line}: {obs['_load_error']}"]

    if not _valid_schema_version(obs.get("schema_version")):
        errors.append(
            f"line {line}: unsupported schema_version "
            f"{obs.get('schema_version', 'missing')}"
        )
    for key in ("session_id", "frame_id"):
        if not isinstance(obs.get(key), str) or not obs.get(key):
            errors.append(f"line {line}: missing/non-string {key}")
        elif _is_placeholder(obs[key]):
            errors.append(f"line {line}: placeholder {key}")
    for key in ("capture_device", "capture_app_version"):
        if not isinstance(obs.get(key), str) or not obs.get(key):
            errors.append(f"line {line}: missing/non-string {key}")
        elif _is_placeholder(obs[key]):
            errors.append(f"line {line}: placeholder {key}")
    if not _valid_utc_date(obs.get("capture_date_utc")):
        errors.append(f"line {line}: capture_date_utc must be a real UTC date in YYYY-MM-DD format")
    if not _valid_capture_index(obs.get("capture_index")):
        errors.append(f"line {line}: capture_index must be a non-negative integer")
    if "wheel_index" in obs and obs["wheel_index"] is not None:
        if not isinstance(obs["wheel_index"], int) or isinstance(obs["wheel_index"], bool) or obs["wheel_index"] < 0:
            errors.append(f"line {line}: wheel_index must be a non-negative integer when present")
    if "wheel_track_id" in obs and obs["wheel_track_id"] is not None:
        if not isinstance(obs["wheel_track_id"], str) or _is_placeholder(obs["wheel_track_id"]):
            errors.append(f"line {line}: wheel_track_id must be a non-placeholder string when present")
    camera_transform = obs.get("camera_transform")
    camera_pose_ref = obs.get("camera_pose_ref")
    has_camera_transform = _valid_camera_transform(camera_transform)
    has_camera_pose_ref = isinstance(camera_pose_ref, str) and not _is_placeholder(camera_pose_ref)
    if camera_transform is not None and not isinstance(camera_transform, dict):
        errors.append(f"line {line}: camera_transform must be object or null")
    elif isinstance(camera_transform, dict) and not _valid_camera_transform(camera_transform):
        errors.append(f"line {line}: camera_transform must contain finite numeric R 3x3 and t vec3")
    if camera_pose_ref is not None and not isinstance(camera_pose_ref, str):
        errors.append(f"line {line}: camera_pose_ref must be string or null")
    if isinstance(camera_pose_ref, str) and _is_placeholder(camera_pose_ref):
        errors.append(f"line {line}: placeholder camera_pose_ref")
    if has_camera_transform and has_camera_pose_ref:
        errors.append(f"line {line}: camera_transform and camera_pose_ref are mutually exclusive")
    if not has_camera_transform and not has_camera_pose_ref:
        errors.append(f"line {line}: missing camera_transform or camera_pose_ref")

    screen_points = obs.get("screen_points")
    if not isinstance(screen_points, dict):
        errors.append(f"line {line}: missing screen_points object")
    else:
        for name in SCREEN_POINT_NAMES:
            if not _valid_point(screen_points.get(name), 2):
                errors.append(f"line {line}: invalid screen_points.{name}")

    floor_hits = obs.get("floor_raycast_hits")
    if not isinstance(floor_hits, dict):
        errors.append(f"line {line}: missing floor_raycast_hits object")
    else:
        for name in FLOOR_HIT_NAMES:
            hit = floor_hits.get(name)
            if hit is not None and not _valid_point(hit, 3):
                errors.append(f"line {line}: invalid floor_raycast_hits.{name}")

    if "residual" in obs and obs["residual"] is not None:
        if not _is_number(obs["residual"]):
            errors.append(f"line {line}: residual must be numeric when present")
        elif float(obs["residual"]) < 0:
            errors.append(f"line {line}: residual must be non-negative")
    if "inlier" in obs and obs["inlier"] is not None and not isinstance(obs["inlier"], bool):
        errors.append(f"line {line}: inlier must be boolean when present")
    if "recovered_plane" in obs and obs["recovered_plane"] is not None and not _valid_recovered_plane(obs["recovered_plane"]):
        errors.append(
            f"line {line}: recovered_plane must contain unit normal vec3, point vec3, and positive support"
        )
    if "c_plane_hit" in obs and obs["c_plane_hit"] is not None and not _valid_point(obs["c_plane_hit"], 3):
        errors.append(f"line {line}: c_plane_hit must be [x,y,z] when present")
    if "c_height_value" in obs and obs["c_height_value"] is not None:
        if not _is_number(obs["c_height_value"]):
            errors.append(f"line {line}: c_height_value must be numeric when present")
        elif float(obs["c_height_value"]) < 0:
            errors.append(f"line {line}: c_height_value must be non-negative")
    if (
        "final_disc_bottom_position" in obs
        and obs["final_disc_bottom_position"] is not None
        and not _valid_point(obs["final_disc_bottom_position"], 3)
    ):
        errors.append(f"line {line}: final_disc_bottom_position must be [x,y,z] when present")
    return errors


def build_report(
    observations: list[dict[str, Any]],
    thresholds: ReplayThresholds,
    *,
    source: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    valid_observations: list[dict[str, Any]] = []
    for obs in observations:
        obs_errors = validate_observation(obs)
        if obs_errors:
            errors.extend(obs_errors)
        else:
            valid_observations.append(obs)

    last_capture_index_by_session: dict[str, tuple[int, str, int]] = {}
    observations_by_frame: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for obs in valid_observations:
        session_id = obs["session_id"]
        frame_id = obs["frame_id"]
        capture_index = int(obs["capture_index"])
        line = int(obs.get("_line_no", 0) or 0)
        previous = last_capture_index_by_session.get(session_id)
        if previous is not None:
            previous_index, previous_frame_id, previous_line = previous
            if capture_index < previous_index:
                errors.append(
                    "line "
                    f"{line or '?'}: capture_index must not decrease within session "
                    f"{session_id}: {capture_index} < {previous_index} "
                    f"(previous line {previous_line or '?'})"
                )
            elif capture_index == previous_index and frame_id != previous_frame_id:
                errors.append(
                    "line "
                    f"{line or '?'}: repeated capture_index {capture_index} within session "
                    f"{session_id} must keep frame_id {previous_frame_id}, got {frame_id}"
                )
        last_capture_index_by_session[session_id] = (capture_index, frame_id, line)
        observations_by_frame.setdefault((session_id, capture_index, frame_id), []).append(obs)

    for (session_id, capture_index, frame_id), frame_observations in observations_by_frame.items():
        if len(frame_observations) <= 1:
            continue
        identities: list[str | None] = [_wheel_identity(obs) for obs in frame_observations]
        if any(identity is None for identity in identities):
            lines = ",".join(str(obs.get("_line_no", "?")) for obs in frame_observations)
            errors.append(
                f"lines {lines}: repeated frame {session_id}/{frame_id}/"
                f"{capture_index} requires wheel_index or wheel_track_id on every row"
            )
            continue
        duplicates = sorted(
            identity for identity in set(identities) if identities.count(identity) > 1
        )
        if duplicates:
            lines = ",".join(str(obs.get("_line_no", "?")) for obs in frame_observations)
            errors.append(
                f"lines {lines}: repeated frame {session_id}/{frame_id}/"
                f"{capture_index} has duplicate wheel identities {duplicates}"
            )

    sessions = {obs["session_id"] for obs in valid_observations}
    complete_floor_hits = 0
    production_source_observations = 0
    inliers = 0
    explicit_outliers = 0
    residuals: list[float] = []
    recovered_planes = 0
    c_plane_hits = 0
    c_height_values = 0
    final_positions = 0

    for obs in valid_observations:
        source_type = str(obs.get("source_type", "")).strip()
        capture_device = str(obs.get("capture_device", "")).strip()
        capture_app_version = str(obs.get("capture_app_version", "")).strip()
        if (
            source_type in PRODUCTION_SOURCE_TYPES
            and not _is_placeholder(capture_device)
            and not _is_placeholder(capture_app_version)
            and _valid_utc_date(obs.get("capture_date_utc"))
        ):
            production_source_observations += 1
        floor_hits = obs.get("floor_raycast_hits", {})
        if all(_valid_point(floor_hits.get(name), 3) for name in FLOOR_HIT_NAMES):
            complete_floor_hits += 1
        if obs.get("inlier") is True:
            inliers += 1
        elif obs.get("inlier") is False:
            explicit_outliers += 1
        if _is_number(obs.get("residual")):
            residuals.append(float(obs["residual"]))
        if _valid_recovered_plane(obs.get("recovered_plane")):
            recovered_planes += 1
        if _valid_point(obs.get("c_plane_hit"), 3):
            c_plane_hits += 1
        if _is_number(obs.get("c_height_value")):
            c_height_values += 1
        if _valid_point(obs.get("final_disc_bottom_position"), 3):
            final_positions += 1

    total_valid = len(valid_observations)
    floor_hit_rate = complete_floor_hits / total_valid if total_valid else 0.0
    labelled_ransac = inliers + explicit_outliers
    inlier_rate = inliers / labelled_ransac if labelled_ransac else None
    median_residual = float(median(residuals)) if residuals else None
    p95_residual = _percentile(residuals, 0.95)

    failures: list[str] = []
    if errors:
        failures.append("schema_errors")
    if total_valid < thresholds.min_observations:
        failures.append(
            f"too_few_observations: {total_valid} < {thresholds.min_observations}"
        )
    if len(sessions) < thresholds.min_sessions:
        failures.append(f"too_few_sessions: {len(sessions)} < {thresholds.min_sessions}")
    if thresholds.require_production_source and production_source_observations < total_valid:
        failures.append(
            "missing_production_source: "
            f"{production_source_observations}/{total_valid} observations carry "
            f"source_type in {sorted(PRODUCTION_SOURCE_TYPES)}, capture_device, "
            "capture_app_version, and capture_date_utc"
        )
    if floor_hit_rate < thresholds.min_floor_hit_rate:
        failures.append(
            f"floor_hit_rate_low: {floor_hit_rate:.3f} < {thresholds.min_floor_hit_rate:.3f}"
        )
    if thresholds.require_ransac:
        if inlier_rate is None:
            failures.append("missing_ransac_inlier_labels")
        elif inlier_rate < thresholds.min_inlier_rate:
            failures.append(
                f"inlier_rate_low: {inlier_rate:.3f} < {thresholds.min_inlier_rate:.3f}"
            )
        if median_residual is None:
            failures.append("missing_residuals")
        elif median_residual > thresholds.max_median_residual:
            failures.append(
                f"median_residual_high: {median_residual:.6f} > {thresholds.max_median_residual:.6f}"
            )
        if p95_residual is None:
            failures.append("missing_p95_residual")
        elif p95_residual > thresholds.max_p95_residual:
            failures.append(
                f"p95_residual_high: {p95_residual:.6f} > {thresholds.max_p95_residual:.6f}"
            )
        if (
            median_residual is not None
            and p95_residual is not None
            and p95_residual < median_residual
        ):
            failures.append(
                f"p95_residual_less_than_median: {p95_residual:.6f} < {median_residual:.6f}"
            )
        if recovered_planes < total_valid:
            failures.append(f"missing_recovered_planes: {recovered_planes} < {total_valid}")
        if c_plane_hits < total_valid:
            failures.append(f"missing_c_plane_hits: {c_plane_hits} < {total_valid}")
        if c_height_values < total_valid:
            failures.append(f"missing_c_height_values: {c_height_values} < {total_valid}")
        if final_positions < thresholds.min_final_positions:
            failures.append(
                f"too_few_final_positions: {final_positions} < {thresholds.min_final_positions}"
            )

    return {
        "ok": not failures,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "thresholds": thresholds.__dict__,
        "counts": {
            "observations_total": len(observations),
            "observations_valid": total_valid,
            "schema_errors": len(errors),
            "sessions": len(sessions),
            "floor_hits_complete": complete_floor_hits,
            "production_source_observations": production_source_observations,
            "ransac_labelled": labelled_ransac,
            "inliers": inliers,
            "outliers": explicit_outliers,
            "residuals": len(residuals),
            "recovered_planes": recovered_planes,
            "c_plane_hits": c_plane_hits,
            "c_height_values": c_height_values,
            "final_disc_bottom_positions": final_positions,
        },
        "metrics": {
            "floor_hit_rate": floor_hit_rate,
            "inlier_rate": inlier_rate,
            "median_residual": median_residual,
            "p95_residual": p95_residual,
        },
        "failures": failures,
        "errors": errors[:100],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/production_audit/ar_3d_replay_eval.json"))
    parser.add_argument("--min-observations", type=int, default=ReplayThresholds.min_observations)
    parser.add_argument("--min-sessions", type=int, default=ReplayThresholds.min_sessions)
    parser.add_argument(
        "--no-require-production-source",
        action="store_true",
        help="Allow synthetic/template replay logs. Do not use for the production gate.",
    )
    parser.add_argument("--min-floor-hit-rate", type=float, default=ReplayThresholds.min_floor_hit_rate)
    parser.add_argument("--no-require-ransac", action="store_true")
    parser.add_argument("--min-inlier-rate", type=float, default=ReplayThresholds.min_inlier_rate)
    parser.add_argument("--max-median-residual", type=float, default=ReplayThresholds.max_median_residual)
    parser.add_argument("--max-p95-residual", type=float, default=ReplayThresholds.max_p95_residual)
    parser.add_argument("--min-final-positions", type=int, default=ReplayThresholds.min_final_positions)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.jsonl.is_file():
        raise FileNotFoundError(args.jsonl)
    thresholds = ReplayThresholds(
        min_observations=args.min_observations,
        min_sessions=args.min_sessions,
        require_production_source=not args.no_require_production_source,
        min_floor_hit_rate=args.min_floor_hit_rate,
        require_ransac=not args.no_require_ransac,
        min_inlier_rate=args.min_inlier_rate,
        max_median_residual=args.max_median_residual,
        max_p95_residual=args.max_p95_residual,
        min_final_positions=args.min_final_positions,
    )
    report = build_report(load_jsonl(args.jsonl), thresholds, source=args.jsonl)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(
        f"ok={report['ok']} observations={report['counts']['observations_valid']}/"
        f"{report['counts']['observations_total']} sessions={report['counts']['sessions']} "
        f"floor_hit_rate={report['metrics']['floor_hit_rate']:.3f} "
        f"inlier_rate={report['metrics']['inlier_rate']}"
    )
    print(f"report={args.out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
