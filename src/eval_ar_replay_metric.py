"""Offline quality metric for AR-side floor-raycast replay logs.

`src/validate_ar_replay.py` verifies that a JSONL replay log is valid
production evidence. This module scores the geometry quality inside a valid
log: per-session/per-wheel inlier rate, residuals, recovered-plane stability,
verticality, and C-projection stability.

It does not run model inference, does not change the ML -> AR JSON contract,
and does not add 3D outputs to the model. The AR client owns raycasts,
RANSAC, and plane recovery; this script evaluates the recorded result offline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.validate_ar_replay import load_jsonl, validate_observation


@dataclass(frozen=True)
class MetricConfig:
    min_observations_per_wheel: int = 30
    floor_normal: tuple[float, float, float] = (0.0, 1.0, 0.0)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _valid_point(value: Any, dims: int) -> bool:
    return isinstance(value, list) and len(value) == dims and all(_is_number(v) for v in value)


def _valid_recovered_plane(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    support = value.get("support")
    normal = value.get("normal")
    if not _valid_point(normal, 3):
        return False
    norm = np.linalg.norm(np.asarray(normal, dtype=float))
    return (
        abs(norm - 1.0) <= 0.05
        and _valid_point(value.get("point"), 3)
        and isinstance(support, int)
        and not isinstance(support, bool)
        and support > 0
    )


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q * 100.0))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def _unit(vec: Any) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(arr)
    if norm <= 1e-12:
        raise ValueError("zero-length vector")
    return arr / norm


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.clip(np.dot(_unit(a), _unit(b)), -1.0, 1.0))
    return float(math.degrees(math.acos(dot)))


def _aligned_normals(normals: list[list[float]]) -> list[np.ndarray]:
    if not normals:
        return []
    aligned = [_unit(normals[0])]
    reference = aligned[0]
    for normal in normals[1:]:
        candidate = _unit(normal)
        if float(np.dot(candidate, reference)) < 0.0:
            candidate = -candidate
        aligned.append(candidate)
    return aligned


def _normal_stability_deg(normals: list[list[float]]) -> float | None:
    aligned = _aligned_normals(normals)
    if len(aligned) < 2:
        return None
    mean_normal = _unit(np.mean(np.stack(aligned), axis=0))
    angles = [_angle_deg(normal, mean_normal) for normal in aligned]
    return float(np.std(np.asarray(angles, dtype=float)))


def _mean_normal(normals: list[list[float]]) -> np.ndarray | None:
    aligned = _aligned_normals(normals)
    if not aligned:
        return None
    return _unit(np.mean(np.stack(aligned), axis=0))


def _point_std(points: list[list[float]]) -> float | None:
    if len(points) < 2:
        return None
    arr = np.asarray(points, dtype=float)
    return float(np.sqrt(np.sum(np.var(arr, axis=0))))


def _wheel_id(obs: dict[str, Any]) -> str:
    wheel_index = obs.get("wheel_index")
    if isinstance(wheel_index, int) and not isinstance(wheel_index, bool) and wheel_index >= 0:
        return f"wheel_index:{wheel_index}"
    wheel_track_id = obs.get("wheel_track_id")
    if isinstance(wheel_track_id, str) and wheel_track_id.strip():
        return f"wheel_track_id:{wheel_track_id.strip()}"
    return "wheel_index:0"


def _bool_text(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return ""


def _score_wheel(
    observations: list[dict[str, Any]],
    *,
    config: MetricConfig,
    floor_normal: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observations = sorted(observations, key=lambda item: int(item.get("capture_index", 0)))
    failures: list[str] = []
    metric_field_errors: list[str] = []
    per_frame_rows: list[dict[str, Any]] = []

    inlier_observations: list[dict[str, Any]] = []
    outliers = 0
    residuals: list[float] = []
    inlier_residuals: list[float] = []
    inlier_normals: list[list[float]] = []
    inlier_c_hits: list[list[float]] = []
    inlier_heights: list[float] = []
    final_positions: list[list[float]] = []
    complete_floor_hits = 0

    for obs in observations:
        line = obs.get("_line_no", "?")
        floor_hits = obs.get("floor_raycast_hits", {})
        if isinstance(floor_hits, dict) and _valid_point(floor_hits.get("a"), 3) and _valid_point(floor_hits.get("b"), 3):
            complete_floor_hits += 1

        if not isinstance(obs.get("inlier"), bool):
            metric_field_errors.append(f"line {line}: missing/non-boolean inlier")
        if not _is_number(obs.get("residual")):
            metric_field_errors.append(f"line {line}: missing/non-numeric residual")
        if not _valid_recovered_plane(obs.get("recovered_plane")):
            metric_field_errors.append(f"line {line}: missing/invalid recovered_plane")
        if not _valid_point(obs.get("c_plane_hit"), 3):
            metric_field_errors.append(f"line {line}: missing/invalid c_plane_hit")
        if not _is_number(obs.get("c_height_value")):
            metric_field_errors.append(f"line {line}: missing/non-numeric c_height_value")

        if obs.get("inlier") is True:
            inlier_observations.append(obs)
            if _is_number(obs.get("residual")):
                value = float(obs["residual"])
                residuals.append(value)
                inlier_residuals.append(value)
            plane = obs.get("recovered_plane")
            if _valid_recovered_plane(plane):
                inlier_normals.append([float(v) for v in plane["normal"]])
            if _valid_point(obs.get("c_plane_hit"), 3):
                inlier_c_hits.append([float(v) for v in obs["c_plane_hit"]])
            if _is_number(obs.get("c_height_value")):
                inlier_heights.append(float(obs["c_height_value"]))
        elif obs.get("inlier") is False:
            outliers += 1
            if _is_number(obs.get("residual")):
                residuals.append(float(obs["residual"]))

        if _valid_point(obs.get("final_disc_bottom_position"), 3):
            final_positions.append([float(v) for v in obs["final_disc_bottom_position"]])

        c_hit = obs.get("c_plane_hit") if _valid_point(obs.get("c_plane_hit"), 3) else ["", "", ""]
        per_frame_rows.append(
            {
                "session_id": obs["session_id"],
                "wheel_id": _wheel_id(obs),
                "frame_id": obs["frame_id"],
                "capture_index": int(obs["capture_index"]),
                "line_no": obs.get("_line_no", ""),
                "inlier": _bool_text(obs.get("inlier")),
                "residual": obs.get("residual", ""),
                "c_plane_hit_x": c_hit[0],
                "c_plane_hit_y": c_hit[1],
                "c_plane_hit_z": c_hit[2],
                "c_height_value": obs.get("c_height_value", ""),
                "final_position_present": _bool_text(_valid_point(obs.get("final_disc_bottom_position"), 3)),
            }
        )

    if len(observations) < config.min_observations_per_wheel:
        failures.append(
            f"too_few_observations: {len(observations)} < {config.min_observations_per_wheel}"
        )
    if metric_field_errors:
        failures.append("missing_metric_fields")
    if not final_positions:
        failures.append("missing_final_disc_bottom_position")

    labelled = len(inlier_observations) + outliers
    inlier_ratio = len(inlier_observations) / labelled if labelled else None
    mean_normal = _mean_normal(inlier_normals)
    plane_verticality = _angle_deg(mean_normal, floor_normal) if mean_normal is not None else None

    report = {
        "ok": not failures,
        "counts": {
            "observations": len(observations),
            "floor_hits_complete": complete_floor_hits,
            "inliers": len(inlier_observations),
            "outliers": outliers,
            "residuals": len(residuals),
            "inlier_residuals": len(inlier_residuals),
            "recovered_planes_inlier": len(inlier_normals),
            "c_plane_hits_inlier": len(inlier_c_hits),
            "c_height_values_inlier": len(inlier_heights),
            "final_disc_bottom_positions": len(final_positions),
        },
        "metrics": {
            "floor_hit_rate": complete_floor_hits / len(observations) if observations else 0.0,
            "inlier_ratio": inlier_ratio,
            "median_residual": _median(inlier_residuals),
            "p95_residual": _percentile(inlier_residuals, 0.95),
            "plane_normal_stability_deg": _normal_stability_deg(inlier_normals),
            "plane_verticality_deg": plane_verticality,
            "c_plane_hit_std": _point_std(inlier_c_hits),
            "c_height_std": float(np.std(np.asarray(inlier_heights, dtype=float))) if len(inlier_heights) >= 2 else None,
            "final_disc_bottom_position_std": _point_std(final_positions),
        },
        "failures": failures,
        "metric_field_errors": metric_field_errors[:100],
    }
    return report, per_frame_rows


def _aggregate_numeric(wheels: list[dict[str, Any]], metric: str) -> dict[str, float | None]:
    values = [
        float(wheel["metrics"][metric])
        for wheel in wheels
        if wheel.get("metrics", {}).get(metric) is not None
    ]
    return {
        "median": _median(values),
        "p95": _percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def build_metric_report(
    observations: list[dict[str, Any]],
    config: MetricConfig | None = None,
    *,
    source: Path,
) -> dict[str, Any]:
    if config is None:
        config = MetricConfig()
    floor_normal = _unit(config.floor_normal)

    schema_errors: list[str] = []
    valid_observations: list[dict[str, Any]] = []
    for obs in observations:
        errors = validate_observation(obs)
        if errors:
            schema_errors.extend(errors)
        else:
            valid_observations.append(obs)

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for obs in valid_observations:
        grouped.setdefault(obs["session_id"], {}).setdefault(_wheel_id(obs), []).append(obs)

    sessions: dict[str, Any] = {}
    per_frame_rows: list[dict[str, Any]] = []
    top_failures: set[str] = set()
    wheel_reports: list[dict[str, Any]] = []

    for session_id in sorted(grouped):
        wheels: dict[str, Any] = {}
        for wheel_id in sorted(grouped[session_id]):
            wheel_report, wheel_rows = _score_wheel(
                grouped[session_id][wheel_id],
                config=config,
                floor_normal=floor_normal,
            )
            wheels[wheel_id] = wheel_report
            wheel_reports.append(wheel_report)
            per_frame_rows.extend(wheel_rows)
            for failure in wheel_report["failures"]:
                if failure.startswith("too_few_observations"):
                    top_failures.add("too_few_observations_per_wheel")
                else:
                    top_failures.add(failure)
        sessions[session_id] = {
            "counts": {
                "wheels": len(wheels),
                "observations": sum(item["counts"]["observations"] for item in wheels.values()),
                "failed_wheels": sum(1 for item in wheels.values() if not item["ok"]),
            },
            "wheels": wheels,
        }

    if schema_errors:
        top_failures.add("schema_errors")
    if not wheel_reports:
        top_failures.add("no_wheels")

    failed_wheels = sum(1 for wheel in wheel_reports if not wheel["ok"])
    report = {
        "schema_version": 1,
        "ok": not top_failures,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "config": {
            "min_observations_per_wheel": config.min_observations_per_wheel,
            "floor_normal": list(config.floor_normal),
        },
        "counts": {
            "observations_total": len(observations),
            "observations_valid": len(valid_observations),
            "schema_errors": len(schema_errors),
            "sessions": len(sessions),
            "wheels": len(wheel_reports),
            "failed_wheels": failed_wheels,
            "per_frame_rows": len(per_frame_rows),
        },
        "aggregate": {
            "failure_rate": failed_wheels / len(wheel_reports) if wheel_reports else 1.0,
            "inlier_ratio": _aggregate_numeric(wheel_reports, "inlier_ratio"),
            "median_residual": _aggregate_numeric(wheel_reports, "median_residual"),
            "p95_residual": _aggregate_numeric(wheel_reports, "p95_residual"),
            "plane_normal_stability_deg": _aggregate_numeric(wheel_reports, "plane_normal_stability_deg"),
            "plane_verticality_deg": _aggregate_numeric(wheel_reports, "plane_verticality_deg"),
            "c_plane_hit_std": _aggregate_numeric(wheel_reports, "c_plane_hit_std"),
            "c_height_std": _aggregate_numeric(wheel_reports, "c_height_std"),
        },
        "sessions": sessions,
        "failures": sorted(top_failures),
        "schema_error_examples": schema_errors[:100],
        "per_frame_rows": per_frame_rows,
    }
    return report


def write_per_frame_csv(rows: list[dict[str, Any]], out: Path) -> None:
    fieldnames = [
        "session_id",
        "wheel_id",
        "frame_id",
        "capture_index",
        "line_no",
        "inlier",
        "residual",
        "c_plane_hit_x",
        "c_plane_hit_y",
        "c_plane_hit_z",
        "c_height_value",
        "final_position_present",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_floor_normal(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("floor normal must be formatted as x,y,z")
    try:
        vec = tuple(float(part) for part in parts)
        _unit(vec)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return vec  # type: ignore[return-value]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/ar_replay/ar_replay_metric.json"))
    parser.add_argument(
        "--per-frame-csv",
        type=Path,
        default=Path("outputs/ar_replay/ar_replay_metric_per_frame.csv"),
    )
    parser.add_argument("--min-observations-per-wheel", type=int, default=MetricConfig.min_observations_per_wheel)
    parser.add_argument(
        "--floor-normal",
        type=_parse_floor_normal,
        default=MetricConfig.floor_normal,
        help="World floor-up normal used for plane verticality, formatted x,y,z. AR logs usually use 0,1,0.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.jsonl.is_file():
        raise FileNotFoundError(args.jsonl)
    config = MetricConfig(
        min_observations_per_wheel=args.min_observations_per_wheel,
        floor_normal=args.floor_normal,
    )
    report = build_metric_report(load_jsonl(args.jsonl), config, source=args.jsonl)
    per_frame_rows = report.pop("per_frame_rows")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_per_frame_csv(per_frame_rows, args.per_frame_csv)
    print(
        f"ok={report['ok']} sessions={report['counts']['sessions']} "
        f"wheels={report['counts']['wheels']} failed_wheels={report['counts']['failed_wheels']} "
        f"failure_rate={report['aggregate']['failure_rate']:.3f}"
    )
    print(f"report={args.out}")
    print(f"per_frame_csv={args.per_frame_csv}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
