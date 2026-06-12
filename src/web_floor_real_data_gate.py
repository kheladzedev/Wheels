"""Production-data gate for web floor training manifests.

The web floor model can only move past fixture proof once a real web/phone
dataset exists with wheel labels plus floor ``pitch/roll/distance`` labels.
This module turns that rule into a machine-readable audit instead of relying
on status prose.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from web_floor_dataset import WebFloorDataset, WebFloorDatasetError


@dataclass(frozen=True)
class WebFloorRealDataGateConfig:
    min_frames: int = 50
    min_wheels: int = 80
    required_splits: tuple[str, ...] = ("train", "holdout")
    require_non_fixture: bool = True
    require_real_source: bool = True
    require_provenance: bool = True
    require_known_distance_mode: bool = True
    min_distance_span: float = 0.5
    min_angle_span_rad: float = 0.05


def _check(name: str, passed: bool, detail: str, observed: Any = None) -> dict[str, Any]:
    out = {"name": name, "passed": bool(passed), "detail": detail}
    if observed is not None:
        out["observed"] = observed
    return out


def _split_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        split = item.get("split")
        if split is None:
            split = "missing"
        counts[str(split)] = counts.get(str(split), 0) + 1
    return dict(sorted(counts.items()))


def _frame_ids(items: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("frame_id", "")) for item in items]


def _floor_values(items: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for item in items:
        floor = item.get("floor", {})
        if isinstance(floor, dict) and key in floor:
            values.append(float(floor[key]))
    return values


def _span(values: list[float]) -> float:
    return float(max(values) - min(values)) if values else 0.0


def _has_provenance(item: dict[str, Any]) -> bool:
    value = item.get("provenance")
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return bool(value)
    return False


_NON_REAL_SOURCE_MARKERS = ("synthetic", "unreal", "fixture", "generated")


def _source_marker_matches(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    lowered = value.lower()
    return [marker for marker in _NON_REAL_SOURCE_MARKERS if marker in lowered]


def _non_real_source_evidence(
    manifest: dict[str, Any],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    hits: list[dict[str, Any]] = []

    for key in ("source_type", "source_name", "source_format"):
        markers = _source_marker_matches(manifest.get(key))
        if markers:
            hits.append({"scope": "manifest", "key": key, "markers": markers})

    for index, item in enumerate(items):
        provenance = item.get("provenance")
        if not isinstance(provenance, dict):
            continue
        for key in ("source_type", "source", "source_format"):
            markers = _source_marker_matches(provenance.get(key))
            if markers:
                hits.append(
                    {
                        "scope": f"items[{index}].provenance",
                        "key": key,
                        "markers": markers,
                    }
                )
                break

    return {
        "non_real_source_count": len(hits),
        "examples": hits[:5],
    }


def audit_web_floor_real_data(
    config: str | Path,
    *,
    gate: WebFloorRealDataGateConfig | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Audit whether a web-floor manifest is ready for production training."""
    gate = gate or WebFloorRealDataGateConfig()
    config_path = Path(config)
    checks: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema": "web_floor_real_data_gate_v1",
        "config": str(config_path),
        "gate": {
            "min_frames": gate.min_frames,
            "min_wheels": gate.min_wheels,
            "required_splits": list(gate.required_splits),
            "require_non_fixture": gate.require_non_fixture,
            "require_real_source": gate.require_real_source,
            "require_provenance": gate.require_provenance,
            "require_known_distance_mode": gate.require_known_distance_mode,
            "min_distance_span": gate.min_distance_span,
            "min_angle_span_rad": gate.min_angle_span_rad,
        },
    }

    try:
        dataset = WebFloorDataset(config_path)
    except (OSError, WebFloorDatasetError, ValueError) as exc:
        report.update(
            {
                "production_data_ready": False,
                "dataset_loaded": False,
                "checks": [
                    _check("dataset_loads", False, str(exc)),
                ],
            }
        )
        if output_json is not None:
            path = Path(output_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    items = [item for item in dataset.items if isinstance(item, dict)]
    total_wheels = sum(len(item.get("wheels", [])) for item in items)
    split_counts = _split_counts(items)
    distance_modes = sorted(
        {
            str(item.get("floor", {}).get("distance_mode", "missing"))
            for item in items
            if isinstance(item.get("floor"), dict)
        }
    )
    unknown_distance_count = sum(
        1
        for item in items
        if isinstance(item.get("floor"), dict)
        and item["floor"].get("distance_mode") in {"unknown", None}
    )
    frame_ids = _frame_ids(items)
    missing_frame_ids = sum(1 for frame_id in frame_ids if not frame_id)
    duplicate_frame_ids = len(frame_ids) - len(set(frame_ids))
    missing_provenance = sum(1 for item in items if not _has_provenance(item))
    distance_span = _span(_floor_values(items, "distance"))
    pitch_span = _span(_floor_values(items, "pitch"))
    roll_span = _span(_floor_values(items, "roll"))
    angle_span = max(pitch_span, roll_span)
    non_real_evidence = _non_real_source_evidence(dataset.manifest, items)

    checks.append(_check("dataset_loads", True, "dataset loaded and every item validates"))
    checks.append(
        _check(
            "not_fixture",
            (not dataset.fixture_only) or (not gate.require_non_fixture),
            "manifest must declare fixture_only=false for production data",
            dataset.fixture_only,
        )
    )
    checks.append(
        _check(
            "real_source",
            non_real_evidence["non_real_source_count"] == 0 or not gate.require_real_source,
            "production training requires real web/phone capture, not synthetic/generated sources",
            non_real_evidence,
        )
    )
    checks.append(
        _check(
            "minimum_frames",
            len(dataset) >= gate.min_frames,
            f"requires at least {gate.min_frames} frames",
            len(dataset),
        )
    )
    checks.append(
        _check(
            "minimum_wheels",
            total_wheels >= gate.min_wheels,
            f"requires at least {gate.min_wheels} labelled wheels",
            total_wheels,
        )
    )
    checks.append(
        _check(
            "required_splits",
            all(split in split_counts for split in gate.required_splits),
            f"requires split(s): {', '.join(gate.required_splits)}",
            split_counts,
        )
    )
    checks.append(
        _check(
            "frame_ids_unique",
            missing_frame_ids == 0 and duplicate_frame_ids == 0,
            "frame_id must be present and unique for every frame",
            {"missing": missing_frame_ids, "duplicates": duplicate_frame_ids},
        )
    )
    checks.append(
        _check(
            "provenance_present",
            missing_provenance == 0 or not gate.require_provenance,
            "every frame needs provenance before production training",
            {"missing": missing_provenance},
        )
    )
    checks.append(
        _check(
            "distance_mode_known",
            unknown_distance_count == 0 or not gate.require_known_distance_mode,
            "distance_mode must not be unknown for production data",
            {"unknown": unknown_distance_count, "modes": distance_modes},
        )
    )
    checks.append(
        _check(
            "distance_span",
            distance_span >= gate.min_distance_span,
            f"requires distance span >= {gate.min_distance_span}",
            distance_span,
        )
    )
    checks.append(
        _check(
            "angle_span",
            angle_span >= gate.min_angle_span_rad,
            f"requires pitch or roll span >= {gate.min_angle_span_rad} rad",
            {"pitch_span": pitch_span, "roll_span": roll_span, "max": angle_span},
        )
    )

    report.update(
        {
            "production_data_ready": all(check["passed"] for check in checks),
            "dataset_loaded": True,
            "fixture_only": dataset.fixture_only,
            "dataset_items": len(dataset),
            "total_wheels": total_wheels,
            "split_counts": split_counts,
            "distance_modes": distance_modes,
            "distance_span": distance_span,
            "pitch_span": pitch_span,
            "roll_span": roll_span,
            "checks": checks,
        }
    )
    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
