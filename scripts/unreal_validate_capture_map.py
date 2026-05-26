"""Validate NeuralData wheel capture actors inside Unreal Editor.

Run with UnrealEditor-Cmd, for example:

    VSBL_UNREAL_VALIDATE_MAP=/Game/Wheels/maps/standartWheelsRoom \
    VSBL_UNREAL_VALIDATE_OUT=outputs/unreal_control/unreal_map_validation.json \
    "/Users/Shared/Epic Games/UE_5.7/Engine/Binaries/Mac/UnrealEditor-Cmd" \
      "/Users/edward/Desktop/VSBL/NeuralData1 2/NeuralData.uproject" \
      -run=pythonscript -script=scripts/unreal_validate_capture_map.py \
      -unattended -nop4 -nosplash -NullRHI

This script intentionally does not modify the level. It inspects the capture
map, records the export-order wheel actor list, checks required wheel
components, and performs best-effort floor traces from A/B/C source points.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import unreal


DEFAULT_MAP = "/Game/Wheels/maps/standartWheelsRoom"
DEFAULT_OUT = (
    "/Users/edward/Desktop/VSBL/outputs/unreal_control/"
    "unreal_map_validation.json"
)

REQUIRED_COMPONENTS = (
    "Center",
    "SphereLeft",
    "SphereRight",
    "SphereLeftTop",
    "SphereRightTop",
)
FLOOR_TRACE_COMPONENTS = ("SphereLeft", "SphereRight", "Center")
MIN_BOTTOM_WIDTH_CM = 5.0
MIN_TOP_HEIGHT_CM = 5.0


def _vec(v: Any) -> list[float]:
    return [float(v.x), float(v.y), float(v.z)]


def _dist(a: Any, b: Any) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _actor_label(actor: Any) -> str:
    try:
        return str(actor.get_actor_label())
    except Exception:
        return str(actor.get_name())


def _actor_class(actor: Any) -> str:
    try:
        return str(actor.get_class().get_name())
    except Exception:
        return type(actor).__name__


def _is_wheel_actor(actor: Any) -> bool:
    cls = _actor_class(actor).lower()
    label = _actor_label(actor).lower()
    name = str(actor.get_name()).lower()
    return cls == "wheels_c" or label.startswith("wheels") or name.startswith("wheels")


def _is_capture_actor(actor: Any) -> bool:
    cls = _actor_class(actor).lower()
    label = _actor_label(actor).lower()
    return cls == "cameracapturewheels_c" or "cameracapturewheels" in label


def _get_prop(actor: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        try:
            return actor.get_editor_property(name)
        except Exception:
            continue
    return None


def _components_by_name(actor: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        comps = actor.get_components_by_class(unreal.SceneComponent)
    except Exception:
        return out

    for comp in comps:
        name = str(comp.get_name())
        out.setdefault(name, comp)
    return out


def _get_all_level_actors() -> list[Any]:
    try:
        actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        return list(actor_subsystem.get_all_level_actors())
    except Exception:
        return list(unreal.EditorLevelLibrary.get_all_level_actors())


def _trace_down(actor: Any, comp: Any) -> dict[str, Any]:
    start = comp.get_world_location()
    end = unreal.Vector(start.x, start.y, start.z - 5000.0)
    out: dict[str, Any] = {"start": _vec(start), "end": _vec(end)}

    try:
        if hasattr(unreal.SystemLibrary, "line_trace_single_by_profile"):
            trace_result = unreal.SystemLibrary.line_trace_single_by_profile(
                actor,
                start,
                end,
                "BlockAll",
                False,
                [actor],
                unreal.DrawDebugTrace.NONE,
                True,
                unreal.LinearColor(1.0, 0.0, 0.0, 1.0),
                unreal.LinearColor(0.0, 1.0, 0.0, 1.0),
                0.0,
            )
        else:
            trace_result = unreal.SystemLibrary.line_trace_single(
                actor,
                start,
                end,
                unreal.TraceTypeQuery.TRACE_TYPE_QUERY1,
                False,
                [actor],
                unreal.DrawDebugTrace.NONE,
                True,
                unreal.LinearColor(1.0, 0.0, 0.0, 1.0),
                unreal.LinearColor(0.0, 1.0, 0.0, 1.0),
                0.0,
            )

        if isinstance(trace_result, tuple):
            hit, hit_result = trace_result
        else:
            hit_result = trace_result
            hit = hit_result is not None

        if hit_result is not None:
            try:
                hit = bool(hit_result.get_editor_property("blocking_hit"))
            except Exception:
                pass

        out["hit"] = bool(hit)
        if hit:
            try:
                loc = hit_result.get_editor_property("location")
                out["hit_location"] = _vec(loc)
            except Exception:
                pass
    except Exception as exc:
        out["hit"] = None
        out["error"] = str(exc)
    return out


def _validate_wheel(actor: Any, export_order: int) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "export_object_id": str(export_order),
        "actor_label": _actor_label(actor),
        "actor_name": str(actor.get_name()),
        "actor_class": _actor_class(actor),
        "actor_path": str(actor.get_path_name()),
        "location": _vec(actor.get_actor_location()),
        "size": _get_prop(actor, ("Size", "size")),
        "height": _get_prop(actor, ("Height", "height")),
        "components": {},
        "distances_cm": {},
        "floor_traces": {},
        "errors": [],
        "warnings": [],
    }

    comps = _components_by_name(actor)
    for required in REQUIRED_COMPONENTS:
        comp = comps.get(required)
        if comp is None:
            entry["errors"].append(f"missing_component:{required}")
            continue
        loc = comp.get_world_location()
        entry["components"][required] = {
            "class": str(comp.get_class().get_name()),
            "world_location": _vec(loc),
        }

    if entry["size"] is not None:
        entry["size"] = float(entry["size"])
        if entry["size"] <= 0:
            entry["errors"].append("non_positive_size")
    if entry["height"] is not None:
        entry["height"] = float(entry["height"])
        if entry["height"] < 0:
            entry["errors"].append("negative_height")

    for comp_name in FLOOR_TRACE_COMPONENTS:
        comp = comps.get(comp_name)
        if comp is not None:
            trace = _trace_down(actor, comp)
            entry["floor_traces"][comp_name] = trace
            if trace.get("hit") is False:
                entry["warnings"].append(f"floor_trace_no_hit:{comp_name}")
            if trace.get("error"):
                entry["warnings"].append(f"floor_trace_error:{comp_name}")

    left = comps.get("SphereLeft")
    right = comps.get("SphereRight")
    left_top = comps.get("SphereLeftTop")
    right_top = comps.get("SphereRightTop")
    center = comps.get("Center")

    if left is not None and right is not None:
        width = _dist(left.get_world_location(), right.get_world_location())
        entry["distances_cm"]["bottom_left_right"] = width
        if width < MIN_BOTTOM_WIDTH_CM:
            entry["errors"].append("bottom_points_collapsed")

    if left is not None and left_top is not None:
        h = left_top.get_world_location().z - left.get_world_location().z
        entry["distances_cm"]["left_top_minus_bottom_z"] = h
        if abs(h) < MIN_TOP_HEIGHT_CM:
            entry["warnings"].append("left_top_height_small")

    if right is not None and right_top is not None:
        h = right_top.get_world_location().z - right.get_world_location().z
        entry["distances_cm"]["right_top_minus_bottom_z"] = h
        if abs(h) < MIN_TOP_HEIGHT_CM:
            entry["warnings"].append("right_top_height_small")

    if center is not None and left is not None:
        entry["distances_cm"]["center_to_left"] = _dist(
            center.get_world_location(), left.get_world_location()
        )
    if center is not None and right is not None:
        entry["distances_cm"]["center_to_right"] = _dist(
            center.get_world_location(), right.get_world_location()
        )

    entry["status"] = "PASS" if not entry["errors"] else "FAIL"
    return entry


def _write_markdown(report: dict[str, Any], json_path: Path) -> None:
    md_path = json_path.with_suffix(".md")
    lines = [
        "# Unreal Wheel Capture Map Validation",
        "",
        f"- Status: **{report['status']}**",
        f"- Map: `{report['map']}`",
        f"- Wheel actors: `{report['wheel_actor_count']}`",
        f"- CameraCaptureWheels actors: `{report['camera_capture_actor_count']}`",
        "",
        "## Wheel Actors",
        "",
        "| Export object id | Actor | Status | Errors | Warnings |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in report["wheel_actors"]:
        lines.append(
            "| {id} | `{actor}` | {status} | {errors} | {warnings} |".format(
                id=item["export_object_id"],
                actor=item["actor_label"],
                status=item["status"],
                errors=", ".join(item["errors"]) or "-",
                warnings=", ".join(item["warnings"]) or "-",
            )
        )

    if report.get("notes"):
        lines += ["", "## Notes", ""]
        lines.extend(f"- {note}" for note in report["notes"])

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    map_path = os.environ.get("VSBL_UNREAL_VALIDATE_MAP", DEFAULT_MAP)
    out_path = Path(os.environ.get("VSBL_UNREAL_VALIDATE_OUT", DEFAULT_OUT))
    report: dict[str, Any] = {
        "map": map_path,
        "status": "FAIL",
        "camera_capture_actors": [],
        "wheel_actors": [],
        "errors": [],
        "notes": [
            "This validation is read-only and runs inside Unreal.",
            "export_object_id is the observed actor iteration order used to map actors to keyPoint object ids.",
        ],
    }

    try:
        if not unreal.EditorAssetLibrary.does_asset_exist(map_path):
            raise RuntimeError(f"map asset does not exist: {map_path}")

        unreal.EditorLoadingAndSavingUtils.load_map(map_path)
        actors = _get_all_level_actors()
        camera_actors = [actor for actor in actors if _is_capture_actor(actor)]
        wheel_actors = [actor for actor in actors if _is_wheel_actor(actor)]

        report["camera_capture_actors"] = [
            {
                "actor_label": _actor_label(actor),
                "actor_name": str(actor.get_name()),
                "actor_class": _actor_class(actor),
                "actor_path": str(actor.get_path_name()),
                "location": _vec(actor.get_actor_location()),
            }
            for actor in camera_actors
        ]
        report["wheel_actors"] = [
            _validate_wheel(actor, idx) for idx, actor in enumerate(wheel_actors)
        ]
        report["actor_count"] = len(actors)
        report["camera_capture_actor_count"] = len(camera_actors)
        report["wheel_actor_count"] = len(wheel_actors)

        if len(camera_actors) != 1:
            report["errors"].append(
                f"expected_one_camera_capture_actor_found_{len(camera_actors)}"
            )
        if not wheel_actors:
            report["errors"].append("no_wheel_actors")

        failed_wheels = [
            item for item in report["wheel_actors"] if item["status"] != "PASS"
        ]
        if failed_wheels:
            report["errors"].append(f"failed_wheel_actor_count:{len(failed_wheels)}")

        report["status"] = "PASS" if not report["errors"] else "FAIL"
    except Exception as exc:
        report["errors"].append(str(exc))
        report["status"] = "FAIL"
        raise
    finally:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        _write_markdown(report, out_path)
        print(f"WROTE {out_path}")
        print(f"WROTE {out_path.with_suffix('.md')}")


main()
