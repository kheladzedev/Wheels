"""Create an Unreal capture map with auto-placed wheel annotation actors.

This commandlet script is conservative by design:
- duplicates a source capture map;
- keeps the original map untouched;
- removes previous wheel markers and known vehicle actors from the copy;
- spawns a small curated set of complete vehicle meshes;
- places `wheels` Blueprint actors on the camera-facing side of each vehicle;
- disables vehicle rotation so generated markers stay aligned with the mesh;
- writes JSON/Markdown reports for review.

Run with UnrealEditor-Cmd:

    VSBL_UNREAL_AUTO_SOURCE_MAP=/Game/Wheels/maps/standartWheelsRoom_capture_clean_v3_two_wheels \
    VSBL_UNREAL_AUTO_MAP=/Game/Wheels/maps/standartWheelsRoom_auto_wheels_v1 \
    VSBL_UNREAL_AUTO_VEHICLE_LIMIT=3 \
    VSBL_UNREAL_AUTO_OUT=outputs/unreal_control/create_auto_wheel_capture_map_v1.json \
    "/Users/Shared/Epic Games/UE_5.7/Engine/Binaries/Mac/UnrealEditor-Cmd" \
      "/Users/edward/Desktop/VSBL/NeuralData1 2/NeuralData.uproject" \
      -run=pythonscript -script=/Users/edward/Desktop/VSBL/scripts/unreal_create_auto_wheel_capture_map.py \
      -unattended -nop4 -nosplash -NullRHI
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import unreal


DEFAULT_SOURCE_MAP = "/Game/Wheels/maps/standartWheelsRoom_capture_clean_v3_two_wheels"
DEFAULT_TARGET_MAP = "/Game/Wheels/maps/standartWheelsRoom_auto_wheels_v1"
DEFAULT_OUT = (
    "/Users/edward/Desktop/VSBL/outputs/unreal_control/"
    "create_auto_wheel_capture_map_v1.json"
)
DEFAULT_WHEEL_BP = "/Game/Wheels/main/wheels"
DEFAULT_FLOOR_Z = 53.0
DEFAULT_METHOD = "duplicate_existing"

VEHICLE_CANDIDATES = [
    {
        "label": "auto_demo_derby",
        "asset": "/Game/Wheels/assets/demolition_derby_car/StaticMeshes/demolition_derby_car",
        "location": [40.0, -700.0, DEFAULT_FLOOR_Z],
        "yaw": -27.0,
    },
    {
        "label": "auto_police_car",
        "asset": "/Game/Wheels/assets/police_car/StaticMeshes/police_car",
        "location": [850.0, -360.0, DEFAULT_FLOOR_Z],
        "yaw": 148.0,
    },
    {
        "label": "auto_city_car02",
        "asset": "/Game/CitySampleVehicles/vehicle02_Car/Mesh/SM_vehCar_vehicle02_LOD",
        "location": [-850.0, -360.0, DEFAULT_FLOOR_Z],
        "yaw": 32.0,
    },
    {
        "label": "auto_city_van01",
        "asset": "/Game/CitySampleVehicles/vehicle01_Van/Mesh/SM_vehVan_vehicle01_LOD",
        "location": [0.0, 520.0, DEFAULT_FLOOR_Z],
        "yaw": 180.0,
    },
    {
        "label": "auto_city_truck02",
        "asset": "/Game/CitySampleVehicles/vehicle02_Truck/Mesh/SM_vehTruck_vehicle02_LOD",
        "location": [1250.0, 450.0, DEFAULT_FLOOR_Z],
        "yaw": -145.0,
    },
]


def _vec(v: Any) -> list[float]:
    return [float(v.x), float(v.y), float(v.z)]


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


def _asset_path_for_static_actor(actor: Any) -> str:
    try:
        comp = actor.get_component_by_class(unreal.StaticMeshComponent)
        mesh = comp.get_editor_property("static_mesh") if comp is not None else None
        if mesh is not None:
            return str(mesh.get_path_name()).split(".")[0]
    except Exception:
        pass
    return ""


def _is_wheel_actor(actor: Any) -> bool:
    cls = _actor_class(actor).lower()
    label = _actor_label(actor).lower()
    name = str(actor.get_name()).lower()
    return cls == "wheels_c" or label.startswith("wheels") or name.startswith("wheels")


def _is_capture_actor(actor: Any) -> bool:
    cls = _actor_class(actor).lower()
    label = _actor_label(actor).lower()
    return cls == "cameracapturewheels_c" or "cameracapturewheels" in label


def _is_known_vehicle_actor(actor: Any) -> bool:
    if not isinstance(actor, unreal.StaticMeshActor):
        return False
    label = _actor_label(actor).lower()
    asset = _asset_path_for_static_actor(actor)
    if label.startswith("auto_"):
        return True
    return any(
        needle in asset
        for needle in (
            "/Game/Wheels/assets/demolition_derby_car/",
            "/Game/Wheels/assets/police_car/",
            "/Game/CitySampleVehicles/",
            "/Game/VehicleVarietyPack/Meshes/",
            "/Game/SketchfabCars/",
        )
    )


def _get_all_level_actors() -> list[Any]:
    try:
        actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        return list(actor_subsystem.get_all_level_actors())
    except Exception:
        return list(unreal.EditorLevelLibrary.get_all_level_actors())


def _destroy_actor(actor: Any) -> bool:
    try:
        actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        return bool(actor_subsystem.destroy_actor(actor))
    except Exception:
        return bool(unreal.EditorLevelLibrary.destroy_actor(actor))


def _duplicate_actor(actor_subsystem: Any, actor: Any) -> Any:
    try:
        duplicate = actor_subsystem.duplicate_actor(actor)
        if duplicate is not None:
            return duplicate
    except Exception:
        pass
    try:
        duplicate = unreal.EditorLevelLibrary.duplicate_actor(actor)
        if duplicate is not None:
            return duplicate
    except Exception:
        pass
    raise RuntimeError(f"failed to duplicate actor: {_actor_label(actor)}")


def _load_wheel_class(path: str) -> Any:
    try:
        cls = unreal.EditorAssetLibrary.load_blueprint_class(path)
        if cls is not None:
            return cls
    except Exception:
        pass
    asset = unreal.EditorAssetLibrary.load_asset(path)
    if asset is not None:
        try:
            return asset.generated_class
        except Exception:
            pass
    raise RuntimeError(f"failed to load wheel Blueprint class: {path}")


def _get_static_mesh_bounds(mesh: Any) -> tuple[Any, Any]:
    try:
        bounds = mesh.get_bounds()
        return bounds.origin, bounds.box_extent
    except Exception:
        box = mesh.get_bounding_box()
        origin = unreal.Vector(
            (box.min.x + box.max.x) * 0.5,
            (box.min.y + box.max.y) * 0.5,
            (box.min.z + box.max.z) * 0.5,
        )
        extent = unreal.Vector(
            abs(box.max.x - box.min.x) * 0.5,
            abs(box.max.y - box.min.y) * 0.5,
            abs(box.max.z - box.min.z) * 0.5,
        )
        return origin, extent


def _axis_value(vec: Any, axis: str) -> float:
    return float(getattr(vec, axis))


def _set_axis_value(vec: Any, axis: str, value: float) -> None:
    setattr(vec, axis, float(value))


def _axis_unit(axis: str, sign: float = 1.0) -> Any:
    if axis == "x":
        return unreal.Vector(sign, 0.0, 0.0)
    if axis == "y":
        return unreal.Vector(0.0, sign, 0.0)
    return unreal.Vector(0.0, 0.0, sign)


def _transform_position(transform: Any, vector: Any) -> Any:
    try:
        return transform.transform_position(vector)
    except Exception:
        return transform.transform_location(vector)


def _transform_vector(transform: Any, vector: Any) -> Any:
    try:
        return transform.transform_vector(vector)
    except Exception:
        p0 = _transform_position(transform, unreal.Vector(0.0, 0.0, 0.0))
        p1 = _transform_position(transform, vector)
        return unreal.Vector(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z)


def _inverse_transform(transform: Any) -> Any:
    try:
        return transform.inverse()
    except Exception:
        return transform.get_inverse()


def _yaw_for_local_y_to_world_vector(vec: Any) -> float:
    angle = math.degrees(math.atan2(float(vec.y), float(vec.x)))
    return angle - 90.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _find_capture_actor(actors: list[Any]) -> Any | None:
    for actor in actors:
        if _is_capture_actor(actor):
            return actor
    return None


def _clear_capture_rotation_lists(capture: Any, report: dict[str, Any]) -> None:
    if capture is None:
        return
    try:
        arr = unreal.Array(unreal.Actor)
        capture.set_editor_property("RotateObjects", arr)
        report["capture_rotate_objects_cleared"] = True
    except Exception as exc:
        report["warnings"].append(f"failed_to_clear_rotate_objects:{exc}")
    try:
        capture.set_editor_property("ToRotate", None)
        report["capture_to_rotate_cleared"] = True
    except Exception:
        report["capture_to_rotate_cleared"] = False


def _spawn_vehicle(actor_subsystem: Any, spec: dict[str, Any], floor_z: float) -> Any:
    mesh = unreal.EditorAssetLibrary.load_asset(spec["asset"])
    if mesh is None:
        raise RuntimeError(f"failed to load vehicle mesh: {spec['asset']}")
    loc = spec["location"]
    actor = actor_subsystem.spawn_actor_from_object(
        mesh,
        unreal.Vector(float(loc[0]), float(loc[1]), float(loc[2] or floor_z)),
        unreal.Rotator(0.0, 0.0, float(spec["yaw"])),
    )
    if actor is None:
        raise RuntimeError(f"failed to spawn vehicle: {spec['label']}")
    actor.set_actor_label(spec["label"])
    return actor


def _spawn_wheels_for_vehicle(
    *,
    actor_subsystem: Any,
    wheel_class: Any,
    vehicle: Any,
    spec: dict[str, Any],
    floor_z: float,
    capture_location: Any | None,
    rerun_construction: bool,
) -> list[dict[str, Any]]:
    mesh = unreal.EditorAssetLibrary.load_asset(spec["asset"])
    if mesh is None:
        raise RuntimeError(f"failed to load bounds mesh: {spec['asset']}")

    origin, extent = _get_static_mesh_bounds(mesh)
    long_axis = "x" if abs(extent.x) >= abs(extent.y) else "y"
    side_axis = "y" if long_axis == "x" else "x"
    long_extent = abs(_axis_value(extent, long_axis))
    side_extent = abs(_axis_value(extent, side_axis))
    z_extent = abs(float(extent.z))
    diameter = _clamp(min(max(z_extent * 0.34, side_extent * 0.42), 125.0), 58.0, 125.0)
    center_z = floor_z + diameter * 0.18

    transform = vehicle.get_actor_transform()
    side_sign = 1.0
    if capture_location is not None:
        try:
            local_capture = _transform_position(_inverse_transform(transform), capture_location)
            side_sign = 1.0 if _axis_value(local_capture, side_axis) >= _axis_value(origin, side_axis) else -1.0
        except Exception:
            side_sign = 1.0

    length_dir = _transform_vector(transform, _axis_unit(long_axis, 1.0))
    marker_yaw = _yaw_for_local_y_to_world_vector(length_dir)
    marker_rotation = unreal.Rotator(0.0, 0.0, marker_yaw)

    spawned: list[dict[str, Any]] = []
    for idx, length_sign in enumerate((-1.0, 1.0)):
        local = unreal.Vector(float(origin.x), float(origin.y), float(origin.z))
        _set_axis_value(
            local,
            long_axis,
            _axis_value(origin, long_axis) + length_sign * long_extent * 0.52,
        )
        _set_axis_value(
            local,
            side_axis,
            _axis_value(origin, side_axis) + side_sign * side_extent * 0.82,
        )
        local.z = float(origin.z - extent.z + diameter * 0.18)
        world = _transform_position(transform, local)
        world.z = center_z

        label = f"{spec['label']}_wheel_{idx}"
        actor = actor_subsystem.spawn_actor_from_class(wheel_class, world, marker_rotation)
        if actor is None:
            raise RuntimeError(f"failed to spawn wheel marker: {label}")
        actor.set_actor_label(label)
        actor.set_editor_property("Size", float(diameter))
        actor.set_editor_property("Height", float(diameter))
        if rerun_construction:
            try:
                actor.rerun_construction_scripts()
            except Exception:
                pass
        spawned.append(
            {
                "label": label,
                "vehicle": spec["label"],
                "vehicle_asset": spec["asset"],
                "location": _vec(world),
                "rotation_yaw": marker_yaw,
                "size": diameter,
                "height": diameter,
                "long_axis": long_axis,
                "side_axis": side_axis,
                "side_sign": side_sign,
                "mesh_bounds_origin": _vec(origin),
                "mesh_bounds_extent": _vec(extent),
            }
        )
    return spawned


def _write_markdown(report: dict[str, Any], json_path: Path) -> None:
    md_path = json_path.with_suffix(".md")
    lines = [
        "# Auto Wheel Capture Map",
        "",
        f"- Status: **{report['status']}**",
        f"- Source map: `{report['source_map']}`",
        f"- Target map: `{report['target_map']}`",
        f"- Vehicle limit: `{report['vehicle_limit']}`",
        f"- Method: `{report.get('method', 'spawn')}`",
        f"- Spawned vehicles: `{len(report.get('spawned_vehicles', []))}`",
        f"- Spawned wheel markers: `{len(report.get('spawned_wheels', []))}`",
        f"- Deleted existing wheel actors: `{len(report.get('deleted_wheel_actors', []))}`",
        f"- Deleted existing vehicle actors: `{len(report.get('deleted_vehicle_actors', []))}`",
        "",
        "## Spawned Vehicles",
        "",
        "| Label | Asset | Location | Yaw |",
        "| --- | --- | ---: | ---: |",
    ]
    for item in report.get("spawned_vehicles", []):
        lines.append(
            f"| `{item['label']}` | `{item['asset']}` | `{item['location']}` | `{item['yaw']}` |"
        )
    lines += [
        "",
        "## Spawned Wheel Markers",
        "",
        "| Label | Vehicle | Location | Size | Axes |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for item in report.get("spawned_wheels", []):
        axes = f"{item.get('long_axis', '-')}/{item.get('side_axis', '-')}"
        lines.append(
            f"| `{item['label']}` | `{item['vehicle']}` | `{item['location']}` | "
            f"`{round(item['size'], 2)}` | `{axes}` |"
        )
    if report.get("warnings"):
        lines += ["", "## Warnings", ""]
        lines.extend(f"- {w}" for w in report["warnings"])
    if report.get("errors"):
        lines += ["", "## Errors", ""]
        lines.extend(f"- {e}" for e in report["errors"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _checkpoint(report: dict[str, Any], out_path: Path, step: str) -> None:
    report["last_step"] = step
    report.setdefault("status", "RUNNING")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"VSBL_AUTO_WHEELS step={step}", flush=True)


def _sort_wheels_for_template(wheels: list[Any]) -> list[Any]:
    return sorted(wheels, key=lambda actor: (_actor_label(actor), str(actor.get_name())))


def _run_duplicate_existing(
    *,
    actor_subsystem: Any,
    vehicle_limit: int,
    floor_z: float,
    capture: Any | None,
    clear_capture_rotation: bool,
    report: dict[str, Any],
    out_path: Path,
) -> None:
    _checkpoint(report, out_path, "duplicate_existing_scan_templates")
    actors = _get_all_level_actors()
    template_vehicles = [actor for actor in actors if _is_known_vehicle_actor(actor)]
    template_wheels = _sort_wheels_for_template([actor for actor in actors if _is_wheel_actor(actor)])
    if not template_vehicles:
        raise RuntimeError("duplicate_existing requires one template vehicle actor")
    if len(template_wheels) < 2:
        raise RuntimeError("duplicate_existing requires at least two template wheel actors")

    template_vehicle = template_vehicles[0]
    template_wheels = template_wheels[:2]
    base_loc = template_vehicle.get_actor_location()
    report["template_vehicle"] = {
        "label": _actor_label(template_vehicle),
        "class": _actor_class(template_vehicle),
        "asset": _asset_path_for_static_actor(template_vehicle),
        "location": _vec(base_loc),
    }
    report["template_wheels"] = [
        {
            "label": _actor_label(actor),
            "class": _actor_class(actor),
            "location": _vec(actor.get_actor_location()),
        }
        for actor in template_wheels
    ]

    _checkpoint(report, out_path, "duplicate_existing_delete_old_clones")
    for actor in list(_get_all_level_actors()):
        label = _actor_label(actor)
        if label.startswith("auto_clone_"):
            report.setdefault("deleted_clone_actors", []).append(
                {
                    "label": label,
                    "class": _actor_class(actor),
                    "location": _vec(actor.get_actor_location()),
                }
            )
            _destroy_actor(actor)

    step_x = float(os.environ.get("VSBL_UNREAL_AUTO_DUPLICATE_STEP_X", "420"))
    step_y = float(os.environ.get("VSBL_UNREAL_AUTO_DUPLICATE_STEP_Y", "0"))
    for idx in range(max(1, vehicle_limit)):
        _checkpoint(report, out_path, f"duplicate_existing_place:{idx}")
        if idx == 0:
            vehicle = template_vehicle
            wheels = template_wheels
        else:
            vehicle = _duplicate_actor(actor_subsystem, template_vehicle)
            wheels = [_duplicate_actor(actor_subsystem, actor) for actor in template_wheels]

        offset = unreal.Vector(step_x * idx, step_y * idx, 0.0)
        vehicle.set_actor_location(
            unreal.Vector(base_loc.x + offset.x, base_loc.y + offset.y, float(floor_z)),
            False,
            False,
        )
        vehicle.set_actor_label(f"auto_clone_{idx:02d}_vehicle")
        report["spawned_vehicles"].append(
            {
                "label": _actor_label(vehicle),
                "asset": _asset_path_for_static_actor(vehicle),
                "location": _vec(vehicle.get_actor_location()),
                "yaw": float(vehicle.get_actor_rotation().yaw),
                "source": "duplicated_existing_actor" if idx else "existing_template_actor",
            }
        )
        for wheel_idx, wheel in enumerate(wheels):
            source_loc = template_wheels[wheel_idx].get_actor_location()
            loc = unreal.Vector(source_loc.x + offset.x, source_loc.y + offset.y, source_loc.z)
            wheel.set_actor_location(loc, False, False)
            wheel.set_actor_label(f"auto_clone_{idx:02d}_wheel_{wheel_idx}")
            report["spawned_wheels"].append(
                {
                    "label": _actor_label(wheel),
                    "vehicle": _actor_label(vehicle),
                    "vehicle_asset": _asset_path_for_static_actor(vehicle),
                    "location": _vec(wheel.get_actor_location()),
                    "rotation_yaw": float(wheel.get_actor_rotation().yaw),
                    "size": float(wheel.get_editor_property("Size")),
                    "height": float(wheel.get_editor_property("Height")),
                    "source": "duplicated_existing_actor" if idx else "existing_template_actor",
                }
            )

    if clear_capture_rotation:
        _clear_capture_rotation_lists(capture, report)
    else:
        report["capture_rotation_preserved"] = True


def main() -> None:
    source_map = os.environ.get("VSBL_UNREAL_AUTO_SOURCE_MAP", DEFAULT_SOURCE_MAP)
    target_map = os.environ.get("VSBL_UNREAL_AUTO_MAP", DEFAULT_TARGET_MAP)
    out_path = Path(os.environ.get("VSBL_UNREAL_AUTO_OUT", DEFAULT_OUT))
    wheel_bp = os.environ.get("VSBL_UNREAL_AUTO_WHEEL_BP", DEFAULT_WHEEL_BP)
    floor_z = float(os.environ.get("VSBL_UNREAL_AUTO_FLOOR_Z", str(DEFAULT_FLOOR_Z)))
    vehicle_limit = int(os.environ.get("VSBL_UNREAL_AUTO_VEHICLE_LIMIT", "3"))
    rerun_construction = os.environ.get("VSBL_UNREAL_AUTO_RERUN_CONSTRUCTION", "0") == "1"
    skip_duplicate = os.environ.get("VSBL_UNREAL_AUTO_SKIP_DUPLICATE", "0") == "1"
    method = os.environ.get("VSBL_UNREAL_AUTO_METHOD", DEFAULT_METHOD)
    clear_capture_rotation = (
        os.environ.get("VSBL_UNREAL_AUTO_CLEAR_CAPTURE_ROTATION", "0") == "1"
    )

    report: dict[str, Any] = {
        "source_map": source_map,
        "target_map": target_map,
        "wheel_blueprint": wheel_bp,
        "floor_z": floor_z,
        "vehicle_limit": vehicle_limit,
        "method": method,
        "rerun_construction": rerun_construction,
        "skip_duplicate": skip_duplicate,
        "clear_capture_rotation": clear_capture_rotation,
        "deleted_wheel_actors": [],
        "deleted_vehicle_actors": [],
        "spawned_vehicles": [],
        "spawned_wheels": [],
        "warnings": [],
        "errors": [],
        "status": "FAIL",
    }

    try:
        _checkpoint(report, out_path, "start")
        if not skip_duplicate:
            _checkpoint(report, out_path, "check_source_map")
            if not unreal.EditorAssetLibrary.does_asset_exist(source_map):
                raise RuntimeError(f"source map does not exist: {source_map}")
            _checkpoint(report, out_path, "delete_existing_target")
            if unreal.EditorAssetLibrary.does_asset_exist(target_map):
                unreal.EditorAssetLibrary.delete_asset(target_map)
            _checkpoint(report, out_path, "duplicate_map")
            if not unreal.EditorAssetLibrary.duplicate_asset(source_map, target_map):
                raise RuntimeError(f"failed to duplicate map: {source_map} -> {target_map}")
        else:
            _checkpoint(report, out_path, "skip_duplicate_check_target")
            if not unreal.EditorAssetLibrary.does_asset_exist(target_map):
                raise RuntimeError(f"target map does not exist for skip mode: {target_map}")

        _checkpoint(report, out_path, "load_map")
        unreal.EditorLoadingAndSavingUtils.load_map(target_map)
        actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        _checkpoint(report, out_path, "load_wheel_class")
        wheel_class = _load_wheel_class(wheel_bp)

        _checkpoint(report, out_path, "scan_existing_actors")
        actors = _get_all_level_actors()
        capture = _find_capture_actor(actors)
        capture_location = capture.get_actor_location() if capture is not None else None
        if capture is None:
            report["warnings"].append("missing_camera_capture_actor")

        if method == "duplicate_existing":
            _run_duplicate_existing(
                actor_subsystem=actor_subsystem,
                vehicle_limit=vehicle_limit,
                floor_z=floor_z,
                capture=capture,
                clear_capture_rotation=clear_capture_rotation,
                report=report,
                out_path=out_path,
            )
            _checkpoint(report, out_path, "save_current_level")
            saved = unreal.EditorLoadingAndSavingUtils.save_current_level()
            _checkpoint(report, out_path, "save_dirty_packages")
            unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)
            report["saved"] = bool(saved)
            report["status"] = "PASS" if saved and report["spawned_wheels"] else "FAIL"
            return

        if method != "spawn":
            raise RuntimeError(f"unsupported auto wheel method: {method}")

        for actor in list(actors):
            if _is_wheel_actor(actor):
                report["deleted_wheel_actors"].append(
                    {
                        "label": _actor_label(actor),
                        "class": _actor_class(actor),
                        "location": _vec(actor.get_actor_location()),
                    }
                )
                _destroy_actor(actor)

        _checkpoint(report, out_path, "delete_existing_vehicles")
        for actor in list(_get_all_level_actors()):
            if _is_known_vehicle_actor(actor):
                report["deleted_vehicle_actors"].append(
                    {
                        "label": _actor_label(actor),
                        "class": _actor_class(actor),
                        "asset": _asset_path_for_static_actor(actor),
                        "location": _vec(actor.get_actor_location()),
                    }
                )
                _destroy_actor(actor)

        selected = []
        for spec in VEHICLE_CANDIDATES:
            if len(selected) >= vehicle_limit:
                break
            if not unreal.EditorAssetLibrary.does_asset_exist(spec["asset"]):
                report["warnings"].append(f"missing_vehicle_asset:{spec['asset']}")
                continue
            selected.append(spec)

        if not selected:
            raise RuntimeError("no vehicle assets available to spawn")

        for spec in selected:
            _checkpoint(report, out_path, f"spawn_vehicle:{spec['label']}")
            vehicle = _spawn_vehicle(actor_subsystem, spec, floor_z)
            report["spawned_vehicles"].append(
                {
                    "label": spec["label"],
                    "asset": spec["asset"],
                    "location": _vec(vehicle.get_actor_location()),
                    "yaw": float(spec["yaw"]),
                }
            )
            _checkpoint(report, out_path, f"spawn_wheels:{spec['label']}")
            report["spawned_wheels"].extend(
                _spawn_wheels_for_vehicle(
                    actor_subsystem=actor_subsystem,
                    wheel_class=wheel_class,
                    vehicle=vehicle,
                    spec=spec,
                    floor_z=floor_z,
                    capture_location=capture_location,
                    rerun_construction=rerun_construction,
                )
            )

        _checkpoint(report, out_path, "clear_capture_rotation")
        _clear_capture_rotation_lists(capture, report)
        _checkpoint(report, out_path, "save_current_level")
        saved = unreal.EditorLoadingAndSavingUtils.save_current_level()
        _checkpoint(report, out_path, "save_dirty_packages")
        unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)
        report["saved"] = bool(saved)
        report["status"] = "PASS" if saved and report["spawned_wheels"] else "FAIL"
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
        try:
            if hasattr(unreal.SystemLibrary, "quit_editor"):
                unreal.SystemLibrary.quit_editor()
        except Exception:
            pass


main()
