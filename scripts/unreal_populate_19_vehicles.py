"""Populate standartWheelsRoom with up to 19 vehicle meshes (VehicleVarietyPack
+ CitySampleVehicles) in a circular layout around the capture origin, so the
CameraCaptureWheels spline sees a diverse fleet on every run.

Run via:

  "/Users/Shared/Epic Games/UE_5.7/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor" \\
    "/Users/edward/Desktop/VSBL/NeuralData1 2/NeuralData.uproject" \\
    -ExecutePythonScript="/Users/edward/Desktop/VSBL/scripts/unreal_populate_19_vehicles.py" \\
    -nosplash

Saves the map in place and calls quit_editor() at the end so the shell
command returns without a human touching the GUI.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import unreal


MAP_PATH = os.environ.get("VSBL_POPULATE_MAP", "/Game/Wheels/maps/standartWheelsRoom")
OUT_PATH = Path(
    os.environ.get(
        "VSBL_POPULATE_OUT",
        "/Users/edward/Desktop/VSBL/outputs/unreal_control/populate_19_vehicles.json",
    )
)

# (label, mesh path). The first batch is VehicleVarietyPack (already in map);
# the second batch is CitySampleVehicles vehicle01..13 main LOD body meshes.
ALL_VEHICLES = [
    ("vv_pickup", "/Game/VehicleVarietyPack/Meshes/SM_Pickup"),
    ("vv_hatchback", "/Game/VehicleVarietyPack/Meshes/SM_Hatchback"),
    ("vv_sportscar", "/Game/VehicleVarietyPack/Meshes/SM_SportsCar"),
    ("vv_truck_box", "/Game/VehicleVarietyPack/Meshes/SM_Truck_Box"),
    ("vv_suv", "/Game/VehicleVarietyPack/Meshes/SM_SUV"),
    ("cs_van01", "/Game/CitySampleVehicles/vehicle01_Van/Mesh/SM_vehVan_vehicle01_LOD"),
    ("cs_car02", "/Game/CitySampleVehicles/vehicle02_Car/Mesh/SM_vehCar_vehicle02_LOD"),
    ("cs_car03", "/Game/CitySampleVehicles/vehicle03_Car/Mesh/SM_vehCar_vehicle03_LOD"),
    (
        "cs_truck04",
        "/Game/CitySampleVehicles/vehicle04_Truck/Mesh/SM_vehTruck_vehicle04_LOD",
    ),
    ("cs_car05", "/Game/CitySampleVehicles/vehicle05_Car/Mesh/SM_vehCar_vehicle05_LOD"),
    ("cs_car06", "/Game/CitySampleVehicles/vehicle06_Car/Mesh/SM_vehCar_vehicle06_LOD"),
    ("cs_car07", "/Game/CitySampleVehicles/vehicle07_Car/Mesh/SM_vehCar_vehicle07_LOD"),
    (
        "cs_truck08",
        "/Game/CitySampleVehicles/vehicle08_Truck/Mesh/SM_vehTruck_vehicle08_LOD",
    ),
    ("cs_van09", "/Game/CitySampleVehicles/vehicle09_Van/Mesh/SM_vehVan_vehicle09_LOD"),
    ("cs_bus10", "/Game/CitySampleVehicles/vehicle10_Bus/Mesh/SM_vehBus_vehicle10_LOD"),
    (
        "cs_truck11",
        "/Game/CitySampleVehicles/vehicle11_Truck/Mesh/SM_vehTruck_vehicle11_LOD",
    ),
    ("cs_car12", "/Game/CitySampleVehicles/vehicle12_Car/Mesh/SM_vehCar_vehicle12_LOD"),
    ("cs_car13", "/Game/CitySampleVehicles/vehicle13_Car/Mesh/SM_vehCar_vehicle13_LOD"),
]

VEHICLE_LABEL_HINTS = (
    "car",
    "truck",
    "bus",
    "vehicle",
    "van",
    "suv",
    "sedan",
    "pickup",
    "hatchback",
    "sports",
    "demolition",
    "vv_",
    "cs_",
)
RADIUS_CM = 1500.0
JITTER_CM = 80.0


def _label(a: Any) -> str:
    try:
        return str(a.get_actor_label())
    except Exception:
        return str(a.get_name())


def _is_vehicle_mesh(actor: Any) -> bool:
    if not isinstance(actor, unreal.StaticMeshActor):
        return False
    lower = _label(actor).lower()
    return any(h in lower for h in VEHICLE_LABEL_HINTS)


def _find_capture(actors: list[Any]) -> Any | None:
    for a in actors:
        if str(a.get_class().get_name()).lower() == "cameracapturewheels_c":
            return a
    return None


def main() -> None:
    report: dict[str, Any] = {
        "map": MAP_PATH,
        "spawned": [],
        "skipped": [],
        "status": "FAIL",
    }
    try:
        unreal.EditorLoadingAndSavingUtils.load_map(MAP_PATH)
        actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        existing_labels = {_label(a) for a in actor_subsystem.get_all_level_actors()}

        n = len(ALL_VEHICLES)
        for i, (label, mesh_path) in enumerate(ALL_VEHICLES):
            if label in existing_labels:
                report["skipped"].append({"label": label, "reason": "already_present"})
                continue
            mesh = unreal.EditorAssetLibrary.load_asset(mesh_path)
            if mesh is None:
                report["skipped"].append(
                    {"label": label, "reason": f"mesh_load_failed:{mesh_path}"}
                )
                continue
            angle = 2.0 * math.pi * (i / n)
            x = RADIUS_CM * math.cos(angle)
            y = RADIUS_CM * math.sin(angle)
            # Tiny offset so identical labels don't perfectly stack.
            x += JITTER_CM * math.cos(angle * 5.0)
            y += JITTER_CM * math.sin(angle * 5.0)
            yaw = math.degrees(angle) + 90.0
            loc = unreal.Vector(x, y, 0.0)
            rot = unreal.Rotator(0.0, 0.0, yaw)
            actor = actor_subsystem.spawn_actor_from_object(mesh, loc, rot)
            if actor is None:
                report["skipped"].append(
                    {"label": label, "reason": "spawn_returned_none"}
                )
                continue
            try:
                actor.set_actor_label(label)
            except Exception:
                pass
            report["spawned"].append(
                {
                    "label": label,
                    "mesh": mesh_path,
                    "location": [round(x, 1), round(y, 1), 0.0],
                    "yaw": round(yaw, 1),
                }
            )

        actors = list(actor_subsystem.get_all_level_actors())
        capture = _find_capture(actors)
        vehicles = [a for a in actors if _is_vehicle_mesh(a)]
        report["vehicle_inventory"] = sorted(_label(v) for v in vehicles)

        if capture is not None and vehicles:
            arr = unreal.Array(unreal.Actor)
            for v in vehicles:
                arr.append(v)
            capture.set_editor_property("RotateObjects", arr)
            try:
                capture.set_editor_property("ToRotate", None)
            except Exception:
                pass
            report["rotate_objects_count"] = len(vehicles)

        saved = unreal.EditorLoadingAndSavingUtils.save_current_level()
        report["saved"] = bool(saved)
        report["status"] = "PASS" if saved else "PARTIAL_SAVE_FAILED"
    except Exception as exc:
        report["error"] = str(exc)
        report["status"] = "FAIL"
    finally:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"WROTE {OUT_PATH}")
        try:
            if hasattr(unreal.SystemLibrary, "quit_editor"):
                unreal.SystemLibrary.quit_editor()
        except Exception:
            pass


main()
