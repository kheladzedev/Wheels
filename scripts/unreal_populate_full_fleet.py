"""Final inventory populate: 22 vehicles + 1 office chair (negative sample
distractor) into standartWheelsRoom.

After the 18-vehicle ring blocked the CameraCaptureWheels spline, this
script uses a tighter dual-ring layout: inner ring radius 600cm (for VVP +
small CS cars), outer ring 1400cm (for trucks / bus / vans). Office chair
spawned at +100 from origin as a within-frame distractor.

Run via -ExecutePythonScript so UE renders normally. Calls quit_editor() at
the end.
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
        "/Users/edward/Desktop/VSBL/outputs/unreal_control/populate_full_fleet.json",
    )
)

# (label, mesh path, ring: "inner"/"outer", is_negative)
ALL_ASSETS = [
    # VVP small fleet — inner ring
    ("vv_pickup", "/Game/VehicleVarietyPack/Meshes/SM_Pickup", "inner", False),
    ("vv_hatchback", "/Game/VehicleVarietyPack/Meshes/SM_Hatchback", "inner", False),
    ("vv_sportscar", "/Game/VehicleVarietyPack/Meshes/SM_SportsCar", "inner", False),
    ("vv_suv", "/Game/VehicleVarietyPack/Meshes/SM_SUV", "inner", False),
    # CS Cars — inner ring
    (
        "cs_car02",
        "/Game/CitySampleVehicles/vehicle02_Car/Mesh/SM_vehCar_vehicle02_LOD",
        "inner",
        False,
    ),
    (
        "cs_car03",
        "/Game/CitySampleVehicles/vehicle03_Car/Mesh/SM_vehCar_vehicle03_LOD",
        "inner",
        False,
    ),
    (
        "cs_car05",
        "/Game/CitySampleVehicles/vehicle05_Car/Mesh/SM_vehCar_vehicle05_LOD",
        "inner",
        False,
    ),
    (
        "cs_car06",
        "/Game/CitySampleVehicles/vehicle06_Car/Mesh/SM_vehCar_vehicle06_LOD",
        "inner",
        False,
    ),
    (
        "cs_car12",
        "/Game/CitySampleVehicles/vehicle12_Car/Mesh/SM_vehCar_vehicle12_LOD",
        "inner",
        False,
    ),
    (
        "cs_car13",
        "/Game/CitySampleVehicles/vehicle13_Car/Mesh/SM_vehCar_vehicle13_LOD",
        "inner",
        False,
    ),
    # Previously-missed Wheels/assets fleet — inner ring (legacy hero cars)
    (
        "legacy_police_car",
        "/Game/Wheels/assets/police_car/StaticMeshes/police_car",
        "inner",
        False,
    ),
    ("legacy_medik", "/Game/Wheels/assets/medik/StaticMeshes/source", "inner", False),
    ("legacy_scan", "/Game/Wheels/assets/scan/StaticMeshes/scan", "inner", False),
    (
        "legacy_modelsCars",
        "/Game/Wheels/assets/modelsCars/StaticMeshes/modelsCars",
        "inner",
        False,
    ),
    # VVP large + CS commercial — outer ring
    ("vv_truck_box", "/Game/VehicleVarietyPack/Meshes/SM_Truck_Box", "outer", False),
    (
        "cs_truck04",
        "/Game/CitySampleVehicles/vehicle04_Truck/Mesh/SM_vehTruck_vehicle04_LOD",
        "outer",
        False,
    ),
    (
        "cs_truck08",
        "/Game/CitySampleVehicles/vehicle08_Truck/Mesh/SM_vehTruck_vehicle08_LOD",
        "outer",
        False,
    ),
    (
        "cs_truck11",
        "/Game/CitySampleVehicles/vehicle11_Truck/Mesh/SM_vehTruck_vehicle11_LOD",
        "outer",
        False,
    ),
    (
        "cs_van01",
        "/Game/CitySampleVehicles/vehicle01_Van/Mesh/SM_vehVan_vehicle01_LOD",
        "outer",
        False,
    ),
    (
        "cs_van09",
        "/Game/CitySampleVehicles/vehicle09_Van/Mesh/SM_vehVan_vehicle09_LOD",
        "outer",
        False,
    ),
    (
        "cs_bus10",
        "/Game/CitySampleVehicles/vehicle10_Bus/Mesh/SM_vehBus_vehicle10_LOD",
        "outer",
        False,
    ),
    # Negative-sample: office chair with caster wheels — placed at origin
    (
        "neg_office_chair",
        "/Game/FreeFurniturePack/Meshes/SM_Modern_Office_chair",
        "center",
        True,
    ),
]

INNER_RADIUS = 600.0
OUTER_RADIUS = 1400.0
JITTER = 40.0
CHAIR_OFFSET = [100.0, 100.0, 0.0]


def _label(a: Any) -> str:
    try:
        return str(a.get_actor_label())
    except Exception:
        return str(a.get_name())


def _is_vehicle(actor: Any) -> bool:
    if not isinstance(actor, unreal.StaticMeshActor):
        return False
    label = _label(actor).lower()
    return any(
        h in label
        for h in (
            "car",
            "truck",
            "bus",
            "van",
            "vehicle",
            "pickup",
            "hatchback",
            "sports",
            "demolition",
            "vv_",
            "cs_",
            "legacy_",
            "suv",
        )
    )


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

        # Count per ring for angle calc
        inner_assets = [a for a in ALL_ASSETS if a[2] == "inner"]
        outer_assets = [a for a in ALL_ASSETS if a[2] == "outer"]
        n_inner = len(inner_assets)
        n_outer = len(outer_assets)

        for asset_idx, (label, mesh_path, ring, is_neg) in enumerate(ALL_ASSETS):
            if label in existing_labels:
                report["skipped"].append({"label": label, "reason": "already_present"})
                continue
            mesh = unreal.EditorAssetLibrary.load_asset(mesh_path)
            if mesh is None:
                report["skipped"].append(
                    {"label": label, "reason": f"mesh_load_failed:{mesh_path}"}
                )
                continue
            if ring == "center":
                loc = unreal.Vector(*CHAIR_OFFSET)
                yaw = 45.0
            else:
                if ring == "inner":
                    i = inner_assets.index((label, mesh_path, ring, is_neg))
                    angle = 2.0 * math.pi * (i / n_inner)
                    r = INNER_RADIUS
                else:
                    i = outer_assets.index((label, mesh_path, ring, is_neg))
                    angle = 2.0 * math.pi * (i / n_outer) + math.pi / n_outer
                    r = OUTER_RADIUS
                x = r * math.cos(angle) + JITTER * math.cos(angle * 5)
                y = r * math.sin(angle) + JITTER * math.sin(angle * 5)
                loc = unreal.Vector(x, y, 0.0)
                yaw = math.degrees(angle) + 90.0
            actor = actor_subsystem.spawn_actor_from_object(
                mesh, loc, unreal.Rotator(0.0, 0.0, yaw)
            )
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
                {"label": label, "mesh": mesh_path, "ring": ring, "negative": is_neg}
            )

        # Refresh RotateObjects with vehicles only (exclude office chair so it stays still
        # as a stable negative distractor).
        actors = list(actor_subsystem.get_all_level_actors())
        capture = _find_capture(actors)
        vehicles = [a for a in actors if _is_vehicle(a)]
        report["vehicle_inventory"] = sorted(_label(v) for v in vehicles)
        report["chair_present"] = any(_label(a) == "neg_office_chair" for a in actors)

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
