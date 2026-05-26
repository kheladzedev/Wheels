"""Spawn additional vehicle StaticMeshActors into standartWheelsRoom for visual
diversity, and refresh CameraCaptureWheels.RotateObjects so every car rotates
each frame.

The existing wheels markers remain unchanged. They are scene-anchored and
already produce correct A/B/C labels; the new vehicles are background variety
so the network stops over-fitting on a single mesh + lighting combination.

Idempotent: if a vehicle with the desired label already exists, the script
leaves it in place.

  VSBL_POPULATE_MAP=/Game/Wheels/maps/standartWheelsRoom \
  "/Users/Shared/Epic Games/UE_5.7/Engine/Binaries/Mac/UnrealEditor-Cmd" \
    "/Users/edward/Desktop/VSBL/NeuralData1 2/NeuralData.uproject" \
    -run=pythonscript -script=/Users/edward/Desktop/VSBL/scripts/unreal_populate_vehicles.py \
    -unattended -nop4 -nosplash -NullRHI
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import unreal


# Each entry: (label, mesh asset path, world location, yaw degrees)
# Positions are spread around the existing wheels markers so each capture frame
# is likely to see several vehicles. demolition_derby_car already exists.
NEW_VEHICLES = [
    (
        "vv_pickup",
        "/Game/VehicleVarietyPack/Meshes/SM_Pickup",
        [-300.0, 200.0, 0.0],
        30.0,
    ),
    (
        "vv_hatchback",
        "/Game/VehicleVarietyPack/Meshes/SM_Hatchback",
        [300.0, -100.0, 0.0],
        -75.0,
    ),
    (
        "vv_sportscar",
        "/Game/VehicleVarietyPack/Meshes/SM_SportsCar",
        [-200.0, 450.0, 0.0],
        110.0,
    ),
    (
        "vv_truck_box",
        "/Game/VehicleVarietyPack/Meshes/SM_Truck_Box",
        [400.0, 350.0, 0.0],
        -160.0,
    ),
    ("vv_suv", "/Game/VehicleVarietyPack/Meshes/SM_SUV", [-100.0, -350.0, 0.0], 200.0),
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
)


def _label(a: Any) -> str:
    try:
        return str(a.get_actor_label())
    except Exception:
        return str(a.get_name())


def _is_vehicle_mesh(actor: Any) -> bool:
    if not isinstance(actor, unreal.StaticMeshActor):
        return False
    label = _label(actor).lower()
    return any(h in label for h in VEHICLE_LABEL_HINTS)


def _find_capture(actors: list[Any]) -> Any | None:
    for a in actors:
        cls = str(a.get_class().get_name()).lower()
        if cls == "cameracapturewheels_c":
            return a
    return None


def main() -> None:
    map_path = os.environ.get(
        "VSBL_POPULATE_MAP", "/Game/Wheels/maps/standartWheelsRoom"
    )
    out_path = Path(
        os.environ.get(
            "VSBL_POPULATE_OUT",
            "/Users/edward/Desktop/VSBL/outputs/unreal_control/populate_vehicles.json",
        )
    )

    report: dict[str, Any] = {
        "map": map_path,
        "spawned": [],
        "skipped": [],
        "status": "FAIL",
    }

    try:
        unreal.EditorLoadingAndSavingUtils.load_map(map_path)
        actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        existing_labels = {_label(a) for a in actor_subsystem.get_all_level_actors()}

        for label, mesh_path, loc, yaw in NEW_VEHICLES:
            if label in existing_labels:
                report["skipped"].append({"label": label, "reason": "already_present"})
                continue
            mesh = unreal.EditorAssetLibrary.load_asset(mesh_path)
            if mesh is None:
                report["skipped"].append(
                    {"label": label, "reason": f"mesh_load_failed:{mesh_path}"}
                )
                continue
            world_loc = unreal.Vector(loc[0], loc[1], loc[2])
            rot = unreal.Rotator(0.0, 0.0, yaw)
            actor = actor_subsystem.spawn_actor_from_object(mesh, world_loc, rot)
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
                    "location": list(loc),
                    "yaw": yaw,
                }
            )

        # Refresh RotateObjects with every vehicle now in the scene.
        actors = list(actor_subsystem.get_all_level_actors())
        capture = _find_capture(actors)
        vehicles = [a for a in actors if _is_vehicle_mesh(a)]
        report["vehicle_inventory"] = [_label(v) for v in vehicles]

        if capture is not None and vehicles:
            new_arr = unreal.Array(unreal.Actor)
            for v in vehicles:
                new_arr.append(v)
            capture.set_editor_property("RotateObjects", new_arr)
            try:
                capture.set_editor_property("ToRotate", None)
            except Exception:
                pass
            report["camera_capture_actor"] = _label(capture)
            report["rotate_objects_count"] = len(vehicles)

        saved = unreal.EditorLoadingAndSavingUtils.save_current_level()
        report["saved"] = bool(saved)
        report["status"] = "PASS" if saved else "PARTIAL_SAVE_FAILED"
    except Exception as exc:
        report["error"] = str(exc)
        report["status"] = "FAIL"
    finally:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"WROTE {out_path}")
        # When run via -ExecutePythonScript in a normal editor process,
        # quit the editor when finished so the shell command returns.
        try:
            if hasattr(unreal.SystemLibrary, "quit_editor"):
                unreal.SystemLibrary.quit_editor()
        except Exception:
            pass


main()
