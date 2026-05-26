"""Discover every StaticMesh under /Game/VehicleVarietyPack, /Game/CitySampleVehicles,
/Game/Wheels/assets and /Game/SketchfabCars, then spawn each into
standartWheelsRoom in a multi-ring layout sized to the fleet count.

This is the fleet-agnostic populate. After importing more vehicles (Sketchfab,
new packs, etc.) just run this script and the map adapts.

  "/Users/Shared/Epic Games/UE_5.7/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor" \\
    "/Users/edward/Desktop/VSBL/NeuralData1 2/NeuralData.uproject" \\
    -ExecutePythonScript="/Users/edward/Desktop/VSBL/scripts/unreal_populate_all_imported.py" \\
    -nosplash
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
        "/Users/edward/Desktop/VSBL/outputs/unreal_control/populate_all_imported.json",
    )
)

# Source folders + a filename predicate so we pick the vehicle BODY mesh and
# skip per-wheel components / no-wheel proxies. Each entry: (root, body_filter).
SOURCE_ROOTS = [
    (
        "/Game/VehicleVarietyPack/Meshes",
        lambda n: n.startswith("SM_") and "Floorplane" not in n,
    ),
    (
        "/Game/CitySampleVehicles",
        lambda n: n.startswith("SM_veh") and n.endswith("_LOD"),
    ),
    (
        "/Game/Wheels/assets",
        lambda n: (
            not any(s in n for s in ("Material", "Texture", "_Mat", "_Inst"))
            and n not in ("modelsCars",)
            or n
            in ("demolition_derby_car", "police_car", "source", "scan", "modelsCars")
        ),
    ),
    (
        "/Game/SketchfabCars",
        lambda n: n.startswith("SM_") or "_LOD" in n or True,
    ),  # take everything imported
]

INNER_RADIUS = 600.0
OUTER_RADIUS = 1400.0
EXTRA_RADIUS = 2200.0


def _list_static_meshes(root: str, predicate) -> list[str]:
    if not unreal.EditorAssetLibrary.does_directory_exist(root):
        return []
    asset_paths = unreal.EditorAssetLibrary.list_assets(
        root, recursive=True, include_folder=False
    )
    out: list[str] = []
    for p in asset_paths:
        # asset paths look like "/Game/.../Name.Name" — strip class suffix
        clean = p.split(".")[0]
        name = clean.rsplit("/", 1)[-1]
        if not predicate(name):
            continue
        try:
            asset = unreal.EditorAssetLibrary.load_asset(clean)
        except Exception:
            continue
        if isinstance(asset, unreal.StaticMesh):
            out.append(clean)
    return sorted(set(out))


def _label_from_path(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    return name[:60]


def _is_vehicle(actor: Any) -> bool:
    if not isinstance(actor, unreal.StaticMeshActor):
        return False
    try:
        lbl = str(actor.get_actor_label()).lower()
    except Exception:
        return False
    return any(
        h in lbl
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
            "suv",
            "sm_",
            "sk_",
            "modelscars",
            "police",
            "source",
            "scan",
        )
    )


def _find_capture(actors):
    for a in actors:
        if str(a.get_class().get_name()).lower() == "cameracapturewheels_c":
            return a
    return None


def main() -> None:
    report: dict[str, Any] = {
        "map": MAP_PATH,
        "discovered": {},
        "spawned": [],
        "skipped": [],
        "status": "FAIL",
    }
    try:
        unreal.EditorLoadingAndSavingUtils.load_map(MAP_PATH)
        all_mesh_paths: list[str] = []
        for root, pred in SOURCE_ROOTS:
            meshes = _list_static_meshes(root, pred)
            report["discovered"][root] = len(meshes)
            all_mesh_paths.extend(meshes)
        all_mesh_paths = sorted(set(all_mesh_paths))
        report["total_meshes"] = len(all_mesh_paths)

        actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        existing_labels = {
            str(a.get_actor_label()) for a in actor_subsystem.get_all_level_actors()
        }

        # Split into 3 rings.
        n = len(all_mesh_paths)
        rings = [[], [], []]
        for i, mesh_path in enumerate(all_mesh_paths):
            rings[i % 3].append(mesh_path)
        radii = [INNER_RADIUS, OUTER_RADIUS, EXTRA_RADIUS]

        for ring_idx, ring in enumerate(rings):
            r = radii[ring_idx]
            m = len(ring)
            for i, mesh_path in enumerate(ring):
                label = _label_from_path(mesh_path)
                if label in existing_labels:
                    report["skipped"].append(
                        {"label": label, "reason": "already_present"}
                    )
                    continue
                mesh = unreal.EditorAssetLibrary.load_asset(mesh_path)
                if mesh is None:
                    report["skipped"].append({"label": label, "reason": "load_failed"})
                    continue
                angle = 2.0 * math.pi * (i / max(1, m)) + (
                    ring_idx * math.pi / max(1, m)
                )
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                yaw = math.degrees(angle) + 90.0
                actor = actor_subsystem.spawn_actor_from_object(
                    mesh, unreal.Vector(x, y, 0.0), unreal.Rotator(0.0, 0.0, yaw)
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
                    {
                        "label": label,
                        "mesh": mesh_path,
                        "ring": ring_idx,
                        "loc": [round(x, 1), round(y, 1), 0.0],
                    }
                )

        actors = list(actor_subsystem.get_all_level_actors())
        capture = _find_capture(actors)
        vehicles = [a for a in actors if _is_vehicle(a)]
        report["vehicle_inventory_count"] = len(vehicles)

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
