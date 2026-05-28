"""Find a StaticMesh asset in /Game/SketchfabCars and spawn it next to the
CameraCapture, then aim camera and capture a frame."""

import math
import time
from pathlib import Path

import unreal

OUT = Path("/Users/codefactory/Desktop/ML/VSBL/Wheels/outputs/ue_capture_test")
OUT.mkdir(parents=True, exist_ok=True)


def list_static_meshes_under(folder: str, limit: int = 30) -> list:
    ar = unreal.AssetRegistryHelpers.get_asset_registry()
    assets = ar.get_assets_by_path(folder, recursive=True)
    out = []
    for a in assets:
        try:
            cls_name = a.asset_class_path.asset_name
        except AttributeError:
            cls_name = ""
        if str(cls_name) == "StaticMesh":
            out.append(str(a.package_name) + "." + str(a.asset_name))
        if len(out) >= limit:
            break
    return out


def look_at(eye: unreal.Vector, target: unreal.Vector) -> unreal.Rotator:
    dx, dy, dz = target.x - eye.x, target.y - eye.y, target.z - eye.z
    yaw = math.degrees(math.atan2(dy, dx))
    horiz = math.sqrt(dx * dx + dy * dy)
    pitch = math.degrees(math.atan2(dz, horiz))
    return unreal.Rotator(roll=0.0, pitch=pitch, yaw=yaw)


meshes = list_static_meshes_under("/Game/SketchfabCars", limit=40)
unreal.log(f"[mesh] {len(meshes)} StaticMesh under /Game/SketchfabCars")
for m in meshes[:10]:
    unreal.log(f"  - {m}")

if not meshes:
    unreal.log_error("[mesh] no car StaticMeshes — try other path")
else:
    sub = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    world = sub.get_editor_world()
    actors = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor)
    cap = next(
        (
            a
            for a in actors
            if (a.get_actor_label() or "").lower().startswith("cameracapture")
        ),
        None,
    )
    cap_loc = cap.get_actor_location() if cap else unreal.Vector(0, 0, 0)

    # Spawn first 4 mesh assets near camera (offsets in a row)
    for idx, mesh_path in enumerate(meshes[:4]):
        mesh = unreal.EditorAssetLibrary.load_asset(mesh_path)
        if mesh is None:
            unreal.log_warning(f"  cant load {mesh_path}")
            continue
        loc = unreal.Vector(
            cap_loc.x + 250.0 + idx * 350.0, cap_loc.y + 600.0, cap_loc.z - 50.0
        )
        actor = unreal.EditorLevelLibrary.spawn_actor_from_object(mesh, loc)
        actor.set_actor_label(f"CarSpawn_{idx}")
        bounds = actor.get_actor_bounds(True)
        unreal.log(
            f"  spawned {mesh.get_name()} at ({loc.x:.0f},{loc.y:.0f},{loc.z:.0f}) "
            f"ext=({bounds[1].x:.0f},{bounds[1].y:.0f},{bounds[1].z:.0f})"
        )

    # Aim camera at first spawned car
    actors = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor)
    first_car = next(
        (a for a in actors if (a.get_actor_label() or "").startswith("CarSpawn_0")),
        None,
    )
    if first_car and cap:
        origin, ext = first_car.get_actor_bounds(True)
        radius = max(math.sqrt(ext.x**2 + ext.y**2 + ext.z**2), 200.0)
        distance = radius * 3.0
        eye = unreal.Vector(
            origin.x + distance * 0.6,
            origin.y - distance * 0.8,
            origin.z + distance * 0.4,
        )
        rot = look_at(eye, origin)
        cap.set_actor_location(eye, sweep=False, teleport=True)
        cap.set_actor_rotation(rot, teleport_physics=True)
        unreal.log(
            f"[aim] cam=({eye.x:.0f},{eye.y:.0f},{eye.z:.0f}) -> car at ({origin.x:.0f},{origin.y:.0f},{origin.z:.0f}) radius={radius:.1f}"
        )
        comp = cap.get_components_by_class(unreal.SceneCaptureComponent2D)[0]
        comp.capture_scene()
        time.sleep(0.4)
        rt = comp.get_editor_property("texture_target")
        out_path = OUT / f"car_{time.strftime('%H%M%S')}.png"
        unreal.RenderingLibrary.export_render_target(
            world, rt, str(out_path.parent), out_path.name
        )
        if out_path.is_file():
            unreal.log(f"[capture] -> {out_path} bytes={out_path.stat().st_size}")
