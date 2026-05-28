"""Find vehicle actors in NewMap3, move CameraCapture to look at one of them, capture PNG.

A "vehicle" is recognized by label substring or by having a SkeletalMeshComponent
with a mesh path under /Game/SketchfabCars or /Game/CitySampleVehicles or
/Game/VehicleVarietyPack.
"""

import math
import time
from pathlib import Path

import unreal

OUT = Path("/Users/codefactory/Desktop/ML/VSBL/Wheels/outputs/ue_capture_test")
OUT.mkdir(parents=True, exist_ok=True)

VEHICLE_HINTS = (
    "sketchfab",
    "vehicle",
    "car",
    "asset",
    "bmw",
    "audi",
    "ford",
    "lambo",
    "toyota",
    "chevy",
    "drifter",
)
VEHICLE_MESH_PATHS = (
    "/Game/SketchfabCars/",
    "/Game/CitySampleVehicles/",
    "/Game/VehicleVarietyPack/",
)


def is_vehicle(actor) -> bool:
    label = (actor.get_actor_label() or "").lower()
    if any(h in label for h in VEHICLE_HINTS):
        return True
    for comp in actor.get_components_by_class(unreal.StaticMeshComponent):
        mesh = comp.get_editor_property("static_mesh")
        if mesh is None:
            continue
        path = mesh.get_path_name()
        if any(path.startswith(p) for p in VEHICLE_MESH_PATHS):
            return True
    for comp in actor.get_components_by_class(unreal.SkeletalMeshComponent):
        mesh = comp.get_editor_property("skeletal_mesh") or comp.get_editor_property(
            "skinned_asset"
        )
        if mesh is None:
            continue
        path = mesh.get_path_name()
        if any(path.startswith(p) for p in VEHICLE_MESH_PATHS):
            return True
    return False


def actor_extent_radius(actor) -> tuple[unreal.Vector, float]:
    origin, box_extent = actor.get_actor_bounds(True)
    radius = math.sqrt(box_extent.x**2 + box_extent.y**2 + box_extent.z**2)
    return origin, radius


def look_at_rotation(eye: unreal.Vector, target: unreal.Vector) -> unreal.Rotator:
    dx, dy, dz = target.x - eye.x, target.y - eye.y, target.z - eye.z
    yaw = math.degrees(math.atan2(dy, dx))
    horiz = math.sqrt(dx * dx + dy * dy)
    pitch = math.degrees(math.atan2(dz, horiz))
    return unreal.Rotator(roll=0.0, pitch=pitch, yaw=yaw)


sub = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
world = sub.get_editor_world()
actors = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor)

vehicles = [a for a in actors if a is not None and is_vehicle(a)]
unreal.log(f"[aim] found {len(vehicles)} vehicle candidates")
for v in vehicles[:10]:
    origin, radius = actor_extent_radius(v)
    unreal.log(
        f"  - {v.get_actor_label()} <{v.get_class().get_name()}> "
        f"loc=({v.get_actor_location().x:.0f},{v.get_actor_location().y:.0f},{v.get_actor_location().z:.0f}) "
        f"radius={radius:.1f}"
    )

cap = next(
    (
        a
        for a in actors
        if (a.get_actor_label() or "").lower().startswith("cameracapture")
    ),
    None,
)
if cap is None:
    unreal.log_error("[aim] no CameraCapture")
elif not vehicles:
    unreal.log_warning("[aim] no vehicles to aim at — falling back to scene origin")
else:
    target_actor = vehicles[0]
    origin, radius = actor_extent_radius(target_actor)
    # Place camera ~radius*3 away on a 30deg pitched orbit
    distance = max(radius * 3.0, 250.0)
    yaw_deg = 35.0
    pitch_deg = -15.0
    yr = math.radians(yaw_deg)
    pr = math.radians(pitch_deg)
    offset = unreal.Vector(
        x=distance * math.cos(pr) * math.cos(yr),
        y=distance * math.cos(pr) * math.sin(yr),
        z=-distance * math.sin(pr),
    )
    eye = unreal.Vector(origin.x + offset.x, origin.y + offset.y, origin.z + offset.z)
    rot = look_at_rotation(eye, origin)
    cap.set_actor_location(eye, sweep=False, teleport=True)
    cap.set_actor_rotation(rot, teleport_physics=True)
    unreal.log(
        f"[aim] camera at ({eye.x:.0f},{eye.y:.0f},{eye.z:.0f}) rot=({rot.pitch:.1f},{rot.yaw:.1f}) "
        f"looking at {target_actor.get_actor_label()}"
    )

    comp = cap.get_components_by_class(unreal.SceneCaptureComponent2D)[0]
    comp.capture_scene()
    time.sleep(0.4)
    rt = comp.get_editor_property("texture_target")
    out_path = OUT / f"aim_{time.strftime('%H%M%S')}.png"
    unreal.RenderingLibrary.export_render_target(
        world, rt, str(out_path.parent), out_path.name
    )
    if out_path.is_file():
        unreal.log(f"[aim] saved -> {out_path} bytes={out_path.stat().st_size}")
    else:
        unreal.log_error(f"[aim] no file at {out_path}")
