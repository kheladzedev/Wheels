"""Try spawning a standalone wheel Blueprint (BP_SC_Wheel_Front).

If wheels are non-Pawn BPs (just StaticMeshActor + WheelComponent), they should
spawn fine in editor world. A single wheel on a floor is enough for keypoint
training — it is literally a wheel-detection dataset.
"""

import math
import time
from pathlib import Path

import unreal

OUT = Path("/Users/codefactory/Desktop/ML/VSBL/Wheels/outputs/ue_capture_test")
OUT.mkdir(parents=True, exist_ok=True)


def look_at(eye, target):
    dx, dy, dz = target.x - eye.x, target.y - eye.y, target.z - eye.z
    yaw = math.degrees(math.atan2(dy, dx))
    horiz = math.sqrt(dx * dx + dy * dy)
    pitch = math.degrees(math.atan2(dz, horiz))
    return unreal.Rotator(roll=0.0, pitch=pitch, yaw=yaw)


candidates = [
    "/Game/VehicleVarietyPack/Blueprints/SportsCar/BP_SC_Wheel_Front",
    "/Game/VehicleVarietyPack/Blueprints/SportsCar/BP_SC_Wheel_Rear",
    "/Game/VehicleVarietyPack/Blueprints/Pickup/BP_PU_Wheel_Front",
    "/Game/VehicleVarietyPack/Blueprints/Pickup/BP_PU_Wheel_Rear",
    "/Game/VehicleVarietyPack/Blueprints/SUV/BP_SUV_Wheel_Front",
    "/Game/VehicleVarietyPack/Blueprints/SUV/BP_SUV_Wheel_Rear",
    "/Game/VehicleVarietyPack/Blueprints/BoxTruck/BP_BT_RearTire",
]

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
cap_loc = cap.get_actor_location()
base = unreal.Vector(cap_loc.x + 350.0, cap_loc.y + 100.0, 44610.0)

spawned: list = []
for i, pkg in enumerate(candidates):
    asset = unreal.EditorAssetLibrary.load_asset(pkg)
    if asset is None:
        unreal.log_warning(f"[wheel] cant load {pkg}")
        continue
    loc = unreal.Vector(base.x + i * 200.0, base.y, base.z)
    actor = None
    try:
        actor = unreal.EditorLevelLibrary.spawn_actor_from_object(asset, loc)
    except Exception as exc:
        unreal.log_warning(f"[wheel] spawn_actor_from_object({pkg}) failed: {exc}")
    if actor is None:
        # Try the alt _C path
        try:
            cls_obj = unreal.load_object(None, f"{pkg}.{pkg.rsplit('/', 1)[-1]}_C")
            actor = unreal.EditorLevelLibrary.spawn_actor_from_class(cls_obj, loc)
            unreal.log(f"[wheel] spawn via _C class -> {actor}")
        except Exception as exc:
            unreal.log_warning(f"[wheel] _C spawn failed: {exc}")
    if actor is None:
        unreal.log_error(f"[wheel] ALL spawn methods failed for {pkg}")
        continue
    actor.set_actor_label(f"WheelTest_{i}")
    spawned.append(actor)
    origin, ext = actor.get_actor_bounds(True)
    unreal.log(
        f"[wheel] spawned {pkg.split('/')[-1]} at ({loc.x:.0f},{loc.y:.0f},{loc.z:.0f})  "
        f"ext=({ext.x:.0f},{ext.y:.0f},{ext.z:.0f})"
    )

unreal.log(f"[wheel] total spawned: {len(spawned)}")

if spawned:
    # Aim camera at first wheel
    first = spawned[0]
    origin, ext = first.get_actor_bounds(True)
    radius = max(math.sqrt(ext.x**2 + ext.y**2 + ext.z**2), 100.0)
    distance = radius * 4.0
    eye = unreal.Vector(
        origin.x + distance * 0.5, origin.y - distance * 0.8, origin.z + distance * 0.3
    )
    rot = look_at(eye, origin)
    cap.set_actor_location(eye, sweep=False, teleport=True)
    cap.set_actor_rotation(rot, teleport_physics=True)
    comp = cap.get_components_by_class(unreal.SceneCaptureComponent2D)[0]
    comp.capture_scene()
    time.sleep(0.5)
    rt = comp.get_editor_property("texture_target")
    out_path = OUT / f"wheel_only_{time.strftime('%H%M%S')}.png"
    unreal.RenderingLibrary.export_render_target(
        world, rt, str(out_path.parent), out_path.name
    )
    if out_path.is_file():
        unreal.log(f"[capture] -> {out_path} bytes={out_path.stat().st_size}")
