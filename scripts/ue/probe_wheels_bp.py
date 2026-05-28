"""Probe what /Game/Wheels/main/wheels actually is + try to spawn it."""

import math
import time
from pathlib import Path

import unreal

OUT = Path("/Users/codefactory/Desktop/ML/VSBL/Wheels/outputs/ue_capture_test")
OUT.mkdir(parents=True, exist_ok=True)

WHEELS_BP = "/Game/Wheels/main/wheels"


def look_at(eye, target):
    dx, dy, dz = target.x - eye.x, target.y - eye.y, target.z - eye.z
    yaw = math.degrees(math.atan2(dy, dx))
    horiz = math.sqrt(dx * dx + dy * dy)
    pitch = math.degrees(math.atan2(dz, horiz))
    return unreal.Rotator(roll=0.0, pitch=pitch, yaw=yaw)


bp = unreal.EditorAssetLibrary.load_asset(WHEELS_BP)
unreal.log(f"[probe] wheels asset = {bp}")
if bp is not None:
    parent = bp.get_editor_property("parent_class")
    unreal.log(f"  parent_class = {parent}")

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
spawn_loc = unreal.Vector(cap_loc.x + 400.0, cap_loc.y, 44590.0)

spawned = None
try:
    spawned = unreal.EditorLevelLibrary.spawn_actor_from_object(bp, spawn_loc)
    unreal.log(f"[probe] spawn_actor_from_object -> {spawned}")
except Exception as exc:
    unreal.log_warning(f"[probe] spawn failed: {exc}")

if spawned is not None:
    spawned.set_actor_label("WheelsBP_test")
    origin, ext = spawned.get_actor_bounds(True)
    unreal.log(
        f"[spawn] wheels ext=({ext.x:.0f},{ext.y:.0f},{ext.z:.0f}) origin=({origin.x:.0f},{origin.y:.0f},{origin.z:.0f})"
    )
    radius = max(math.sqrt(ext.x**2 + ext.y**2 + ext.z**2), 200.0)
    distance = radius * 3.0
    eye = unreal.Vector(
        origin.x + distance * 0.5, origin.y - distance * 0.8, origin.z + distance * 0.3
    )
    rot = look_at(eye, origin)
    cap.set_actor_location(eye, sweep=False, teleport=True)
    cap.set_actor_rotation(rot, teleport_physics=True)
    comp = cap.get_components_by_class(unreal.SceneCaptureComponent2D)[0]
    comp.capture_scene()
    time.sleep(0.6)
    rt = comp.get_editor_property("texture_target")
    out_path = OUT / f"wheels_{time.strftime('%H%M%S')}.png"
    unreal.RenderingLibrary.export_render_target(
        world, rt, str(out_path.parent), out_path.name
    )
    if out_path.is_file():
        unreal.log(f"[capture] -> {out_path} bytes={out_path.stat().st_size}")
