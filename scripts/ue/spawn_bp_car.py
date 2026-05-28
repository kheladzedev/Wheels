"""Spawn one BP_SUV using whichever UE-Python spawn API works in 5.7."""

import math
import time
from pathlib import Path

import unreal

OUT = Path("/Users/codefactory/Desktop/ML/VSBL/Wheels/outputs/ue_capture_test")
OUT.mkdir(parents=True, exist_ok=True)

PACKAGE = "/Game/VehicleVarietyPack/Blueprints/SUV/BP_SUV"


def look_at(eye, target):
    dx, dy, dz = target.x - eye.x, target.y - eye.y, target.z - eye.z
    yaw = math.degrees(math.atan2(dy, dx))
    horiz = math.sqrt(dx * dx + dy * dy)
    pitch = math.degrees(math.atan2(dz, horiz))
    return unreal.Rotator(roll=0.0, pitch=pitch, yaw=yaw)


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
if cap is None:
    raise SystemExit("no CameraCapture")
cap_loc = cap.get_actor_location()
spawn_loc = unreal.Vector(cap_loc.x + 500.0, cap_loc.y, 44590.0)
spawn_rot = unreal.Rotator(0.0, 30.0, 0.0)

bp_asset = unreal.EditorAssetLibrary.load_asset(PACKAGE)
unreal.log(f"[bp] asset={bp_asset}")

suv = None
# Attempt 1: spawn_actor_from_object — accepts the Blueprint asset directly
try:
    suv = unreal.EditorLevelLibrary.spawn_actor_from_object(
        bp_asset, spawn_loc, spawn_rot
    )
    unreal.log(f"[spawn] spawn_actor_from_object -> {suv}")
except Exception as exc:
    unreal.log_warning(f"[spawn] spawn_actor_from_object failed: {exc}")

# Attempt 2: via the EditorActorSubsystem with the same signature
if suv is None:
    try:
        eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        suv = eas.spawn_actor_from_object(bp_asset, spawn_loc, spawn_rot)
        unreal.log(f"[spawn] EditorActorSubsystem.spawn_actor_from_object -> {suv}")
    except Exception as exc:
        unreal.log_warning(f"[spawn] EAS spawn_actor_from_object failed: {exc}")

# Attempt 3: pull the generated class via property/load_object and use spawn_actor_from_class
if suv is None:
    bp_class = None
    try:
        bp_class = bp_asset.get_editor_property("generated_class")
    except Exception:
        pass
    if bp_class is None:
        try:
            bp_class = unreal.load_object(None, f"{PACKAGE}.BP_SUV_C")
        except Exception:
            pass
    if bp_class is None:
        try:
            cls_path = unreal.SoftClassPath(f"{PACKAGE}.BP_SUV_C")
            bp_class = cls_path.try_load_class()
        except Exception:
            pass
    unreal.log(f"[bp] resolved class -> {bp_class}")
    if bp_class is not None:
        try:
            suv = unreal.EditorLevelLibrary.spawn_actor_from_class(
                bp_class, spawn_loc, spawn_rot
            )
        except Exception as exc:
            unreal.log_warning(f"[spawn] spawn_actor_from_class failed: {exc}")

if suv is None:
    unreal.log_error("ALL spawn attempts failed")
else:
    suv.set_actor_label("BP_SUV_test")
    origin, ext = suv.get_actor_bounds(True)
    unreal.log(
        f"[spawn] SUV at loc=({spawn_loc.x:.0f},{spawn_loc.y:.0f},{spawn_loc.z:.0f}) "
        f"bounds origin=({origin.x:.0f},{origin.y:.0f},{origin.z:.0f}) ext=({ext.x:.0f},{ext.y:.0f},{ext.z:.0f})"
    )

    radius = max(math.sqrt(ext.x**2 + ext.y**2 + ext.z**2), 250.0)
    distance = radius * 2.5
    eye = unreal.Vector(
        origin.x + distance * 0.6,
        origin.y - distance * 0.7,
        origin.z + distance * 0.35,
    )
    rot = look_at(eye, origin)
    cap.set_actor_location(eye, sweep=False, teleport=True)
    cap.set_actor_rotation(rot, teleport_physics=True)

    comp = cap.get_components_by_class(unreal.SceneCaptureComponent2D)[0]
    comp.capture_scene()
    time.sleep(0.6)
    rt = comp.get_editor_property("texture_target")
    out_path = OUT / f"bp_suv_{time.strftime('%H%M%S')}.png"
    unreal.RenderingLibrary.export_render_target(
        world, rt, str(out_path.parent), out_path.name
    )
    if out_path.is_file():
        unreal.log(f"[capture] -> {out_path} bytes={out_path.stat().st_size}")
