"""Delete previously-spawned debris + spawn ONE wheel mesh + aim camera close."""

import math
import time
from pathlib import Path

import unreal

OUT = Path("/Users/codefactory/Desktop/ML/VSBL/Wheels/outputs/ue_capture_test")

sub = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
world = sub.get_editor_world()

# 1. Clean my old spawns
removed = 0
for a in unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor):
    lbl = a.get_actor_label() or ""
    if lbl.startswith(("BP_SUV_test", "WheelTest_", "Car_", "CarSpawn_")):
        unreal.EditorLevelLibrary.destroy_actor(a)
        removed += 1
unreal.log(f"[clean] {removed} removed")

# 2. Spawn ONE wheel mesh from CitySampleVehicles
WHEEL_PATH = (
    "/Game/CitySampleVehicles/vehicle05_Car/Mesh/SM_Wheel_Front_L_vehCar_vehicle05"
)
mesh = unreal.EditorAssetLibrary.load_asset(WHEEL_PATH)
if mesh is None:
    raise SystemExit(f"cannot load {WHEEL_PATH}")
unreal.log(f"[mesh] asset = {mesh}, class = {type(mesh).__name__}")

# Use the existing CameraCapture position as anchor — spawn wheel right where
# the camera looks, slightly away.
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

# 2 m in front of camera, on the floor (z=44610 we know floor is around 44590)
wheel_pos = unreal.Vector(cap_loc.x + 200.0, cap_loc.y, 44610.0)
wheel = unreal.EditorLevelLibrary.spawn_actor_from_object(mesh, wheel_pos)
if wheel is None:
    raise SystemExit("spawn returned None")
wheel.set_actor_label("OneWheel_test")
origin, ext = wheel.get_actor_bounds(True)
unreal.log(
    f"[wheel] at ({wheel_pos.x:.0f},{wheel_pos.y:.0f},{wheel_pos.z:.0f}) "
    f"bounds origin=({origin.x:.0f},{origin.y:.0f},{origin.z:.0f}) "
    f"ext=({ext.x:.1f},{ext.y:.1f},{ext.z:.1f})"
)

# 3. Aim camera at the wheel from 1.5 m back, on its right
distance = max(ext.x, ext.y, ext.z) * 6.0 if ext.x > 0 else 200.0
eye = unreal.Vector(
    origin.x - distance * 0.8, origin.y + distance * 0.5, origin.z + distance * 0.3
)
dx, dy, dz = origin.x - eye.x, origin.y - eye.y, origin.z - eye.z
yaw = math.degrees(math.atan2(dy, dx))
horiz = math.sqrt(dx * dx + dy * dy)
pitch = math.degrees(math.atan2(dz, horiz))
cap.set_actor_location(eye, sweep=False, teleport=True)
cap.set_actor_rotation(
    unreal.Rotator(roll=0.0, pitch=pitch, yaw=yaw), teleport_physics=True
)
unreal.log(
    f"[aim] eye=({eye.x:.0f},{eye.y:.0f},{eye.z:.0f}) yaw={yaw:.1f} pitch={pitch:.1f} dist={distance:.1f}"
)

# 4. Capture
comp = cap.get_components_by_class(unreal.SceneCaptureComponent2D)[0]
comp.capture_scene()
time.sleep(0.5)
rt = comp.get_editor_property("texture_target")
out_path = OUT / f"one_wheel_{time.strftime('%H%M%S')}.png"
unreal.RenderingLibrary.export_render_target(
    world, rt, str(out_path.parent), out_path.name
)
unreal.log(
    f"[capture] -> {out_path}  bytes={out_path.stat().st_size if out_path.is_file() else 'NO'}"
)
