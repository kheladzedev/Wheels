"""Assemble a static (non-Chaos) car from CitySampleVehicles parts.

Takes one vehicle05_Car package, spawns:
  - body (SM_vehCar_vehicle05_No_Wheel)
  - 4 wheels (SM_Wheel_{Front,Rear}_{L,R}_vehCar_vehicle05) placed at body bbox corners
Stores the world-space 3D positions of each wheel centroid so a later
keypoint-extraction pass can project them to screen-space.

Output: /tmp wheel positions JSON for the renderer.
"""

import json
import math
import time
from pathlib import Path

import unreal

VEHICLE_ROOT = "/Game/CitySampleVehicles/vehicle05_Car/Mesh"
BODY_MESH = f"{VEHICLE_ROOT}/SM_vehCar_vehicle05_No_Wheel"
WHEELS = {
    "FL": f"{VEHICLE_ROOT}/SM_Wheel_Front_L_vehCar_vehicle05",
    "FR": f"{VEHICLE_ROOT}/SM_Wheel_Front_R_vehCar_vehicle05",
    "RL": f"{VEHICLE_ROOT}/SM_Wheel_Rear_L_vehCar_vehicle05",
    "RR": f"{VEHICLE_ROOT}/SM_Wheel_Rear_R_vehCar_vehicle05",
}

OUT = Path("/Users/codefactory/Desktop/ML/VSBL/Wheels/outputs/ue_capture_test")
OUT.mkdir(parents=True, exist_ok=True)


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
cap_loc = cap.get_actor_location()

# Spawn point — open area near camera
base = unreal.Vector(cap_loc.x + 600.0, cap_loc.y + 200.0, 44590.0)

# Body first
body_mesh = unreal.EditorAssetLibrary.load_asset(BODY_MESH)
if body_mesh is None:
    raise SystemExit(f"cannot load {BODY_MESH}")

body = unreal.EditorLevelLibrary.spawn_actor_from_object(body_mesh, base)
if body is None:
    raise SystemExit("spawn body failed")
body.set_actor_label("Car_Body")
b_origin, b_ext = body.get_actor_bounds(True)
unreal.log(
    f"[body] {body_mesh.get_name()} at ({base.x:.0f},{base.y:.0f},{base.z:.0f}) "
    f"ext=({b_ext.x:.0f},{b_ext.y:.0f},{b_ext.z:.0f})"
)

# Wheels at body bbox corners on the ground.
# Convention from CitySample: X is forward, Y is right, Z is up.
# We offset by ~75 % of the body extent to put wheels at corners.
half_x = b_ext.x * 0.72
half_y = b_ext.y * 0.78
wheel_z = base.z  # on the floor
positions = {
    "FL": unreal.Vector(b_origin.x + half_x, b_origin.y - half_y, wheel_z),
    "FR": unreal.Vector(b_origin.x + half_x, b_origin.y + half_y, wheel_z),
    "RL": unreal.Vector(b_origin.x - half_x, b_origin.y - half_y, wheel_z),
    "RR": unreal.Vector(b_origin.x - half_x, b_origin.y + half_y, wheel_z),
}

wheel_centroids: dict[str, list[float]] = {}
spawned_wheels = []
for tag, pkg in WHEELS.items():
    mesh = unreal.EditorAssetLibrary.load_asset(pkg)
    if mesh is None:
        unreal.log_warning(f"  cant load wheel {pkg}")
        continue
    pos = positions[tag]
    w = unreal.EditorLevelLibrary.spawn_actor_from_object(mesh, pos)
    if w is None:
        unreal.log_warning(f"  spawn wheel {tag} failed")
        continue
    w.set_actor_label(f"Car_Wheel_{tag}")
    w_origin, w_ext = w.get_actor_bounds(True)
    wheel_centroids[tag] = [w_origin.x, w_origin.y, w_origin.z]
    spawned_wheels.append(w)
    unreal.log(
        f"  wheel {tag} at ({pos.x:.0f},{pos.y:.0f},{pos.z:.0f})  "
        f"radius ~ {max(w_ext.x, w_ext.y, w_ext.z):.1f}"
    )

unreal.log(f"[assembly] body + {len(spawned_wheels)} wheels")

# Aim camera at the assembly from 3-quarter view
target = b_origin
radius = max(b_ext.x, b_ext.y, b_ext.z)
distance = radius * 3.5
yaw_deg = 35.0
pitch_deg = -15.0
yr = math.radians(yaw_deg)
pr = math.radians(pitch_deg)
eye = unreal.Vector(
    target.x + distance * math.cos(pr) * math.cos(yr),
    target.y + distance * math.cos(pr) * math.sin(yr),
    target.z + distance * math.sin(-pr),
)
rot = look_at(eye, target)
cap.set_actor_location(eye, sweep=False, teleport=True)
cap.set_actor_rotation(rot, teleport_physics=True)

comp = cap.get_components_by_class(unreal.SceneCaptureComponent2D)[0]
comp.capture_scene()
time.sleep(0.6)
rt = comp.get_editor_property("texture_target")
out_path = OUT / f"assembly_{time.strftime('%H%M%S')}.png"
unreal.RenderingLibrary.export_render_target(
    world, rt, str(out_path.parent), out_path.name
)
if out_path.is_file():
    unreal.log(f"[capture] -> {out_path} bytes={out_path.stat().st_size}")

# Save wheel centroids for the next pass (screen-space projection)
out_json = Path(
    "/Users/codefactory/Desktop/ML/VSBL/Wheels/outputs/ue_capture_test/last_assembly.json"
)
out_json.write_text(
    json.dumps(
        {
            "body": [b_origin.x, b_origin.y, b_origin.z],
            "body_extent": [b_ext.x, b_ext.y, b_ext.z],
            "wheels": wheel_centroids,
            "camera_eye": [eye.x, eye.y, eye.z],
            "camera_rot_ypr": [rot.yaw, rot.pitch, rot.roll],
            "camera_fov_deg": comp.get_editor_property("fov_angle"),
        },
        indent=2,
    )
)
unreal.log(f"[meta] saved {out_json}")
