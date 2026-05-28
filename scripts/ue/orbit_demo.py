"""Demo: orbit CameraCapture around the interior centroid in 6 stops with
visible pauses, so the user can watch the actor teleport in viewport."""

import math
import time

import unreal

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

# Interior centroid is around (74900, 85200, 44600) per audit
CENTER = unreal.Vector(74900.0, 85200.0, 44900.0)
RADIUS = 350.0  # close enough to see the room
PITCH = -10.0  # look down a bit
STOPS = 6
PAUSE_S = 1.5


def look_at(eye, target):
    dx, dy, dz = target.x - eye.x, target.y - eye.y, target.z - eye.z
    yaw = math.degrees(math.atan2(dy, dx))
    horiz = math.sqrt(dx * dx + dy * dy)
    pitch = math.degrees(math.atan2(dz, horiz))
    return unreal.Rotator(roll=0.0, pitch=pitch, yaw=yaw)


unreal.log(f"[orbit] starting {STOPS} stops at radius {RADIUS}cm around {CENTER}")
for i in range(STOPS):
    deg = i * (360.0 / STOPS)
    r = math.radians(deg)
    eye = unreal.Vector(
        CENTER.x + RADIUS * math.cos(r),
        CENTER.y + RADIUS * math.sin(r),
        CENTER.z + 120.0,
    )
    rot = look_at(eye, CENTER)
    cap.set_actor_location(eye, sweep=False, teleport=True)
    cap.set_actor_rotation(rot, teleport_physics=True)
    unreal.log(
        f"[orbit] stop {i + 1}/{STOPS}  deg={deg:.0f}  eye=({eye.x:.0f},{eye.y:.0f},{eye.z:.0f})"
    )
    time.sleep(PAUSE_S)

unreal.log("[orbit] done")
