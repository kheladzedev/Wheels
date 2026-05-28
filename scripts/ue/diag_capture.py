"""Diagnose: move camera to a known-good position (the one that gave the first
clean capture mean=121/107/99) and re-capture. If it's still grey, something
broke in the RT pipeline since the spawns; if it's clean, the spawn-aim
geometry is the bug."""

import time
from pathlib import Path

import unreal

OUT = Path("/Users/codefactory/Desktop/ML/VSBL/Wheels/outputs/ue_capture_test")

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

# This is the original CameraCapture pose from the first known-good capture
known_good_loc = unreal.Vector(74769.0, 85346.0, 44590.0)
known_good_rot = unreal.Rotator(roll=0.0, pitch=0.0, yaw=0.0)
cap.set_actor_location(known_good_loc, sweep=False, teleport=True)
cap.set_actor_rotation(known_good_rot, teleport_physics=True)
unreal.log("[diag] camera set to known-good pose")

comp = cap.get_components_by_class(unreal.SceneCaptureComponent2D)[0]
unreal.log(
    f"[diag] capture_every_frame = {comp.get_editor_property('capture_every_frame')}"
)
unreal.log(f"[diag] capture_source = {comp.get_editor_property('capture_source')}")

# Force capture x3 with sleeps so the engine has time to render
for i in range(3):
    comp.capture_scene()
    time.sleep(0.4)

rt = comp.get_editor_property("texture_target")
out_path = OUT / f"diag_{time.strftime('%H%M%S')}.png"
unreal.RenderingLibrary.export_render_target(
    world, rt, str(out_path.parent), out_path.name
)
unreal.log(
    f"[diag] -> {out_path} bytes={out_path.stat().st_size if out_path.is_file() else 'NO'}"
)
