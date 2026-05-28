"""After-spawn capture with multi-second wait + retry, to test if the grey
frames are a render-pipeline timing problem after camera teleport."""

import time
from pathlib import Path

import unreal

OUT = Path("/Users/codefactory/Desktop/ML/VSBL/Wheels/outputs/ue_capture_test")
OUT.mkdir(parents=True, exist_ok=True)

sub = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
world = sub.get_editor_world()
cap = next(
    (
        a
        for a in unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor)
        if (a.get_actor_label() or "").lower().startswith("cameracapture")
    ),
    None,
)
loc = cap.get_actor_location()
rot = cap.get_actor_rotation()
unreal.log(
    f"[wait] current pose loc=({loc.x:.0f},{loc.y:.0f},{loc.z:.0f}) yaw={rot.yaw:.1f} pitch={rot.pitch:.1f}"
)

comp = cap.get_components_by_class(unreal.SceneCaptureComponent2D)[0]
rt = comp.get_editor_property("texture_target")

# Long settle delay first
time.sleep(2.0)

# Multiple captures with delays — let async render kick in
for i in range(5):
    comp.capture_scene()
    time.sleep(0.6)

out_path = OUT / f"longwait_{time.strftime('%H%M%S')}.png"
unreal.RenderingLibrary.export_render_target(
    world, rt, str(out_path.parent), out_path.name
)
sz = out_path.stat().st_size if out_path.is_file() else 0
unreal.log(f"[wait] saved -> {out_path}  bytes={sz}")
