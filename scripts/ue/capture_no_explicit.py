"""Verify that capture_every_frame=True actually fills the RenderTarget
without an explicit CaptureScene() call — that was the original bug."""

import time
from pathlib import Path

import unreal

OUT = Path("/Users/codefactory/Desktop/ML/VSBL/Wheels/outputs/ue_capture_test")
OUT.mkdir(parents=True, exist_ok=True)

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
    unreal.log_error("no CameraCapture in level")
else:
    comp = cap.get_components_by_class(unreal.SceneCaptureComponent2D)[0]
    every = comp.get_editor_property("capture_every_frame")
    on_move = comp.get_editor_property("capture_on_movement")
    unreal.log(f"[verify] capture_every_frame={every} capture_on_movement={on_move}")
    rt = comp.get_editor_property("texture_target")
    # Just save RT — no explicit CaptureScene. If capture_every_frame is honoured
    # in Editor (not just PIE), the RT should already contain a fresh frame.
    out_path = OUT / f"no_explicit_{time.strftime('%H%M%S')}.png"
    unreal.RenderingLibrary.export_render_target(
        world, rt, str(out_path.parent), out_path.name
    )
    if out_path.is_file():
        unreal.log(f"[verify] saved {out_path} bytes={out_path.stat().st_size}")
    else:
        unreal.log_error(f"[verify] no file at {out_path}")
