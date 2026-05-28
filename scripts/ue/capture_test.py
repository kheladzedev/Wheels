"""UE 5 Editor Python — force CameraCapture to render now and save to PNG.

Run from UE Output Log Cmd:
    py /Users/codefactory/Desktop/ML/VSBL/Wheels/scripts/ue/capture_test.py

Finds the CameraCapture_C actor in the current level, locates its
SceneCaptureComponent2D, forces a CaptureScene() call (the rig has
capture_every_frame=False, so the render target only updates when this
is invoked), then exports the target to PNG so we can verify whether
the RGB pipeline produces a real image or stays black.

Output:
    Wheels/outputs/ue_capture_test/<timestamp>.png
"""

from __future__ import annotations

import time
from pathlib import Path

import unreal

OUT_DIR = Path("/Users/codefactory/Desktop/ML/VSBL/Wheels/outputs/ue_capture_test")


def _find_capture_actor():
    editor_sub = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    world = editor_sub.get_editor_world()
    actors = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor)
    for a in actors:
        if a is None:
            continue
        label = (a.get_actor_label() or "").lower()
        cls = a.get_class().get_name()
        if "cameracapture" in label or cls.startswith("CameraCapture"):
            return a
    return None


def _save_render_target_png(rt, out_path: Path) -> bool:
    """Save a UTextureRenderTarget2D to PNG. Tries multiple UE 5.x APIs."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    editor_sub = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    world = editor_sub.get_editor_world()
    errors: list[str] = []
    try:
        unreal.RenderingLibrary.export_render_target(
            world, rt, str(out_path.parent), out_path.name
        )
        if out_path.is_file():
            return True
        errors.append("RenderingLibrary.export_render_target wrote no file")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"RenderingLibrary: {exc}")
    try:
        unreal.AutomationLibrary.take_high_res_screenshot(
            int(rt.get_editor_property("size_x")),
            int(rt.get_editor_property("size_y")),
            str(out_path),
            camera=None,
            mask_enabled=False,
            capture_hdr=False,
            comparison_tolerance=unreal.ComparisonTolerance.LOW,
        )
        if out_path.is_file():
            return True
        errors.append("take_high_res_screenshot wrote no file")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"AutomationLibrary: {exc}")
    for e in errors:
        unreal.log_error(f"[capture] {e}")
    return False


def main() -> None:
    actor = _find_capture_actor()
    if actor is None:
        unreal.log_error("[capture] CameraCapture actor not found in current level")
        return
    unreal.log(
        f"[capture] actor: {actor.get_actor_label()} loc={actor.get_actor_location()}"
    )

    sc_components = actor.get_components_by_class(unreal.SceneCaptureComponent2D)
    if not sc_components:
        unreal.log_error("[capture] no SceneCaptureComponent2D")
        return
    sc = sc_components[0]
    rt = sc.get_editor_property("texture_target")
    if rt is None:
        unreal.log_error("[capture] no texture_target on capture component")
        return

    unreal.log(
        f"[capture] forcing CaptureScene -> RT {rt.get_name()} "
        f"{rt.get_editor_property('size_x')}x{rt.get_editor_property('size_y')}"
    )
    sc.capture_scene()
    time.sleep(0.5)  # let the GPU flush

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%H%M%S")
    out_file = OUT_DIR / f"capture_{stamp}.png"
    ok = _save_render_target_png(rt, out_file)
    if ok and out_file.is_file():
        unreal.log(f"[capture] saved -> {out_file} ({out_file.stat().st_size} bytes)")
    else:
        unreal.log_error(f"[capture] save failed; expected {out_file}")


main()
