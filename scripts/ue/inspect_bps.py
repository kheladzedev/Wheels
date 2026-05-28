"""Inspect both capture Blueprints: their components + SceneCaptureComponent2D defaults."""

import unreal


def inspect_bp(path: str) -> None:
    bp = unreal.EditorAssetLibrary.load_asset(path)
    if bp is None:
        unreal.log_warning(f"[bp] cannot load {path}")
        return
    unreal.log(f"--- {path} ---")
    scs = getattr(bp, "simple_construction_script", None)
    if scs is None:
        unreal.log("  no SimpleConstructionScript")
        return
    nodes = scs.get_all_nodes()
    unreal.log(f"  {len(nodes)} component nodes")
    for node in nodes:
        comp = node.component_template
        if comp is None:
            unreal.log(f"  - {node.variable_name} <None>")
            continue
        cls_name = comp.get_class().get_name()
        unreal.log(f"  - {node.variable_name} <{cls_name}>")
        if isinstance(comp, unreal.SceneCaptureComponent2D):
            unreal.log(
                f"      capture_every_frame = {comp.get_editor_property('capture_every_frame')}"
            )
            unreal.log(
                f"      capture_on_movement = {comp.get_editor_property('capture_on_movement')}"
            )
            unreal.log(
                f"      capture_source      = {comp.get_editor_property('capture_source')}"
            )
            rt = comp.get_editor_property("texture_target")
            if rt is None:
                unreal.log("      texture_target      = None")
            else:
                unreal.log(
                    f"      texture_target      = {rt.get_name()} "
                    f"{rt.get_editor_property('size_x')}x{rt.get_editor_property('size_y')} "
                    f"fmt={rt.get_editor_property('render_target_format')}"
                )


for p in ("/Game/Wheels/main/CameraCaptureWheels", "/Game/Capturing/CameraCapture"):
    try:
        inspect_bp(p)
    except Exception as exc:
        unreal.log_error(f"[inspect] {p}: {exc}")
