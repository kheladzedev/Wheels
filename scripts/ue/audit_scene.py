"""UE 5 Editor Python — audit currently open scene + capture-rig diagnosis.

Run from UE Output Log Cmd:
    py /Users/codefactory/Desktop/ML/VSBL/Wheels/scripts/ue/audit_scene.py

Does NOT switch levels (NewMap3 is expected to be already open — avoids
modal dialogs about unsaved work). Enumerates actors, classifies them,
and for any SceneCapture / CameraCapture* actor dumps the parameters
that decide whether the render target receives a usable RGB image
(CaptureSource, TextureTarget format, PostProcessSettings exposure).
Writes JSON report to VSBL/Wheels/outputs/ue_scene_audit.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import unreal

REPORT_PATH = Path(
    "/Users/codefactory/Desktop/ML/VSBL/Wheels/outputs/ue_scene_audit.json"
)


def _classify(actor) -> str:
    label = (actor.get_actor_label() or "").lower()
    cls = actor.get_class().get_name()
    if "cameracapture" in label or "scenecapture" in label or "capture" in label:
        return "capture"
    if any(
        tag in label
        for tag in ("vehicle", "car", "sketchfab", "auto", "bmw", "audi", "ford")
    ):
        return "vehicle"
    if "wheel" in label:
        return "wheel"
    if "Light" in cls or "light" in label:
        return "light"
    if "PostProcess" in cls or "postprocess" in label:
        return "postprocess"
    if "Camera" in cls:
        return "camera"
    return "other"


def _actor_basic(actor) -> dict:
    loc = actor.get_actor_location()
    rot = actor.get_actor_rotation()
    return {
        "label": actor.get_actor_label(),
        "name": actor.get_name(),
        "class": actor.get_class().get_name(),
        "location": [loc.x, loc.y, loc.z],
        "rotation_yaw_pitch_roll": [rot.yaw, rot.pitch, rot.roll],
    }


def _inspect_scene_capture_component(comp) -> dict:
    """Pull settings that decide whether the render target gets RGB."""
    info: dict = {"component_class": comp.get_class().get_name()}
    for attr in (
        "capture_source",
        "fov_angle",
        "max_view_distance_override",
        "primitive_render_mode",
        "always_persist_rendering_state",
        "capture_every_frame",
        "capture_on_movement",
    ):
        try:
            info[attr] = repr(comp.get_editor_property(attr))
        except Exception as exc:  # noqa: BLE001
            info[attr] = f"<err: {exc}>"
    try:
        tt = comp.get_editor_property("texture_target")
        if tt is None:
            info["texture_target"] = None
        else:
            info["texture_target"] = {
                "name": tt.get_name(),
                "size_x": tt.get_editor_property("size_x"),
                "size_y": tt.get_editor_property("size_y"),
                "render_target_format": repr(
                    tt.get_editor_property("render_target_format")
                ),
                "clear_color": repr(tt.get_editor_property("clear_color")),
            }
    except Exception as exc:  # noqa: BLE001
        info["texture_target"] = f"<err: {exc}>"
    try:
        pp = comp.get_editor_property("post_process_settings")
        pp_view: dict = {}
        for key in (
            "auto_exposure_method",
            "auto_exposure_bias",
            "auto_exposure_min_brightness",
            "auto_exposure_max_brightness",
            "override_auto_exposure_bias",
            "override_auto_exposure_method",
        ):
            try:
                pp_view[key] = repr(pp.get_editor_property(key))
            except Exception:  # noqa: BLE001
                pp_view[key] = "<missing>"
        info["post_process_settings"] = pp_view
    except Exception as exc:  # noqa: BLE001
        info["post_process_settings"] = f"<err: {exc}>"
    return info


def _inspect_capture_actor(actor) -> dict:
    out: dict = _actor_basic(actor)
    out["components"] = []
    try:
        for comp in actor.get_components_by_class(unreal.SceneCaptureComponent2D):
            out["components"].append(_inspect_scene_capture_component(comp))
    except Exception as exc:  # noqa: BLE001
        out["components_error"] = repr(exc)
    return out


def main() -> None:
    editor_sub = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    world = editor_sub.get_editor_world()
    level_name = world.get_name() if world else "<no_world>"
    actors = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor)

    buckets: dict[str, list[dict]] = {
        "vehicle": [],
        "wheel": [],
        "capture": [],
        "camera": [],
        "light": [],
        "postprocess": [],
        "other": [],
    }
    for actor in actors:
        if actor is None:
            continue
        category = _classify(actor)
        if category == "capture":
            buckets["capture"].append(_inspect_capture_actor(actor))
        else:
            buckets[category].append(_actor_basic(actor))

    report = {
        "level_name": level_name,
        "total_actors": len(actors),
        "counts": {k: len(v) for k, v in buckets.items()},
        "buckets": buckets,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    unreal.log(
        f"[audit] level={level_name} actors={len(actors)} counts={report['counts']}"
    )
    unreal.log(f"[audit] report -> {REPORT_PATH}")


main()
