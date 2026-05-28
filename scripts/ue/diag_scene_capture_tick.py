"""Diagnose SceneCapture export on a fresh TextureRenderTarget2D.

Runs as a tiny Slate-tick state machine so the render thread gets ticks
between capture_scene() and export_render_target().
"""

from __future__ import annotations

import json
import math
import traceback
from pathlib import Path

import unreal

OUT_DIR = Path("/Users/codefactory/Desktop/ML/VSBL/Wheels/outputs/ue_capture_test")
STATUS = OUT_DIR / "diag_scene_capture_tick_status.json"
OUT_FILE = OUT_DIR / "diag_scene_capture_tick.png"
_TICK_HANDLE = None
_TASK = None


def _write_status(state: str, **payload) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({"state": state, **payload}, indent=2), encoding="utf-8")


def _world():
    return unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()


def _look_at(eye: unreal.Vector, target: unreal.Vector) -> unreal.Rotator:
    dx, dy, dz = target.x - eye.x, target.y - eye.y, target.z - eye.z
    yaw = math.degrees(math.atan2(dy, dx))
    horiz = math.sqrt(dx * dx + dy * dy)
    pitch = math.degrees(math.atan2(dz, horiz))
    return unreal.Rotator(roll=0.0, pitch=pitch, yaw=yaw)


def _capture_actor():
    actors = unreal.GameplayStatics.get_all_actors_of_class(_world(), unreal.Actor)
    for actor in actors:
        label = (actor.get_actor_label() or "").lower()
        cls_name = actor.get_class().get_name().lower()
        if "cameracapture" in label or cls_name.startswith("cameracapture"):
            return actor
    return None


class Task:
    def __init__(self) -> None:
        self.phase = "setup"
        self.wait_ticks = 0
        self.done = False
        self.actors = []
        self.old_rt = None
        self.sc = None

    def cleanup(self) -> None:
        if self.sc is not None and self.old_rt is not None:
            self.sc.set_editor_property("texture_target", self.old_rt)
        for actor in self.actors:
            unreal.EditorLevelLibrary.destroy_actor(actor)
        self.actors = []

    def setup(self) -> None:
        cap = _capture_actor()
        if cap is None:
            raise RuntimeError("CameraCapture actor not found")
        components = cap.get_components_by_class(unreal.SceneCaptureComponent2D)
        if not components:
            raise RuntimeError("CameraCapture has no SceneCaptureComponent2D")
        self.sc = components[0]
        self.old_rt = self.sc.get_editor_property("texture_target")
        rt = unreal.RenderingLibrary.create_render_target2d(
            _world(),
            1024,
            1024,
            unreal.TextureRenderTargetFormat.RTF_RGBA8,
            unreal.LinearColor(0.0, 0.0, 0.0, 1.0),
            False,
        )
        self.sc.set_editor_property("texture_target", rt)
        self.sc.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
        self.sc.set_editor_property("fov_angle", 45.0)
        self.sc.set_editor_property("post_process_blend_weight", 0.0)

        mesh = unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Cube.Cube")
        cube = unreal.EditorLevelLibrary.spawn_actor_from_object(mesh, unreal.Vector(0.0, 0.0, 100.0))
        cube.set_actor_label("VSBL_DiagTick_Cube")
        self.actors.append(cube)

        light = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.PointLight, unreal.Vector(300.0, -300.0, 500.0))
        light.set_actor_label("VSBL_DiagTick_Light")
        component = light.get_component_by_class(unreal.PointLightComponent)
        component.set_editor_property("intensity", 200000.0)
        component.set_editor_property("attenuation_radius", 2000.0)
        self.actors.append(light)

        eye = unreal.Vector(500.0, -600.0, 350.0)
        target = unreal.Vector(0.0, 0.0, 100.0)
        cap.set_actor_location(eye, sweep=False, teleport=True)
        cap.set_actor_rotation(_look_at(eye, target), teleport_physics=True)
        self.sc.update_content()
        self.sc.capture_scene()
        self.wait_ticks = 5
        self.phase = "export"
        _write_status("capturing")

    def export(self) -> None:
        if self.wait_ticks > 0:
            self.wait_ticks -= 1
            return
        rt = self.sc.get_editor_property("texture_target")
        unreal.RenderingLibrary.export_render_target(_world(), rt, str(OUT_FILE.parent), OUT_FILE.name)
        size = OUT_FILE.stat().st_size if OUT_FILE.is_file() else 0
        self.cleanup()
        self.done = True
        _write_status("done", out=str(OUT_FILE), bytes=size)

    def step(self) -> bool:
        if self.done:
            return True
        if self.phase == "setup":
            self.setup()
        elif self.phase == "export":
            self.export()
        return self.done


def _tick(delta_time) -> None:  # noqa: ARG001
    global _TICK_HANDLE, _TASK
    try:
        if _TASK is None:
            _TASK = Task()
        if _TASK.step() and _TICK_HANDLE is not None:
            unreal.unregister_slate_post_tick_callback(_TICK_HANDLE)
            _TICK_HANDLE = None
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        _write_status("error", error=str(exc), traceback=tb)
        if _TASK is not None:
            _TASK.cleanup()
        if _TICK_HANDLE is not None:
            unreal.unregister_slate_post_tick_callback(_TICK_HANDLE)
            _TICK_HANDLE = None


_write_status("scheduled")
_TICK_HANDLE = unreal.register_slate_post_tick_callback(_tick)
unreal.log(f"[diag-capture-tick] scheduled; status={STATUS}")
