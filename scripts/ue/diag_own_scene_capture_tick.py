"""Diagnose a freshly spawned SceneCapture2D actor."""

from __future__ import annotations

import json
import math
import traceback
from pathlib import Path

import unreal

OUT_DIR = Path("/Users/codefactory/Desktop/ML/VSBL/Wheels/outputs/ue_capture_test")
STATUS = OUT_DIR / "diag_own_scene_capture_tick_status.json"
OUT_FILE = OUT_DIR / "diag_own_scene_capture_tick.png"
_TICK_HANDLE = None
_TASK = None


def _write_status(state: str, **payload) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({"state": state, **payload}, indent=2), encoding="utf-8")


def _world():
    return unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()


def _look_at(eye: unreal.Vector, target: unreal.Vector) -> unreal.Rotator:
    dx, dy, dz = target.x - eye.x, target.y - eye.y, target.z - eye.z
    return unreal.Rotator(
        roll=0.0,
        pitch=math.degrees(math.atan2(dz, math.sqrt(dx * dx + dy * dy))),
        yaw=math.degrees(math.atan2(dy, dx)),
    )


class Task:
    def __init__(self) -> None:
        self.phase = "setup"
        self.wait_ticks = 0
        self.done = False
        self.actors = []
        self.sc = None
        self.rt = None

    def cleanup(self) -> None:
        for actor in self.actors:
            unreal.EditorLevelLibrary.destroy_actor(actor)
        self.actors = []

    def setup(self) -> None:
        self.rt = unreal.RenderingLibrary.create_render_target2d(
            _world(),
            1024,
            1024,
            unreal.TextureRenderTargetFormat.RTF_RGBA8,
            unreal.LinearColor(0.0, 0.0, 0.0, 1.0),
            False,
        )
        capture = unreal.EditorLevelLibrary.spawn_actor_from_class(
            unreal.SceneCapture2D,
            unreal.Vector(500.0, -600.0, 350.0),
        )
        capture.set_actor_label("VSBL_DiagOwnCapture")
        self.actors.append(capture)
        self.sc = capture.get_component_by_class(unreal.SceneCaptureComponent2D)
        self.sc.set_editor_property("texture_target", self.rt)
        self.sc.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
        self.sc.set_editor_property("fov_angle", 45.0)
        self.sc.set_editor_property("capture_every_frame", False)
        self.sc.set_editor_property("post_process_blend_weight", 0.0)

        mesh = unreal.EditorAssetLibrary.load_asset("/Engine/BasicShapes/Cube.Cube")
        cube = unreal.EditorLevelLibrary.spawn_actor_from_object(mesh, unreal.Vector(0.0, 0.0, 100.0))
        cube.set_actor_label("VSBL_DiagOwn_Cube")
        self.actors.append(cube)

        light = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.PointLight, unreal.Vector(300.0, -300.0, 500.0))
        light.set_actor_label("VSBL_DiagOwn_Light")
        component = light.get_component_by_class(unreal.PointLightComponent)
        component.set_editor_property("intensity", 200000.0)
        component.set_editor_property("attenuation_radius", 2000.0)
        self.actors.append(light)

        target = unreal.Vector(0.0, 0.0, 100.0)
        capture.set_actor_rotation(_look_at(capture.get_actor_location(), target), teleport_physics=True)
        self.sc.capture_scene()
        self.wait_ticks = 5
        self.phase = "export"
        _write_status("capturing")

    def export(self) -> None:
        if self.wait_ticks > 0:
            self.wait_ticks -= 1
            return
        unreal.RenderingLibrary.export_render_target(_world(), self.rt, str(OUT_FILE.parent), OUT_FILE.name)
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
unreal.log(f"[diag-own-capture] scheduled; status={STATUS}")
