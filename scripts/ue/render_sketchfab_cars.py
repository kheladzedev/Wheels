"""Render imported Sketchfab car StaticMeshes through the CameraCapture rig.

Run through UnrealMCP after GLBs are imported:
    ./.venv/bin/python scripts/ue/_send.py exec_file scripts/ue/render_sketchfab_cars.py

Expected output:
    outputs/ue_sketchfab_renders/images/<mesh>__view_<n>.png

Optional environment:
    VSBL_UE_RENDER_SOURCE=/Game/SketchfabCars
    VSBL_UE_RENDER_OUT=/Users/codefactory/Desktop/ML/VSBL/Wheels/outputs/ue_sketchfab_renders/images
    VSBL_UE_RENDER_LIMIT=300
    VSBL_UE_RENDER_VIEWS=4
    VSBL_UE_RENDER_SLEEP=0.25
"""

from __future__ import annotations

import math
import os
import re
import json
import time
import traceback
from pathlib import Path

import unreal

SCRIPT_PATH = Path(
    globals().get(
        "__file__",
        "/Users/codefactory/Desktop/ML/VSBL/Wheels/scripts/ue/render_sketchfab_cars.py",
    )
).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
SOURCE_ROOT = os.environ.get("VSBL_UE_RENDER_SOURCE", "/Game/SketchfabCars")
OUT_DIR = Path(
    os.environ.get(
        "VSBL_UE_RENDER_OUT",
        REPO_ROOT / "outputs/ue_sketchfab_renders/images",
    )
)
LIMIT = int(os.environ.get("VSBL_UE_RENDER_LIMIT", "300"))
VIEWS_PER_MODEL = int(os.environ.get("VSBL_UE_RENDER_VIEWS", "4"))
SLEEP_S = float(os.environ.get("VSBL_UE_RENDER_SLEEP", "0.25"))
MAX_PARTS_PER_MODEL = int(os.environ.get("VSBL_UE_RENDER_MAX_PARTS_PER_MODEL", "80"))
ACTOR_LABEL_PREFIX = "VSBL_RenderCar_"
STATUS_PATH = REPO_ROOT / "outputs/ue_tasks/render_sketchfab_cars_status.json"
_TICK_HANDLE = None


def _write_status(state: str, **payload) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {"task": "render_sketchfab_cars", "state": state, **payload}
    STATUS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:96] or "mesh"


def _editor_world():
    sub = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
    return sub.get_editor_world()


def _asset_class_name(asset_data) -> str:
    try:
        return str(asset_data.asset_class_path.asset_name)
    except AttributeError:
        return ""


def list_static_mesh_groups(folder: str, limit: int) -> list[tuple[str, list[str]]]:
    registry = unreal.AssetRegistryHelpers.get_asset_registry()
    assets = registry.get_assets_by_path(folder, recursive=True)
    grouped: dict[str, list] = {}
    for asset in assets:
        if _asset_class_name(asset) != "StaticMesh":
            continue
        package_path = str(asset.package_path)
        rel = package_path[len(folder) :].strip("/") if package_path.startswith(folder) else package_path
        model_key = rel.split("/", 1)[0] if rel else str(asset.asset_name)
        grouped.setdefault(model_key, []).append(asset)

    def pick_group(candidates: list) -> list[str]:
        ranked = sorted(
            candidates,
            key=lambda item: (
                str(item.asset_name).lower().startswith(("material", "color", "texture")),
                str(item.package_name),
            ),
        )
        return [
            str(asset.package_name) + "." + str(asset.asset_name)
            for asset in ranked[:MAX_PARTS_PER_MODEL]
        ]

    groups = [
        (model_key, pick_group(candidates))
        for model_key, candidates in sorted(grouped.items())
    ]
    return groups[:limit]


def find_capture_actor():
    world = _editor_world()
    actors = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor)
    for actor in actors:
        label = (actor.get_actor_label() or "").lower()
        cls_name = actor.get_class().get_name().lower()
        if "cameracapture" in label or cls_name.startswith("cameracapture"):
            return actor
    return None


def cleanup_previous_render_actors() -> None:
    world = _editor_world()
    actors = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor)
    for actor in actors:
        if (actor.get_actor_label() or "").startswith(ACTOR_LABEL_PREFIX):
            unreal.EditorLevelLibrary.destroy_actor(actor)


def look_at(eye: unreal.Vector, target: unreal.Vector) -> unreal.Rotator:
    dx, dy, dz = target.x - eye.x, target.y - eye.y, target.z - eye.z
    yaw = math.degrees(math.atan2(dy, dx))
    horiz = math.sqrt(dx * dx + dy * dy)
    pitch = math.degrees(math.atan2(dz, horiz))
    return unreal.Rotator(roll=0.0, pitch=pitch, yaw=yaw)


def _save_render_target_png(world, rt, out_path: Path) -> bool:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        unreal.RenderingLibrary.export_render_target(
            world, rt, str(out_path.parent), out_path.name
        )
        return out_path.is_file()
    except Exception as exc:  # noqa: BLE001
        unreal.log_error(f"[render] export_render_target failed: {exc}")
        return False


def _spawn_mesh(mesh_path: str, idx: int, part_idx: int):
    mesh = unreal.EditorAssetLibrary.load_asset(mesh_path)
    if mesh is None:
        unreal.log_warning(f"[render] cannot load {mesh_path}")
        return None
    actor = unreal.EditorLevelLibrary.spawn_actor_from_object(
        mesh, unreal.Vector(0.0, 0.0, 0.0)
    )
    if actor is None:
        unreal.log_warning(f"[render] cannot spawn {mesh_path}")
        return None
    actor.set_actor_label(f"{ACTOR_LABEL_PREFIX}{idx:04d}_{part_idx:03d}")
    return actor


def _spawn_mesh_group(mesh_paths: list[str], idx: int) -> list:
    actors = []
    for part_idx, mesh_path in enumerate(mesh_paths):
        actor = _spawn_mesh(mesh_path, idx, part_idx)
        if actor is not None:
            actors.append(actor)
    return actors


def _group_bounds(actors: list) -> tuple[unreal.Vector, unreal.Vector] | None:
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    for actor in actors:
        origin, ext = actor.get_actor_bounds(True)
        mins[0] = min(mins[0], origin.x - ext.x)
        mins[1] = min(mins[1], origin.y - ext.y)
        mins[2] = min(mins[2], origin.z - ext.z)
        maxs[0] = max(maxs[0], origin.x + ext.x)
        maxs[1] = max(maxs[1], origin.y + ext.y)
        maxs[2] = max(maxs[2], origin.z + ext.z)
    if not actors or any(not math.isfinite(v) for v in mins + maxs):
        return None
    origin = unreal.Vector(
        (mins[0] + maxs[0]) * 0.5,
        (mins[1] + maxs[1]) * 0.5,
        (mins[2] + maxs[2]) * 0.5,
    )
    ext = unreal.Vector(
        max((maxs[0] - mins[0]) * 0.5, 1.0),
        max((maxs[1] - mins[1]) * 0.5, 1.0),
        max((maxs[2] - mins[2]) * 0.5, 1.0),
    )
    return origin, ext


def _capture_actor_views(cap_actor, car_actors: list, mesh_name: str) -> int:
    world = _editor_world()
    sc_components = cap_actor.get_components_by_class(unreal.SceneCaptureComponent2D)
    if not sc_components:
        unreal.log_error("[render] CameraCapture has no SceneCaptureComponent2D")
        return 0
    sc = sc_components[0]
    rt = sc.get_editor_property("texture_target")
    if rt is None:
        unreal.log_error("[render] CameraCapture SceneCapture has no texture_target")
        return 0

    bounds = _group_bounds(car_actors)
    if bounds is None:
        unreal.log_warning(f"[render] cannot compute bounds for {mesh_name}")
        return 0
    origin, ext = bounds
    radius = max(math.sqrt(ext.x * ext.x + ext.y * ext.y + ext.z * ext.z), 80.0)
    distance = max(radius * 2.8, 220.0)
    height = max(ext.z * 0.65, 80.0)
    saved = 0
    views = max(1, VIEWS_PER_MODEL)

    for view_idx in range(views):
        deg = view_idx * (360.0 / views) + 30.0
        rad = math.radians(deg)
        eye = unreal.Vector(
            origin.x + distance * math.cos(rad),
            origin.y + distance * math.sin(rad),
            origin.z + height,
        )
        target = unreal.Vector(origin.x, origin.y, origin.z + ext.z * 0.15)
        cap_actor.set_actor_location(eye, sweep=False, teleport=True)
        cap_actor.set_actor_rotation(look_at(eye, target), teleport_physics=True)
        sc.capture_scene()
        time.sleep(SLEEP_S)
        out_path = OUT_DIR / f"{mesh_name}__view_{view_idx:02d}.png"
        if _save_render_target_png(world, rt, out_path):
            saved += 1
            unreal.log(f"[render] saved {out_path}")
        else:
            unreal.log_warning(f"[render] no file written: {out_path}")
    return saved


def main() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mesh_groups = list_static_mesh_groups(SOURCE_ROOT, LIMIT)
    unreal.log(
        f"[render] source={SOURCE_ROOT} mesh_groups={len(mesh_groups)} "
        f"views={VIEWS_PER_MODEL} max_parts={MAX_PARTS_PER_MODEL} out={OUT_DIR}"
    )
    if not mesh_groups:
        unreal.log_error("[render] no StaticMesh assets found")
        return {"ok": False, "error": "no StaticMesh assets found", "meshes": 0}

    cap_actor = find_capture_actor()
    if cap_actor is None:
        unreal.log_error("[render] CameraCapture actor not found")
        return {
            "ok": False,
            "error": "CameraCapture actor not found",
            "meshes": len(mesh_groups),
        }

    cleanup_previous_render_actors()
    total_saved = 0
    rendered_meshes = 0
    for idx, (model_key, mesh_paths) in enumerate(mesh_groups):
        _write_status(
            "running",
            meshes=len(mesh_groups),
            mesh_index=idx + 1,
            images_saved=total_saved,
            rendered_meshes=rendered_meshes,
            parts=len(mesh_paths),
        )
        actors = _spawn_mesh_group(mesh_paths, idx)
        if not actors:
            continue
        mesh_name = _safe_stem(model_key)
        saved = _capture_actor_views(cap_actor, actors, mesh_name)
        total_saved += saved
        rendered_meshes += 1 if saved else 0
        for actor in actors:
            unreal.EditorLevelLibrary.destroy_actor(actor)

    unreal.log(
        f"[render] done rendered_meshes={rendered_meshes} images_saved={total_saved}"
    )
    return {
        "ok": total_saved > 0,
        "meshes": len(mesh_groups),
        "rendered_meshes": rendered_meshes,
        "images_saved": total_saved,
        "out_dir": str(OUT_DIR),
    }


def _run_on_slate_tick(delta_time) -> None:  # noqa: ARG001
    global _TICK_HANDLE
    if _TICK_HANDLE is not None:
        unreal.unregister_slate_post_tick_callback(_TICK_HANDLE)
        _TICK_HANDLE = None
    try:
        _write_status("running")
        result = main()
        _write_status("done", **result)
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        unreal.log_error(f"[render] failed: {exc}\n{tb}")
        _write_status("error", error=str(exc), traceback=tb)


_write_status("scheduled")
_TICK_HANDLE = unreal.register_slate_post_tick_callback(_run_on_slate_tick)
unreal.log(f"[render] scheduled on Slate tick; status={STATUS_PATH}")
