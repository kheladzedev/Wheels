"""Render imported car model groups and export geometry-derived wheel labels.

This is the production-oriented alternative to model pseudo-labeling:
for each imported model folder under /Game/SketchfabCars, spawn its mesh
parts, identify wheel/tire/rim StaticMesh parts by asset path/name, project
their 3D bounds through the CameraCapture pose, and write plugin-format
incoming annotations.

Run through UnrealMCP:
    ./.venv/bin/python scripts/ue/_send.py exec_file scripts/ue/render_sketchfab_geometry_labels.py

Output:
    data/incoming/ue_sketchfab_geometry/{images,annotations,metadata}/
    outputs/ue_tasks/render_sketchfab_geometry_labels_status.json
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path

import unreal

SCRIPT_PATH = Path(
    globals().get(
        "__file__",
        "/Users/codefactory/Desktop/ML/VSBL/Wheels/scripts/ue/render_sketchfab_geometry_labels.py",
    )
).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ue_wheel_asset_filter import classify_wheel_asset_paths, wheel_classifier_metadata

SOURCE_ROOT = os.environ.get("VSBL_UE_GEOM_SOURCE", "/Game/SketchfabCars")
OUT_ROOT = Path(
    os.environ.get(
        "VSBL_UE_GEOM_OUT",
        REPO_ROOT / "data/incoming/ue_sketchfab_geometry",
    )
)
LIMIT = int(os.environ.get("VSBL_UE_GEOM_LIMIT", "300"))
VIEWS_PER_MODEL = int(os.environ.get("VSBL_UE_GEOM_VIEWS", "4"))
SLEEP_S = float(os.environ.get("VSBL_UE_GEOM_SLEEP", "0.15"))
TICKS_AFTER_CAPTURE = int(os.environ.get("VSBL_UE_GEOM_TICKS_AFTER_CAPTURE", "3"))
CAPTURE_SIZE = int(os.environ.get("VSBL_UE_GEOM_SIZE", "1024"))
MAX_PARTS_PER_MODEL = int(os.environ.get("VSBL_UE_GEOM_MAX_PARTS_PER_MODEL", "120"))
MIN_BBOX_SIDE_PX = float(os.environ.get("VSBL_UE_GEOM_MIN_BBOX_SIDE", "18"))
MIN_BBOX_AREA_FRAC = float(os.environ.get("VSBL_UE_GEOM_MIN_BBOX_AREA_FRAC", "0.00035"))
MIN_VISIBLE_POINTS = int(os.environ.get("VSBL_UE_GEOM_MIN_VISIBLE_POINTS", "4"))
MAX_WHEELS_PER_VIEW = int(os.environ.get("VSBL_UE_GEOM_MAX_WHEELS_PER_VIEW", "6"))
FOV_DEG = float(os.environ.get("VSBL_UE_GEOM_FOV", "45"))
OVERWRITE = os.environ.get("VSBL_UE_GEOM_OVERWRITE", "1") not in ("0", "false", "False")
ACTOR_LABEL_PREFIX = "VSBL_GeomCar_"
STATUS_PATH = REPO_ROOT / "outputs/ue_tasks/render_sketchfab_geometry_labels_status.json"

_TICK_HANDLE = None


def _write_status(state: str, **payload) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {"task": "render_sketchfab_geometry_labels", "state": state, **payload}
    STATUS_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")[:96] or "mesh"


def _editor_world():
    return unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()


def _asset_class_name(asset_data) -> str:
    try:
        return str(asset_data.asset_class_path.asset_name)
    except AttributeError:
        return ""


def list_static_mesh_groups(folder: str, limit: int) -> list[tuple[str, list[tuple[str, bool]]]]:
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

    groups: list[tuple[str, list[tuple[str, bool]]]] = []
    for model_key, candidates in sorted(grouped.items()):
        ranked = sorted(
            candidates,
            key=lambda item: (
                str(item.asset_name).lower().startswith(("material", "color", "texture")),
                str(item.package_name),
            ),
        )
        paths = []
        for asset in ranked[:MAX_PARTS_PER_MODEL]:
            path = str(asset.package_name) + "." + str(asset.asset_name)
            paths.append(path)
        flags = classify_wheel_asset_paths(paths)
        parts = list(zip(paths, flags, strict=True))
        if any(is_wheel for _, is_wheel in parts):
            groups.append((model_key, parts))
        if len(groups) >= limit:
            break
    return groups


def find_capture_actor():
    actors = unreal.GameplayStatics.get_all_actors_of_class(_editor_world(), unreal.Actor)
    for actor in actors:
        label = (actor.get_actor_label() or "").lower()
        cls_name = actor.get_class().get_name().lower()
        if "cameracapture" in label or cls_name.startswith("cameracapture"):
            return actor
    return None


def cleanup_previous_render_actors() -> None:
    actors = unreal.GameplayStatics.get_all_actors_of_class(_editor_world(), unreal.Actor)
    for actor in actors:
        if (actor.get_actor_label() or "").startswith(ACTOR_LABEL_PREFIX):
            unreal.EditorLevelLibrary.destroy_actor(actor)


def reset_output_dirs() -> None:
    if not OVERWRITE:
        return
    for sub in ("images", "annotations"):
        root = OUT_ROOT / sub
        if not root.is_dir():
            continue
        for path in root.glob("*"):
            if path.is_file():
                path.unlink()


def setup_capture(scene_capture) -> None:
    try:
        scene_capture.set_editor_property("fov_angle", FOV_DEG)
        scene_capture.set_editor_property("capture_source", unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
        scene_capture.set_editor_property("post_process_blend_weight", 0.0)
    except Exception as exc:  # noqa: BLE001
        unreal.log_warning(f"[geom-label] cannot configure capture component: {exc}")


def spawn_capture_actor() -> tuple[object, object, object]:
    capture = unreal.EditorLevelLibrary.spawn_actor_from_class(
        unreal.SceneCapture2D,
        unreal.Vector(0.0, 0.0, 0.0),
    )
    if capture is None:
        raise RuntimeError("could not spawn SceneCapture2D")
    capture.set_actor_label(f"{ACTOR_LABEL_PREFIX}SceneCapture")
    scene_capture = capture.get_component_by_class(unreal.SceneCaptureComponent2D)
    if scene_capture is None:
        raise RuntimeError("spawned SceneCapture2D has no SceneCaptureComponent2D")
    rt = unreal.RenderingLibrary.create_render_target2d(
        _editor_world(),
        CAPTURE_SIZE,
        CAPTURE_SIZE,
        unreal.TextureRenderTargetFormat.RTF_RGBA8,
        unreal.LinearColor(0.0, 0.0, 0.0, 1.0),
        False,
    )
    scene_capture.set_editor_property("texture_target", rt)
    scene_capture.set_editor_property("capture_every_frame", False)
    scene_capture.set_editor_property("capture_on_movement", False)
    setup_capture(scene_capture)
    return capture, scene_capture, rt


def spawn_lights(origin: unreal.Vector, radius: float) -> list:
    world_z = origin.z + max(radius * 0.75, 120.0)
    positions = (
        unreal.Vector(origin.x - radius, origin.y - radius, world_z),
        unreal.Vector(origin.x + radius, origin.y + radius * 0.5, world_z),
        unreal.Vector(origin.x, origin.y, origin.z + max(radius * 1.4, 220.0)),
    )
    lights = []
    for idx, position in enumerate(positions):
        light = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.PointLight, position)
        if light is None:
            continue
        light.set_actor_label(f"{ACTOR_LABEL_PREFIX}Light_{idx:02d}")
        try:
            component = light.get_component_by_class(unreal.PointLightComponent)
            component.set_editor_property("intensity", 90000.0)
            component.set_editor_property("attenuation_radius", max(radius * 6.0, 1200.0))
            component.set_editor_property("source_radius", max(radius * 0.08, 20.0))
        except Exception as exc:  # noqa: BLE001
            unreal.log_warning(f"[geom-label] cannot configure point light: {exc}")
        lights.append(light)
    return lights


def _vec_add(a, b):
    return unreal.Vector(a.x + b.x, a.y + b.y, a.z + b.z)


def _vec_sub(a, b):
    return unreal.Vector(a.x - b.x, a.y - b.y, a.z - b.z)


def _vec_mul(a, scalar: float):
    return unreal.Vector(a.x * scalar, a.y * scalar, a.z * scalar)


def _dot(a, b) -> float:
    return float(a.x * b.x + a.y * b.y + a.z * b.z)


def _cross(a, b):
    return unreal.Vector(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )


def _norm(a):
    length = math.sqrt(max(_dot(a, a), 1e-12))
    return _vec_mul(a, 1.0 / length)


def look_at(eye: unreal.Vector, target: unreal.Vector) -> unreal.Rotator:
    dx, dy, dz = target.x - eye.x, target.y - eye.y, target.z - eye.z
    yaw = math.degrees(math.atan2(dy, dx))
    horiz = math.sqrt(dx * dx + dy * dy)
    pitch = math.degrees(math.atan2(dz, horiz))
    return unreal.Rotator(roll=0.0, pitch=pitch, yaw=yaw)


def _spawn_mesh(mesh_path: str, idx: int, part_idx: int):
    mesh = unreal.EditorAssetLibrary.load_asset(mesh_path)
    if mesh is None:
        unreal.log_warning(f"[geom-label] cannot load {mesh_path}")
        return None
    actor = unreal.EditorLevelLibrary.spawn_actor_from_object(
        mesh, unreal.Vector(0.0, 0.0, 0.0)
    )
    if actor is None:
        unreal.log_warning(f"[geom-label] cannot spawn {mesh_path}")
        return None
    actor.set_actor_label(f"{ACTOR_LABEL_PREFIX}{idx:04d}_{part_idx:03d}")
    return actor


def _spawn_group(parts: list[tuple[str, bool]], idx: int) -> list[tuple[object, str, bool]]:
    out = []
    for part_idx, (mesh_path, is_wheel) in enumerate(parts):
        actor = _spawn_mesh(mesh_path, idx, part_idx)
        if actor is not None:
            out.append((actor, mesh_path, is_wheel))
    return out


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


def _bounds_corners(actor) -> list:
    origin, ext = actor.get_actor_bounds(True)
    corners = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corners.append(
                    unreal.Vector(
                        origin.x + ext.x * sx,
                        origin.y + ext.y * sy,
                        origin.z + ext.z * sz,
                    )
                )
    return corners


def _project(point, eye, forward, right, up, width: int, height: int, hfov_deg: float):
    rel = _vec_sub(point, eye)
    depth = _dot(rel, forward)
    if depth <= 1.0:
        return None
    aspect = width / max(float(height), 1.0)
    tan_h = math.tan(math.radians(hfov_deg) * 0.5)
    tan_v = tan_h / aspect
    nx = _dot(rel, right) / (depth * tan_h)
    ny = _dot(rel, up) / (depth * tan_v)
    px = (0.5 + nx * 0.5) * width
    py = (0.5 - ny * 0.5) * height
    return px, py


def _bbox_iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def _dedupe_wheels(wheels: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for wheel in sorted(
        wheels,
        key=lambda item: (item["bbox_xyxy"][2] - item["bbox_xyxy"][0])
        * (item["bbox_xyxy"][3] - item["bbox_xyxy"][1]),
        reverse=True,
    ):
        if all(_bbox_iou(wheel["bbox_xyxy"], other["bbox_xyxy"]) < 0.45 for other in kept):
            kept.append(wheel)
    return kept[:MAX_WHEELS_PER_VIEW]


def _wheel_from_actor(actor, mesh_path: str, eye, forward, right, up, width, height, hfov_deg):
    projected = []
    for corner in _bounds_corners(actor):
        point = _project(corner, eye, forward, right, up, width, height, hfov_deg)
        if point is not None:
            projected.append(point)
    if len(projected) < MIN_VISIBLE_POINTS:
        return None

    xs = [p[0] for p in projected]
    ys = [p[1] for p in projected]
    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(float(width - 1), max(xs))
    y2 = min(float(height - 1), max(ys))
    if (x2 - x1) < MIN_BBOX_SIDE_PX or (y2 - y1) < MIN_BBOX_SIDE_PX:
        return None
    if ((x2 - x1) * (y2 - y1)) < float(width * height) * MIN_BBOX_AREA_FRAC:
        return None

    # The plugin wants A/B as bottom floor-ray points and C as disc bottom.
    # For arbitrary third-party meshes we can only derive a conservative
    # geometry proxy from the projected wheel-part bounds; keep review flags.
    a = [round(x1, 3), round(y2, 3)]
    b = [round(x2, 3), round(y2, 3)]
    c = [round((x1 + x2) * 0.5, 3), round(y2, 3)]
    return {
        "bbox_xyxy": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
        "points": {"a": a, "b": b, "c_disc_bottom": c},
        "_draft": True,
        "_needs_review": True,
        "_review_reasons": ["ue_geometry_projected_wheel_part_bounds"],
        "_source_mesh": mesh_path,
    }


def _save_render_target_png(world, rt, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        unreal.RenderingLibrary.export_render_target(world, rt, str(out_path.parent), out_path.name)
        return out_path.is_file()
    except Exception as exc:  # noqa: BLE001
        unreal.log_error(f"[geom-label] export_render_target failed: {exc}")
        return False


def _write_annotation(frame_id: str, image_name: str, wheels: list[dict], model_key: str) -> None:
    annotation = {
        "frame_id": frame_id,
        "image": image_name,
        "wheels": wheels,
        "_draft": True,
        "_warning": "UE_GEOMETRY_PROJECTED_WHEEL_PART_BOUNDS_REQUIRES_REVIEW",
        "_source_model": model_key,
    }
    out_path = OUT_ROOT / "annotations" / f"{frame_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(annotation, indent=2, ensure_ascii=False), encoding="utf-8")


def _render_group(cap_actor, scene_capture, rt, model_key: str, spawned_parts: list, idx: int) -> tuple[int, int]:
    world = _editor_world()
    actors = [part[0] for part in spawned_parts]
    wheel_parts = [part for part in spawned_parts if part[2]]
    group_bounds = _group_bounds(actors)
    wheel_bounds = _group_bounds([part[0] for part in wheel_parts])
    if group_bounds is None or wheel_bounds is None or not wheel_parts:
        return 0, 0

    origin, ext = wheel_bounds
    ext = unreal.Vector(max(ext.x * 1.8, 45.0), max(ext.y * 1.8, 45.0), max(ext.z * 2.0, 35.0))
    radius = max(math.sqrt(ext.x * ext.x + ext.y * ext.y + ext.z * ext.z), 70.0)
    distance = max(radius * 1.45, 115.0)
    height = max(ext.z * 0.8, 45.0)
    width = int(rt.get_editor_property("size_x"))
    height_px = int(rt.get_editor_property("size_y"))
    hfov = float(scene_capture.get_editor_property("fov_angle"))
    saved = 0
    wheels_written = 0
    lights = spawn_lights(origin, radius)

    try:
        for view_idx in range(max(1, VIEWS_PER_MODEL)):
            deg = view_idx * (360.0 / max(1, VIEWS_PER_MODEL)) + 30.0
            rad = math.radians(deg)
            eye = unreal.Vector(
                origin.x + distance * math.cos(rad),
                origin.y + distance * math.sin(rad),
                origin.z + height,
            )
            target = unreal.Vector(origin.x, origin.y, origin.z + ext.z * 0.08)
            forward = _norm(_vec_sub(target, eye))
            world_up = unreal.Vector(0.0, 0.0, 1.0)
            right = _norm(_cross(world_up, forward))
            up = _norm(_cross(forward, right))

            cap_actor.set_actor_location(eye, sweep=False, teleport=True)
            cap_actor.set_actor_rotation(look_at(eye, target), teleport_physics=True)
            scene_capture.capture_scene()
            time.sleep(SLEEP_S)

            wheels = []
            for actor, mesh_path, _ in wheel_parts:
                wheel = _wheel_from_actor(actor, mesh_path, eye, forward, right, up, width, height_px, hfov)
                if wheel is not None:
                    wheels.append(wheel)
            wheels = _dedupe_wheels(wheels)
            if not wheels:
                continue

            frame_id = f"{_safe_stem(model_key)}__view_{view_idx:02d}"
            image_name = f"{frame_id}.png"
            out_img = OUT_ROOT / "images" / image_name
            if _save_render_target_png(world, rt, out_img):
                _write_annotation(frame_id, image_name, wheels, model_key)
                saved += 1
                wheels_written += len(wheels)
    finally:
        for light in lights:
            unreal.EditorLevelLibrary.destroy_actor(light)
    return saved, wheels_written


def _write_source_info(total_groups: int, frames_written: int, wheels_written: int) -> None:
    meta_dir = OUT_ROOT / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "source_name": "ue_sketchfab_geometry",
        "annotation_method": "ue_geometry_projected_wheel_part_bounds",
        "_warning": "GEOMETRY_PROXY_REQUIRES_REVIEW",
        "source_root": SOURCE_ROOT,
        "model_groups_seen": total_groups,
        "frames_written": frames_written,
        "wheels_written": wheels_written,
        "wheel_classifier": wheel_classifier_metadata(),
        "views_per_model": VIEWS_PER_MODEL,
        "max_parts_per_model": MAX_PARTS_PER_MODEL,
    }
    (meta_dir / "source_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


class GeometryRenderTask:
    def __init__(self) -> None:
        for sub in ("images", "annotations", "metadata"):
            (OUT_ROOT / sub).mkdir(parents=True, exist_ok=True)
        reset_output_dirs()

        self.groups = list_static_mesh_groups(SOURCE_ROOT, LIMIT)
        cleanup_previous_render_actors()
        self.cap_actor = None
        self.scene_capture = None
        self.rt = None
        self.done = False
        self.result: dict | None = None
        self.idx = 0
        self.view_idx = 0
        self.spawned_parts: list = []
        self.lights: list = []
        self.model_key = ""
        self.origin = None
        self.ext = None
        self.radius = 0.0
        self.distance = 0.0
        self.eye = None
        self.forward = None
        self.right = None
        self.up = None
        self.width = 0
        self.height_px = 0
        self.hfov = FOV_DEG
        self.pending_frame: tuple[str, str, list[dict], str] | None = None
        self.wait_ticks = 0
        self.phase = "group"
        self.frames_written = 0
        self.wheels_written = 0
        self.groups_rendered = 0
        self.current_group_saved = 0

        try:
            self.cap_actor, self.scene_capture, self.rt = spawn_capture_actor()
        except Exception as exc:  # noqa: BLE001
            self._finish({"ok": False, "error": str(exc), "groups": len(self.groups)})
            return

    def _finish(self, result: dict) -> None:
        self._cleanup_group()
        self.done = True
        self.result = result
        _write_status("done", **result)

    def _cleanup_group(self) -> None:
        for light in self.lights:
            unreal.EditorLevelLibrary.destroy_actor(light)
        self.lights = []
        for actor, _, _ in self.spawned_parts:
            unreal.EditorLevelLibrary.destroy_actor(actor)
        self.spawned_parts = []

    def _cleanup_capture(self) -> None:
        if self.cap_actor is not None:
            unreal.EditorLevelLibrary.destroy_actor(self.cap_actor)
            self.cap_actor = None

    def _start_group(self) -> bool:
        while self.idx < len(self.groups):
            self.model_key, parts = self.groups[self.idx]
            _write_status(
                "running",
                groups=len(self.groups),
                group_index=self.idx + 1,
                frames_written=self.frames_written,
                wheels_written=self.wheels_written,
                parts=len(parts),
                wheel_parts=sum(1 for _, is_wheel in parts if is_wheel),
            )
            self.spawned_parts = _spawn_group(parts, self.idx)
            actors = [part[0] for part in self.spawned_parts]
            wheel_parts = [part for part in self.spawned_parts if part[2]]
            wheel_bounds = _group_bounds([part[0] for part in wheel_parts])
            if not actors or not wheel_parts or wheel_bounds is None:
                self._cleanup_group()
                self.idx += 1
                continue

            self.origin, raw_ext = wheel_bounds
            self.ext = unreal.Vector(
                max(raw_ext.x * 1.8, 45.0),
                max(raw_ext.y * 1.8, 45.0),
                max(raw_ext.z * 2.0, 35.0),
            )
            self.radius = max(
                math.sqrt(self.ext.x * self.ext.x + self.ext.y * self.ext.y + self.ext.z * self.ext.z),
                70.0,
            )
            self.distance = max(self.radius * 1.45, 115.0)
            self.lights = spawn_lights(self.origin, self.radius)
            self.width = int(self.rt.get_editor_property("size_x"))
            self.height_px = int(self.rt.get_editor_property("size_y"))
            self.hfov = float(self.scene_capture.get_editor_property("fov_angle"))
            self.view_idx = 0
            self.current_group_saved = 0
            self.phase = "capture"
            return False

        _write_source_info(len(self.groups), self.frames_written, self.wheels_written)
        self._cleanup_capture()
        self._finish(
            {
                "ok": self.frames_written > 0 and self.wheels_written > 0,
                "groups": len(self.groups),
                "groups_rendered": self.groups_rendered,
                "frames_written": self.frames_written,
                "wheels_written": self.wheels_written,
                "out_root": str(OUT_ROOT),
            }
        )
        return True

    def _prepare_capture(self) -> None:
        if self.view_idx >= max(1, VIEWS_PER_MODEL):
            if self.current_group_saved:
                self.groups_rendered += 1
            self._cleanup_group()
            self.idx += 1
            self.phase = "group"
            return

        deg = self.view_idx * (360.0 / max(1, VIEWS_PER_MODEL)) + 30.0
        rad = math.radians(deg)
        height = max(self.ext.z * 0.8, 45.0)
        self.eye = unreal.Vector(
            self.origin.x + self.distance * math.cos(rad),
            self.origin.y + self.distance * math.sin(rad),
            self.origin.z + height,
        )
        target = unreal.Vector(self.origin.x, self.origin.y, self.origin.z + self.ext.z * 0.08)
        self.forward = _norm(_vec_sub(target, self.eye))
        world_up = unreal.Vector(0.0, 0.0, 1.0)
        self.right = _norm(_cross(world_up, self.forward))
        self.up = _norm(_cross(self.forward, self.right))

        self.cap_actor.set_actor_location(self.eye, sweep=False, teleport=True)
        self.cap_actor.set_actor_rotation(look_at(self.eye, target), teleport_physics=True)

        wheels = []
        for actor, mesh_path, is_wheel in self.spawned_parts:
            if not is_wheel:
                continue
            wheel = _wheel_from_actor(
                actor,
                mesh_path,
                self.eye,
                self.forward,
                self.right,
                self.up,
                self.width,
                self.height_px,
                self.hfov,
            )
            if wheel is not None:
                wheels.append(wheel)
        wheels = _dedupe_wheels(wheels)
        if not wheels:
            self.view_idx += 1
            return

        frame_id = f"{_safe_stem(self.model_key)}__view_{self.view_idx:02d}"
        image_name = f"{frame_id}.png"
        self.pending_frame = (frame_id, image_name, wheels, self.model_key)
        self.scene_capture.update_content()
        self.scene_capture.capture_scene()
        self.wait_ticks = max(1, TICKS_AFTER_CAPTURE)
        self.phase = "export"

    def _export_pending(self) -> None:
        if self.wait_ticks > 0:
            self.wait_ticks -= 1
            return
        if self.pending_frame is None:
            self.view_idx += 1
            self.phase = "capture"
            return
        frame_id, image_name, wheels, model_key = self.pending_frame
        out_img = OUT_ROOT / "images" / image_name
        if _save_render_target_png(_editor_world(), self.rt, out_img):
            _write_annotation(frame_id, image_name, wheels, model_key)
            self.frames_written += 1
            self.wheels_written += len(wheels)
            self.current_group_saved += 1
        self.pending_frame = None
        self.view_idx += 1
        self.phase = "capture"

    def step(self) -> bool:
        if self.done:
            return True
        if self.phase == "group":
            return self._start_group()
        if self.phase == "capture":
            self._prepare_capture()
            return False
        if self.phase == "export":
            self._export_pending()
            return False
        return False


def main() -> dict:
    for sub in ("images", "annotations", "metadata"):
        (OUT_ROOT / sub).mkdir(parents=True, exist_ok=True)
    reset_output_dirs()

    groups = list_static_mesh_groups(SOURCE_ROOT, LIMIT)
    cap_actor = find_capture_actor()
    if cap_actor is None:
        return {"ok": False, "error": "CameraCapture actor not found", "groups": len(groups)}
    sc_components = cap_actor.get_components_by_class(unreal.SceneCaptureComponent2D)
    if not sc_components:
        return {"ok": False, "error": "CameraCapture has no SceneCaptureComponent2D", "groups": len(groups)}
    scene_capture = sc_components[0]
    rt = scene_capture.get_editor_property("texture_target")
    if rt is None:
        return {"ok": False, "error": "CameraCapture has no texture_target", "groups": len(groups)}
    setup_capture(scene_capture)

    cleanup_previous_render_actors()
    frames_written = 0
    wheels_written = 0
    groups_rendered = 0

    for idx, (model_key, parts) in enumerate(groups):
        _write_status(
            "running",
            groups=len(groups),
            group_index=idx + 1,
            frames_written=frames_written,
            wheels_written=wheels_written,
            parts=len(parts),
            wheel_parts=sum(1 for _, is_wheel in parts if is_wheel),
        )
        spawned_parts = _spawn_group(parts, idx)
        try:
            saved, wheels = _render_group(cap_actor, scene_capture, rt, model_key, spawned_parts, idx)
            if saved:
                groups_rendered += 1
                frames_written += saved
                wheels_written += wheels
        finally:
            for actor, _, _ in spawned_parts:
                unreal.EditorLevelLibrary.destroy_actor(actor)

    _write_source_info(len(groups), frames_written, wheels_written)
    return {
        "ok": frames_written > 0 and wheels_written > 0,
        "groups": len(groups),
        "groups_rendered": groups_rendered,
        "frames_written": frames_written,
        "wheels_written": wheels_written,
        "out_root": str(OUT_ROOT),
    }


def _run_on_slate_tick(delta_time) -> None:  # noqa: ARG001
    global _TICK_HANDLE, _TASK
    try:
        if _TASK is None:
            _write_status("running")
            _TASK = GeometryRenderTask()
        if _TASK.step() and _TICK_HANDLE is not None:
            unreal.unregister_slate_post_tick_callback(_TICK_HANDLE)
            _TICK_HANDLE = None
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        unreal.log_error(f"[geom-label] failed: {exc}\n{tb}")
        _write_status("error", error=str(exc), traceback=tb)
        if _TICK_HANDLE is not None:
            unreal.unregister_slate_post_tick_callback(_TICK_HANDLE)
            _TICK_HANDLE = None


_write_status("scheduled")
_TASK = None
_TICK_HANDLE = unreal.register_slate_post_tick_callback(_run_on_slate_tick)
unreal.log(f"[geom-label] scheduled on Slate tick; status={STATUS_PATH}")
