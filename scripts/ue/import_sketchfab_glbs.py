"""Import downloaded Sketchfab GLBs into Unreal under /Game/SketchfabCars.

Run through UnrealMCP:
    ./.venv/bin/python scripts/ue/_send.py exec_file scripts/ue/import_sketchfab_glbs.py

Optional environment:
    VSBL_SKETCHFAB_GLB_ROOT=/path/to/data/sketchfab_cars
    VSBL_UE_IMPORT_LIMIT=300
    VSBL_UE_IMPORT_DEST=/Game/SketchfabCars
"""

from __future__ import annotations

import os
import re
import json
import traceback
from pathlib import Path

import unreal

SCRIPT_PATH = Path(
    globals().get(
        "__file__",
        "/Users/codefactory/Desktop/ML/VSBL/Wheels/scripts/ue/import_sketchfab_glbs.py",
    )
).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
GLB_ROOT = Path(os.environ.get("VSBL_SKETCHFAB_GLB_ROOT", REPO_ROOT / "data/sketchfab_cars"))
DEST_ROOT = os.environ.get("VSBL_UE_IMPORT_DEST", "/Game/SketchfabCars").rstrip("/")
IMPORT_LIMIT = int(os.environ.get("VSBL_UE_IMPORT_LIMIT", "300"))
BATCH_SIZE = int(os.environ.get("VSBL_UE_IMPORT_BATCH_SIZE", "25"))
STATUS_PATH = REPO_ROOT / "outputs/ue_tasks/import_sketchfab_glbs_status.json"
_TICK_HANDLE = None


def _write_status(state: str, **payload) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {"task": "import_sketchfab_glbs", "state": state, **payload}
    STATUS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_asset_folder(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    return safe[:80] or "model"


def _existing_destination_paths() -> set[str]:
    registry = unreal.AssetRegistryHelpers.get_asset_registry()
    existing = set()
    for asset in registry.get_assets_by_path(DEST_ROOT, recursive=True):
        existing.add(str(asset.package_path))
    return existing


def _make_task(glb: Path) -> unreal.AssetImportTask:
    task = unreal.AssetImportTask()
    destination = f"{DEST_ROOT}/{_safe_asset_folder(glb.stem)}"
    task.set_editor_property("filename", str(glb))
    task.set_editor_property("destination_path", destination)
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", False)
    task.set_editor_property("save", True)
    return task


def _count_static_meshes() -> int:
    registry = unreal.AssetRegistryHelpers.get_asset_registry()
    count = 0
    for asset in registry.get_assets_by_path(DEST_ROOT, recursive=True):
        try:
            class_name = str(asset.asset_class_path.asset_name)
        except AttributeError:
            class_name = ""
        if class_name == "StaticMesh":
            count += 1
    return count


def main() -> dict:
    if not GLB_ROOT.is_dir():
        unreal.log_error(f"[sketchfab-import] missing GLB root: {GLB_ROOT}")
        return {"ok": False, "error": f"missing GLB root: {GLB_ROOT}"}

    glbs = sorted(GLB_ROOT.glob("*.glb"))[:IMPORT_LIMIT]
    existing_paths = _existing_destination_paths()
    tasks = []
    skipped_existing = 0
    for glb in glbs:
        destination = f"{DEST_ROOT}/{_safe_asset_folder(glb.stem)}"
        if destination in existing_paths:
            skipped_existing += 1
            continue
        tasks.append(_make_task(glb))

    unreal.log(
        "[sketchfab-import] "
        f"root={GLB_ROOT} dest={DEST_ROOT} glbs={len(glbs)} "
        f"tasks={len(tasks)} skipped_existing={skipped_existing}"
    )
    if not tasks:
        static_meshes = _count_static_meshes()
        unreal.log(f"[sketchfab-import] static_meshes={static_meshes}")
        return {
            "ok": True,
            "glbs": len(glbs),
            "tasks": 0,
            "skipped_existing": skipped_existing,
            "static_meshes": static_meshes,
        }

    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    for start in range(0, len(tasks), BATCH_SIZE):
        _write_status(
            "running",
            glbs=len(glbs),
            tasks=len(tasks),
            skipped_existing=skipped_existing,
            batch_start=start + 1,
            batch_end=min(start + BATCH_SIZE, len(tasks)),
        )
        batch = tasks[start : start + BATCH_SIZE]
        asset_tools.import_asset_tasks(batch)
        unreal.EditorAssetLibrary.save_directory(DEST_ROOT, only_if_is_dirty=True, recursive=True)
        unreal.log(
            f"[sketchfab-import] imported batch {start + 1}-{start + len(batch)} "
            f"of {len(tasks)}"
        )

    static_meshes = _count_static_meshes()
    unreal.log(f"[sketchfab-import] static_meshes={static_meshes}")
    return {
        "ok": True,
        "glbs": len(glbs),
        "tasks": len(tasks),
        "skipped_existing": skipped_existing,
        "static_meshes": static_meshes,
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
        unreal.log_error(f"[sketchfab-import] failed: {exc}\n{tb}")
        _write_status("error", error=str(exc), traceback=tb)


_write_status("scheduled")
_TICK_HANDLE = unreal.register_slate_post_tick_callback(_run_on_slate_tick)
unreal.log(f"[sketchfab-import] scheduled on Slate tick; status={STATUS_PATH}")
