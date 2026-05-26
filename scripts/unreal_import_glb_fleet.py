"""Import every .glb file in data/raw/sketchfab_cars/ into the NeuralData
Unreal project under /Game/SketchfabCars/.

Each .glb is imported via unreal.AssetImportTask, producing one or more
StaticMesh uassets per file (a GLB file may contain a single mesh or a
nested mesh graph; Unreal's GLTF importer flattens it into one or more
StaticMeshes inside the destination folder).

Run as a normal editor session via -ExecutePythonScript so renderer is
available and import succeeds; calls quit_editor() at the end.

  "/Users/Shared/Epic Games/UE_5.7/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor" \\
    "/Users/edward/Desktop/VSBL/NeuralData1 2/NeuralData.uproject" \\
    -ExecutePythonScript="/Users/edward/Desktop/VSBL/scripts/unreal_import_glb_fleet.py" \\
    -nosplash
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import unreal


GLB_DIR = Path(
    os.environ.get("VSBL_GLB_DIR", "/Users/edward/Desktop/VSBL/data/raw/sketchfab_cars")
)
DEST_DIR = "/Game/SketchfabCars"
OUT_PATH = Path(
    os.environ.get(
        "VSBL_IMPORT_OUT",
        "/Users/edward/Desktop/VSBL/outputs/unreal_control/import_glb_fleet.json",
    )
)


def _slug(stem: str) -> str:
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in stem)
    if safe and safe[0].isdigit():
        safe = "sk_" + safe
    return safe[:60]


def _import_one(glb_path: Path, dest_pkg: str) -> tuple[bool, list[str]]:
    task = unreal.AssetImportTask()
    task.filename = str(glb_path)
    task.destination_path = dest_pkg
    task.replace_existing = True
    task.replace_existing_settings = True
    task.automated = True
    task.save = True
    task.factory = (
        unreal.GLTFImportFactory() if hasattr(unreal, "GLTFImportFactory") else None
    )
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    try:
        asset_tools.import_asset_tasks([task])
    except Exception as exc:
        return False, [f"import_exc:{exc!r}"]
    imported: list[str] = []
    try:
        for o in task.imported_object_paths:
            imported.append(str(o))
    except Exception:
        pass
    return len(imported) > 0, imported


def main() -> None:
    report: dict[str, Any] = {
        "glb_dir": str(GLB_DIR),
        "destination": DEST_DIR,
        "imported": [],
        "failed": [],
    }
    try:
        if not GLB_DIR.is_dir():
            raise FileNotFoundError(f"glb dir missing: {GLB_DIR}")
        glbs = sorted(GLB_DIR.glob("*.glb"))
        report["glb_count"] = len(glbs)
        for glb in glbs:
            slug = _slug(glb.stem)
            dest_pkg = f"{DEST_DIR}/{slug}"
            unreal.EditorAssetLibrary.make_directory(dest_pkg)
            ok, imported_paths = _import_one(glb, dest_pkg)
            entry = {
                "glb": glb.name,
                "slug": slug,
                "imported_paths": imported_paths,
            }
            if ok:
                report["imported"].append(entry)
            else:
                report["failed"].append(entry)
        report["status"] = "PASS"
    except Exception as exc:
        report["error"] = str(exc)
        report["status"] = "FAIL"
    finally:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"WROTE {OUT_PATH}")
        try:
            if hasattr(unreal.SystemLibrary, "quit_editor"):
                unreal.SystemLibrary.quit_editor()
        except Exception:
            pass


main()
