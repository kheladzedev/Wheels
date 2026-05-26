"""Rebuild lighting for standartWheelsRoom so captures are not all-black.

Triggers a preview-quality lighting build on the active map. This regenerates
the _BuiltData.uasset alongside the .umap. Production-quality bakes take
hours; preview is enough to make the capture frames legible.

Run via -ExecutePythonScript.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import unreal


MAP_PATH = os.environ.get("VSBL_MAP", "/Game/Wheels/maps/standartWheelsRoom")
OUT_PATH = Path(
    os.environ.get(
        "VSBL_OUT",
        "/Users/edward/Desktop/VSBL/outputs/unreal_control/rebuild_lighting.json",
    )
)


def main() -> None:
    report = {"map": MAP_PATH, "status": "FAIL"}
    try:
        unreal.EditorLoadingAndSavingUtils.load_map(MAP_PATH)
        world = None
        try:
            world = unreal.EditorLevelLibrary.get_editor_world()
        except Exception:
            pass

        # Trigger a preview-quality lighting build.
        if hasattr(unreal, "EditorBuildLibrary"):
            try:
                build_type = (
                    unreal.EditorBuildType.LIGHTING_QUALITY_PREVIEW
                    if hasattr(unreal.EditorBuildType, "LIGHTING_QUALITY_PREVIEW")
                    else 0
                )
                ok = unreal.EditorBuildLibrary.editor_build(world, build_type)
                report["editor_build"] = bool(ok)
            except Exception as exc:
                report["editor_build_error"] = str(exc)

        # Fallback / belt-and-braces: BuildLighting via legacy lib.
        if hasattr(unreal, "EditorLevelLibrary") and hasattr(
            unreal.EditorLevelLibrary, "build_lighting"
        ):
            try:
                unreal.EditorLevelLibrary.build_lighting()
            except Exception as exc:
                report["build_lighting_error"] = str(exc)

        saved = unreal.EditorLoadingAndSavingUtils.save_current_level()
        report["saved"] = bool(saved)
        report["status"] = "PASS" if saved else "PARTIAL_SAVE_FAILED"
    except Exception as exc:
        report["error"] = str(exc)
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
