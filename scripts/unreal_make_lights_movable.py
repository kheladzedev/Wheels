"""Convert Static/Stationary lights in the active map to Movable so capture
works without a baked lightmap.

Capture frames went all-black after every populate because every new
StaticMesh actor invalidated the prebaked lightmap. Setting lights to
Movable means UE renders dynamic lighting at runtime — no bake needed and
populates are safe.

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
        "/Users/edward/Desktop/VSBL/outputs/unreal_control/lights_movable.json",
    )
)


LIGHT_CLASSES = (
    "DirectionalLight",
    "SkyLight",
    "PointLight",
    "SpotLight",
    "RectLight",
)


def main() -> None:
    report = {"map": MAP_PATH, "changed": [], "status": "FAIL"}
    try:
        unreal.EditorLoadingAndSavingUtils.load_map(MAP_PATH)
        subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        for actor in subsystem.get_all_level_actors():
            cls = str(actor.get_class().get_name())
            if cls not in LIGHT_CLASSES:
                continue
            try:
                root = actor.root_component
            except Exception:
                root = None
            try:
                if root is not None:
                    root.set_editor_property(
                        "mobility", unreal.ComponentMobility.MOVABLE
                    )
            except Exception as exc:
                report["changed"].append(
                    {"actor": str(actor.get_actor_label()), "error": str(exc)}
                )
                continue
            # Some Light actors hold their LightComponent under the root; iterate
            try:
                comps = actor.get_components_by_class(unreal.LightComponent)
            except Exception:
                comps = []
            for c in comps:
                try:
                    c.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
                except Exception:
                    pass
            report["changed"].append(
                {
                    "actor": str(actor.get_actor_label()),
                    "class": cls,
                }
            )
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
