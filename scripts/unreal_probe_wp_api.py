"""Probe what World Partition / DataLayer Python APIs are exposed in UE 5.7
against the loaded NeuralData project, and dump actor inventory of map 03
before/after attempting to force-load streaming.

Read-only / diagnostic.

  VSBL_PROBE_MAP=/Game/Wheels/maps/03 \
  VSBL_PROBE_OUT=outputs/unreal_control/wp_probe.json \
  "/Users/Shared/Epic Games/UE_5.7/Engine/Binaries/Mac/UnrealEditor-Cmd" \
    "/Users/edward/Desktop/VSBL/NeuralData1 2/NeuralData.uproject" \
    -run=pythonscript -script=/Users/edward/Desktop/VSBL/scripts/unreal_probe_wp_api.py \
    -unattended -nop4 -nosplash -NullRHI
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import unreal


def _safe_call(callable_, *a, **kw):
    try:
        return callable_(*a, **kw)
    except Exception as exc:
        return f"<error: {exc!r}>"


def _has_attr(mod, name: str) -> bool:
    return hasattr(mod, name)


def _list_actors_summary() -> dict[str, Any]:
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = list(actor_subsystem.get_all_level_actors())
    by_class: dict[str, int] = {}
    labels_sample = []
    for a in actors:
        cls = str(a.get_class().get_name())
        by_class[cls] = by_class.get(cls, 0) + 1
        if len(labels_sample) < 20:
            try:
                labels_sample.append(a.get_actor_label())
            except Exception:
                pass
    return {"total": len(actors), "by_class": by_class, "labels_sample": labels_sample}


def main() -> None:
    map_path = os.environ.get("VSBL_PROBE_MAP", "/Game/Wheels/maps/03")
    out_path = Path(
        os.environ.get(
            "VSBL_PROBE_OUT",
            "/Users/edward/Desktop/VSBL/outputs/unreal_control/wp_probe.json",
        )
    )

    report: dict[str, Any] = {
        "map": map_path,
        "available_apis": {},
        "before": {},
        "after_force_load": {},
        "attempts": [],
    }

    # Discover candidate APIs.
    candidates = [
        "WorldPartitionEditorSubsystem",
        "WorldPartitionEditorPerProjectUserSettings",
        "WorldPartition",
        "WorldPartitionLevelHelper",
        "DataLayerSubsystem",
        "DataLayerEditorSubsystem",
        "WorldPartitionRuntimeCellRegistry",
        "ContentBundleManager",
        "WorldPartitionRuntimeSpatialHash",
    ]
    for name in candidates:
        report["available_apis"][name] = _has_attr(unreal, name)

    # 1. Load the map normally.
    unreal.EditorLoadingAndSavingUtils.load_map(map_path)
    report["before"] = _list_actors_summary()

    # 2. Try to enumerate streaming sublevels via the editor world.
    try:
        editor_world = unreal.EditorLevelLibrary.get_editor_world()
        streaming_levels = (
            unreal.GameplayStatics.get_streaming_levels(editor_world)
            if _has_attr(unreal.GameplayStatics, "get_streaming_levels")
            else None
        )
        if streaming_levels is not None:
            report["streaming_levels"] = [
                {
                    "name": str(sl.get_world_asset_package_name())
                    if hasattr(sl, "get_world_asset_package_name")
                    else str(sl),
                    "should_be_loaded": _safe_call(sl.get_should_be_loaded)
                    if hasattr(sl, "get_should_be_loaded")
                    else None,
                    "should_be_visible": _safe_call(sl.get_should_be_visible)
                    if hasattr(sl, "get_should_be_visible")
                    else None,
                }
                for sl in streaming_levels
            ]
            report["attempts"].append(
                {"step": "enumerate_streaming_levels", "count": len(streaming_levels)}
            )
    except Exception as exc:
        report["attempts"].append(
            {"step": "enumerate_streaming_levels", "error": str(exc)}
        )

    # 3. Try DataLayerEditorSubsystem.
    try:
        if _has_attr(unreal, "DataLayerEditorSubsystem"):
            dls = unreal.get_editor_subsystem(unreal.DataLayerEditorSubsystem)
            data_layers = (
                dls.get_all_data_layers()
                if hasattr(dls, "get_all_data_layers")
                else None
            )
            if data_layers is not None:
                report["data_layers"] = []
                for dl in data_layers:
                    info = {
                        "name": str(dl.get_name())
                        if hasattr(dl, "get_name")
                        else str(dl)
                    }
                    try:
                        info["asset_label"] = (
                            str(dl.get_data_layer_label())
                            if hasattr(dl, "get_data_layer_label")
                            else None
                        )
                    except Exception:
                        pass
                    report["data_layers"].append(info)
                    # Try to make it active and loaded.
                    try:
                        if hasattr(dls, "set_data_layer_initial_runtime_state"):
                            dls.set_data_layer_initial_runtime_state(
                                dl, unreal.DataLayerRuntimeState.ACTIVATED
                            )
                        if hasattr(dls, "set_data_layer_is_loaded_in_editor"):
                            dls.set_data_layer_is_loaded_in_editor(dl, True, False)
                        if hasattr(dls, "set_data_layer_visible"):
                            dls.set_data_layer_visible(dl, True)
                    except Exception as exc2:
                        info["activate_error"] = str(exc2)
                report["attempts"].append(
                    {"step": "activate_data_layers", "count": len(data_layers)}
                )
    except Exception as exc:
        report["attempts"].append({"step": "data_layers", "error": str(exc)})

    # 4. Force-load all editor cells via WorldPartitionEditorSubsystem.
    try:
        if _has_attr(unreal, "WorldPartitionEditorSubsystem"):
            wpe = unreal.get_editor_subsystem(unreal.WorldPartitionEditorSubsystem)
            wpe_methods = [m for m in dir(wpe) if not m.startswith("_")]
            report["wp_editor_methods"] = wpe_methods[:40]
            for m in (
                "load_cells",
                "load_region",
                "load_visible_region",
                "load_all_cells",
            ):
                if hasattr(wpe, m):
                    try:
                        getattr(wpe, m)()
                        report["attempts"].append({"step": f"wpe.{m}", "ok": True})
                    except Exception as exc2:
                        report["attempts"].append(
                            {"step": f"wpe.{m}", "error": str(exc2)}
                        )
    except Exception as exc:
        report["attempts"].append({"step": "wpe", "error": str(exc)})

    # 5. Inspect after.
    report["after_force_load"] = _list_actors_summary()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"WROTE {out_path}")


main()
