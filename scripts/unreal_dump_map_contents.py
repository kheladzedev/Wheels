"""Dump all actors and the CameraCaptureWheels configuration of a map.

Run via UnrealEditor-Cmd:

  VSBL_DUMP_MAP=/Game/Wheels/maps/standartWheelsRoom \
  VSBL_DUMP_OUT=outputs/unreal_control/dump_standartWheelsRoom.json \
  "/Users/Shared/Epic Games/UE_5.7/Engine/Binaries/Mac/UnrealEditor-Cmd" \
    "/Users/edward/Desktop/VSBL/NeuralData1 2/NeuralData.uproject" \
    -run=pythonscript -script=/Users/edward/Desktop/VSBL/scripts/unreal_dump_map_contents.py \
    -unattended -nop4 -nosplash -NullRHI

Read-only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import unreal


def _vec(v: Any) -> list[float]:
    try:
        return [float(v.x), float(v.y), float(v.z)]
    except Exception:
        return []


def _label(actor: Any) -> str:
    try:
        return str(actor.get_actor_label())
    except Exception:
        return str(actor.get_name())


def _cls(actor: Any) -> str:
    try:
        return str(actor.get_class().get_name())
    except Exception:
        return type(actor).__name__


def _dump_capture(actor: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "label": _label(actor),
        "class": _cls(actor),
        "location": _vec(actor.get_actor_location()),
    }
    # ToRotate, Floor, etc are Blueprint exposed properties; try a few.
    for prop in (
        "ToRotate",
        "Floor",
        "Wall",
        "Floors",
        "RotateObjects",
        "Array",
        "Table",
        "TablesStatic",
        "Chair",
        "ChairsStatic",
        "Dir",
        "Carpets",
        "Struct",
    ):
        try:
            v = actor.get_editor_property(prop)
        except Exception:
            continue
        out[prop] = _describe(v)
    return out


def _describe(v: Any) -> Any:
    if v is None:
        return None
    # Actor reference
    if hasattr(v, "get_actor_label"):
        try:
            return {"actor_ref": _label(v), "class": _cls(v)}
        except Exception:
            return str(v)
    if isinstance(v, (list, tuple)):
        return [_describe(x) for x in v]
    try:
        # Unreal Array proxy
        if hasattr(v, "__iter__"):
            return [_describe(x) for x in v]
    except Exception:
        pass
    return str(v)


def main() -> None:
    map_path = os.environ.get("VSBL_DUMP_MAP", "/Game/Wheels/maps/standartWheelsRoom")
    out_path = Path(os.environ.get("VSBL_DUMP_OUT", "outputs/unreal_control/dump.json"))

    unreal.EditorLoadingAndSavingUtils.load_map(map_path)
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = list(actor_subsystem.get_all_level_actors())

    report: dict[str, Any] = {
        "map": map_path,
        "total_actors": len(actors),
        "by_class": {},
        "capture_actors": [],
        "vehicles": [],
        "wheels": [],
    }

    for a in actors:
        cls = _cls(a)
        report["by_class"][cls] = report["by_class"].get(cls, 0) + 1
        label_lower = _label(a).lower()
        if cls == "CameraCaptureWheels_C" or "cameracapturewheels" in label_lower:
            report["capture_actors"].append(_dump_capture(a))
        elif cls == "wheels_C" or label_lower.startswith("wheels"):
            report["wheels"].append(
                {"label": _label(a), "class": cls, "loc": _vec(a.get_actor_location())}
            )
        elif (
            "vehicle" in cls.lower()
            or "vehicle" in label_lower
            or "car" in label_lower
            or "truck" in label_lower
            or "bus" in label_lower
        ):
            report["vehicles"].append(
                {"label": _label(a), "class": cls, "loc": _vec(a.get_actor_location())}
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"WROTE {out_path}")


main()
