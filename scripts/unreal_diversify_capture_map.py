"""Modify CameraCaptureWheels actor in a map to maximize per-run diversity.

Loads the map, finds the CameraCaptureWheels actor, clears ToRotate (so the
camera rotates randomly per the Blueprint's documented behaviour), and fills
RotateObjects with all StaticMeshActor vehicles in the scene so they rotate
each captured frame. Then saves the map in place.

Idempotent. Run via:

  VSBL_DIVERSIFY_MAP=/Game/Wheels/maps/standartWheelsRoom \
  VSBL_DIVERSIFY_OUT=outputs/unreal_control/diversify_report.json \
  "/Users/Shared/Epic Games/UE_5.7/Engine/Binaries/Mac/UnrealEditor-Cmd" \
    "/Users/edward/Desktop/VSBL/NeuralData1 2/NeuralData.uproject" \
    -run=pythonscript -script=/Users/edward/Desktop/VSBL/scripts/unreal_diversify_capture_map.py \
    -unattended -nop4 -nosplash -NullRHI
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import unreal


VEHICLE_LABEL_HINTS = (
    "car",
    "truck",
    "bus",
    "vehicle",
    "van",
    "suv",
    "sedan",
    "demolition",
)


def _label(a: Any) -> str:
    try:
        return str(a.get_actor_label())
    except Exception:
        return str(a.get_name())


def _is_vehicle_mesh(actor: Any) -> bool:
    if not isinstance(actor, unreal.StaticMeshActor):
        return False
    label = _label(actor).lower()
    return any(h in label for h in VEHICLE_LABEL_HINTS)


def _find_capture(actors: list[Any]) -> Any | None:
    for a in actors:
        cls = str(a.get_class().get_name()).lower()
        label = _label(a).lower()
        if cls == "cameracapturewheels_c" or "cameracapturewheels" in label:
            return a
    return None


def main() -> None:
    map_path = os.environ.get(
        "VSBL_DIVERSIFY_MAP", "/Game/Wheels/maps/standartWheelsRoom"
    )
    out_path = Path(
        os.environ.get(
            "VSBL_DIVERSIFY_OUT",
            "/Users/edward/Desktop/VSBL/outputs/unreal_control/diversify_report.json",
        )
    )

    report: dict[str, Any] = {"map": map_path, "changes": [], "status": "FAIL"}

    try:
        unreal.EditorLoadingAndSavingUtils.load_map(map_path)
        actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        actors = list(actor_subsystem.get_all_level_actors())

        capture = _find_capture(actors)
        if capture is None:
            raise RuntimeError("CameraCaptureWheels not found")
        report["capture_actor"] = _label(capture)

        # 1. Clear ToRotate so camera rotation becomes random per Blueprint spec.
        try:
            current = capture.get_editor_property("ToRotate")
            if current is not None:
                capture.set_editor_property("ToRotate", None)
                report["changes"].append(
                    {
                        "prop": "ToRotate",
                        "from": _label(current)
                        if hasattr(current, "get_actor_label")
                        else str(current),
                        "to": None,
                    }
                )
            else:
                report["changes"].append({"prop": "ToRotate", "noop": True})
        except Exception as exc:
            report["changes"].append({"prop": "ToRotate", "error": str(exc)})

        # 2. Fill RotateObjects with all vehicle StaticMeshActors.
        vehicles = [a for a in actors if _is_vehicle_mesh(a)]
        report["vehicles_detected"] = [_label(v) for v in vehicles]
        try:
            current_list = capture.get_editor_property("RotateObjects")
            current_count = 0
            try:
                current_count = (
                    len(list(current_list)) if current_list is not None else 0
                )
            except Exception:
                pass
            # Wrap in unreal.Array to match property type.
            new_arr = unreal.Array(unreal.Actor)
            for v in vehicles:
                new_arr.append(v)
            capture.set_editor_property("RotateObjects", new_arr)
            report["changes"].append(
                {
                    "prop": "RotateObjects",
                    "from_count": current_count,
                    "to_count": len(vehicles),
                    "actors": [_label(v) for v in vehicles],
                }
            )
        except Exception as exc:
            report["changes"].append({"prop": "RotateObjects", "error": str(exc)})

        # Save the map in place.
        saved = unreal.EditorLoadingAndSavingUtils.save_current_level()
        report["saved"] = bool(saved)
        report["status"] = "PASS" if saved else "PARTIAL_SAVE_FAILED"
    except Exception as exc:
        report["error"] = str(exc)
        report["status"] = "FAIL"
        raise
    finally:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"WROTE {out_path}")


main()
