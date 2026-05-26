"""Create a cleaned Unreal capture-map copy by removing bad wheel actors.

This commandlet script is intentionally conservative:
- duplicates the source map;
- deletes wheel actors by observed export object id;
- leaves the original map untouched;
- writes a JSON/Markdown report.

Run with UnrealEditor-Cmd:

    VSBL_UNREAL_SOURCE_MAP=/Game/Wheels/maps/standartWheelsRoom \
    VSBL_UNREAL_CLEAN_MAP=/Game/Wheels/maps/standartWheelsRoom_capture_clean_v2 \
    VSBL_UNREAL_REMOVE_EXPORT_IDS=1,4,5 \
    VSBL_UNREAL_CLEAN_OUT=outputs/unreal_control/create_clean_capture_map_v2_result.json \
    UnrealEditor-Cmd <project.uproject> -run=pythonscript \
      -script=scripts/unreal_create_clean_capture_map.py -unattended -nop4 -NullRHI
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import unreal


DEFAULT_SOURCE_MAP = "/Game/Wheels/maps/standartWheelsRoom"
DEFAULT_CLEAN_MAP = "/Game/Wheels/maps/standartWheelsRoom_capture_clean_v2"
DEFAULT_REMOVE_IDS = "1,4,5"
DEFAULT_OUT = (
    "/Users/edward/Desktop/VSBL/outputs/unreal_control/"
    "create_clean_capture_map_v2_result.json"
)


def _actor_label(actor: Any) -> str:
    try:
        return str(actor.get_actor_label())
    except Exception:
        return str(actor.get_name())


def _actor_class(actor: Any) -> str:
    try:
        return str(actor.get_class().get_name())
    except Exception:
        return type(actor).__name__


def _vec(v: Any) -> list[float]:
    return [float(v.x), float(v.y), float(v.z)]


def _is_wheel_actor(actor: Any) -> bool:
    cls = _actor_class(actor).lower()
    label = _actor_label(actor).lower()
    name = str(actor.get_name()).lower()
    return cls == "wheels_c" or label.startswith("wheels") or name.startswith("wheels")


def _is_capture_actor(actor: Any) -> bool:
    cls = _actor_class(actor).lower()
    label = _actor_label(actor).lower()
    return cls == "cameracapturewheels_c" or "cameracapturewheels" in label


def _get_all_level_actors() -> list[Any]:
    try:
        actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        return list(actor_subsystem.get_all_level_actors())
    except Exception:
        return list(unreal.EditorLevelLibrary.get_all_level_actors())


def _destroy_actor(actor: Any) -> bool:
    try:
        actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        return bool(actor_subsystem.destroy_actor(actor))
    except Exception:
        return bool(unreal.EditorLevelLibrary.destroy_actor(actor))


def _parse_ids(raw: str) -> set[int]:
    out = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def _write_markdown(report: dict[str, Any], json_path: Path) -> None:
    md_path = json_path.with_suffix(".md")
    lines = [
        "# Clean Unreal Capture Map",
        "",
        f"- Status: **{report['status']}**",
        f"- Source map: `{report['source_map']}`",
        f"- Clean map: `{report['clean_map']}`",
        f"- Removed export ids: `{report['remove_export_ids']}`",
        "",
        "## Deleted Wheel Actors",
        "",
        "| Export object id | Actor | Class | Location |",
        "| --- | --- | --- | --- |",
    ]
    for item in report.get("deleted_wheel_actors", []):
        lines.append(
            f"| {item['export_object_id']} | `{item['actor_label']}` | "
            f"`{item['actor_class']}` | `{item['location']}` |"
        )
    lines += [
        "",
        "## Kept Wheel Actors",
        "",
        "| Export object id | Actor | Class | Location |",
        "| --- | --- | --- | --- |",
    ]
    for item in report.get("kept_wheel_actors", []):
        lines.append(
            f"| {item['export_object_id']} | `{item['actor_label']}` | "
            f"`{item['actor_class']}` | `{item['location']}` |"
        )
    if report.get("errors"):
        lines += ["", "## Errors", ""]
        lines.extend(f"- {err}" for err in report["errors"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    source_map = os.environ.get("VSBL_UNREAL_SOURCE_MAP", DEFAULT_SOURCE_MAP)
    clean_map = os.environ.get("VSBL_UNREAL_CLEAN_MAP", DEFAULT_CLEAN_MAP)
    remove_ids = _parse_ids(os.environ.get("VSBL_UNREAL_REMOVE_EXPORT_IDS", DEFAULT_REMOVE_IDS))
    out_path = Path(os.environ.get("VSBL_UNREAL_CLEAN_OUT", DEFAULT_OUT))

    report: dict[str, Any] = {
        "source_map": source_map,
        "clean_map": clean_map,
        "remove_export_ids": sorted(remove_ids),
        "deleted_wheel_actors": [],
        "kept_wheel_actors": [],
        "camera_capture_actors": [],
        "errors": [],
        "status": "FAIL",
    }

    try:
        if not unreal.EditorAssetLibrary.does_asset_exist(source_map):
            raise RuntimeError(f"source map does not exist: {source_map}")

        if unreal.EditorAssetLibrary.does_asset_exist(clean_map):
            unreal.EditorAssetLibrary.delete_asset(clean_map)

        if not unreal.EditorAssetLibrary.duplicate_asset(source_map, clean_map):
            raise RuntimeError(f"failed to duplicate {source_map} -> {clean_map}")

        unreal.EditorLoadingAndSavingUtils.load_map(clean_map)
        actors = _get_all_level_actors()
        wheel_actors = [actor for actor in actors if _is_wheel_actor(actor)]

        for actor in actors:
            if _is_capture_actor(actor):
                report["camera_capture_actors"].append(
                    {
                        "actor_label": _actor_label(actor),
                        "actor_name": str(actor.get_name()),
                        "actor_class": _actor_class(actor),
                        "location": _vec(actor.get_actor_location()),
                    }
                )

        for idx, actor in enumerate(wheel_actors):
            payload = {
                "export_object_id": str(idx),
                "actor_label": _actor_label(actor),
                "actor_name": str(actor.get_name()),
                "actor_class": _actor_class(actor),
                "actor_path": str(actor.get_path_name()),
                "location": _vec(actor.get_actor_location()),
            }
            if idx in remove_ids:
                ok = _destroy_actor(actor)
                payload["destroyed"] = ok
                report["deleted_wheel_actors"].append(payload)
                if not ok:
                    report["errors"].append(f"failed_to_destroy_export_id:{idx}")
            else:
                report["kept_wheel_actors"].append(payload)

        unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True)
        if not report["camera_capture_actors"]:
            report["errors"].append("missing_camera_capture_actor")
        report["status"] = "PASS" if not report["errors"] else "FAIL"
    except Exception as exc:
        report["errors"].append(str(exc))
        report["status"] = "FAIL"
        raise
    finally:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        _write_markdown(report, out_path)
        print(f"WROTE {out_path}")
        print(f"WROTE {out_path.with_suffix('.md')}")


main()
