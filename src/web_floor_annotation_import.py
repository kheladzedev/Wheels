"""Import real web-floor CSV annotations into the training manifest format."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
from typing import Any

import yaml

from web_floor_contract import validate_web_floor_payload


REQUIRED_COLUMNS = (
    "frame_id",
    "split",
    "image",
    "provenance_source",
    "provenance_device",
    "provenance_annotator",
    "pitch",
    "roll",
    "distance",
    "distance_mode",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "confidence",
    "a_x",
    "a_y",
    "b_x",
    "b_y",
    "c_disc_bottom_x",
    "c_disc_bottom_y",
)


class WebFloorAnnotationImportError(ValueError):
    """Raised when CSV annotations cannot be converted into a valid manifest."""


def _required_str(row: dict[str, str], key: str, row_number: int) -> str:
    value = (row.get(key) or "").strip()
    if not value:
        raise WebFloorAnnotationImportError(f"row {row_number}: missing required {key!r}")
    return value


def _float(row: dict[str, str], key: str, row_number: int) -> float:
    value = _required_str(row, key, row_number)
    try:
        return float(value)
    except ValueError as exc:
        raise WebFloorAnnotationImportError(f"row {row_number}: {key!r} must be numeric") from exc


def _optional_str(row: dict[str, str], key: str) -> str | None:
    value = (row.get(key) or "").strip()
    return value or None


def _same_frame_value(frame: dict[str, Any], key: str, value: Any, row_number: int) -> None:
    if frame[key] != value:
        raise WebFloorAnnotationImportError(
            f"row {row_number}: frame {frame['frame_id']!r} has inconsistent {key}"
        )


def _copy_image(*, source: Path, image_name: str, dataset_root: Path) -> str:
    source_path = Path(image_name)
    if not source_path.is_absolute():
        source_path = source / source_path
    if not source_path.is_file():
        raise WebFloorAnnotationImportError(f"image does not exist: {source_path}")
    images_dir = dataset_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    out_path = images_dir / source_path.name
    if source_path.resolve() != out_path.resolve():
        shutil.copy2(source_path, out_path)
    return str(out_path.relative_to(dataset_root))


def import_web_floor_csv_annotations(
    *,
    csv_path: str | Path,
    image_root: str | Path,
    dataset_root: str | Path,
    config_out: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert CSV rows into a web-floor manifest and optional dataset config."""
    csv_file = Path(csv_path)
    image_root_path = Path(image_root)
    dataset_root_path = Path(dataset_root)
    manifest_path = dataset_root_path / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise WebFloorAnnotationImportError(f"{manifest_path} exists; pass overwrite=True")
    dataset_root_path.mkdir(parents=True, exist_ok=True)

    with csv_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = sorted(set(REQUIRED_COLUMNS) - set(fieldnames))
        if missing:
            raise WebFloorAnnotationImportError(f"CSV missing required column(s): {missing}")

        frames: dict[str, dict[str, Any]] = {}
        for row_number, row in enumerate(reader, start=2):
            frame_id = _required_str(row, "frame_id", row_number)
            split = _required_str(row, "split", row_number)
            image_rel = _copy_image(
                source=image_root_path,
                image_name=_required_str(row, "image", row_number),
                dataset_root=dataset_root_path,
            )
            provenance = {
                "source": _required_str(row, "provenance_source", row_number),
                "device": _required_str(row, "provenance_device", row_number),
                "annotator": _required_str(row, "provenance_annotator", row_number),
            }
            capture_date = _optional_str(row, "provenance_capture_date")
            if capture_date is not None:
                provenance["capture_date"] = capture_date
            floor = {
                "pitch": _float(row, "pitch", row_number),
                "roll": _float(row, "roll", row_number),
                "distance": _float(row, "distance", row_number),
                "distance_mode": _required_str(row, "distance_mode", row_number),
                "fov_mode": _optional_str(row, "fov_mode") or "unknown",
            }
            if frame_id not in frames:
                frames[frame_id] = {
                    "frame_id": frame_id,
                    "split": split,
                    "image": image_rel,
                    "provenance": provenance,
                    "floor": floor,
                    "wheels": [],
                }
            else:
                frame = frames[frame_id]
                _same_frame_value(frame, "split", split, row_number)
                _same_frame_value(frame, "image", image_rel, row_number)
                _same_frame_value(frame, "provenance", provenance, row_number)
                _same_frame_value(frame, "floor", floor, row_number)

            wheel = {
                "bbox_xyxy": [
                    _float(row, "bbox_x1", row_number),
                    _float(row, "bbox_y1", row_number),
                    _float(row, "bbox_x2", row_number),
                    _float(row, "bbox_y2", row_number),
                ],
                "confidence": _float(row, "confidence", row_number),
                "points": {
                    "a": [_float(row, "a_x", row_number), _float(row, "a_y", row_number)],
                    "b": [_float(row, "b_x", row_number), _float(row, "b_y", row_number)],
                    "c_disc_bottom": [
                        _float(row, "c_disc_bottom_x", row_number),
                        _float(row, "c_disc_bottom_y", row_number),
                    ],
                },
            }
            frames[frame_id]["wheels"].append(wheel)

    manifest = {
        "schema": "web_floor_manifest_v1",
        "fixture_only": False,
        "items": list(frames.values()),
    }
    for item in manifest["items"]:
        validate_web_floor_payload(
            {
                "frame_id": item["frame_id"],
                "runtime_scope": "single_forward_no_depth_no_ransac",
                "floor": item["floor"],
                "wheels": item["wheels"],
            },
            require_frame_id=True,
        )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if config_out is not None:
        cfg = {
            "path": str(dataset_root_path),
            "manifest": "manifest.json",
            "image_size": [512, 512],
            "fixture_only": False,
            "runtime_scope": "single_forward_no_depth_no_ransac",
        }
        config_path = Path(config_out)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    return manifest
