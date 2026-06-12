#!/usr/bin/env python3
"""Import a raw Unreal/plugin export into the web-floor manifest format."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
from pathlib import Path
import random
import shutil
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import import_unreal_export as ue  # noqa: E402
import inspect_unreal_export as ix  # noqa: E402
from web_floor_contract import validate_web_floor_payload  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SOURCE_TYPE = "synthetic_unreal_plugin_export"


class UnrealWebFloorImportError(ValueError):
    """Raised when the Unreal export cannot become a web-floor manifest."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config-out", type=Path, required=True)
    parser.add_argument(
        "--source-name",
        default=None,
        help="Source slug recorded in manifest provenance. Defaults to unreal_<source-root-name>_web_floor.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("absolute", "copy"),
        default="absolute",
        help="Use absolute source image paths or copy images into dataset-root/images.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--holdout-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--margin", type=int, default=ue.DEFAULT_MARGIN_PX)
    parser.add_argument(
        "--distance-mode",
        choices=("scale_relative", "metric_anchor", "normalized", "unknown"),
        default="scale_relative",
        help="How to label Unreal DeltaZ in the public web contract.",
    )
    parser.add_argument(
        "--right-left-mapping",
        choices=(
            ue.RIGHT_LEFT_MAPPING_AUTO,
            ue.RIGHT_LEFT_MAPPING_CONFIRMED,
            ue.RIGHT_LEFT_MAPPING_SCREEN_SIDES,
        ),
        default=ue.RIGHT_LEFT_MAPPING_AUTO,
    )
    parser.add_argument(
        "--swap-right-left",
        action="store_true",
        help="Alias for --right-left-mapping screen-sides.",
    )
    return parser.parse_args(argv)


def _prepare_dataset_root(dataset_root: Path, overwrite: bool) -> None:
    if dataset_root.exists() and any(dataset_root.iterdir()):
        if not overwrite:
            raise UnrealWebFloorImportError(
                f"{dataset_root} exists and is not empty; pass --overwrite"
            )
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)


def _copy_or_reference_image(
    image_path: Path,
    *,
    dataset_root: Path,
    image_mode: str,
) -> str:
    if image_mode == "absolute":
        return str(image_path.resolve())

    images_dir = dataset_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    out = images_dir / image_path.name
    shutil.copy2(image_path, out)
    return str(out.relative_to(dataset_root))


def _split_map(images: list[Path], holdout_ratio: float, seed: int) -> dict[str, str]:
    if not 0.0 <= holdout_ratio < 1.0:
        raise UnrealWebFloorImportError("--holdout-ratio must be in [0, 1)")
    frame_ids = [image.stem for image in images]
    if len(frame_ids) < 2 or holdout_ratio == 0.0:
        return {frame_id: "train" for frame_id in frame_ids}

    rng = random.Random(seed)
    holdout_ids = list(frame_ids)
    rng.shuffle(holdout_ids)
    holdout_count = max(1, int(round(len(frame_ids) * holdout_ratio)))
    holdout_count = min(holdout_count, len(frame_ids) - 1)
    holdout = set(holdout_ids[:holdout_count])
    return {
        frame_id: "holdout" if frame_id in holdout else "train"
        for frame_id in frame_ids
    }


def _floor_from_ground(gmeta: ix.GroundMeta, distance_mode: str) -> dict[str, Any]:
    return {
        "pitch": math.radians(gmeta.pitch),
        "roll": math.radians(gmeta.roll),
        "distance": float(gmeta.delta_z),
        "distance_mode": distance_mode,
        "fov_mode": "provided",
    }


def _read_ground(ground_path: Path) -> ix.GroundMeta | None:
    if not ground_path.is_file():
        return None
    try:
        return ix.parse_ground_text(ground_path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return None


def import_unreal_web_floor_export(
    *,
    source_root: str | Path,
    dataset_root: str | Path,
    config_out: str | Path,
    source_name: str | None = None,
    image_mode: str = "absolute",
    overwrite: bool = False,
    holdout_ratio: float = 0.15,
    seed: int = 42,
    margin: int = ue.DEFAULT_MARGIN_PX,
    distance_mode: str = "scale_relative",
    right_left_mapping: str = ue.RIGHT_LEFT_MAPPING_AUTO,
    swap_right_left: bool = False,
) -> dict[str, Any]:
    """Convert ``Images/Ground/keyPoint`` Unreal exports to web-floor JSON."""
    src = Path(source_root).expanduser().resolve()
    dataset_root_path = Path(dataset_root).expanduser().resolve()
    config_path = Path(config_out).expanduser().resolve()
    name = source_name or f"unreal_{src.name}_web_floor"

    images_dir = src / "Images"
    ground_dir = src / "Ground"
    kp_root = src / "keyPoint"
    if not images_dir.is_dir():
        raise UnrealWebFloorImportError(f"missing directory: {images_dir}")
    if not ground_dir.is_dir():
        raise UnrealWebFloorImportError(f"missing directory: {ground_dir}")
    if not kp_root.is_dir():
        raise UnrealWebFloorImportError(f"missing directory: {kp_root}")

    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise UnrealWebFloorImportError(f"no images found under {images_dir}")

    _prepare_dataset_root(dataset_root_path, overwrite)
    split_by_frame = _split_map(images, holdout_ratio=holdout_ratio, seed=seed)

    mapping_args = argparse.Namespace(
        source_root=src,
        right_left_mapping=right_left_mapping,
        swap_right_left=swap_right_left,
    )
    ue._resolve_right_left_mapping(src, images, kp_root, mapping_args)

    summary = ue.ImportSummary()
    items: list[dict[str, Any]] = []
    skipped_missing_ground = 0
    skipped_bad_image = 0
    imported_at = _dt.datetime.now(_dt.timezone.utc).isoformat()

    for image_path in images:
        frame_id = image_path.stem
        size = ue._read_image_size(image_path)
        if size is None:
            skipped_bad_image += 1
            continue
        image_w, image_h = size

        ground_path = ground_dir / f"{frame_id}.txt"
        ground = _read_ground(ground_path)
        if ground is None:
            skipped_missing_ground += 1
            continue
        summary.ground_meta_parsed += 1
        summary.ground_meta[frame_id] = {
            "delta_z": ground.delta_z,
            "roll": ground.roll,
            "pitch": ground.pitch,
            "fov": ground.fov,
        }

        wheels: list[dict[str, Any]] = []
        kp_dir = kp_root / frame_id
        if kp_dir.is_dir():
            for kp_file in sorted(kp_dir.iterdir(), key=lambda p: p.name):
                if kp_file.suffix.lower() != ".txt" or not kp_file.is_file():
                    continue
                summary.keypoint_object_files_found += 1
                try:
                    text = kp_file.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    ue._drop(summary, ue.DROP_PARSE_ERROR)
                    continue
                wheel = ue._try_build_wheel(
                    text,
                    image_w,
                    image_h,
                    margin,
                    summary,
                    right_left_mapping=mapping_args.right_left_mapping_resolved,
                    allow_synthetic_bbox=True,
                )
                if wheel is None:
                    continue
                wheel["confidence"] = 1.0
                wheels.append(wheel)
                summary.valid_wheels += 1

        image_ref = _copy_or_reference_image(
            image_path,
            dataset_root=dataset_root_path,
            image_mode=image_mode,
        )
        item = {
            "frame_id": frame_id,
            "split": split_by_frame[frame_id],
            "image": image_ref,
            "provenance": {
                "source": name,
                "source_type": SOURCE_TYPE,
                "source_root": str(src),
                "image_file": str(image_path),
                "ground_file": str(ground_path),
                "keypoint_dir": str(kp_dir),
                "imported_at": imported_at,
                "importer": "scripts/import_unreal_web_floor_export.py",
                "units": {
                    "source_pitch_roll": "degrees",
                    "manifest_pitch_roll": "radians",
                    "source_distance": "Unreal DeltaZ",
                    "manifest_distance_mode": distance_mode,
                },
            },
            "floor": _floor_from_ground(ground, distance_mode),
            "source_floor": {
                "pitch_degrees": ground.pitch,
                "roll_degrees": ground.roll,
                "delta_z": ground.delta_z,
                "fov_degrees": ground.fov,
            },
            "wheels": wheels,
        }
        validate_web_floor_payload(
            {
                "frame_id": item["frame_id"],
                "runtime_scope": "single_forward_no_depth_no_ransac",
                "floor": item["floor"],
                "wheels": item["wheels"],
            },
            require_frame_id=True,
        )
        items.append(item)
        summary.images_imported += 1

    if not items:
        raise UnrealWebFloorImportError("no frames with parseable ground metadata were imported")

    manifest = {
        "schema": "web_floor_manifest_v1",
        "fixture_only": False,
        "source_type": SOURCE_TYPE,
        "source_name": name,
        "source_root": str(src),
        "image_mode": image_mode,
        "created_at": imported_at,
        "right_left_mapping_requested": mapping_args.right_left_mapping_requested,
        "right_left_mapping_resolved": mapping_args.right_left_mapping_resolved,
        "right_left_mapping_basis": mapping_args.right_left_mapping_basis,
        "right_left_mapping_counts": mapping_args.right_left_mapping_counts,
        "floor_units": {
            "pitch": "radians",
            "roll": "radians",
            "distance": "Unreal DeltaZ",
            "distance_mode": distance_mode,
        },
        "import_report": {
            "images_found": len(images),
            "images_imported": summary.images_imported,
            "frames_skipped_missing_ground": skipped_missing_ground,
            "frames_skipped_bad_image": skipped_bad_image,
            "keypoint_object_files_found": summary.keypoint_object_files_found,
            "valid_wheels": summary.valid_wheels,
            "bbox_strategy_counts": {
                "top_points": summary.bbox_from_top_points,
                "floorray": summary.bbox_from_floorray,
            },
            "drop_counts": summary.drop_counts,
            "ground_meta_parsed": summary.ground_meta_parsed,
        },
        "items": items,
    }

    manifest_path = dataset_root_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    config = {
        "path": str(dataset_root_path),
        "manifest": "manifest.json",
        "image_size": [512, 512],
        "fixture_only": False,
        "runtime_scope": "single_forward_no_depth_no_ransac",
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = import_unreal_web_floor_export(
        source_root=args.source_root,
        dataset_root=args.dataset_root,
        config_out=args.config_out,
        source_name=args.source_name,
        image_mode=args.image_mode,
        overwrite=args.overwrite,
        holdout_ratio=args.holdout_ratio,
        seed=args.seed,
        margin=args.margin,
        distance_mode=args.distance_mode,
        right_left_mapping=args.right_left_mapping,
        swap_right_left=args.swap_right_left,
    )
    report = manifest["import_report"]
    print(
        json.dumps(
            {
                "manifest": str(Path(args.dataset_root) / "manifest.json"),
                "config": str(args.config_out),
                "source_type": manifest["source_type"],
                "image_mode": manifest["image_mode"],
                "items": len(manifest["items"]),
                "wheels": report["valid_wheels"],
                "splits": {
                    split: sum(1 for item in manifest["items"] if item.get("split") == split)
                    for split in ("train", "holdout")
                },
                "right_left_mapping": manifest["right_left_mapping_resolved"],
                "bbox_strategy_counts": report["bbox_strategy_counts"],
                "drop_counts": report["drop_counts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
