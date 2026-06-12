from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.import_unreal_web_floor_export import (
    SOURCE_TYPE,
    import_unreal_web_floor_export,
    parse_args,
)
from web_floor_dataset import WebFloorDataset
from web_floor_real_data_gate import WebFloorRealDataGateConfig, audit_web_floor_real_data


def _kp_text(right, left, center, left_top, right_top) -> str:
    return (
        "{\n"
        f'{{name:"Right",XY:{right[0]},{right[1]}\n}},\n'
        f'{{name:"Left",XY:{left[0]},{left[1]}\n}},\n'
        f'{{name:"Center",XY:{center[0]},{center[1]}\n}},\n'
        f'{{name:"LeftTop",XY:{left_top[0]},{left_top[1]}\n}},\n'
        f'{{name:"RightTop",XY:{right_top[0]},{right_top[1]}\n}}\n}}'
    )


def _build_export(root: Path) -> None:
    (root / "Images").mkdir(parents=True)
    (root / "Ground").mkdir(parents=True)
    (root / "keyPoint").mkdir(parents=True)
    image = np.full((480, 640, 3), 200, dtype=np.uint8)
    floor_rows = [
        ("0", 100.0, -5.0, 60.0, 54.0),
        ("1", 150.0, 0.0, 66.0, 60.0),
        ("2", 220.0, 4.0, 72.0, 70.0),
    ]
    for frame_id, delta_z, roll, pitch, fov in floor_rows:
        assert cv2.imwrite(str(root / "Images" / f"{frame_id}.jpg"), image)
        (root / "Ground" / f"{frame_id}.txt").write_text(
            f"DeltaZ{{{delta_z}}},Roll:{roll},Pitch:{pitch},FOV:{fov}",
            encoding="utf-8",
        )
        kp_dir = root / "keyPoint" / frame_id
        kp_dir.mkdir()
        (kp_dir / "0.txt").write_text(
            _kp_text(
                right=(300.0, 420.0),
                left=(100.0, 420.0),
                center=(200.0, 330.0),
                left_top=(90.0, 120.0),
                right_top=(310.0, 120.0),
            ),
            encoding="utf-8",
        )


def test_parse_args_defaults_to_absolute_image_mode() -> None:
    args = parse_args(
        [
            "--source-root",
            "0003",
            "--dataset-root",
            "data/web_floor_unreal_0003",
            "--config-out",
            "configs/pose_dataset_web_floor_unreal_0003.yaml",
        ]
    )

    assert args.image_mode == "absolute"
    assert args.distance_mode == "scale_relative"


def test_import_unreal_export_writes_web_floor_manifest_and_blocks_real_gate(tmp_path: Path) -> None:
    source = tmp_path / "0003"
    _build_export(source)
    dataset_root = tmp_path / "web_floor_unreal_0003"
    config = tmp_path / "pose_dataset_web_floor_unreal_0003.yaml"

    manifest = import_unreal_web_floor_export(
        source_root=source,
        dataset_root=dataset_root,
        config_out=config,
        source_name="unreal_0003_web_floor_source",
        image_mode="absolute",
        overwrite=True,
        holdout_ratio=0.34,
    )

    assert manifest["source_type"] == SOURCE_TYPE
    assert manifest["fixture_only"] is False
    assert manifest["right_left_mapping_resolved"] == "screen-sides"
    assert manifest["import_report"]["valid_wheels"] == 3
    assert manifest["import_report"]["bbox_strategy_counts"] == {"top_points": 3, "floorray": 0}
    assert {item["split"] for item in manifest["items"]} == {"train", "holdout"}
    assert all(Path(item["image"]).is_absolute() for item in manifest["items"])
    assert manifest["items"][0]["floor"]["pitch"] == pytest.approx(math.radians(60.0))
    assert manifest["items"][0]["floor"]["roll"] == pytest.approx(math.radians(-5.0))
    assert manifest["items"][0]["floor"]["distance"] == pytest.approx(100.0)
    assert manifest["items"][0]["floor"]["distance_mode"] == "scale_relative"

    dataset = WebFloorDataset(config)
    image, target = dataset[0]
    assert dataset.fixture_only is False
    assert list(image.shape) == [3, 512, 512]
    assert target["boxes"].shape == (1, 4)
    assert target["floor_meta"]["fov_mode"] == "provided"

    report = audit_web_floor_real_data(
        config,
        gate=WebFloorRealDataGateConfig(
            min_frames=3,
            min_wheels=3,
            min_distance_span=100.0,
            min_angle_span_rad=0.1,
        ),
    )
    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert report["production_data_ready"] is False
    assert failed == {"real_source"}
