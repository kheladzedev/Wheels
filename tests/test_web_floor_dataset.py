from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.create_web_floor_fixture import create_fixture
from web_floor_dataset import WebFloorDataset, WebFloorDatasetError, summarize_sample


ROOT_CONFIG = Path("configs/pose_dataset_web_floor_fixture.yaml")


def _write_config(
    tmp_path: Path,
    root: Path,
    manifest: str = "manifest.json",
    fixture_only: bool | None = True,
) -> Path:
    data = {
        "path": str(root),
        "manifest": manifest,
        "image_size": [128, 128],
        "runtime_scope": "single_forward_no_depth_no_ransac",
    }
    if fixture_only is not None:
        data["fixture_only"] = fixture_only
    cfg = tmp_path / "web_floor_fixture.yaml"
    cfg.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )
    return cfg


def test_fixture_config_loads_checked_in_dataset() -> None:
    dataset = WebFloorDataset(ROOT_CONFIG)

    assert len(dataset) == 4
    image, target = dataset[0]
    assert list(image.shape) == [3, 128, 128]
    assert list(target["boxes"].shape) == [1, 4]
    assert list(target["keypoints"].shape) == [1, 3, 2]
    assert list(target["visibility"].shape) == [1, 3]
    assert list(target["floor"].shape) == [3]
    assert target["floor_meta"]["distance_mode"] == "scale_relative"
    assert target["floor_meta"]["fixture_only"] is True


def test_empty_wheel_frame_shapes_are_stable() -> None:
    dataset = WebFloorDataset(ROOT_CONFIG)
    image, target = dataset[2]

    assert target["frame_id"] == "fixture-empty-0003"
    assert list(image.shape) == [3, 128, 128]
    assert list(target["boxes"].shape) == [0, 4]
    assert list(target["keypoints"].shape) == [0, 3, 2]
    assert list(target["visibility"].shape) == [0, 3]


def test_summarize_sample_reports_shapes() -> None:
    dataset = WebFloorDataset(ROOT_CONFIG)

    summary = summarize_sample(dataset, 1)

    assert summary["frame_id"] == "fixture-multi-wheel-0002"
    assert summary["image_shape"] == [3, 128, 128]
    assert summary["boxes_shape"] == [2, 4]
    assert summary["keypoints_shape"] == [2, 3, 2]
    assert summary["fixture_only"] is True


def test_fixture_generator_creates_deterministic_manifest(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    manifest = create_fixture(root, overwrite=True)

    assert len(manifest["items"]) == 4
    assert (root / "manifest.json").is_file()
    assert (root / "manifest_invalid_missing_floor.json").is_file()
    assert sorted(p.name for p in (root / "images").glob("*.png")) == [
        "empty_no_wheel.png",
        "multi_wheel.png",
        "normalized_distance.png",
        "wheel_floor.png",
    ]
    written = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert written["fixture_only"] is True
    assert "not production" in (root / "README.md").read_text(encoding="utf-8")


def test_missing_floor_metadata_fails_with_clear_error(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    create_fixture(root, overwrite=True)
    cfg = _write_config(tmp_path, root, "manifest_invalid_missing_floor.json")

    with pytest.raises(WebFloorDatasetError, match="missing floor"):
        WebFloorDataset(cfg)


def test_invalid_distance_mode_fails_with_clear_error(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    create_fixture(root, overwrite=True)
    cfg = _write_config(tmp_path, root, "manifest_invalid_distance_mode.json")

    with pytest.raises(WebFloorDatasetError, match="distance_mode"):
        WebFloorDataset(cfg)


def test_non_finite_floor_values_fail_with_clear_error(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    create_fixture(root, overwrite=True)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["items"][0]["floor"]["pitch"] = float("nan")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cfg = _write_config(tmp_path, root)

    with pytest.raises(WebFloorDatasetError, match="finite"):
        WebFloorDataset(cfg)


def test_non_fixture_manifest_loads_when_config_matches(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    create_fixture(root, overwrite=True)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fixture_only"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cfg = _write_config(tmp_path, root, fixture_only=False)

    dataset = WebFloorDataset(cfg)

    assert dataset.fixture_only is False
    _image, target = dataset[0]
    assert target["floor_meta"]["fixture_only"] is False


def test_config_manifest_fixture_mismatch_fails(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    create_fixture(root, overwrite=True)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fixture_only"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cfg = _write_config(tmp_path, root, fixture_only=True)

    with pytest.raises(WebFloorDatasetError, match="disagrees"):
        WebFloorDataset(cfg)


def test_manifest_must_declare_fixture_flag(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    create_fixture(root, overwrite=True)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["fixture_only"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cfg = _write_config(tmp_path, root, fixture_only=None)

    with pytest.raises(WebFloorDatasetError, match="fixture_only"):
        WebFloorDataset(cfg)
