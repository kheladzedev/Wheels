from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_web_floor_real_data import parse_args
from scripts.create_web_floor_fixture import create_fixture
from web_floor_real_data_gate import WebFloorRealDataGateConfig, audit_web_floor_real_data


ROOT_CONFIG = Path("configs/pose_dataset_web_floor_fixture.yaml")


def _write_config(tmp_path: Path, root: Path, *, fixture_only: bool) -> Path:
    config = tmp_path / "web_floor_real.yaml"
    config.write_text(
        "\n".join(
            [
                f"path: {root}",
                "manifest: manifest.json",
                "image_size: [128, 128]",
                f"fixture_only: {str(fixture_only).lower()}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config


def _make_real_like_manifest(root: Path) -> None:
    create_fixture(root, overwrite=True)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fixture_only"] = False
    for index, item in enumerate(manifest["items"]):
        item["split"] = "holdout" if index == len(manifest["items"]) - 1 else "train"
        item["provenance"] = {
            "source": "test_phone_capture",
            "device": "unit-test-camera",
            "annotator": "unit-test",
        }
        item["floor"]["distance_mode"] = "scale_relative"
        item["floor"]["distance"] = 0.2 + index * 0.4
        item["floor"]["pitch"] = -0.04 + index * 0.04
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_parse_args_defaults_to_production_thresholds() -> None:
    args = parse_args([])

    assert args.config == ROOT_CONFIG
    assert args.min_frames == 50
    assert args.min_wheels == 80
    assert args.required_splits is None
    assert args.fail_on_not_ready is False


def test_fixture_manifest_fails_production_data_gate(tmp_path: Path) -> None:
    out = tmp_path / "gate.json"
    report = audit_web_floor_real_data(
        ROOT_CONFIG,
        gate=WebFloorRealDataGateConfig(min_frames=1, min_wheels=1, min_distance_span=0.0, min_angle_span_rad=0.0),
        output_json=out,
    )

    assert out.is_file()
    assert report["production_data_ready"] is False
    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert "not_fixture" in failed
    assert "required_splits" in failed
    assert "provenance_present" in failed


def test_synthetic_source_fails_real_source_gate(tmp_path: Path) -> None:
    root = tmp_path / "synthetic_like"
    _make_real_like_manifest(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_type"] = "synthetic_unreal_plugin_export"
    for item in manifest["items"]:
        item["provenance"]["source_type"] = "synthetic_unreal_plugin_export"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    config = _write_config(tmp_path, root, fixture_only=False)

    report = audit_web_floor_real_data(
        config,
        gate=WebFloorRealDataGateConfig(
            min_frames=4,
            min_wheels=4,
            min_distance_span=0.5,
            min_angle_span_rad=0.05,
        ),
    )

    assert report["production_data_ready"] is False
    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert failed == {"real_source"}


def test_real_like_manifest_passes_data_gate_with_test_thresholds(tmp_path: Path) -> None:
    root = tmp_path / "real_like"
    _make_real_like_manifest(root)
    config = _write_config(tmp_path, root, fixture_only=False)

    report = audit_web_floor_real_data(
        config,
        gate=WebFloorRealDataGateConfig(
            min_frames=4,
            min_wheels=4,
            min_distance_span=0.5,
            min_angle_span_rad=0.05,
        ),
    )

    assert report["production_data_ready"] is True
    assert report["fixture_only"] is False
    assert report["dataset_items"] == 4
    assert report["total_wheels"] == 4
    assert report["split_counts"] == {"holdout": 1, "train": 3}


def test_missing_real_dataset_reports_load_failure(tmp_path: Path) -> None:
    config = _write_config(tmp_path, tmp_path / "missing", fixture_only=False)

    report = audit_web_floor_real_data(config)

    assert report["production_data_ready"] is False
    assert report["dataset_loaded"] is False
    assert report["checks"][0]["name"] == "dataset_loads"
