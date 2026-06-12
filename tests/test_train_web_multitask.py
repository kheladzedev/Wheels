from __future__ import annotations

import json
from pathlib import Path

import pytest

from models.web_multitask import WebMultiTaskModel
from scripts.train_web_multitask import parse_args
from web_floor_training import WebFloorTrainConfig, floor_head_trainable, run_fixture_training


ROOT_CONFIG = Path("configs/pose_dataset_web_floor_fixture.yaml")


def test_parse_args_exposes_fixture_friendly_defaults() -> None:
    args = parse_args([])

    assert args.config == ROOT_CONFIG
    assert args.stage == "floor"
    assert args.epochs == 1
    assert args.batch_size == 2
    assert args.imgsz == 128
    assert args.device == "cpu"


def test_floor_head_trainability_matches_stage() -> None:
    model = WebMultiTaskModel(pretrained=False)

    model.set_stage("2d")
    assert floor_head_trainable(model) is False

    model.set_stage("floor")
    assert floor_head_trainable(model) is True

    model.set_stage("joint")
    assert floor_head_trainable(model) is True


def test_fixture_dry_run_writes_checkpoint_metrics_and_snapshot(tmp_path: Path) -> None:
    out_dir = tmp_path / "train_fixture"
    metrics = run_fixture_training(
        WebFloorTrainConfig(
            config=ROOT_CONFIG,
            stage="floor",
            epochs=1,
            batch_size=2,
            imgsz=128,
            out_dir=out_dir,
            device="cpu",
            seed=7,
        )
    )

    checkpoint = out_dir / "web_floor_fixture_checkpoint.pt"
    metrics_path = out_dir / "metrics.json"
    snapshot = out_dir / "config_snapshot.yaml"
    assert checkpoint.is_file()
    assert metrics_path.is_file()
    assert snapshot.is_file()
    written = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert written["stage"] == "floor"
    assert written["dataset_items"] == 4
    assert written["image_size"] == [128, 128]
    assert written["fixture_only"] is True
    assert written["production_ready"] is False
    assert written["trained_model_ready"] is False
    assert written["distance_modes"] == ["normalized", "scale_relative"]
    assert written["runtime_scope"] == "single_forward_no_depth_no_ransac"
    assert written["losses"][0]["pose"] > 0
    assert written["losses"][0]["floor"] >= 0
    assert metrics == written


def test_training_imgsz_overrides_dataset_config_resize(tmp_path: Path) -> None:
    out_dir = tmp_path / "train_fixture_96"
    metrics = run_fixture_training(
        WebFloorTrainConfig(
            config=ROOT_CONFIG,
            stage="floor",
            epochs=1,
            batch_size=2,
            imgsz=96,
            out_dir=out_dir,
            device="cpu",
        )
    )

    assert metrics["image_size"] == [96, 96]


def test_2d_stage_freezes_floor_head_in_metrics(tmp_path: Path) -> None:
    out_dir = tmp_path / "train_fixture_2d"
    metrics = run_fixture_training(
        WebFloorTrainConfig(
            config=ROOT_CONFIG,
            stage="2d",
            epochs=1,
            batch_size=2,
            imgsz=128,
            out_dir=out_dir,
            device="cpu",
        )
    )

    assert metrics["stage"] == "2d"
    assert metrics["floor_head_trainable"] is False
    assert metrics["production_ready"] is False


def test_recon_stage_requires_explicit_offline_flag(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires --enable-reconstruction-loss"):
        run_fixture_training(
            WebFloorTrainConfig(
                config=ROOT_CONFIG,
                stage="recon",
                out_dir=tmp_path / "recon",
                device="cpu",
            )
        )
