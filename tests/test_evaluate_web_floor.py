from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from evaluate_web_floor import evaluate_web_floor_fixture
from scripts.eval_web_floor import parse_args
from scripts.create_web_floor_fixture import create_fixture
from web_floor_dataset import WebFloorDatasetError
from web_floor_postprocess import decode_web_floor_payload


ROOT_CONFIG = Path("configs/pose_dataset_web_floor_fixture.yaml")
ROOT_CHECKPOINT = Path("outputs/web_floor_network/train_fixture/web_floor_fixture_checkpoint.pt")


def test_eval_cli_defaults() -> None:
    args = parse_args([])

    assert args.config == ROOT_CONFIG
    assert args.output_json == Path("outputs/web_floor_network/eval_fixture/web_floor_eval.json")
    assert args.device == "cpu"


def test_fixture_eval_writes_honest_readiness_report(tmp_path: Path) -> None:
    out = tmp_path / "web_floor_eval.json"
    report = evaluate_web_floor_fixture(
        config=ROOT_CONFIG,
        checkpoint=ROOT_CHECKPOINT,
        output_json=out,
        device="cpu",
    )

    assert out.is_file()
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written == report
    assert report["pipeline_ready"] is True
    assert report["trained_model_ready"] is False
    assert report["production_ready"] is False
    assert report["fixture_only"] is True
    assert report["wheel_metrics"]["total_wheels"] == 4
    assert report["wheel_metrics"]["frames_with_empty_wheels"] == 1
    assert set(report["floor_metrics"]["mae"]) == {"pitch", "roll", "distance"}
    assert report["finite_outputs"] is True


def test_runtime_scope_marks_depth_and_ransac_not_required(tmp_path: Path) -> None:
    report = evaluate_web_floor_fixture(
        config=ROOT_CONFIG,
        checkpoint=ROOT_CHECKPOINT,
        output_json=tmp_path / "eval.json",
        device="cpu",
    )

    assert report["runtime_requirements"] == {
        "depth": "not_required_for_runtime",
        "segmentation": "not_required_for_runtime",
        "ransac": "not_required_for_runtime",
        "multi_frame_accumulation": "not_required_for_runtime",
        "heavy_backend_postprocess": "not_required_for_runtime",
    }
    assert report["optional_3d_reconstruction"]["status"] == "not_required_for_runtime"


def test_decode_rejects_invalid_model_floor_output() -> None:
    with pytest.raises(Exception, match="finite"):
        decode_web_floor_payload(
            frame_id="bad-floor",
            floor_values=[float("nan"), 0.0, 1.0],
            wheels=[],
        )


def test_decode_accepts_empty_wheel_frame() -> None:
    payload = decode_web_floor_payload(
        frame_id="empty",
        floor_values=[0.0, 0.0, 1.0],
        wheels=[],
        distance_mode="scale_relative",
    )

    assert payload["wheels"] == []
    assert payload["floor"]["distance"] == 1.0


def test_eval_rejects_missing_floor_metadata(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    create_fixture(root, overwrite=True)
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "manifest": "manifest_invalid_missing_floor.json",
                "image_size": [128, 128],
                "fixture_only": True,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(WebFloorDatasetError, match="missing floor"):
        evaluate_web_floor_fixture(
            config=cfg,
            checkpoint=ROOT_CHECKPOINT,
            output_json=tmp_path / "eval.json",
            device="cpu",
        )
