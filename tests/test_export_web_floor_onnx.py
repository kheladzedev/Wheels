from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from models.web_multitask import WebMultiTaskModel
from scripts.export_web_floor_onnx import parse_args, main as export_main
from web_floor_export import OUTPUT_NAMES, WebFloorOnnxWrapper, onnxruntime_shape_smoke


ROOT_CONFIG = Path("configs/pose_dataset_web_floor_fixture.yaml")


def test_export_cli_defaults() -> None:
    args = parse_args([])

    assert args.checkpoint == Path("outputs/web_floor_network/train_fixture/web_floor_fixture_checkpoint.pt")
    assert args.config == ROOT_CONFIG
    assert args.imgsz == 512
    assert args.out_dir == Path("outputs/web_floor_network/handoff")


def test_wrapper_returns_stable_tuple_outputs() -> None:
    model = WebFloorOnnxWrapper(WebMultiTaskModel(pretrained=False))
    with torch.no_grad():
        outputs = model(torch.zeros(1, 3, 64, 64))

    assert len(outputs) == len(OUTPUT_NAMES)
    assert outputs[-1].shape == (1, 3)


def test_export_main_writes_manifest_smoke_and_decoded_sample(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    model = WebMultiTaskModel(pretrained=False)
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    out_dir = tmp_path / "handoff"

    rc = export_main([
        "--checkpoint",
        str(checkpoint),
        "--config",
        str(ROOT_CONFIG),
        "--out-dir",
        str(out_dir),
        "--imgsz",
        "64",
        "--device",
        "cpu",
    ])

    assert rc == 0
    onnx_path = out_dir / "web_floor_multitask.onnx"
    manifest_path = out_dir / "manifest.json"
    smoke_path = out_dir / "python_onnx_smoke.json"
    sample_path = out_dir / "sample_decoded.json"
    assert onnx_path.is_file()
    assert manifest_path.is_file()
    assert smoke_path.is_file()
    assert sample_path.is_file()

    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    assert smoke["ok"] is True
    assert smoke["input_name"] == "image"
    assert set(smoke["outputs"]) == set(OUTPUT_NAMES)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_sha256"]
    assert manifest["input_shape"] == [1, 3, 64, 64]
    assert manifest["output_names"] == list(OUTPUT_NAMES)
    assert manifest["distance_mode"] in {"scale_relative", "normalized"}
    assert manifest["runtime_scope"] == "single_forward_no_depth_no_ransac"
    assert manifest["runtime_requires"] == {
        "depth": False,
        "segmentation": False,
        "ransac": False,
        "heavy_backend_postprocess": False,
    }
    assert manifest["production_ready"] is False

    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    assert set(sample["floor"]) == {"pitch", "roll", "distance", "distance_mode", "fov_mode"}
    assert sample["wheels"]


def test_onnxruntime_shape_smoke_reports_named_outputs(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    model = WebMultiTaskModel(pretrained=False)
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    out_dir = tmp_path / "handoff"
    export_main([
        "--checkpoint", str(checkpoint),
        "--out-dir", str(out_dir),
        "--imgsz", "64",
        "--device", "cpu",
    ])

    smoke = onnxruntime_shape_smoke(out_dir / "web_floor_multitask.onnx", imgsz=64)
    assert smoke["ok"] is True
    assert smoke["input_name"] == "image"
    assert smoke["outputs"]["floor"]["shape"] == [1, 3]
