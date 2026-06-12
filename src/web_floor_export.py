"""ONNX export helpers for the web floor multi-task model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from models.web_multitask import WebMultiTaskModel
from web_floor_dataset import WebFloorDataset
from web_floor_postprocess import decode_web_floor_payload, wheels_from_target

OUTPUT_NAMES = ("cls", "bbox", "kpt", "vis", "floor")
RUNTIME_SCOPE = "single_forward_no_depth_no_ransac"


class WebFloorOnnxWrapper(nn.Module):
    """Tuple-output wrapper so exported ONNX names stay stable."""

    def __init__(self, model: WebMultiTaskModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out = self.model(image)
        return out["cls"], out["bbox"], out["kpt"], out["vis"], out["floor"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_web_floor_model(checkpoint: Path | None, device: torch.device) -> WebMultiTaskModel:
    model = WebMultiTaskModel(pretrained=False).to(device)
    if checkpoint is not None and checkpoint.is_file():
        data = torch.load(checkpoint, map_location=device)
        state = data.get("model_state_dict", data)
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


def export_web_floor_onnx(
    *,
    checkpoint: str | Path | None,
    onnx_path: str | Path,
    imgsz: int = 512,
    opset: int = 17,
    device: str = "cpu",
) -> Path:
    torch_device = torch.device(device)
    ckpt = Path(checkpoint) if checkpoint is not None else None
    model = load_web_floor_model(ckpt, torch_device)
    wrapper = WebFloorOnnxWrapper(model).to(torch_device)
    wrapper.eval()
    out = Path(onnx_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, imgsz, imgsz, dtype=torch.float32, device=torch_device)
    torch.onnx.export(
        wrapper,
        dummy,
        str(out),
        input_names=["image"],
        output_names=list(OUTPUT_NAMES),
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    if not out.is_file():
        raise FileNotFoundError(f"ONNX export did not create file: {out}")
    return out


def onnxruntime_shape_smoke(onnx_path: str | Path, imgsz: int = 512) -> dict[str, Any]:
    import onnxruntime as ort

    path = Path(onnx_path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    image = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
    outputs = session.run(None, {"image": image})
    output_report: dict[str, Any] = {}
    finite = True
    for name, value in zip(OUTPUT_NAMES, outputs):
        arr = np.asarray(value)
        finite = finite and bool(np.isfinite(arr).all())
        output_report[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    return {
        "ok": finite,
        "provider": "CPUExecutionProvider",
        "input_name": session.get_inputs()[0].name,
        "input_shape": [1, 3, imgsz, imgsz],
        "outputs": output_report,
    }


def decoded_sample_from_onnx(
    *,
    onnx_path: str | Path,
    config: str | Path,
    sample_index: int = 0,
    imgsz: int = 512,
) -> dict[str, Any]:
    import onnxruntime as ort

    dataset = WebFloorDataset(config)
    image, target = dataset[sample_index]
    image = torch.nn.functional.interpolate(
        image.unsqueeze(0), size=(imgsz, imgsz), mode="bilinear", align_corners=False
    ).numpy().astype(np.float32)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"image": image})
    floor = np.asarray(outputs[list(OUTPUT_NAMES).index("floor")])[0].tolist()
    return decode_web_floor_payload(
        frame_id=target["frame_id"],
        floor_values=[float(v) for v in floor],
        wheels=wheels_from_target(target),
        distance_mode=target["floor_meta"]["distance_mode"],
        fov_mode=target["floor_meta"]["fov_mode"],
    )


def write_handoff_manifest(
    *,
    manifest_path: str | Path,
    onnx_path: str | Path,
    smoke: dict[str, Any],
    distance_mode: str = "scale_relative",
) -> dict[str, Any]:
    model_path = Path(onnx_path)
    manifest = {
        "schema": "web_floor_handoff_v1",
        "model_file": model_path.name,
        "model_sha256": sha256_file(model_path),
        "input_name": "image",
        "input_shape": smoke["input_shape"],
        "output_names": list(OUTPUT_NAMES),
        "output_shapes": {name: info["shape"] for name, info in smoke["outputs"].items()},
        "distance_mode": distance_mode,
        "runtime_scope": RUNTIME_SCOPE,
        "runtime_requires": {
            "depth": False,
            "segmentation": False,
            "ransac": False,
            "heavy_backend_postprocess": False,
        },
        "production_ready": False,
        "caveat": "Fixture handoff only; needs real web/phone holdout with floor angle/distance labels before production.",
    }
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
