"""Evaluate web floor fixture readiness without overclaiming production quality."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from models.web_multitask import WebMultiTaskModel
from web_floor_dataset import WebFloorDataset
from web_floor_postprocess import decode_web_floor_payload, wheels_from_target

RUNTIME_SCOPE = "single_forward_no_depth_no_ransac"
NOT_REQUIRED = "not_required_for_runtime"


def _load_model(checkpoint: Path | None, device: torch.device) -> WebMultiTaskModel:
    model = WebMultiTaskModel(pretrained=False).to(device)
    if checkpoint is not None and checkpoint.is_file():
        data = torch.load(checkpoint, map_location=device)
        state = data.get("model_state_dict", data)
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _mae(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def evaluate_web_floor_fixture(
    *,
    config: str | Path = "configs/pose_dataset_web_floor_fixture.yaml",
    checkpoint: str | Path | None = "outputs/web_floor_network/train_fixture/web_floor_fixture_checkpoint.pt",
    output_json: str | Path = "outputs/web_floor_network/eval_fixture/web_floor_eval.json",
    device: str = "cpu",
) -> dict[str, Any]:
    dataset = WebFloorDataset(config)
    torch_device = torch.device(device)
    ckpt_path = Path(checkpoint) if checkpoint is not None else None
    model = _load_model(ckpt_path, torch_device)

    pitch_errors: list[float] = []
    roll_errors: list[float] = []
    distance_errors: list[float] = []
    frames: list[dict[str, Any]] = []
    finite_outputs = True
    total_wheels = 0

    with torch.no_grad():
        for index in range(len(dataset)):
            image, target = dataset[index]
            outputs = model(image.unsqueeze(0).to(torch_device))
            floor_pred = outputs["floor"][0].detach().cpu()
            finite_outputs = finite_outputs and bool(torch.isfinite(floor_pred).all())
            floor_gt = target["floor"].detach().cpu()
            distance_mode = target["floor_meta"]["distance_mode"]
            fov_mode = target["floor_meta"]["fov_mode"]
            wheels = wheels_from_target(target)
            decoded = decode_web_floor_payload(
                frame_id=target["frame_id"],
                floor_values=[float(v) for v in floor_pred],
                wheels=wheels,
                distance_mode=distance_mode,
                fov_mode=fov_mode,
            )
            errors = (floor_pred - floor_gt).abs().tolist()
            pitch_errors.append(float(errors[0]))
            roll_errors.append(float(errors[1]))
            distance_errors.append(float(errors[2]))
            total_wheels += len(wheels)
            frames.append(
                {
                    "index": index,
                    "frame_id": target["frame_id"],
                    "wheel_count": len(wheels),
                    "floor_gt": [float(v) for v in floor_gt],
                    "floor_pred": decoded["floor"],
                    "floor_abs_error": {
                        "pitch": float(errors[0]),
                        "roll": float(errors[1]),
                        "distance": float(errors[2]),
                    },
                    "decoded_contract_valid": True,
                }
            )

    production_blockers = [
        "wheel metrics currently use fixture proxy targets, not a production pose decoder report",
        "no accepted production thresholds configured for web floor holdout",
    ]
    if dataset.fixture_only:
        production_blockers.insert(0, "fixture data only")
        production_blockers.insert(1, "no real web/phone holdout with floor angle/distance labels")

    report = {
        "schema": "web_floor_eval_v1",
        "config": str(config),
        "checkpoint": str(ckpt_path) if ckpt_path is not None else None,
        "runtime_scope": RUNTIME_SCOPE,
        "dataset_items": len(dataset),
        "fixture_only": dataset.fixture_only,
        "finite_outputs": finite_outputs,
        "wheel_metrics": {
            "total_wheels": total_wheels,
            "frames_with_empty_wheels": sum(1 for frame in frames if frame["wheel_count"] == 0),
            "bbox_keypoint_metric_source": "fixture_proxy_targets_until_web_pose_decoder_lands",
            "proxy_bbox_mae_px": 0.0,
            "proxy_keypoint_mae_px": 0.0,
        },
        "floor_metrics": {
            "mae": {
                "pitch": _mae(pitch_errors),
                "roll": _mae(roll_errors),
                "distance": _mae(distance_errors),
            },
            "distance_modes": sorted({frame["floor_pred"]["distance_mode"] for frame in frames}),
        },
        "runtime_requirements": {
            "depth": NOT_REQUIRED,
            "segmentation": NOT_REQUIRED,
            "ransac": NOT_REQUIRED,
            "multi_frame_accumulation": NOT_REQUIRED,
            "heavy_backend_postprocess": NOT_REQUIRED,
        },
        "optional_3d_reconstruction": {
            "status": NOT_REQUIRED,
            "reason": "Web runtime target is direct floor angle/distance prediction; 3D tools are offline validation only.",
        },
        "pipeline_ready": finite_outputs,
        "trained_model_ready": False,
        "production_ready": False,
        "production_blockers": production_blockers,
        "frames": frames,
    }
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
