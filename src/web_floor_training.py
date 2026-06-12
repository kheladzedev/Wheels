"""Training utilities for the lightweight web floor multi-task path.

The fixture dry-run proves data/model/optimizer/checkpoint plumbing. It is not
quality evidence and intentionally keeps reconstruction loss disabled unless a
caller opts into an offline experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from models.web_multitask import MultiTaskLoss, WebMultiTaskModel
from web_floor_dataset import WebFloorDataset

RUNTIME_SCOPE = "single_forward_no_depth_no_ransac"
ALLOWED_STAGES = ("2d", "floor", "joint", "recon")


@dataclass(frozen=True)
class WebFloorTrainConfig:
    config: Path
    stage: str = "floor"
    epochs: int = 1
    batch_size: int = 2
    imgsz: int = 128
    out_dir: Path = Path("outputs/web_floor_network/train_fixture")
    device: str = "cpu"
    lr: float = 1e-4
    seed: int = 123
    pretrained: bool = False
    enable_reconstruction_loss: bool = False


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def collate_web_floor(batch: list[tuple[torch.Tensor, dict[str, Any]]]) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    images = torch.stack([image for image, _target in batch], dim=0)
    targets = [target for _image, target in batch]
    return images, targets


def fixture_pose_proxy_loss(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Small differentiable pose-head proxy used only for fixture dry-runs."""
    cls_term = outputs["cls"].sigmoid().mean()
    bbox_term = outputs["bbox"].abs().mean()
    kpt_term = outputs["kpt"].abs().mean()
    vis_term = outputs["vis"].abs().mean()
    return cls_term + 0.01 * (bbox_term + kpt_term + vis_term)


def _floor_targets(targets: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    return torch.stack([target["floor"] for target in targets], dim=0).to(device)


def _distance_modes(targets: list[dict[str, Any]]) -> list[str]:
    return sorted({target["floor_meta"]["distance_mode"] for target in targets})


def _runtime_scopes(targets: list[dict[str, Any]]) -> list[str]:
    return sorted({target["floor_meta"]["runtime_scope"] for target in targets})


def floor_head_trainable(model: WebMultiTaskModel) -> bool:
    return any(param.requires_grad for param in model.floor_head.parameters())


def _training_image_size(imgsz: int) -> tuple[int, int]:
    if imgsz <= 0:
        raise ValueError("imgsz must be positive")
    return imgsz, imgsz


def run_fixture_training(config: WebFloorTrainConfig) -> dict[str, Any]:
    if config.stage not in ALLOWED_STAGES:
        raise ValueError(f"stage must be one of {ALLOWED_STAGES}, got {config.stage!r}")
    if config.stage == "recon" and not config.enable_reconstruction_loss:
        raise ValueError("stage='recon' requires --enable-reconstruction-loss for offline experiments")
    if config.enable_reconstruction_loss:
        raise ValueError("offline reconstruction loss is not wired in this lightweight runtime trainer yet")

    seed_everything(config.seed)
    device = resolve_device(config.device)
    dataset = WebFloorDataset(config.config)
    dataset.image_size = _training_image_size(config.imgsz)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_web_floor,
    )
    model = WebMultiTaskModel(pretrained=config.pretrained).to(device)
    effective_stage = "joint" if config.stage == "recon" else config.stage
    model.set_stage(effective_stage)
    criterion = MultiTaskLoss().to(device)
    optimizer = torch.optim.Adam(
        [p for p in list(model.parameters()) + list(criterion.parameters()) if p.requires_grad],
        lr=config.lr,
    )

    epoch_summaries: list[dict[str, float]] = []
    all_distance_modes: set[str] = set()
    all_runtime_scopes: set[str] = set()
    for epoch in range(config.epochs):
        losses = {"total": 0.0, "pose": 0.0, "floor": 0.0, "reconstruction": 0.0}
        n_batches = 0
        model.train()
        for images, targets in loader:
            images = images.to(device)
            floor_gt = _floor_targets(targets, device)
            all_distance_modes.update(_distance_modes(targets))
            all_runtime_scopes.update(_runtime_scopes(targets))

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            pose_loss = fixture_pose_proxy_loss(outputs)
            result = criterion(
                pose_loss=pose_loss,
                floor_pred=outputs["floor"],
                gt_floor=floor_gt,
                detach_floor=config.stage == "2d",
            )
            result["total"].backward()
            optimizer.step()

            losses["total"] += float(result["total"].detach().cpu())
            losses["pose"] += float(result["pose"].detach().cpu())
            losses["floor"] += float(result["floor"].detach().cpu())
            n_batches += 1
        epoch_summaries.append({key: value / max(n_batches, 1) for key, value in losses.items()})

    config.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = "web_floor_fixture_checkpoint.pt" if dataset.fixture_only else "web_floor_checkpoint.pt"
    checkpoint_path = config.out_dir / checkpoint_name
    metrics_path = config.out_dir / "metrics.json"
    snapshot_path = config.out_dir / "config_snapshot.yaml"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "stage": config.stage,
            "runtime_scope": RUNTIME_SCOPE,
            "fixture_only": dataset.fixture_only,
        },
        checkpoint_path,
    )
    yaml.safe_dump(
        {**asdict(config), "config": str(config.config), "out_dir": str(config.out_dir)},
        snapshot_path.open("w", encoding="utf-8"),
        sort_keys=False,
    )
    metrics = {
        "stage": config.stage,
        "epochs": config.epochs,
        "dataset_items": len(dataset),
        "batch_size": config.batch_size,
        "image_size": list(dataset.image_size),
        "device": str(device),
        "distance_modes": sorted(all_distance_modes),
        "runtime_scope": RUNTIME_SCOPE,
        "runtime_scopes_seen": sorted(all_runtime_scopes),
        "fixture_only": dataset.fixture_only,
        "production_data_seen": not dataset.fixture_only,
        "trained_model_ready": False,
        "production_ready": False,
        "reconstruction_loss_enabled": config.enable_reconstruction_loss,
        "floor_head_trainable": floor_head_trainable(model),
        "losses": epoch_summaries,
        "checkpoint": str(checkpoint_path),
        "config_snapshot": str(snapshot_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
