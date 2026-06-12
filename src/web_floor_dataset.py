"""Dataset utilities for the web floor angles + distance training path.

This loader reads a small JSON-manifest contract that couples RGB frames,
wheel A/B/C labels, and direct floor metadata. The checked-in fixture is for
pipeline tests only; it is not production training data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import torch
from torch.utils.data import Dataset
import yaml

from web_floor_contract import POINT_KEYS, validate_web_floor_payload


class WebFloorDatasetError(ValueError):
    """Raised when a web-floor dataset config or item is malformed."""


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise WebFloorDatasetError(f"cannot read config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise WebFloorDatasetError(f"config {path} must be a YAML object")
    return data


def load_web_floor_config(config_path: str | Path) -> dict[str, Any]:
    """Load and normalize a web-floor dataset YAML config."""
    path = Path(config_path)
    cfg = _read_yaml(path)
    root_value = cfg.get("path")
    if not root_value:
        raise WebFloorDatasetError(f"{path}: missing required 'path'")
    root = Path(root_value)
    if not root.is_absolute():
        root = (path.parent.parent / root).resolve()
    manifest_name = cfg.get("manifest", "manifest.json")
    manifest_path = Path(manifest_name)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    cfg["path"] = str(root)
    cfg["manifest_path"] = str(manifest_path)
    cfg.setdefault("image_size", [512, 512])
    if "fixture_only" in cfg and not isinstance(cfg["fixture_only"], bool):
        raise WebFloorDatasetError(f"{path}: fixture_only must be a boolean when present")
    cfg.setdefault("runtime_scope", "single_forward_no_depth_no_ransac")
    return cfg


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise WebFloorDatasetError(f"cannot read manifest {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise WebFloorDatasetError(f"manifest {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise WebFloorDatasetError(f"manifest {path} must be a JSON object")
    if "fixture_only" not in data:
        raise WebFloorDatasetError(
            f"manifest {path} must declare fixture_only as a boolean"
        )
    if not isinstance(data["fixture_only"], bool):
        raise WebFloorDatasetError(f"manifest {path}: fixture_only must be a boolean")
    items = data.get("items")
    if not isinstance(items, list):
        raise WebFloorDatasetError(f"manifest {path} missing items[]")
    return data


def _image_to_tensor(image_path: Path, image_size: tuple[int, int] | None) -> torch.Tensor:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise WebFloorDatasetError(f"cannot read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image_size is not None:
        width, height = image_size
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
    return tensor


def _as_image_size(value: Any) -> tuple[int, int] | None:
    if value in (None, "original"):
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise WebFloorDatasetError("image_size must be [width, height] or 'original'")
    width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        raise WebFloorDatasetError("image_size values must be positive")
    return width, height


class WebFloorDataset(Dataset):
    """Dataset returning image tensors and web floor/wheel targets.

    The format is intentionally explicit rather than reusing YOLO labels because
    floor angles/distance need metadata and provenance. Use this fixture to
    prove pipeline plumbing only; do not report its metrics as production model
    quality.
    """

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.config = load_web_floor_config(self.config_path)
        self.root = Path(self.config["path"])
        self.manifest_path = Path(self.config["manifest_path"])
        self.manifest = _read_manifest(self.manifest_path)
        config_fixture_only = self.config.get("fixture_only")
        self.fixture_only = bool(self.manifest["fixture_only"])
        if config_fixture_only is not None and bool(config_fixture_only) != self.fixture_only:
            raise WebFloorDatasetError(
                f"{self.config_path}: fixture_only={config_fixture_only!r} disagrees "
                f"with manifest fixture_only={self.fixture_only!r}"
            )
        self.config["fixture_only"] = self.fixture_only
        self.image_size = _as_image_size(self.config.get("image_size"))
        self.items = list(self.manifest["items"])
        self.samples = [self._normalize_item(item, index) for index, item in enumerate(self.items)]

    def _normalize_item(self, item: Any, index: int) -> dict[str, Any]:
        if not isinstance(item, dict):
            raise WebFloorDatasetError(f"items[{index}] must be an object")
        image_rel = item.get("image")
        if not isinstance(image_rel, str) or not image_rel:
            raise WebFloorDatasetError(f"items[{index}] missing image path")
        image_path = (self.root / image_rel).resolve()
        payload = {
            "frame_id": item.get("frame_id", f"web_floor_{index:04d}"),
            "runtime_scope": item.get("runtime_scope", self.config["runtime_scope"]),
            "wheels": item.get("wheels", []),
        }
        if "floor" in item:
            payload["floor"] = item["floor"]
        try:
            decoded = validate_web_floor_payload(payload, require_frame_id=True)
        except Exception as exc:
            raise WebFloorDatasetError(f"items[{index}] invalid web-floor target: {exc}") from exc
        return {"image_path": image_path, "decoded": decoded}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        sample = self.samples[index]
        decoded = sample["decoded"]
        image = _image_to_tensor(sample["image_path"], self.image_size)
        wheels = decoded["wheels"]
        boxes = torch.tensor([w["bbox_xyxy"] for w in wheels], dtype=torch.float32)
        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        keypoints = torch.tensor(
            [[w["points"][key] for key in POINT_KEYS] for w in wheels],
            dtype=torch.float32,
        )
        if keypoints.numel() == 0:
            keypoints = torch.zeros((0, len(POINT_KEYS), 2), dtype=torch.float32)
        visibility = torch.ones((len(wheels), len(POINT_KEYS)), dtype=torch.float32)
        floor = decoded["floor"]
        target = {
            "frame_id": decoded["frame_id"],
            "image_path": str(sample["image_path"]),
            "boxes": boxes,
            "labels": torch.zeros((len(wheels),), dtype=torch.long),
            "keypoints": keypoints,
            "visibility": visibility,
            "floor": torch.tensor(
                [floor["pitch"], floor["roll"], floor["distance"]],
                dtype=torch.float32,
            ),
            "floor_meta": {
                "distance_mode": floor["distance_mode"],
                "fov_mode": floor["fov_mode"],
                "runtime_scope": decoded["runtime_scope"],
                "fixture_only": self.fixture_only,
            },
            "decoded_payload": decoded,
        }
        return image, target


def summarize_sample(dataset: WebFloorDataset, index: int = 0) -> dict[str, Any]:
    """Return a compact smoke summary for scripts/tests."""
    image, target = dataset[index]
    return {
        "index": index,
        "frame_id": target["frame_id"],
        "image_shape": list(image.shape),
        "boxes_shape": list(target["boxes"].shape),
        "keypoints_shape": list(target["keypoints"].shape),
        "floor": [float(v) for v in target["floor"]],
        "distance_mode": target["floor_meta"]["distance_mode"],
        "fixture_only": target["floor_meta"]["fixture_only"],
    }
