from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.import_web_floor_annotations import parse_args
from web_floor_annotation_import import (
    REQUIRED_COLUMNS,
    WebFloorAnnotationImportError,
    import_web_floor_csv_annotations,
)
from web_floor_dataset import WebFloorDataset
from web_floor_real_data_gate import WebFloorRealDataGateConfig, audit_web_floor_real_data


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((128, 128, 3), 220, dtype=np.uint8)
    cv2.circle(image, (64, 70), 24, (40, 40, 40), thickness=3)
    cv2.imwrite(str(path), image)


def _row(frame_id: str, image: str, split: str, distance: float, pitch: float) -> dict[str, str]:
    return {
        "frame_id": frame_id,
        "split": split,
        "image": image,
        "provenance_source": "phone_batch",
        "provenance_device": "unit-test-phone",
        "provenance_capture_date": "2026-06-12",
        "provenance_annotator": "unit-test",
        "pitch": str(pitch),
        "roll": "0.01",
        "distance": str(distance),
        "distance_mode": "scale_relative",
        "fov_mode": "provided",
        "bbox_x1": "32",
        "bbox_y1": "32",
        "bbox_x2": "96",
        "bbox_y2": "108",
        "confidence": "1.0",
        "a_x": "42",
        "a_y": "98",
        "b_x": "84",
        "b_y": "99",
        "c_disc_bottom_x": "64",
        "c_disc_bottom_y": "78",
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*REQUIRED_COLUMNS, "provenance_capture_date", "fov_mode"])
        writer.writeheader()
        writer.writerows(rows)


def test_parse_args_requires_csv_inputs() -> None:
    args = parse_args(
        [
            "--csv",
            "annotations.csv",
            "--image-root",
            "images",
            "--dataset-root",
            "data/web_floor_real_v1",
            "--config-out",
            "configs/web_floor_real_v1.yaml",
        ]
    )

    assert args.csv == Path("annotations.csv")
    assert args.image_root == Path("images")
    assert args.dataset_root == Path("data/web_floor_real_v1")


def test_import_csv_writes_manifest_config_and_passes_relaxed_gate(tmp_path: Path) -> None:
    image_root = tmp_path / "raw_images"
    _write_image(image_root / "frame_001.jpg")
    _write_image(image_root / "frame_002.jpg")
    rows = [
        _row("frame-001", "frame_001.jpg", "train", 0.2, -0.04),
        _row("frame-001", "frame_001.jpg", "train", 0.2, -0.04),
        _row("frame-002", "frame_002.jpg", "holdout", 1.0, 0.04),
    ]
    csv_path = tmp_path / "annotations.csv"
    _write_csv(csv_path, rows)
    dataset_root = tmp_path / "dataset"
    config_out = tmp_path / "web_floor_real.yaml"

    manifest = import_web_floor_csv_annotations(
        csv_path=csv_path,
        image_root=image_root,
        dataset_root=dataset_root,
        config_out=config_out,
        overwrite=True,
    )

    assert manifest["fixture_only"] is False
    assert len(manifest["items"]) == 2
    assert sum(len(item["wheels"]) for item in manifest["items"]) == 3
    assert (dataset_root / "images" / "frame_001.jpg").is_file()
    assert json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8")) == manifest

    dataset = WebFloorDataset(config_out)
    assert dataset.fixture_only is False
    report = audit_web_floor_real_data(
        config_out,
        gate=WebFloorRealDataGateConfig(
            min_frames=2,
            min_wheels=3,
            min_distance_span=0.5,
            min_angle_span_rad=0.05,
        ),
    )
    assert report["production_data_ready"] is True


def test_import_rejects_missing_required_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("frame_id,image\nx,frame.jpg\n", encoding="utf-8")

    with pytest.raises(WebFloorAnnotationImportError, match="missing required"):
        import_web_floor_csv_annotations(
            csv_path=csv_path,
            image_root=tmp_path,
            dataset_root=tmp_path / "dataset",
            config_out=tmp_path / "cfg.yaml",
            overwrite=True,
        )


def test_import_rejects_inconsistent_frame_floor(tmp_path: Path) -> None:
    image_root = tmp_path / "raw_images"
    _write_image(image_root / "frame_001.jpg")
    rows = [
        _row("frame-001", "frame_001.jpg", "train", 0.2, -0.04),
        _row("frame-001", "frame_001.jpg", "train", 0.9, -0.04),
    ]
    csv_path = tmp_path / "annotations.csv"
    _write_csv(csv_path, rows)

    with pytest.raises(WebFloorAnnotationImportError, match="inconsistent floor"):
        import_web_floor_csv_annotations(
            csv_path=csv_path,
            image_root=image_root,
            dataset_root=tmp_path / "dataset",
            config_out=tmp_path / "cfg.yaml",
            overwrite=True,
        )
