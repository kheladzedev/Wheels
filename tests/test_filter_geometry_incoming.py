from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import filter_geometry_incoming as fgi


def _write_sample(root: Path, stem: str, wheels: list[dict]) -> None:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    image = np.full((100, 100, 3), 80, dtype=np.uint8)
    cv2.imwrite(str(root / "images" / f"{stem}.png"), image)
    payload = {
        "frame_id": stem,
        "image": f"{stem}.png",
        "wheels": wheels,
    }
    (root / "annotations" / f"{stem}.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _wheel(bbox: list[float]) -> dict:
    return {
        "bbox_xyxy": bbox,
        "points": {
            "a": [bbox[0], bbox[3]],
            "b": [bbox[2], bbox[3]],
            "c_disc_bottom": [(bbox[0] + bbox[2]) / 2, bbox[3]],
        },
    }


def test_wheel_drop_reason_rejects_large_area():
    reason = fgi.wheel_drop_reason(
        _wheel([0, 0, 90, 90]),
        img_w=100,
        img_h=100,
        min_side_frac=0.01,
        max_area_frac=0.08,
        max_width_frac=0.95,
        max_height_frac=0.95,
    )

    assert reason == "bbox_area_too_large"


def test_filter_batch_keeps_good_wheels_and_reports_drops(tmp_path):
    source = tmp_path / "source"
    out = tmp_path / "out"
    _write_sample(
        source,
        "frame",
        [
            _wheel([10, 10, 30, 30]),
            _wheel([0, 0, 95, 95]),
        ],
    )
    args = argparse.Namespace(
        source_root=source,
        output_root=out,
        overwrite=True,
        min_nonblack_frac=0.02,
        min_bbox_side_frac=0.015,
        max_bbox_area_frac=0.08,
        max_bbox_width_frac=0.55,
        max_bbox_height_frac=0.50,
        max_wheels_per_frame=6,
    )

    report = fgi.filter_batch(args)

    assert report["kept_frames"] == 1
    assert report["kept_wheels"] == 1
    assert report["dropped_wheels"] == {"bbox_area_too_large": 1}
    payload = json.loads((out / "annotations/frame.json").read_text(encoding="utf-8"))
    assert len(payload["wheels"]) == 1
    assert payload["_qa_filter"]["source_wheels"] == 2
