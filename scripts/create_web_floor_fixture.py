#!/usr/bin/env python3
"""Create a deterministic web-floor fixture dataset.

The generated data is only for pipeline tests. It contains simple synthetic
images with hand-authored wheel boxes/keypoints and floor angle/distance labels;
it is not production training data.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

RUNTIME_SCOPE = "single_forward_no_depth_no_ransac"


def _draw_frame(path: Path, wheels: list[dict[str, Any]], *, empty: bool = False) -> None:
    image = np.full((128, 128, 3), (242, 242, 236), dtype=np.uint8)
    image[88:, :] = (214, 218, 215)
    cv2.line(image, (0, 88), (127, 94), (150, 155, 150), 1)
    cv2.line(image, (0, 118), (127, 112), (184, 188, 182), 1)
    if empty:
        cv2.rectangle(image, (6, 16), (122, 54), (205, 212, 222), -1)
        cv2.putText(image, "no wheel", (26, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 95, 105), 1)
    for wheel in wheels:
        x1, y1, x2, y2 = [int(round(v)) for v in wheel["bbox_xyxy"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), (46, 52, 62), 2)
        cx = int(round((x1 + x2) / 2))
        cy = int(round((y1 + y2) / 2))
        radius = max(8, int(round(min(x2 - x1, y2 - y1) * 0.32)))
        cv2.circle(image, (cx, cy), radius, (80, 82, 88), 2)
        for name, color in (("a", (0, 80, 255)), ("b", (0, 160, 0)), ("c_disc_bottom", (255, 80, 0))):
            px, py = [int(round(v)) for v in wheel["points"][name]]
            cv2.circle(image, (px, py), 3, color, -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def _wheel(bbox: list[float], conf: float, a: list[float], b: list[float], c: list[float]) -> dict[str, Any]:
    return {
        "bbox_xyxy": bbox,
        "confidence": conf,
        "points": {"a": a, "b": b, "c_disc_bottom": c},
    }


def fixture_manifest() -> dict[str, Any]:
    items = [
        {
            "frame_id": "fixture-wheel-floor-0001",
            "image": "images/wheel_floor.png",
            "floor": {"pitch": 0.035, "roll": -0.012, "distance": 1.55, "distance_mode": "scale_relative", "fov_mode": "unknown"},
            "wheels": [_wheel([34, 34, 86, 106], 0.94, [42, 98], [78, 99], [60, 76])],
        },
        {
            "frame_id": "fixture-multi-wheel-0002",
            "image": "images/multi_wheel.png",
            "floor": {"pitch": 0.055, "roll": 0.018, "distance": 1.82, "distance_mode": "scale_relative", "fov_mode": "unknown"},
            "wheels": [
                _wheel([14, 42, 54, 106], 0.88, [20, 98], [49, 99], [34, 78]),
                _wheel([75, 39, 118, 104], 0.86, [80, 96], [112, 97], [96, 76]),
            ],
        },
        {
            "frame_id": "fixture-empty-0003",
            "image": "images/empty_no_wheel.png",
            "floor": {"pitch": 0.02, "roll": 0.0, "distance": 1.4, "distance_mode": "scale_relative", "fov_mode": "unknown"},
            "wheels": [],
        },
        {
            "frame_id": "fixture-normalized-distance-0004",
            "image": "images/normalized_distance.png",
            "floor": {"pitch": -0.025, "roll": 0.01, "distance": 0.62, "distance_mode": "normalized", "fov_mode": "fixed"},
            "wheels": [_wheel([45, 35, 102, 109], 0.9, [52, 100], [94, 101], [73, 79])],
        },
    ]
    return {
        "schema": "web_floor_fixture_v1",
        "fixture_only": True,
        "runtime_scope": RUNTIME_SCOPE,
        "notes": "Synthetic fixture for tests only; not production training data.",
        "items": items,
    }


def create_fixture(output_root: Path, *, overwrite: bool = False) -> dict[str, Any]:
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = fixture_manifest()
    for item in manifest["items"]:
        _draw_frame(output_root / item["image"], item["wheels"], empty=not item["wheels"])
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    invalid_missing_floor = dict(manifest)
    invalid_missing_floor["items"] = [dict(manifest["items"][0])]
    invalid_missing_floor["items"][0].pop("floor")
    (output_root / "manifest_invalid_missing_floor.json").write_text(
        json.dumps(invalid_missing_floor, indent=2), encoding="utf-8"
    )

    invalid_distance_mode = dict(manifest)
    invalid_distance_mode["items"] = [dict(manifest["items"][0])]
    invalid_distance_mode["items"][0]["floor"] = dict(manifest["items"][0]["floor"])
    invalid_distance_mode["items"][0]["floor"]["distance_mode"] = "metric_magic"
    (output_root / "manifest_invalid_distance_mode.json").write_text(
        json.dumps(invalid_distance_mode, indent=2), encoding="utf-8"
    )

    (output_root / "README.md").write_text(
        "# Web floor fixture\n\n"
        "Synthetic, deterministic fixture for pipeline tests only. "
        "This is not production training or quality evidence.\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path, default=Path("tests/fixtures/web_floor"))
    parser.add_argument("--config-out", type=Path, default=Path("configs/pose_dataset_web_floor_fixture.yaml"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = create_fixture(args.output_root, overwrite=args.overwrite)
    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "path": str(args.output_root),
        "manifest": "manifest.json",
        "image_size": [128, 128],
        "fixture_only": True,
        "runtime_scope": RUNTIME_SCOPE,
        "names": {0: "wheel"},
        "point_names": ["a", "b", "c_disc_bottom"],
    }
    args.config_out.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(json.dumps({"output_root": str(args.output_root), "items": len(manifest["items"]), "config": str(args.config_out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
