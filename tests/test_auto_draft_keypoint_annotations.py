"""End-to-end smoke test: auto-draft output passes the validator and is flagged.

Mirrors `tests/test_create_sample_keypoint_incoming.py` — the cheap
contract check between `src/auto_draft_keypoint_annotations.py` and
`src/check_keypoint_incoming.py`. If one drifts, this fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from auto_draft_keypoint_annotations import main as draft_main
from check_keypoint_incoming import main as check_main


def _write_test_image(path: Path, w: int = 320, h: int = 240) -> None:
    img = np.full((h, w, 3), 180, dtype=np.uint8)
    assert cv2.imwrite(str(path), img), f"could not write {path}"


def test_draft_bundle_is_valid_and_marked(tmp_path: Path) -> None:
    images_dir = tmp_path / "src_imgs"
    images_dir.mkdir()
    # One stem hits the keyword heuristic, one doesn't.
    _write_test_image(images_dir / "car_side_view.jpg")
    _write_test_image(images_dir / "anon.jpg")

    out_root = tmp_path / "draft"
    argv = [
        "auto_draft_keypoint_annotations.py",
        "--images-dir",
        str(images_dir),
        "--output-root",
        str(out_root),
        "--overwrite",
    ]
    with patch.object(sys, "argv", argv):
        assert draft_main() == 0

    # Layout matches the plugin contract.
    assert (out_root / "images" / "car_side_view.jpg").is_file()
    assert (out_root / "images" / "anon.jpg").is_file()
    assert (out_root / "annotations" / "car_side_view.json").is_file()
    assert (out_root / "annotations" / "anon.json").is_file()
    assert (out_root / "metadata" / "source_info.json").is_file()

    # source_info carries the load-bearing draft markers.
    info = json.loads((out_root / "metadata" / "source_info.json").read_text())
    assert info["annotation_method"] == "auto_draft_heuristic"
    assert info["warning"] == "NOT_GROUND_TRUTH_REQUIRES_HUMAN_REVIEW"
    assert info["image_count"] == 2
    assert info["images_with_drafted_wheels"] == 1
    assert info["images_with_empty_wheels"] == 1

    # Per-image annotation shape + draft flags.
    car = json.loads((out_root / "annotations" / "car_side_view.json").read_text())
    assert car["_draft"] is True
    assert car["_warning"] == "NOT_GROUND_TRUTH_REQUIRES_HUMAN_REVIEW"
    assert car["frame_id"] == "car_side_view"
    assert car["image"] == "car_side_view.jpg"
    assert len(car["wheels"]) == 2
    for wheel in car["wheels"]:
        assert set(wheel["points"].keys()) == {"a", "b", "c_disc_bottom"}
        x1, y1, x2, y2 = wheel["bbox_xyxy"]
        assert x1 < x2 and y1 < y2

        # 2026-05-14 floor-ray semantics: A/B in lower band of bbox,
        # below c_disc_bottom, with A on the left and B on the right.
        a_x, a_y = wheel["points"]["a"]
        b_x, b_y = wheel["points"]["b"]
        c_x, c_y = wheel["points"]["c_disc_bottom"]
        mid_y = (y1 + y2) / 2.0
        assert a_y > mid_y, f"a.y={a_y} must be in lower half (mid={mid_y})"
        assert b_y > mid_y, f"b.y={b_y} must be in lower half (mid={mid_y})"
        assert a_y > c_y, f"a.y={a_y} must be below c_disc_bottom.y={c_y}"
        assert b_y > c_y, f"b.y={b_y} must be below c_disc_bottom.y={c_y}"
        assert a_x < b_x, f"a.x={a_x} must be left of b.x={b_x}"
        # All points inside the bbox (strict).
        for name, (px, py) in (
            ("a", (a_x, a_y)),
            ("b", (b_x, b_y)),
            ("c_disc_bottom", (c_x, c_y)),
        ):
            assert x1 <= px <= x2 and y1 <= py <= y2, (
                f"point {name}=({px},{py}) outside bbox [{x1},{y1},{x2},{y2}]"
            )

    anon = json.loads((out_root / "annotations" / "anon.json").read_text())
    assert anon["_draft"] is True
    assert anon["wheels"] == []

    # And the bundle has to pass the plugin validator clean (exit 0).
    check_argv = [
        "check_keypoint_incoming.py",
        "--source-root",
        str(out_root),
    ]
    with patch.object(sys, "argv", check_argv):
        rc = check_main()
    assert rc == 0


def test_overwrite_refused_when_output_root_nonempty(tmp_path: Path) -> None:
    images_dir = tmp_path / "src_imgs"
    images_dir.mkdir()
    _write_test_image(images_dir / "car_side.jpg")

    out_root = tmp_path / "draft"
    out_root.mkdir()
    (out_root / "stale.txt").write_text("x")

    argv = [
        "auto_draft_keypoint_annotations.py",
        "--images-dir",
        str(images_dir),
        "--output-root",
        str(out_root),
    ]
    with patch.object(sys, "argv", argv):
        rc = draft_main()
    assert rc == 1
    # The refusal must not have wiped the existing dir.
    assert (out_root / "stale.txt").is_file()


def test_empty_images_dir(tmp_path: Path) -> None:
    images_dir = tmp_path / "empty"
    images_dir.mkdir()
    out_root = tmp_path / "draft"

    argv = [
        "auto_draft_keypoint_annotations.py",
        "--images-dir",
        str(images_dir),
        "--output-root",
        str(out_root),
        "--overwrite",
    ]
    with patch.object(sys, "argv", argv):
        assert draft_main() == 0

    info = json.loads((out_root / "metadata" / "source_info.json").read_text())
    assert info["image_count"] == 0
    assert info["images_with_drafted_wheels"] == 0
    assert info["images_with_empty_wheels"] == 0


def test_missing_images_dir(tmp_path: Path) -> None:
    argv = [
        "auto_draft_keypoint_annotations.py",
        "--images-dir",
        str(tmp_path / "does_not_exist"),
        "--output-root",
        str(tmp_path / "draft"),
        "--overwrite",
    ]
    with patch.object(sys, "argv", argv):
        rc = draft_main()
    assert rc == 2
