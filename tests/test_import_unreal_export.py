"""Tests for the Unreal/plugin → VSBL incoming adapter.

Covers the per-object drop rules, the bbox builder, and an end-to-end run
on a synthetic ``tmp_path`` export.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import import_unreal_export as imp  # noqa: E402


# ---------- build_bbox_from_points -------------------------------------------


def test_build_bbox_adds_margin_and_orders_corners():
    bbox = imp.build_bbox_from_points(
        (100.0, 200.0),
        (300.0, 200.0),
        (200.0, 250.0),
        image_w=2048,
        image_h=2048,
        margin=80,
    )
    assert bbox is not None
    x1, y1, x2, y2 = bbox
    assert x1 == pytest.approx(20.0)
    assert y1 == pytest.approx(120.0)
    assert x2 == pytest.approx(380.0)
    assert y2 == pytest.approx(330.0)
    # Points must lie inside the bbox.
    for px, py in [(100, 200), (300, 200), (200, 250)]:
        assert x1 <= px <= x2 and y1 <= py <= y2


def test_build_bbox_clips_to_image():
    # Point cluster near the top-left corner; the margin would push the bbox
    # into negative coordinates without clipping.
    bbox = imp.build_bbox_from_points(
        (10.0, 10.0),
        (20.0, 10.0),
        (15.0, 15.0),
        image_w=2048,
        image_h=2048,
        margin=80,
    )
    assert bbox is not None
    x1, y1, x2, y2 = bbox
    assert x1 == 0.0 and y1 == 0.0
    assert x2 == pytest.approx(100.0)
    assert y2 == pytest.approx(95.0)


def test_build_bbox_clips_at_far_edge():
    bbox = imp.build_bbox_from_points(
        (2040.0, 2040.0),
        (2045.0, 2045.0),
        (2042.0, 2042.0),
        image_w=2048,
        image_h=2048,
        margin=80,
    )
    assert bbox is not None
    x1, y1, x2, y2 = bbox
    # check_keypoint_incoming clips half-openly: max valid index is image_w - 1.
    assert x2 == 2047.0 and y2 == 2047.0
    assert x1 == pytest.approx(1960.0)


def test_build_bbox_returns_none_when_all_points_equal_and_no_margin():
    bbox = imp.build_bbox_from_points(
        (10.0, 10.0),
        (10.0, 10.0),
        (10.0, 10.0),
        image_w=2048,
        image_h=2048,
        margin=0,
    )
    assert bbox is None


def test_build_bbox_from_optional_top_points_uses_all_five_points():
    points = {
        "Right": (100.0, 420.0),
        "Left": (300.0, 420.0),
        "Center": (200.0, 330.0),
        "LeftTop": (310.0, 120.0),
        "RightTop": (90.0, 120.0),
    }
    bbox = imp.build_bbox_from_optional_top_points(points, image_w=640, image_h=480)
    assert bbox == pytest.approx((90.0, 120.0, 310.0, 420.0))


def test_build_bbox_from_optional_top_points_rejects_oob_helper():
    points = {
        "Right": (100.0, 420.0),
        "Left": (300.0, 420.0),
        "Center": (200.0, 330.0),
        "LeftTop": (700.0, 120.0),
        "RightTop": (90.0, 120.0),
    }
    assert imp.build_bbox_from_optional_top_points(points, 640, 480) is None


# ---------- _try_build_wheel — drop rules ------------------------------------


def _kp_text(right, left, center) -> str:
    return (
        "{\n"
        f'{{name:"Right",XY:{right[0]},{right[1]}\n}},\n'
        f'{{name:"Left",XY:{left[0]},{left[1]}\n}},\n'
        f'{{name:"Center",XY:{center[0]},{center[1]}\n}}\n}}'
    )


def _kp_text_with_top(right, left, center, left_top, right_top) -> str:
    return (
        "{\n"
        f'{{name:"Right",XY:{right[0]},{right[1]}\n}},\n'
        f'{{name:"Left",XY:{left[0]},{left[1]}\n}},\n'
        f'{{name:"Center",XY:{center[0]},{center[1]}\n}},\n'
        f'{{name:"LeftTop",XY:{left_top[0]},{left_top[1]}\n}},\n'
        f'{{name:"RightTop",XY:{right_top[0]},{right_top[1]}\n}}\n}}'
    )


def test_try_build_wheel_valid():
    summary = imp.ImportSummary()
    wheel = imp._try_build_wheel(
        _kp_text((100.0, 420.0), (300.0, 420.0), (200.0, 330.0)),
        2048,
        2048,
        80,
        summary,
    )
    assert wheel is not None
    assert wheel["points"]["a"] == [100.0, 420.0]
    assert wheel["points"]["b"] == [300.0, 420.0]
    assert wheel["points"]["c_disc_bottom"] == [200.0, 330.0]
    assert len(wheel["bbox_xyxy"]) == 4
    # Drop counters unchanged.
    assert all(v == 0 for v in summary.drop_counts.values())


def test_try_build_wheel_prefers_optional_top_point_bbox():
    summary = imp.ImportSummary()
    wheel = imp._try_build_wheel(
        _kp_text_with_top(
            (100.0, 420.0),
            (300.0, 420.0),
            (200.0, 330.0),
            (310.0, 120.0),
            (90.0, 120.0),
        ),
        640,
        480,
        80,
        summary,
    )
    assert wheel is not None
    assert wheel["bbox_xyxy"] == pytest.approx([90.0, 120.0, 310.0, 420.0])
    assert summary.bbox_from_top_points == 1
    assert summary.bbox_from_floorray == 0
    assert all(v == 0 for v in summary.drop_counts.values())


def test_try_build_wheel_drops_all_zero():
    summary = imp.ImportSummary()
    wheel = imp._try_build_wheel(
        _kp_text((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        2048,
        2048,
        80,
        summary,
    )
    assert wheel is None
    assert summary.drop_counts[imp.DROP_ALL_ZERO] == 1


def test_try_build_wheel_drops_partial_zero_as_out_of_bounds():
    # Plugin author: (0,0) means invisible. We cannot emit only 2/3 points
    # per the contract, so the whole wheel is dropped.
    summary = imp.ImportSummary()
    wheel = imp._try_build_wheel(
        _kp_text((0.0, 0.0), (300.0, 200.0), (200.0, 250.0)),
        2048,
        2048,
        80,
        summary,
    )
    assert wheel is None
    assert summary.drop_counts[imp.DROP_OUT_OF_BOUNDS] == 1


def test_try_build_wheel_drops_negative_oob():
    summary = imp.ImportSummary()
    wheel = imp._try_build_wheel(
        _kp_text((-100.0, 200.0), (300.0, 200.0), (200.0, 250.0)),
        2048,
        2048,
        80,
        summary,
    )
    assert wheel is None
    assert summary.drop_counts[imp.DROP_OUT_OF_BOUNDS] == 1


def test_try_build_wheel_drops_positive_oob():
    summary = imp.ImportSummary()
    wheel = imp._try_build_wheel(
        _kp_text((100.0, 200.0), (5000.0, 200.0), (200.0, 250.0)),
        2048,
        2048,
        80,
        summary,
    )
    assert wheel is None
    assert summary.drop_counts[imp.DROP_OUT_OF_BOUNDS] == 1


def test_try_build_wheel_drops_missing_point():
    summary = imp.ImportSummary()
    wheel = imp._try_build_wheel(
        '{name:"Right",XY:1.0,2.0},{name:"Left",XY:3.0,4.0}',
        2048,
        2048,
        80,
        summary,
    )
    assert wheel is None
    assert summary.drop_counts[imp.DROP_MISSING_POINTS] == 1


def test_try_build_wheel_points_inside_built_bbox():
    """The validator allows 5 px slack — but a margin of 80 guarantees room."""
    summary = imp.ImportSummary()
    wheel = imp._try_build_wheel(
        _kp_text((500.0, 700.0), (700.0, 700.0), (600.0, 610.0)),
        2048,
        2048,
        80,
        summary,
    )
    assert wheel is not None
    x1, y1, x2, y2 = wheel["bbox_xyxy"]
    for pt in wheel["points"].values():
        assert x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2


# ---------- end-to-end --------------------------------------------------------


def _write_kp(path: Path, right, left, center):
    path.write_text(_kp_text(right, left, center))


def _build_fake_export(root: Path):
    (root / "Images").mkdir(parents=True)
    (root / "Ground").mkdir(parents=True)
    (root / "keyPoint").mkdir(parents=True)
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200

    # frame 0: 1 valid + 1 all-zero
    cv2.imwrite(str(root / "Images/0.jpg"), img)
    (root / "Ground/0.txt").write_text("DeltaZ{200.0},Roll:0.0,Pitch:60.0,FOV:55.0")
    (root / "keyPoint/0").mkdir()
    _write_kp(root / "keyPoint/0/0.txt", (100.0, 420.0), (300.0, 420.0), (200.0, 330.0))
    _write_kp(root / "keyPoint/0/1.txt", (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))

    # frame 1: 1 OOB
    cv2.imwrite(str(root / "Images/1.jpg"), img)
    (root / "keyPoint/1").mkdir()
    _write_kp(root / "keyPoint/1/0.txt", (700.0, 100.0), (50.0, 50.0), (60.0, 60.0))

    # frame 2: missing point
    cv2.imwrite(str(root / "Images/2.jpg"), img)
    (root / "keyPoint/2").mkdir()
    (root / "keyPoint/2/0.txt").write_text('{name:"Right",XY:50.0,50.0}')

    # frame 3: image with no keypoint folder at all — must still be imported
    # with wheels: [].
    cv2.imwrite(str(root / "Images/3.jpg"), img)


def test_import_end_to_end(tmp_path: Path):
    src = tmp_path / "export"
    src.mkdir()
    _build_fake_export(src)

    out_root = tmp_path / "out"
    args = imp.parse_args(
        [
            "--source-root",
            str(src),
            "--out-root",
            str(out_root),
            "--overwrite",
        ]
    )
    rc = imp.run(args)
    assert rc == 0

    images_out = sorted((out_root / "images").iterdir())
    assert [p.name for p in images_out] == ["0.jpg", "1.jpg", "2.jpg", "3.jpg"]

    annos_out = sorted((out_root / "annotations").iterdir())
    assert [p.name for p in annos_out] == [
        "0.json",
        "1.json",
        "2.json",
        "3.json",
    ]

    a0 = json.loads((out_root / "annotations/0.json").read_text())
    assert a0["frame_id"] == "0"
    assert a0["image"] == "0.jpg"
    assert len(a0["wheels"]) == 1
    assert a0["wheels"][0]["points"]["a"] == [100.0, 420.0]
    assert a0["wheels"][0]["points"]["b"] == [300.0, 420.0]
    assert a0["wheels"][0]["points"]["c_disc_bottom"] == [200.0, 330.0]

    a3 = json.loads((out_root / "annotations/3.json").read_text())
    assert a3["wheels"] == []

    report = json.loads((out_root / "metadata/import_report.json").read_text())
    assert report["images_found"] == 4
    assert report["images_imported"] == 4
    assert report["valid_wheels"] == 1
    assert report["bbox_strategy_counts"] == {"top_points": 0, "floorray": 1}
    assert report["drop_counts"][imp.DROP_ALL_ZERO] == 1
    assert report["drop_counts"][imp.DROP_OUT_OF_BOUNDS] == 1
    assert report["drop_counts"][imp.DROP_MISSING_POINTS] == 1
    assert "0" in report["ground_meta"]
    assert report["ground_meta"]["0"]["delta_z"] == pytest.approx(200.0)

    src_info = json.loads((out_root / "metadata/source_info.json").read_text())
    assert src_info["source_format"] == "raw_unreal_plugin_export"
    assert src_info["source_name"] == "unreal_export"
    assert src_info["mapping"] == {
        "Right": "a",
        "Left": "b",
        "Center": "c_disc_bottom",
        "LeftTop": "bbox helper when present",
        "RightTop": "bbox helper when present",
    }
    assert src_info["mapping_basis"] == "plugin_author_confirmation"
    assert src_info["not_yet_training_approved"] is True
    assert src_info["requires_human_preview"] is True


def test_import_refuses_to_overwrite_without_flag(tmp_path: Path):
    src = tmp_path / "export"
    src.mkdir()
    _build_fake_export(src)

    out_root = tmp_path / "out"
    out_root.mkdir()
    (out_root / "leftover.txt").write_text("x")

    args = imp.parse_args(["--source-root", str(src), "--out-root", str(out_root)])
    with pytest.raises(SystemExit):
        imp.run(args)


def test_passes_check_keypoint_incoming_validator(tmp_path: Path):
    """Adapter output must satisfy `src/check_keypoint_incoming.py`."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import check_keypoint_incoming as chk  # noqa: E402

    src = tmp_path / "export"
    src.mkdir()
    _build_fake_export(src)
    out_root = tmp_path / "out"

    args = imp.parse_args(
        ["--source-root", str(src), "--out-root", str(out_root), "--overwrite"]
    )
    assert imp.run(args) == 0

    rc = chk.main(["--source-root", str(out_root)])
    assert rc == 0
