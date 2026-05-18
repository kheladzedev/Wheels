"""Tests for the raw Unreal/plugin export inspector.

Covers the parser (both observed text formats), the ground-metadata parser,
the per-object classifier, and a small end-to-end run on a synthetic
``tmp_path`` export.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Make scripts/ importable without installing the project.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import inspect_unreal_export as ix  # noqa: E402


# ---------- parse_keypoint_text -----------------------------------------------


def test_parse_keypoint_unreal_format():
    text = (
        "{\n"
        '{name:"Right",XY:595.78818,1180.397247\n},\n'
        '{name:"Left",XY:1446.774562,1165.815922\n},\n'
        '{name:"Center",XY:1023.827595,1023.984562\n}\n}'
    )
    out = ix.parse_keypoint_text(text)
    assert set(out) == {"Right", "Left", "Center"}
    assert out["Right"] == pytest.approx((595.78818, 1180.397247))
    assert out["Left"] == pytest.approx((1446.774562, 1165.815922))
    assert out["Center"] == pytest.approx((1023.827595, 1023.984562))


def test_parse_keypoint_unreal_format_with_optional_top_points():
    text = (
        "{\n"
        '{name:"Right",XY:100.0,420.0\n},\n'
        '{name:"Left",XY:300.0,420.0\n},\n'
        '{name:"Center",XY:200.0,330.0\n},\n'
        '{name:"LeftTop",XY:310.0,120.0\n},\n'
        '{name:"RightTop",XY:90.0,120.0\n}\n}'
    )
    out = ix.parse_keypoint_text(text)
    assert set(out) == {"Right", "Left", "Center", "LeftTop", "RightTop"}
    assert out["LeftTop"] == pytest.approx((310.0, 120.0))
    assert out["RightTop"] == pytest.approx((90.0, 120.0))


def test_parse_keypoint_simple_format():
    text = "Right: 1.0,2.0\nLeft: 3.0,4.0\nCenter: 5.0,6.0\n"
    out = ix.parse_keypoint_text(text)
    assert out == {"Right": (1.0, 2.0), "Left": (3.0, 4.0), "Center": (5.0, 6.0)}


def test_parse_keypoint_handles_negative_values():
    text = '{name:"Right",XY:-3012.84,1270.80},{name:"Left",XY:-2015.00,-5.0},{name:"Center",XY:-2679.58,1096.91}'
    out = ix.parse_keypoint_text(text)
    assert out["Right"] == pytest.approx((-3012.84, 1270.80))
    assert out["Left"] == pytest.approx((-2015.00, -5.0))
    assert out["Center"] == pytest.approx((-2679.58, 1096.91))


def test_parse_keypoint_missing_one_name_returns_partial():
    text = '{name:"Right",XY:1.0,2.0},{name:"Left",XY:3.0,4.0}'
    out = ix.parse_keypoint_text(text)
    assert set(out) == {"Right", "Left"}
    assert "Center" not in out


def test_parse_keypoint_garbage_returns_empty():
    assert ix.parse_keypoint_text("") == {}
    assert ix.parse_keypoint_text("not a keypoint file") == {}


# ---------- parse_ground_text -------------------------------------------------


def test_parse_ground_meta():
    txt = "DeltaZ{170.000019},Roll:-0.0,Pitch:61.769783,FOV:54.656362"
    g = ix.parse_ground_text(txt)
    assert g is not None
    assert g.delta_z == pytest.approx(170.000019)
    assert g.roll == pytest.approx(0.0)
    assert g.pitch == pytest.approx(61.769783)
    assert g.fov == pytest.approx(54.656362)


def test_parse_ground_meta_garbage_returns_none():
    assert ix.parse_ground_text("nope") is None


# ---------- classify ----------------------------------------------------------


def test_classify_valid():
    pts = {"Right": (100.0, 200.0), "Left": (300.0, 200.0), "Center": (200.0, 250.0)}
    status, _ = ix.classify(pts, 2048, 2048)
    assert status == ix.STATUS_VALID


def test_classify_empty_all_zero():
    pts = {n: (0.0, 0.0) for n in ix.POINT_NAMES}
    status, reason = ix.classify(pts, 2048, 2048)
    assert status == ix.STATUS_EMPTY
    assert "0" in reason


def test_classify_partial_zero():
    pts = {
        "Right": (0.0, 0.0),
        "Left": (300.0, 200.0),
        "Center": (200.0, 250.0),
    }
    status, reason = ix.classify(pts, 2048, 2048)
    assert status == ix.STATUS_PARTIAL_ZERO
    assert "Right" in reason


def test_classify_out_of_bounds_positive():
    pts = {
        "Right": (3677.0, 1733.0),
        "Left": (7914.0, 3574.0),
        "Center": (5220.0, 2247.0),
    }
    status, reason = ix.classify(pts, 2048, 2048)
    assert status == ix.STATUS_OUT_OF_BOUNDS
    assert "Right" in reason or "Left" in reason


def test_classify_out_of_bounds_negative():
    pts = {
        "Right": (-3012.0, 1270.0),
        "Left": (-2015.0, 1253.0),
        "Center": (-2679.0, 1096.0),
    }
    status, _ = ix.classify(pts, 2048, 2048)
    assert status == ix.STATUS_OUT_OF_BOUNDS


def test_classify_missing_points():
    pts = {"Right": (100.0, 100.0)}
    status, reason = ix.classify(pts, 2048, 2048)
    assert status == ix.STATUS_MISSING
    assert "Left" in reason and "Center" in reason


def test_classify_boundary_inclusive():
    pts = {
        "Right": (0.0, 0.0),  # corner — but this is the "empty" sentinel value too
        "Left": (2047.0, 2047.0),
        "Center": (1000.0, 1000.0),
    }
    # First point coincides with the zero sentinel; the classifier therefore
    # reports PARTIAL_ZERO, not VALID. That's the intended behaviour: (0,0)
    # in the export always means "no data" per current observation.
    status, _ = ix.classify(pts, 2048, 2048)
    assert status == ix.STATUS_PARTIAL_ZERO


# ---------- read_jpeg_size ----------------------------------------------------


def test_read_jpeg_size(tmp_path: Path):
    img = (np.random.rand(120, 240, 3) * 255).astype(np.uint8)
    p = tmp_path / "x.jpg"
    assert cv2.imwrite(str(p), img)
    size = ix.read_jpeg_size(p)
    assert size == (240, 120)


# ---------- end-to-end --------------------------------------------------------


def _write_kp(path: Path, right, left, center):
    path.write_text(
        "{\n"
        f'{{name:"Right",XY:{right[0]},{right[1]}\n}},\n'
        f'{{name:"Left",XY:{left[0]},{left[1]}\n}},\n'
        f'{{name:"Center",XY:{center[0]},{center[1]}\n}}\n}}'
    )


def _build_fake_export(root: Path, image_size_wh: tuple[int, int] = (640, 480)):
    w, h = image_size_wh
    img = np.ones((h, w, 3), dtype=np.uint8) * 200
    (root / "Images").mkdir(parents=True)
    (root / "Ground").mkdir(parents=True)
    (root / "keyPoint").mkdir(parents=True)

    # frame 0: one valid wheel, one empty wheel
    cv2.imwrite(str(root / "Images/0.jpg"), img)
    (root / "Ground/0.txt").write_text("DeltaZ{200.0},Roll:0.0,Pitch:60.0,FOV:55.0")
    f0 = root / "keyPoint/0"
    f0.mkdir()
    _write_kp(f0 / "0.txt", (100.0, 200.0), (300.0, 200.0), (200.0, 250.0))
    _write_kp(f0 / "1.txt", (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))

    # frame 1: out-of-bounds wheel
    cv2.imwrite(str(root / "Images/1.jpg"), img)
    f1 = root / "keyPoint/1"
    f1.mkdir()
    _write_kp(f1 / "0.txt", (5000.0, 100.0), (50.0, 50.0), (60.0, 60.0))

    # frame 2: partial zero
    cv2.imwrite(str(root / "Images/2.jpg"), img)
    f2 = root / "keyPoint/2"
    f2.mkdir()
    _write_kp(f2 / "0.txt", (0.0, 0.0), (100.0, 100.0), (110.0, 110.0))

    # frame 3: missing points (only Right present)
    cv2.imwrite(str(root / "Images/3.jpg"), img)
    f3 = root / "keyPoint/3"
    f3.mkdir()
    (f3 / "0.txt").write_text('{name:"Right",XY:50.0,50.0}')


def test_inspect_end_to_end(tmp_path: Path):
    src = tmp_path / "export"
    src.mkdir()
    _build_fake_export(src)
    out = tmp_path / "out"

    args = ix.parse_args(
        [
            "--source-root",
            str(src),
            "--out-dir",
            str(out),
            "--max-preview",
            "10",
        ]
    )
    report = ix.inspect(args)

    counts = report["counts_by_status"]
    assert counts[ix.STATUS_VALID] == 1
    assert counts[ix.STATUS_EMPTY] == 1
    assert counts[ix.STATUS_OUT_OF_BOUNDS] == 1
    assert counts[ix.STATUS_PARTIAL_ZERO] == 1
    assert counts[ix.STATUS_MISSING] == 1
    assert counts[ix.STATUS_PARSE_ERROR] == 0

    assert report["n_images"] == 4
    assert report["n_ground_files"] == 1
    assert report["n_ground_parsed"] == 1
    assert report["n_keypoint_frame_dirs"] == 4
    assert report["n_keypoint_object_files"] == 5

    # Files written.
    assert (out / "report.json").is_file()
    assert (out / "report.md").is_file()
    loaded = json.loads((out / "report.json").read_text())
    assert loaded["counts_by_status"] == counts
    assert loaded["status_previews"][ix.STATUS_VALID]
    assert loaded["status_previews"][ix.STATUS_OUT_OF_BOUNDS]

    md = (out / "report.md").read_text()
    assert "RAW EXPORT" in md or "Raw Unreal export inspection" in md
    assert "Status preview galleries" in md
    assert "Contract notes" in md
    assert "NOT_APPROVED_FOR_TRAINING" in md

    # At least one preview was rendered (frames have signal).
    preview_files = list((out / "previews").glob("*.jpg"))
    assert preview_files, "expected at least one preview to be rendered"


def test_inspect_creates_outputs_when_keypoints_missing(tmp_path: Path):
    src = tmp_path / "export"
    (src / "Images").mkdir(parents=True)
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200
    cv2.imwrite(str(src / "Images/9.jpg"), img)

    out = tmp_path / "out"
    args = ix.parse_args(
        [
            "--source-root",
            str(src),
            "--out-dir",
            str(out),
            "--max-preview",
            "5",
        ]
    )
    report = ix.inspect(args)
    assert report["n_keypoint_object_files"] == 0
    assert report["n_images"] == 1
    assert (out / "report.md").is_file()
