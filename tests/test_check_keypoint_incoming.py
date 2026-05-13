"""Tests for the plugin-format keypoint batch validator.

The validator is `src/check_keypoint_incoming.py`. These tests build tiny
fixtures in `tmp_path` (one or two 64x64 JPEGs + JSON siblings) and drive
the validator in-process via its `main()` entry point. No model, no real
batch data, no network — pure I/O + cv2.imread.

The clean-batch test also exercises the generator end-to-end so we catch
contract drift between the two scripts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from check_keypoint_incoming import main as check_main
from create_sample_keypoint_incoming import main as generator_main

IMG_W = 64
IMG_H = 64


def _write_image(path: Path, w: int = IMG_W, h: int = IMG_H) -> None:
    """Write a synthetic JPEG. Dimensions match the values the JSON encodes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    ok = cv2.imwrite(str(path), img)
    assert ok, f"cv2.imwrite failed for {path}"


def _write_anno(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _good_payload(stem: str) -> dict:
    """A single-wheel annotation that should pass cleanly on a 64x64 image."""
    return {
        "frame_id": stem,
        "image": f"{stem}.jpg",
        "wheels": [
            {
                "bbox_xyxy": [10.0, 10.0, 50.0, 50.0],
                "points": {
                    "a": [12.0, 30.0],
                    "b": [48.0, 30.0],
                    "c_disc_bottom": [30.0, 48.0],
                },
            }
        ],
    }


def _make_batch(tmp_path: Path) -> Path:
    """Lay out a minimal source-root with one good pair. Returns the root."""
    root = tmp_path / "batch"
    _write_image(root / "images" / "frame_0001.jpg")
    _write_anno(root / "annotations" / "frame_0001.json", _good_payload("frame_0001"))
    return root


def _run_check(source_root: Path) -> int:
    argv = ["check_keypoint_incoming.py", "--source-root", str(source_root)]
    with patch.object(sys, "argv", argv):
        return check_main()


def _read_errors_count_via_capsys(capsys) -> tuple[int, int]:
    """Pull the 'Errors:' and 'Warnings:' lines from captured stdout."""
    captured = capsys.readouterr().out
    errors = warnings = 0
    for line in captured.splitlines():
        if line.startswith("Errors:"):
            errors = int(line.split()[-1])
        elif line.startswith("Warnings:"):
            warnings = int(line.split()[-1])
    return errors, warnings


# ---- clean-batch round-trip via the generator ---------------------------


def test_check_keypoint_incoming_clean_batch(tmp_path: Path, capsys) -> None:
    out_root = tmp_path / "gen"
    gen_argv = [
        "create_sample_keypoint_incoming.py",
        "--count",
        "3",
        "--output-root",
        str(out_root),
        "--seed",
        "1",
        "--overwrite",
    ]
    with patch.object(sys, "argv", gen_argv):
        assert generator_main() == 0

    # Discard generator output so the next reader sees only checker output.
    capsys.readouterr()

    rc = _run_check(out_root)
    errors, _ = _read_errors_count_via_capsys(capsys)
    assert rc == 0
    assert errors == 0


# ---- single-rule fixtures (no generator) --------------------------------


def test_check_keypoint_incoming_missing_annotation_is_error(
    tmp_path: Path, capsys
) -> None:
    root = tmp_path / "batch"
    _write_image(root / "images" / "frame_lonely.jpg")
    (root / "annotations").mkdir(parents=True, exist_ok=True)  # empty dir

    rc = _run_check(root)
    errors, _ = _read_errors_count_via_capsys(capsys)
    assert rc == 1
    assert errors >= 1


def test_check_keypoint_incoming_wrong_frame_id_is_error(
    tmp_path: Path, capsys
) -> None:
    root = _make_batch(tmp_path)
    bad = _good_payload("frame_0001")
    bad["frame_id"] = "wrong_id"
    _write_anno(root / "annotations" / "frame_0001.json", bad)

    rc = _run_check(root)
    errors, _ = _read_errors_count_via_capsys(capsys)
    assert rc == 1
    assert errors >= 1


def test_check_keypoint_incoming_wrong_image_field_is_error(
    tmp_path: Path, capsys
) -> None:
    root = _make_batch(tmp_path)
    bad = _good_payload("frame_0001")
    bad["image"] = "not_this_file.jpg"
    _write_anno(root / "annotations" / "frame_0001.json", bad)

    rc = _run_check(root)
    errors, _ = _read_errors_count_via_capsys(capsys)
    assert rc == 1
    assert errors >= 1


def test_check_keypoint_incoming_bbox_inverted_is_error(tmp_path: Path, capsys) -> None:
    root = _make_batch(tmp_path)
    bad = _good_payload("frame_0001")
    # Swap x1/x2 so x2 < x1 — must trip the "x1<x2 and y1<y2" rule.
    bad["wheels"][0]["bbox_xyxy"] = [50.0, 10.0, 10.0, 50.0]
    _write_anno(root / "annotations" / "frame_0001.json", bad)

    rc = _run_check(root)
    errors, _ = _read_errors_count_via_capsys(capsys)
    assert rc == 1
    assert errors >= 1


def test_check_keypoint_incoming_extra_point_key_is_error(
    tmp_path: Path, capsys
) -> None:
    root = _make_batch(tmp_path)
    bad = _good_payload("frame_0001")
    bad["wheels"][0]["points"]["extra"] = [20.0, 20.0]
    _write_anno(root / "annotations" / "frame_0001.json", bad)

    rc = _run_check(root)
    errors, _ = _read_errors_count_via_capsys(capsys)
    assert rc == 1
    assert errors >= 1


def test_check_keypoint_incoming_missing_point_key_is_error(
    tmp_path: Path, capsys
) -> None:
    root = _make_batch(tmp_path)
    bad = _good_payload("frame_0001")
    del bad["wheels"][0]["points"]["c_disc_bottom"]
    _write_anno(root / "annotations" / "frame_0001.json", bad)

    rc = _run_check(root)
    errors, _ = _read_errors_count_via_capsys(capsys)
    assert rc == 1
    assert errors >= 1


def test_check_keypoint_incoming_point_out_of_image_is_error(
    tmp_path: Path, capsys
) -> None:
    root = _make_batch(tmp_path)
    bad = _good_payload("frame_0001")
    bad["wheels"][0]["points"]["a"] = [-10.0, -10.0]
    _write_anno(root / "annotations" / "frame_0001.json", bad)

    rc = _run_check(root)
    errors, _ = _read_errors_count_via_capsys(capsys)
    assert rc == 1
    assert errors >= 1


def test_check_keypoint_incoming_point_outside_bbox_within_tolerance_is_warning(
    tmp_path: Path, capsys
) -> None:
    """Point that sits 4px outside the bbox edge (well under the 5px slack)
    must not produce an error. Point 6px outside should warn.

    We exercise the warning side here: place point a at x = x1 - 6 so it
    crosses the 5px slack threshold but stays inside the image bounds.
    """
    root = _make_batch(tmp_path)
    bad = _good_payload("frame_0001")
    # bbox is (10, 10, 50, 50). a.x = 4 is 6 px outside x1 = 10 -> WARNING.
    bad["wheels"][0]["points"]["a"] = [4.0, 30.0]
    _write_anno(root / "annotations" / "frame_0001.json", bad)

    rc = _run_check(root)
    errors, warnings = _read_errors_count_via_capsys(capsys)
    assert rc == 0
    assert errors == 0
    assert warnings >= 1


def test_check_keypoint_incoming_empty_wheels_is_ok(tmp_path: Path, capsys) -> None:
    root = tmp_path / "batch"
    _write_image(root / "images" / "frame_empty.jpg")
    _write_anno(
        root / "annotations" / "frame_empty.json",
        {"frame_id": "frame_empty", "image": "frame_empty.jpg", "wheels": []},
    )

    rc = _run_check(root)
    errors, _ = _read_errors_count_via_capsys(capsys)
    assert rc == 0
    assert errors == 0


def test_check_keypoint_incoming_missing_source_dirs_returns_2(
    tmp_path: Path,
) -> None:
    # tmp_path itself has no images/ or annotations/ subdir.
    rc = _run_check(tmp_path)
    assert rc == 2


# ---- bonus: orphan annotation surfaces as warning, not error ------------


def test_check_keypoint_incoming_orphan_annotation_is_warning(
    tmp_path: Path, capsys
) -> None:
    root = _make_batch(tmp_path)
    # Stray annotation with no image — must warn, not error.
    _write_anno(
        root / "annotations" / "frame_orphan.json",
        _good_payload("frame_orphan"),
    )

    rc = _run_check(root)
    errors, warnings = _read_errors_count_via_capsys(capsys)
    assert rc == 0
    assert errors == 0
    assert warnings >= 1
