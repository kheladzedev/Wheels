from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import sanitize_unreal_export as sanitize  # noqa: E402


def _write_image(path: Path, size: tuple[int, int] = (100, 100)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


def _write_object(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


VALID_WITH_TOP_POINTS = "\n".join(
    [
        '{name:"Right",XY:20,90},',
        '{name:"Left",XY:80,90},',
        '{name:"Center",XY:50,65},',
        '{name:"RightTop",XY:20,20},',
        '{name:"LeftTop",XY:80,20},',
    ]
)

VALID_WITH_PLUGIN_BBOX = "\n".join(
    [
        '{name:"Right",XY:20,90},',
        '{name:"Left",XY:80,90},',
        '{name:"Center",XY:50,65},',
        '{name:"WheelBBox",XYXY:15,10,85,95},',
    ]
)

ALL_ZERO = "\n".join(
    [
        '{name:"Right",XY:0,0},',
        '{name:"Left",XY:0,0},',
        '{name:"Center",XY:0,0},',
    ]
)


def test_sanitize_keeps_valid_synthetic_debug_and_drops_empty_frame(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "sanitized"
    _write_image(raw / "Images" / "001.jpg")
    _write_image(raw / "Images" / "002.jpg")
    _write_object(raw / "keyPoint" / "001" / "0.txt", VALID_WITH_TOP_POINTS)
    _write_object(raw / "keyPoint" / "001" / "1.txt", ALL_ZERO)
    _write_object(raw / "keyPoint" / "002" / "0.txt", ALL_ZERO)

    rc = sanitize.main(
        [
            "--source-root",
            str(raw),
            "--out-root",
            str(out),
            "--allow-synthetic-bbox",
            "--overwrite",
            "--right-left-mapping",
            "confirmed",
        ]
    )

    assert rc == 0
    assert (out / "Images" / "001.jpg").is_file()
    assert not (out / "Images" / "002.jpg").exists()
    assert (out / "keyPoint" / "001" / "0.txt").read_text() == VALID_WITH_TOP_POINTS
    assert not (out / "keyPoint" / "001" / "1.txt").exists()

    report = json.loads((out / "metadata" / "sanitize_report.json").read_text())
    assert report["frames_seen"] == 2
    assert report["frames_kept"] == 1
    assert report["keypoint_files_seen"] == 3
    assert report["keypoint_files_kept"] == 1
    assert report["empty_frames_dropped"] == 1
    assert report["drop_counts"]["all_zero"] == 2
    assert report["bbox_source_counts"]["synthesized_by_adapter"] == 1
    assert report["training_approved"] is False
    assert report["requires_human_preview"] is True


def test_sanitize_requires_bbox_unless_debug_fallback_enabled(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "sanitized"
    _write_image(raw / "Images" / "001.jpg")
    _write_object(raw / "keyPoint" / "001" / "0.txt", VALID_WITH_TOP_POINTS)

    rc = sanitize.main(
        [
            "--source-root",
            str(raw),
            "--out-root",
            str(out),
            "--overwrite",
            "--right-left-mapping",
            "confirmed",
        ]
    )

    assert rc == 1
    report = json.loads((out / "metadata" / "sanitize_report.json").read_text())
    assert report["status"] == "FAIL_NO_VALID_OBJECTS"
    assert report["drop_counts"]["missing_bbox"] == 1
    assert report["keypoint_files_kept"] == 0


def test_sanitize_keeps_plugin_bbox_without_synthetic_fallback(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "sanitized"
    _write_image(raw / "Images" / "001.jpg")
    _write_object(raw / "keyPoint" / "001" / "0.txt", VALID_WITH_PLUGIN_BBOX)

    rc = sanitize.main(
        [
            "--source-root",
            str(raw),
            "--out-root",
            str(out),
            "--overwrite",
            "--right-left-mapping",
            "confirmed",
        ]
    )

    assert rc == 0
    assert (out / "keyPoint" / "001" / "0.txt").is_file()
    report = json.loads((out / "metadata" / "sanitize_report.json").read_text())
    assert report["bbox_source_counts"]["plugin_provided"] == 1
    assert report["bbox_source_counts"]["synthesized_by_adapter"] == 0
