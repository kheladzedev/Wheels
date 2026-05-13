"""Tests for the YOLO-pose converter's validation and split logic."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from convert_incoming_to_yolo import (
    DROP_REASON_KEYS,
    N_KEYPOINTS,
    assign_splits,
    format_label_line,
    main as convert_main,
    validate_and_convert_bbox,
    validate_and_convert_keypoints,
)


# ---- bbox validation ----------------------------------------------------


def test_bbox_normalized_to_unit_square():
    yolo, warn = validate_and_convert_bbox([10, 20, 110, 220], 200, 400, "x.jpg")
    assert warn is None
    cx, cy, w, h = yolo
    assert cx == 60 / 200 and cy == 120 / 400
    assert w == 100 / 200 and h == 200 / 400


def test_bbox_outside_image_is_clipped_with_warning():
    yolo, warn = validate_and_convert_bbox([-10, -20, 300, 500], 200, 400, "x.jpg")
    assert yolo is not None
    assert warn and "clipped" in warn


def test_bbox_with_inverted_corners_rejected():
    yolo, warn = validate_and_convert_bbox([100, 100, 50, 50], 200, 400, "x.jpg")
    assert yolo is None
    assert "invalid bbox order" in warn


# ---- keypoint validation ------------------------------------------------


def _good_kps():
    return [
        {"name": "rim_left", "xy": [50, 50], "visibility": 2},
        {"name": "rim_right", "xy": [50, 150], "visibility": 2},
        {"name": "disc_bottom", "xy": [50, 160], "visibility": 2},
    ]


def test_keypoints_normalized():
    out, warns = validate_and_convert_keypoints(_good_kps(), 200, 400, "x.jpg")
    assert warns == []
    assert len(out) == N_KEYPOINTS
    assert out[0] == (50 / 200, 50 / 400, 2)


def test_wrong_keypoint_count_rejected():
    out, warns = validate_and_convert_keypoints(_good_kps()[:2], 200, 400, "x.jpg")
    assert out is None
    assert any("must be a list of 3" in w for w in warns)


def test_visibility_zero_emits_zero_coords_regardless_of_xy():
    kps = _good_kps()
    kps[2] = {"name": "disc_bottom", "xy": [123, 456], "visibility": 0}
    out, _ = validate_and_convert_keypoints(kps, 200, 400, "x.jpg")
    assert out[2] == (0.0, 0.0, 0)


def test_invalid_visibility_flag_rejected():
    kps = _good_kps()
    kps[0] = {"name": "rim_left", "xy": [50, 50], "visibility": 7}
    out, _ = validate_and_convert_keypoints(kps, 200, 400, "x.jpg")
    assert out is None


def test_keypoint_outside_image_clipped_with_warning():
    kps = _good_kps()
    kps[2] = {"name": "disc_bottom", "xy": [300, 500], "visibility": 2}
    out, warns = validate_and_convert_keypoints(kps, 200, 400, "x.jpg")
    assert out is not None
    assert out[2] == (200 / 200, 400 / 400, 2)
    assert any("clipped" in w for w in warns)


# ---- label line formatting ---------------------------------------------


def test_label_line_has_expected_field_count():
    bbox = (0.5, 0.5, 0.1, 0.1)
    kps = [(0.5, 0.4, 2), (0.5, 0.6, 2), (0.5, 0.65, 2)]
    line = format_label_line(0, bbox, kps)
    parts = line.split()
    assert len(parts) == 5 + N_KEYPOINTS * 3
    assert parts[0] == "0"


# ---- split assignment ---------------------------------------------------


def test_random_split_respects_val_ratio():
    images = [Path(f"img_{i:03d}.jpg") for i in range(100)]
    assignment, info = assign_splits(images, val_ratio=0.2, seed=42, scene_regex=None)
    n_val = sum(1 for v in assignment.values() if v == "val")
    assert n_val == 20
    assert info["split_strategy"] == "random_per_image"


def test_scene_regex_keeps_scene_images_together():
    images = [
        Path(f"scene_{s:02d}_frame_{f:03d}.jpg") for s in range(5) for f in range(4)
    ]
    assignment, info = assign_splits(
        images,
        val_ratio=0.4,
        seed=42,
        scene_regex=r"^(scene_\d{2})_.*$",
    )
    assert info["split_strategy"] == "scene_regex"
    # Group by scene; every scene must be entirely in one split.
    scene_splits: dict[str, set[str]] = {}
    for p, split in assignment.items():
        scene = p.stem.rsplit("_frame_", 1)[0]
        scene_splits.setdefault(scene, set()).add(split)
    for splits in scene_splits.values():
        assert len(splits) == 1, "scene leaked across splits"


def test_unmatched_stems_become_singleton_scenes():
    images = [Path("scene_01_frame_0.jpg"), Path("anomaly_no_scene.jpg")]
    _, info = assign_splits(
        images, val_ratio=0.5, seed=1, scene_regex=r"^(scene_\d{2})_.*$"
    )
    assert "anomaly_no_scene" in info["unmatched_stems"]


# ---- end-to-end: --min-side filtering and drop_reasons report -----------
#
# These tests drive main() in-process by patching sys.argv. tmp_path provides
# both the source-root and dataset-root, so tests never depend on real data
# under data/wheel_dataset/.


def _good_anno(img_w: int, img_h: int) -> dict:
    """Build a valid annotation for an image of size img_w x img_h."""
    return {
        "objects": [
            {
                "class_name": "wheel",
                "bbox_xyxy": [10, 10, img_w - 10, img_h - 10],
                "keypoints": [
                    {"name": "rim_left", "xy": [20, 20], "visibility": 2},
                    {"name": "rim_right", "xy": [30, 30], "visibility": 2},
                    {"name": "disc_bottom", "xy": [40, 40], "visibility": 2},
                ],
            }
        ]
    }


def _write_image(path: Path, w: int, h: int) -> None:
    """Write a synthetic JPEG of size (w, h) at `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    ok = cv2.imwrite(str(path), img)
    assert ok, f"cv2.imwrite failed for {path}"


def _write_anno(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_source(tmp_path: Path, sources: dict[str, tuple[int, int]]) -> Path:
    """Build an incoming/<source_name>/ layout from a {stem: (w, h)} map.

    Each stem gets a matching valid annotation JSON.
    """
    source_root = tmp_path / "incoming" / "src"
    images_dir = source_root / "images"
    annos_dir = source_root / "annotations"
    for stem, (w, h) in sources.items():
        _write_image(images_dir / f"{stem}.jpg", w, h)
        _write_anno(annos_dir / f"{stem}.json", _good_anno(w, h))
    return source_root


def _run_main(source_root: Path, dataset_root: Path, *extra: str) -> int:
    argv = [
        "convert_incoming_to_yolo.py",
        "--source-root",
        str(source_root),
        "--dataset-root",
        str(dataset_root),
        "--overwrite",
        *extra,
    ]
    with patch.object(sys, "argv", argv):
        return convert_main()


def _load_report(dataset_root: Path) -> dict:
    return json.loads(
        (dataset_root / "metadata" / "conversion_report.json").read_text(
            encoding="utf-8"
        )
    )


def test_min_side_drops_small_images_with_correct_reason_key(tmp_path: Path) -> None:
    source_root = _make_source(
        tmp_path,
        {"too_small": (100, 300), "ok": (800, 600)},
    )
    dataset_root = tmp_path / "dataset"

    rc = _run_main(source_root, dataset_root)
    assert rc == 0

    report = _load_report(dataset_root)
    assert report["drop_reasons"]["image_too_small"] == 1
    assert report["converted"] == 1

    # Find the small-image skip entry and confirm the size + reason_key.
    small_entries = [
        e for e in report["skipped_details"] if e.get("reason_key") == "image_too_small"
    ]
    assert len(small_entries) == 1
    assert small_entries[0]["size"] == [100, 300]
    assert "too_small.jpg" in small_entries[0]["image"]

    # The 800x600 image went through the pipeline — pick whichever split it
    # landed in (assign_splits is random by default).
    labels = list((dataset_root / "labels").rglob("*.txt"))
    assert len(labels) == 1
    assert labels[0].stem.endswith("__ok")


def test_min_side_boundary_kept_at_max_equal_and_dropped_at_max_minus_one(
    tmp_path: Path,
) -> None:
    # Default --min-side is 480. The filter is strict `<`, so max(w, h) == 480
    # must be kept, and max(w, h) == 479 must be dropped.
    source_root = _make_source(
        tmp_path,
        {"keep_equal": (480, 100), "drop_minus_one": (479, 100)},
    )
    dataset_root = tmp_path / "dataset"

    rc = _run_main(source_root, dataset_root)
    assert rc == 0

    report = _load_report(dataset_root)
    assert report["converted"] == 1
    assert report["drop_reasons"]["image_too_small"] == 1

    small_entries = [
        e for e in report["skipped_details"] if e.get("reason_key") == "image_too_small"
    ]
    assert len(small_entries) == 1
    assert "drop_minus_one.jpg" in small_entries[0]["image"]


def test_min_side_zero_disables_filter(tmp_path: Path) -> None:
    source_root = _make_source(
        tmp_path,
        {"too_small": (100, 300), "ok": (800, 600)},
    )
    dataset_root = tmp_path / "dataset"

    rc = _run_main(source_root, dataset_root, "--min-side", "0")
    assert rc == 0

    report = _load_report(dataset_root)
    assert report["drop_reasons"]["image_too_small"] == 0
    assert report["converted"] == 2


def test_drop_reasons_dict_has_all_keys_even_when_zero(tmp_path: Path) -> None:
    source_root = _make_source(
        tmp_path,
        {"a": (640, 480), "b": (640, 480), "c": (640, 480)},
    )
    dataset_root = tmp_path / "dataset"

    rc = _run_main(source_root, dataset_root)
    assert rc == 0

    report = _load_report(dataset_root)
    assert set(report["drop_reasons"].keys()) == set(DROP_REASON_KEYS)
    assert all(v == 0 for v in report["drop_reasons"].values())
    assert report["skipped_details"] == []
    # Invariant: every skipped entry's reason_key must be a registered key.
    # Vacuously true on a clean run, but documents the contract for future tests.
    for entry in report["skipped_details"]:
        assert entry["reason_key"] in report["drop_reasons"]


def test_skipped_details_has_reason_key_alongside_human_reason(tmp_path: Path) -> None:
    # Build a valid image but no matching annotation -> missing_annotation drop.
    source_root = tmp_path / "incoming" / "src"
    _write_image(source_root / "images" / "lonely.jpg", 640, 480)
    # Provide one valid pair so converted > 0 and main() returns 0.
    _write_image(source_root / "images" / "good.jpg", 640, 480)
    _write_anno(source_root / "annotations" / "good.json", _good_anno(640, 480))

    dataset_root = tmp_path / "dataset"
    rc = _run_main(source_root, dataset_root)
    assert rc == 0

    report = _load_report(dataset_root)
    assert report["drop_reasons"]["missing_annotation"] == 1

    missing = [
        e
        for e in report["skipped_details"]
        if e.get("reason_key") == "missing_annotation"
    ]
    assert len(missing) == 1
    # Human-readable text preserved verbatim.
    assert missing[0]["reason"] == "missing annotation JSON"
    assert "lonely.jpg" in missing[0]["image"]
