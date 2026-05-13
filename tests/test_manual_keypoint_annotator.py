"""Headless tests for the pure helpers in manual_keypoint_annotator.

The GUI loop itself (OpenCV mouse + waitKey) is not unit-tested — but
every function that touches the on-disk shape (annotation JSON,
source_info, output packaging) is, so a GUI regression cannot silently
poison the dataset format.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from manual_keypoint_annotator import (
    ANNOTATION_METHOD,
    CLICK_LABELS,
    IMAGE_EXTS,
    SOURCE_NAME,
    SOURCE_NOTE,
    build_annotation,
    build_source_info,
    build_wheel,
    display_to_image_coord,
    list_images,
    normalize_bbox,
    package_output,
    scale_for_display,
    write_annotation,
    write_source_info,
)


# ---- bbox normalisation -------------------------------------------------


def test_normalize_bbox_canonical_order():
    assert normalize_bbox((10, 20), (110, 220)) == [10.0, 20.0, 110.0, 220.0]


def test_normalize_bbox_reversed_corners():
    """User dragged from bottom-right to top-left — must still yield xyxy."""
    assert normalize_bbox((110, 220), (10, 20)) == [10.0, 20.0, 110.0, 220.0]


def test_normalize_bbox_mixed_corners():
    """One axis flipped (top-right → bottom-left)."""
    assert normalize_bbox((110, 20), (10, 220)) == [10.0, 20.0, 110.0, 220.0]


def test_normalize_bbox_returns_floats():
    bbox = normalize_bbox((1, 2), (3, 4))
    assert all(isinstance(v, float) for v in bbox)


def test_normalize_bbox_idempotent_on_canonical_input():
    bbox = normalize_bbox((10, 20), (110, 220))
    again = normalize_bbox((bbox[0], bbox[1]), (bbox[2], bbox[3]))
    assert bbox == again


# ---- wheel + annotation builders ----------------------------------------


def test_build_wheel_matches_plugin_contract_shape():
    w = build_wheel(
        bbox_xyxy=[10, 20, 110, 220],
        point_a=[15, 30],
        point_b=[105, 30],
        point_c=[60, 215],
    )
    # Top-level keys exactly {bbox_xyxy, points} — same as the plugin spec.
    assert set(w.keys()) == {"bbox_xyxy", "points"}
    assert w["bbox_xyxy"] == [10.0, 20.0, 110.0, 220.0]
    # Points contain exactly the three contract keys, no extras.
    assert set(w["points"].keys()) == {"a", "b", "c_disc_bottom"}
    assert w["points"]["a"] == [15.0, 30.0]
    assert w["points"]["b"] == [105.0, 30.0]
    assert w["points"]["c_disc_bottom"] == [60.0, 215.0]


def test_build_wheel_coerces_ints_to_floats():
    """Plugin contract: coords are JSON numbers; converter expects floats."""
    w = build_wheel([0, 0, 1, 1], [0, 0], [1, 1], [0, 1])
    for v in w["bbox_xyxy"]:
        assert isinstance(v, float)
    for name in ("a", "b", "c_disc_bottom"):
        for v in w["points"][name]:
            assert isinstance(v, float)


def test_build_annotation_has_required_top_level_keys():
    anno = build_annotation("img_001", "img_001.jpg", [])
    assert set(anno.keys()) == {"frame_id", "image", "wheels"}
    assert anno["frame_id"] == "img_001"
    assert anno["image"] == "img_001.jpg"
    assert anno["wheels"] == []


def test_build_annotation_with_wheels_preserves_order():
    w1 = build_wheel([0, 0, 10, 10], [1, 1], [9, 1], [5, 9])
    w2 = build_wheel([20, 20, 30, 30], [21, 21], [29, 21], [25, 29])
    anno = build_annotation("img_x", "img_x.jpg", [w1, w2])
    assert anno["wheels"] == [w1, w2]


def test_build_annotation_copies_wheels_list_defensively():
    """Mutating the input wheels list afterwards must not change anno."""
    wheels = [build_wheel([0, 0, 1, 1], [0, 0], [1, 1], [0, 1])]
    anno = build_annotation("x", "x.jpg", wheels)
    wheels.append({"junk": True})
    assert len(anno["wheels"]) == 1


# ---- source_info --------------------------------------------------------


def test_build_source_info_has_required_fields():
    info = build_source_info()
    assert info["source_name"] == SOURCE_NAME == "manual_real"
    assert info["note"] == SOURCE_NOTE
    assert info["annotation_method"] == ANNOTATION_METHOD == "manual clicks"
    # The triple — these are the goal-required fields, none extra.
    assert set(info.keys()) == {"source_name", "note", "annotation_method"}


def test_write_source_info_writes_valid_json(tmp_path: Path):
    out = write_source_info(tmp_path / "metadata")
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload == build_source_info()


def test_write_source_info_creates_dir_if_missing(tmp_path: Path):
    nested = tmp_path / "a" / "b" / "metadata"
    out = write_source_info(nested)
    assert out.is_file()
    assert out.parent == nested


# ---- annotation writer --------------------------------------------------


def test_write_annotation_writes_plugin_shaped_json(tmp_path: Path):
    anno = build_annotation(
        "img_001",
        "img_001.jpg",
        [build_wheel([10, 20, 110, 220], [15, 30], [105, 30], [60, 215])],
    )
    out = tmp_path / "annotations" / "img_001.json"
    write_annotation(out, anno)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload == anno
    # Plugin contract: top-level keys + per-wheel keys.
    assert set(payload.keys()) == {"frame_id", "image", "wheels"}
    w = payload["wheels"][0]
    assert set(w.keys()) == {"bbox_xyxy", "points"}
    assert set(w["points"].keys()) == {"a", "b", "c_disc_bottom"}


def test_write_annotation_atomic_no_tmp_leftover(tmp_path: Path):
    """Verify the writer cleans up its `.tmp` companion."""
    out = tmp_path / "a.json"
    write_annotation(out, {"x": 1})
    assert out.is_file()
    leftovers = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_write_annotation_overwrites_existing(tmp_path: Path):
    out = tmp_path / "a.json"
    write_annotation(out, {"version": 1})
    write_annotation(out, {"version": 2})
    assert json.loads(out.read_text(encoding="utf-8")) == {"version": 2}


# ---- list_images --------------------------------------------------------


def _touch_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((48, 64, 3), 128, dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


def test_list_images_returns_sorted_filtered_files(tmp_path: Path):
    images_dir = tmp_path / "images"
    _touch_image(images_dir / "b.jpg")
    _touch_image(images_dir / "a.png")
    (images_dir / "junk.txt").write_text("not an image")
    out = list_images(images_dir)
    assert [p.name for p in out] == ["a.png", "b.jpg"]


def test_list_images_filters_only_known_extensions(tmp_path: Path):
    images_dir = tmp_path / "images"
    for ext in IMAGE_EXTS:
        _touch_image(images_dir / f"sample{ext}")
    (images_dir / "weird.tiff").write_text("not picked up")
    out = list_images(images_dir)
    assert {p.name for p in out} == {f"sample{ext}" for ext in IMAGE_EXTS}


def test_list_images_honours_start_index(tmp_path: Path):
    images_dir = tmp_path / "images"
    for i in range(5):
        _touch_image(images_dir / f"img_{i}.jpg")
    out = list_images(images_dir, start_index=3)
    assert [p.name for p in out] == ["img_3.jpg", "img_4.jpg"]


def test_list_images_negative_start_index_treated_as_zero(tmp_path: Path):
    images_dir = tmp_path / "images"
    for i in range(3):
        _touch_image(images_dir / f"img_{i}.jpg")
    out = list_images(images_dir, start_index=-5)
    assert len(out) == 3


def test_list_images_missing_dir_returns_empty(tmp_path: Path):
    assert list_images(tmp_path / "nope") == []


# ---- display-scale round trip -------------------------------------------


def test_scale_for_display_no_scale_when_under_max():
    assert scale_for_display(720, 1280, max_side=1280) == 1.0


def test_scale_for_display_scales_down_when_over_max():
    # 4032x3024 phone shot → max_side 1280 → factor 1280/4032.
    factor = scale_for_display(3024, 4032, max_side=1280)
    assert factor == pytest.approx(1280 / 4032)
    # Scaled longer side fits under cap.
    assert max(3024 * factor, 4032 * factor) <= 1280 + 1e-9


def test_scale_for_display_zero_dimension_defaults_to_one():
    assert scale_for_display(0, 0, max_side=1280) == 1.0


def test_display_to_image_coord_round_trip_at_unit_scale():
    assert display_to_image_coord((123, 456), 1.0) == [123.0, 456.0]


def test_display_to_image_coord_round_trip_at_half_scale():
    """If a 4032-wide image was displayed at half-scale, a click at
    (1000, 500) on screen must map to (2000, 1000) on the original.
    """
    img_xy = display_to_image_coord((1000, 500), 0.5)
    assert img_xy == [2000.0, 1000.0]


def test_display_to_image_coord_rejects_zero_scale():
    with pytest.raises(ValueError):
        display_to_image_coord((1, 1), 0.0)


# ---- package_output -----------------------------------------------------


def test_package_output_copies_pairs_only(tmp_path: Path):
    """Only images that have a sibling annotation by stem are mirrored
    into the output bundle — unfinished work stays out.
    """
    images_dir = tmp_path / "manual_real" / "images"
    annos_dir = tmp_path / "manual_real" / "annotations"
    out_root = tmp_path / "incoming" / "manual_real"

    # Two annotated, one un-annotated (still being worked on).
    for stem in ("a", "b", "c"):
        _touch_image(images_dir / f"{stem}.jpg")
    for stem in ("a", "b"):
        write_annotation(
            annos_dir / f"{stem}.json",
            build_annotation(stem, f"{stem}.jpg", []),
        )

    summary = package_output(images_dir, annos_dir, out_root)

    out_images = sorted(p.name for p in (out_root / "images").iterdir())
    out_annos = sorted(p.name for p in (out_root / "annotations").iterdir())
    assert out_images == ["a.jpg", "b.jpg"]
    assert out_annos == ["a.json", "b.json"]
    assert summary["images_copied"] == 2
    assert summary["annotations_copied"] == 2


def test_package_output_writes_source_info(tmp_path: Path):
    images_dir = tmp_path / "manual_real" / "images"
    annos_dir = tmp_path / "manual_real" / "annotations"
    out_root = tmp_path / "incoming" / "manual_real"
    _touch_image(images_dir / "a.jpg")
    write_annotation(annos_dir / "a.json", build_annotation("a", "a.jpg", []))

    summary = package_output(images_dir, annos_dir, out_root)
    source_info_path = Path(summary["source_info"])
    assert source_info_path.is_file()
    payload = json.loads(source_info_path.read_text(encoding="utf-8"))
    assert payload == build_source_info()
    # Path is inside output_root/metadata.
    assert source_info_path.parent == out_root / "metadata"


def test_package_output_creates_layout_when_dirs_missing(tmp_path: Path):
    """Edge case — both source dirs empty / missing. Still write metadata."""
    summary = package_output(
        tmp_path / "no_images",
        tmp_path / "no_annos",
        tmp_path / "out",
    )
    assert summary["images_copied"] == 0
    assert summary["annotations_copied"] == 0
    assert (tmp_path / "out" / "metadata" / "source_info.json").is_file()
    assert (tmp_path / "out" / "images").is_dir()
    assert (tmp_path / "out" / "annotations").is_dir()


def test_package_output_emits_plugin_compatible_bundle(tmp_path: Path):
    """End-to-end: package_output's bundle should pass through
    check_keypoint_incoming.py without raising. Light check — just verify
    a downstream consumer can read the JSON without surprises.
    """
    images_dir = tmp_path / "manual_real" / "images"
    annos_dir = tmp_path / "manual_real" / "annotations"
    out_root = tmp_path / "incoming" / "manual_real"

    _touch_image(images_dir / "frame_001.jpg")
    wheel = build_wheel(
        bbox_xyxy=normalize_bbox((10, 10), (50, 50)),
        point_a=[15.0, 25.0],
        point_b=[45.0, 25.0],
        point_c=[30.0, 48.0],
    )
    write_annotation(
        annos_dir / "frame_001.json",
        build_annotation("frame_001", "frame_001.jpg", [wheel]),
    )

    package_output(images_dir, annos_dir, out_root)

    payload = json.loads(
        (out_root / "annotations" / "frame_001.json").read_text(encoding="utf-8")
    )
    # Plugin contract shape, verbatim.
    assert set(payload.keys()) == {"frame_id", "image", "wheels"}
    assert payload["frame_id"] == "frame_001"
    assert payload["image"] == "frame_001.jpg"
    assert len(payload["wheels"]) == 1
    w = payload["wheels"][0]
    assert set(w.keys()) == {"bbox_xyxy", "points"}
    assert set(w["points"].keys()) == {"a", "b", "c_disc_bottom"}


# ---- constants ----------------------------------------------------------


def test_click_labels_describes_all_five_clicks():
    """If a refactor changes the click sequence, this test fails first.

    Wording is the canonical UI text under the 2026-05-14 spec revision
    (A/B = floor / raycast points, NOT rim edges).
    """
    assert len(CLICK_LABELS) == 5
    assert CLICK_LABELS[0] == "bbox corner 1"
    assert CLICK_LABELS[1] == "bbox corner 2"
    assert CLICK_LABELS[2] == "A floor/raycast point"
    assert CLICK_LABELS[3] == "B floor/raycast point"
    assert CLICK_LABELS[4] == "C disc bottom"


def test_click_labels_ab_describe_floor_raycast():
    """A and B labels must mention floor + raycast — the load-bearing
    semantics under the 2026-05-14 spec revision.
    """
    a_label = CLICK_LABELS[2].lower()
    b_label = CLICK_LABELS[3].lower()
    assert "floor" in a_label and "raycast" in a_label, (
        f"A label {CLICK_LABELS[2]!r} must mention floor + raycast"
    )
    assert "floor" in b_label and "raycast" in b_label, (
        f"B label {CLICK_LABELS[3]!r} must mention floor + raycast"
    )


def test_click_labels_c_describes_disc_bottom():
    """C label must mention disc + bottom — the lower visible metal-rim /
    disc point under the contract.
    """
    c_label = CLICK_LABELS[4].lower()
    assert "disc" in c_label and "bottom" in c_label, (
        f"C label {CLICK_LABELS[4]!r} must mention disc + bottom"
    )


def test_click_labels_do_not_use_rim_left_rim_right():
    """Pin the 2026-05-14 semantic revision: A/B must not be described
    as rim edges in user-facing wording. The legacy literal strings
    `rim_left`/`rim_right` in postprocess_wheels are out of scope.
    """
    forbidden = (
        "rim_left",
        "rim_right",
        "left rim",
        "right rim",
        "rim left",
        "rim right",
        "metal rim left",
        "metal rim right",
        "left point of metal rim",
        "right point of metal rim",
    )
    for i, label in enumerate(CLICK_LABELS):
        lowered = label.lower()
        for needle in forbidden:
            assert needle.lower() not in lowered, (
                f"CLICK_LABELS[{i}] = {label!r} contains forbidden rim "
                f"wording {needle!r}; A/B are floor / raycast points "
                "under the 2026-05-14 contract."
            )
