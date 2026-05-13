"""End-to-end smoke test: generator output passes the validator.

This is the cheap contract check between
`src/create_sample_keypoint_incoming.py` and
`src/check_keypoint_incoming.py`. If one drifts, this fails.

Also pins the 2026-05-14 floor-ray semantics for A/B: the generator
must place them in the lower band of the bbox near the wheel
footprint, below the disc-bottom approximation, not on the rim
centerline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

from check_keypoint_incoming import main as check_main
from create_sample_keypoint_incoming import main as generator_main


def test_generator_produces_valid_batch(tmp_path: Path) -> None:
    out_root = tmp_path / "gen"
    gen_argv = [
        "create_sample_keypoint_incoming.py",
        "--count",
        "5",
        "--output-root",
        str(out_root),
        "--seed",
        "7",
        "--overwrite",
    ]
    with patch.object(sys, "argv", gen_argv):
        assert generator_main() == 0

    # Bucket the files so the rest of the assertions don't depend on the
    # generator's print output.
    images = sorted((out_root / "images").glob("*.jpg"))
    annos = sorted((out_root / "annotations").glob("*.json"))
    meta = out_root / "metadata" / "source_info.json"
    assert len(images) == 5
    assert len(annos) == 5
    assert meta.is_file()

    check_argv = [
        "check_keypoint_incoming.py",
        "--source-root",
        str(out_root),
    ]
    with patch.object(sys, "argv", check_argv):
        rc = check_main()
    assert rc == 0


def test_generator_count_zero_writes_metadata_only(tmp_path: Path) -> None:
    """--count 0 must still produce a metadata file (per the brief).

    No images, no annotations — but exit 0 and source_info.json present.
    """
    out_root = tmp_path / "gen_zero"
    gen_argv = [
        "create_sample_keypoint_incoming.py",
        "--count",
        "0",
        "--output-root",
        str(out_root),
        "--seed",
        "7",
        "--overwrite",
    ]
    with patch.object(sys, "argv", gen_argv):
        assert generator_main() == 0

    assert (out_root / "metadata" / "source_info.json").is_file()
    assert sorted((out_root / "images").glob("*")) == []
    assert sorted((out_root / "annotations").glob("*")) == []


def test_generated_a_b_follow_floor_ray_semantics(tmp_path: Path) -> None:
    """A/B must sit in the lower band of the bbox (footprint area), below
    the disc-bottom approximation and clearly off the rim centerline.

    Pinned values:
      - a.y and b.y are below the bbox midline (lower half) AND below c.y;
      - a is left of b;
      - c (c_disc_bottom) is inside the bbox, below the midline (rim's
        lower edge), and strictly above the bbox bottom (it's on the disc,
        not on the floor).
    """
    out_root = tmp_path / "gen"
    gen_argv = [
        "create_sample_keypoint_incoming.py",
        "--count",
        "12",
        "--output-root",
        str(out_root),
        "--seed",
        "3",
        "--overwrite",
    ]
    with patch.object(sys, "argv", gen_argv):
        assert generator_main() == 0

    wheels_seen = 0
    for anno_path in sorted((out_root / "annotations").glob("*.json")):
        payload = json.loads(anno_path.read_text(encoding="utf-8"))
        for wheel in payload["wheels"]:
            wheels_seen += 1
            x1, y1, x2, y2 = wheel["bbox_xyxy"]
            mid_y = (y1 + y2) / 2.0
            a_x, a_y = wheel["points"]["a"]
            b_x, b_y = wheel["points"]["b"]
            c_x, c_y = wheel["points"]["c_disc_bottom"]

            # A/B in the lower half of the bbox — not on the rim centerline.
            assert a_y > mid_y, (
                f"a.y={a_y} must be below bbox midline {mid_y}; "
                f"bbox={wheel['bbox_xyxy']}"
            )
            assert b_y > mid_y, (
                f"b.y={b_y} must be below bbox midline {mid_y}; "
                f"bbox={wheel['bbox_xyxy']}"
            )
            # A/B below the disc-bottom (footprint sits below the disc).
            assert a_y > c_y, f"a.y={a_y} must be below c_disc_bottom.y={c_y}"
            assert b_y > c_y, f"b.y={b_y} must be below c_disc_bottom.y={c_y}"
            # A on the left, B on the right.
            assert a_x < b_x, f"a.x={a_x} must be left of b.x={b_x}"
            # All three points inside the bbox (strict, no 5px slack — the
            # generator should produce clean points).
            for name, (px, py) in (
                ("a", (a_x, a_y)),
                ("b", (b_x, b_y)),
                ("c_disc_bottom", (c_x, c_y)),
            ):
                assert x1 <= px <= x2 and y1 <= py <= y2, (
                    f"point {name}=({px},{py}) outside bbox [{x1},{y1},{x2},{y2}]"
                )
            # c_disc_bottom sits in the rim region: below the midline but
            # strictly above the bbox bottom (it's on the disc, not on the
            # floor / contact line).
            assert c_y > mid_y, (
                f"c_disc_bottom.y={c_y} must be below bbox midline {mid_y}"
            )
            assert c_y < y2, f"c_disc_bottom.y={c_y} must stay above bbox bottom {y2}"

    assert wheels_seen > 0, "expected at least one wheel across 12 frames"


def test_generator_refuses_to_clobber_without_overwrite(tmp_path: Path) -> None:
    """Without --overwrite, a non-empty output root must fail with exit 1."""
    out_root = tmp_path / "gen_exists"
    out_root.mkdir()
    (out_root / "something.txt").write_text("not empty", encoding="utf-8")

    gen_argv = [
        "create_sample_keypoint_incoming.py",
        "--count",
        "1",
        "--output-root",
        str(out_root),
    ]
    with patch.object(sys, "argv", gen_argv):
        assert generator_main() == 1
    # The existing file must still be there — we refused to touch the tree.
    assert (out_root / "something.txt").is_file()
