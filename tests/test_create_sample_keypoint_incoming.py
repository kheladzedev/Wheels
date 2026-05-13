"""End-to-end smoke test: generator output passes the validator.

This is the cheap contract check between
`src/create_sample_keypoint_incoming.py` and
`src/check_keypoint_incoming.py`. If one drifts, this fails.
"""

from __future__ import annotations

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
