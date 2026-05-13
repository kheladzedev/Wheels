"""Tests for the pure helpers in src/infer_batch.py.

Pure-function tests only: NO ``YOLO()``, no ``cv2.VideoCapture``, no
``model.predict``. The YOLO/cv2 path lives inside ``main()`` and is
exercised by the smoke test in the CLI, not here.

Covers source-type detection, frame-id formatting, timestamp arithmetic
(including the zero-fps guard), and the image-listing helper's sorting
and extension filtering.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from infer_batch import (
    DEFAULT_IMAGE_FPS,
    IMAGE_EXTS,
    frame_id_for_image,
    frame_id_for_video,
    iter_image_paths,
    resolve_source_type,
    timestamp_for_image_index,
    timestamp_for_video,
)


# ---------------------------------------------------------------------------
# resolve_source_type
# ---------------------------------------------------------------------------


def test_resolve_source_type_directory(tmp_path: Path):
    # A real directory must be classified as ``directory`` regardless of
    # whether it contains anything — the caller is responsible for the
    # "directory was empty" message, not us.
    assert resolve_source_type(tmp_path) == "directory"


@pytest.mark.parametrize("ext", [".mp4", ".mov", ".avi", ".mkv"])
def test_resolve_source_type_video_extensions(tmp_path: Path, ext: str):
    # All four documented video extensions must be recognised — case
    # mismatch is handled separately below.
    vid = tmp_path / f"clip{ext}"
    vid.write_bytes(b"")  # empty, but it exists as a file
    assert resolve_source_type(vid) == "video"


def test_resolve_source_type_video_extension_case_insensitive(tmp_path: Path):
    # Uppercase suffix (``.MP4``) is still a video — phones and old
    # cameras love SHOUTY extensions.
    vid = tmp_path / "clip.MP4"
    vid.write_bytes(b"")
    assert resolve_source_type(vid) == "video"


def test_resolve_source_type_unknown_extension(tmp_path: Path):
    # A real file with an extension we don't grok (e.g. ``.txt``) is
    # ``unknown`` — we don't silently treat it as a video.
    txt = tmp_path / "notes.txt"
    txt.write_text("not a video", encoding="utf-8")
    assert resolve_source_type(txt) == "unknown"


def test_resolve_source_type_missing_path_is_unknown(tmp_path: Path):
    # Non-existent path: not a directory, not a file. Must NOT be flagged
    # as ``video`` even if the would-be suffix is ``.mp4`` — there's
    # nothing to infer from.
    missing = tmp_path / "ghost.mp4"
    assert resolve_source_type(missing) == "unknown"


# ---------------------------------------------------------------------------
# frame_id_for_image / frame_id_for_video
# ---------------------------------------------------------------------------


def test_frame_id_for_image_passthrough():
    # The stem flows through unchanged today — this test pins that
    # behaviour so a future "session_id prefix" change is intentional, not
    # accidental.
    assert frame_id_for_image("sample_0007") == "sample_0007"
    assert frame_id_for_image("anything") == "anything"


def test_frame_id_for_video_format():
    # Zero-padded six-digit frame index with the video stem prefix.
    # Picking 42 as a non-boundary, non-zero index so a regression in the
    # padding format (e.g. switching to ``%04d``) is caught.
    assert frame_id_for_video("myvid", 42) == "myvid_frame_000042"


def test_frame_id_for_video_zero_index():
    # Index 0 must still be zero-padded to six digits (not "0" or "00000").
    assert frame_id_for_video("myvid", 0) == "myvid_frame_000000"


# ---------------------------------------------------------------------------
# timestamp_for_video / timestamp_for_image_index
# ---------------------------------------------------------------------------


def test_timestamp_for_video_basic():
    # Index 30 at 30 fps → 1.0 s. The classic sanity check.
    assert timestamp_for_video(30, 30.0) == pytest.approx(1.0)


def test_timestamp_for_video_non_integer_fps():
    # 29.97 fps (NTSC) is realistic input — make sure we don't truncate
    # to int somewhere.
    ts = timestamp_for_video(30, 29.97)
    assert ts == pytest.approx(30 / 29.97)


def test_timestamp_for_video_zero_fps_raises():
    # fps == 0 would make us divide by zero or hand AR ``inf``. Guard.
    with pytest.raises(ValueError, match="fps"):
        timestamp_for_video(10, 0.0)


def test_timestamp_for_video_negative_fps_raises():
    # Same logic for negative fps — Ultralytics/cv2 can return -1 if the
    # video metadata is broken. We refuse to produce a misleading
    # negative timestamp.
    with pytest.raises(ValueError, match="fps"):
        timestamp_for_video(10, -1.0)


def test_timestamp_for_image_index_default_fps():
    # Default fps is the module-level DEFAULT_IMAGE_FPS (30.0). Index 60
    # → 2.0 s.
    assert timestamp_for_image_index(60) == pytest.approx(60 / DEFAULT_IMAGE_FPS)
    assert timestamp_for_image_index(60) == pytest.approx(2.0)


def test_timestamp_for_image_index_custom_fps():
    # User override — fps=15 → index 30 → 2.0 s.
    assert timestamp_for_image_index(30, fps=15.0) == pytest.approx(2.0)


def test_timestamp_for_image_index_zero_fps_raises():
    with pytest.raises(ValueError, match="fps"):
        timestamp_for_image_index(10, fps=0.0)


# ---------------------------------------------------------------------------
# iter_image_paths
# ---------------------------------------------------------------------------


def test_iter_image_paths_sorted_and_filtered(tmp_path: Path):
    # Mix of image extensions (one uppercase) and a non-image .txt file.
    # Spec-required:
    #   - .txt is excluded
    #   - .JPG (uppercase) is included (case-insensitive suffix match)
    #   - returned list is lexicographically sorted
    for name in ("b.jpg", "a.png", "c.txt", "D.JPG", "e.bmp"):
        (tmp_path / name).write_bytes(b"\xff\xd8\xff")

    out = iter_image_paths(tmp_path)
    names = [p.name for p in out]

    assert "c.txt" not in names  # filtered
    assert "D.JPG" in names  # case-insensitive accepted
    # Plain lexicographic sort: uppercase 'D' sorts before lowercase.
    assert names == sorted(names)


def test_iter_image_paths_empty_dir(tmp_path: Path):
    # An empty directory yields an empty list, no crash.
    out = iter_image_paths(tmp_path)
    assert out == []


def test_iter_image_paths_skips_subdirectories(tmp_path: Path):
    # ``iter_image_paths`` is non-recursive: a nested ``a.jpg`` inside a
    # subdir must NOT leak into the result.
    (tmp_path / "top.jpg").write_bytes(b"\xff\xd8\xff")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "inner.jpg").write_bytes(b"\xff\xd8\xff")

    out = iter_image_paths(tmp_path)
    names = [p.name for p in out]

    assert names == ["top.jpg"]


def test_iter_image_paths_respects_custom_extensions(tmp_path: Path):
    # Caller can override the extension whitelist. Useful for AR if they
    # ever start saving heic captures.
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff")
    (tmp_path / "b.heic").write_bytes(b"\x00")

    out = iter_image_paths(tmp_path, extensions=(".heic",))
    names = [p.name for p in out]

    assert names == ["b.heic"]


def test_iter_image_paths_default_extensions_match_constant(tmp_path: Path):
    # All extensions in ``IMAGE_EXTS`` must be accepted by the default
    # call. Defends against the constant and the default arg drifting.
    for ext in IMAGE_EXTS:
        (tmp_path / f"f{ext}").write_bytes(b"\x00")

    out = iter_image_paths(tmp_path)
    assert len(out) == len(IMAGE_EXTS)
