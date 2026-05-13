"""Headless tests for the Wikimedia Commons car-photo scraper.

The HTTP layer is mocked: every network call goes through monkeypatched
``_http_get_json`` / ``_http_download`` shims that read from fixtures
in memory rather than hitting Wikimedia. This keeps the suite offline
and deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import fetch_wikimedia_cars as fwc


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_slugify_title_basic():
    assert fwc.slugify_title("File:Toyota Corolla 2010.jpg") == "Toyota_Corolla_2010"


def test_slugify_title_drops_unicode_and_punctuation():
    assert (
        fwc.slugify_title("File:Cars — édition 2021! (Paris).jpg")
        == "Cars_dition_2021_Paris"
    )


def test_slugify_title_truncates_at_max_len():
    long = "File:" + ("A" * 500) + ".jpg"
    out = fwc.slugify_title(long, max_len=40)
    assert len(out) <= 40
    assert out == "A" * 40


def test_slugify_title_empty_falls_back_to_image():
    assert fwc.slugify_title("File:!!!.jpg") == "image"


@pytest.mark.parametrize(
    "license_str, expected",
    [
        ("CC BY 4.0", True),
        ("CC-BY-SA-3.0", True),
        ("CC0", True),
        ("Public domain", True),
        ("PublicDomain", True),
        ("All rights reserved", False),
        ("", False),
        ("Fair use", False),
    ],
)
def test_license_is_acceptable(license_str: str, expected: bool):
    assert fwc.license_is_acceptable(license_str) is expected


def test_extract_license_prefers_short_name():
    md = {
        "LicenseShortName": {"value": "CC BY-SA 4.0"},
        "License": {"value": "cc-by-sa-4.0"},
        "UsageTerms": {"value": "Creative Commons"},
    }
    assert fwc.extract_license(md) == "CC BY-SA 4.0"


def test_extract_license_falls_through_to_usage_terms():
    md = {"UsageTerms": {"value": "Public domain"}}
    assert fwc.extract_license(md) == "Public domain"


def test_extract_license_empty_blob_returns_empty():
    assert fwc.extract_license({}) == ""


def test_derive_target_filename_has_index_and_slug():
    name = fwc.derive_target_filename("File:Toyota Corolla.jpg", 7)
    assert name == "wmc_0007_Toyota_Corolla.jpg"


def test_already_have_matches_by_url():
    existing = [
        {"file": "x.jpg", "src_url": "https://example.com/a.jpg"},
    ]
    assert fwc.already_have("https://example.com/a.jpg", "y.jpg", existing) is True


def test_already_have_matches_by_filename():
    existing = [
        {"file": "x.jpg", "src_url": "https://example.com/a.jpg"},
    ]
    assert fwc.already_have("https://other/b.jpg", "x.jpg", existing) is True


def test_already_have_false_on_new():
    existing = [{"file": "x.jpg", "src_url": "https://example.com/a.jpg"}]
    assert fwc.already_have("https://example.com/c.jpg", "z.jpg", existing) is False


def test_next_image_index_picks_max_plus_one():
    existing = [
        {"file": "wmc_0001_a.jpg"},
        {"file": "wmc_0007_b.jpg"},
        {"file": "real_005_c.jpg"},  # ignored, different prefix
    ]
    assert fwc.next_image_index(existing) == 8


def test_next_image_index_empty_returns_zero():
    assert fwc.next_image_index([]) == 0


# ---------------------------------------------------------------------------
# SOURCES.json round-trip
# ---------------------------------------------------------------------------


def test_load_sources_json_missing_returns_empty(tmp_path: Path):
    assert fwc.load_sources_json(tmp_path / "no.json") == []


def test_load_sources_json_invalid_returns_empty(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("[broken", encoding="utf-8")
    assert fwc.load_sources_json(p) == []


def test_save_then_load_sources_json_round_trip(tmp_path: Path):
    p = tmp_path / "SOURCES.json"
    entries = [{"file": "x.jpg", "src_url": "https://example.com/x.jpg"}]
    fwc.save_sources_json(p, entries)
    assert fwc.load_sources_json(p) == entries


# ---------------------------------------------------------------------------
# CLI / HTTP-mocked end-to-end
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch):
    """Mock the HTTP layer so the CLI can run offline.

    Returns a small dict the test can mutate / inspect — primarily for
    counting how many downloads occurred.
    """
    state = {
        "downloads": [],  # list of (url, dest)
        "search_titles": [
            "File:Cool_Sedan.jpg",
            "File:Boring_SUV.jpg",
            "File:Locked_Down_Car.jpg",  # this one has no free licence
        ],
    }

    def fake_get_json(url: str, user_agent: str, timeout: float = 30.0) -> dict:
        if "list=search" in url:
            return {
                "query": {
                    "search": [{"title": t} for t in state["search_titles"]],
                }
            }
        if "prop=imageinfo" in url:
            return {
                "query": {
                    "pages": {
                        "100": {
                            "title": "File:Cool_Sedan.jpg",
                            "imageinfo": [
                                {
                                    "thumburl": "https://example.com/cool.jpg",
                                    "url": "https://example.com/cool_orig.jpg",
                                    "thumbwidth": 1280,
                                    "thumbheight": 720,
                                    "mime": "image/jpeg",
                                    "extmetadata": {
                                        "LicenseShortName": {"value": "CC BY 4.0"}
                                    },
                                }
                            ],
                        },
                        "200": {
                            "title": "File:Boring_SUV.jpg",
                            "imageinfo": [
                                {
                                    "thumburl": "https://example.com/boring.jpg",
                                    "url": "https://example.com/boring_orig.jpg",
                                    "thumbwidth": 1280,
                                    "thumbheight": 960,
                                    "mime": "image/jpeg",
                                    "extmetadata": {
                                        "License": {"value": "cc-by-sa-3.0"}
                                    },
                                }
                            ],
                        },
                        "300": {
                            "title": "File:Locked_Down_Car.jpg",
                            "imageinfo": [
                                {
                                    "thumburl": "https://example.com/locked.jpg",
                                    "url": "https://example.com/locked_orig.jpg",
                                    "thumbwidth": 1280,
                                    "thumbheight": 720,
                                    "mime": "image/jpeg",
                                    "extmetadata": {
                                        "LicenseShortName": {
                                            "value": "All rights reserved"
                                        }
                                    },
                                }
                            ],
                        },
                    }
                }
            }
        raise RuntimeError(f"unexpected URL in test: {url}")

    def fake_download(
        url: str, dest: Path, user_agent: str, timeout: float = 60.0
    ) -> int:
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Fake content (a few bytes is fine — no decoding happens here).
        dest.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-bytes")
        state["downloads"].append((url, dest))
        return dest.stat().st_size

    monkeypatch.setattr(fwc, "_http_get_json", fake_get_json)
    monkeypatch.setattr(fwc, "_http_download", fake_download)
    return state


def test_main_downloads_licensed_images_only(tmp_path: Path, fake_api):
    images_dir = tmp_path / "images"
    sources_json = tmp_path / "SOURCES.json"

    exit_code = fwc.main(
        [
            "--target-count",
            "10",
            "--images-dir",
            str(images_dir),
            "--sources-json",
            str(sources_json),
            "--queries",
            "parked car",
            "--sleep",
            "0",
            "--per-query-limit",
            "5",
        ]
    )
    assert exit_code == 0
    # Only the two CC-licensed entries should be downloaded.
    assert len(fake_api["downloads"]) == 2
    sources = json.loads(sources_json.read_text(encoding="utf-8"))
    assert {entry["file"] for entry in sources} == {
        "wmc_0000_Cool_Sedan.jpg",
        "wmc_0001_Boring_SUV.jpg",
    }
    for entry in sources:
        assert (images_dir / entry["file"]).is_file()
        assert fwc.license_is_acceptable(entry["license"])


def test_main_respects_target_count(tmp_path: Path, fake_api):
    images_dir = tmp_path / "images"
    sources_json = tmp_path / "SOURCES.json"

    fwc.main(
        [
            "--target-count",
            "1",  # cap below the 2 available licensed entries
            "--images-dir",
            str(images_dir),
            "--sources-json",
            str(sources_json),
            "--queries",
            "parked car",
            "--sleep",
            "0",
        ]
    )
    assert len(fake_api["downloads"]) == 1


def test_main_dedup_against_existing_sources(tmp_path: Path, fake_api):
    images_dir = tmp_path / "images"
    sources_json = tmp_path / "SOURCES.json"
    # Pre-seed SOURCES.json with one of the two licensed entries by URL.
    fwc.save_sources_json(
        sources_json,
        [
            {
                "file": "already_have.jpg",
                "src_url": "https://example.com/cool.jpg",
                "license": "CC BY 4.0",
            }
        ],
    )

    fwc.main(
        [
            "--target-count",
            "10",
            "--images-dir",
            str(images_dir),
            "--sources-json",
            str(sources_json),
            "--queries",
            "parked car",
            "--sleep",
            "0",
        ]
    )
    # Only the second licensed entry remains; the first was deduped.
    assert len(fake_api["downloads"]) == 1
    downloaded_url = fake_api["downloads"][0][0]
    assert downloaded_url == "https://example.com/boring.jpg"


def test_main_dry_run_writes_nothing(tmp_path: Path, fake_api):
    images_dir = tmp_path / "images"
    sources_json = tmp_path / "SOURCES.json"

    fwc.main(
        [
            "--target-count",
            "10",
            "--images-dir",
            str(images_dir),
            "--sources-json",
            str(sources_json),
            "--queries",
            "parked car",
            "--sleep",
            "0",
            "--dry-run",
        ]
    )
    assert fake_api["downloads"] == []
    # Sources file is not touched in dry-run.
    assert not sources_json.exists()
