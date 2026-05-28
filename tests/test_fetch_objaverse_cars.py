"""Offline helper tests for Objaverse fallback fetcher."""

from __future__ import annotations

import fetch_objaverse_cars as foc


def test_safe_category_name_is_human_readable():
    assert foc._safe_category_name("car_(automobile)") == "car automobile"


def test_candidate_uids_dedupes_preserving_category_order():
    annotations = {
        "car_(automobile)": ["a", "b"],
        "race_car": ["b", "c"],
    }

    out = foc._candidate_uids(
        annotations,
        ["car_(automobile)", "race_car"],
        shuffle_seed=1,
    )

    assert sorted(uid for uid, _ in out) == ["a", "b", "c"]
    assert len(out) == 3


def test_manifest_keeps_objaverse_provenance():
    manifest = foc._manifest_for_objaverse(
        "abc",
        "pickup_truck",
        {"name": "Source Name", "viewerUrl": "https://example.com/abc"},
    )

    assert manifest["uid"] == "ov_abc"
    assert manifest["source_uid"] == "abc"
    assert manifest["source_platform"] == "objaverse"
    assert manifest["source_category"] == "pickup_truck"
    assert "pickup truck" in manifest["name"]
