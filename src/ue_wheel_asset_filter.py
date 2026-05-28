"""Wheel-part asset-name classifier for UE-imported vehicle meshes."""

from __future__ import annotations

import re
from pathlib import PurePosixPath


WHEEL_TERMS = ("wheel", "tire", "tyre", "rim", "pneu", "roue", "reifen")
WHEEL_EXCLUDE_TERMS = ("steering", "flywheel", "handwheel")
WHEEL_ABBREVIATION_RE = re.compile(r"(^|[/_.-])(flw|frw|lfw|lrw|rfw|rlw|rrw)([_/.-]|$)")
RUBBER_MATERIAL_TERMS = ("blackrubber", "_rubber", "-rubber", ".rubber")
MIN_RUBBER_FALLBACK_PARTS = 4
MAX_RUBBER_FALLBACK_PARTS = 8


def is_wheel_asset_path(path: str) -> bool:
    low = path.lower()
    if any(term in low for term in WHEEL_EXCLUDE_TERMS):
        return False
    return any(term in low for term in WHEEL_TERMS) or bool(WHEEL_ABBREVIATION_RE.search(low))


def is_rubber_material_candidate(path: str) -> bool:
    low = path.lower()
    if any(term in low for term in WHEEL_EXCLUDE_TERMS):
        return False
    return any(term in low for term in RUBBER_MATERIAL_TERMS)


def classify_wheel_asset_paths(paths: list[str]) -> list[bool]:
    """Classify wheel meshes with a conservative group-level rubber fallback."""
    direct = [is_wheel_asset_path(path) for path in paths]
    if any(direct):
        return direct

    rubber_candidates = [is_rubber_material_candidate(path) for path in paths]
    rubber_count = sum(rubber_candidates)
    if MIN_RUBBER_FALLBACK_PARTS <= rubber_count <= MAX_RUBBER_FALLBACK_PARTS:
        return rubber_candidates
    return [False for _ in paths]


def wheel_classifier_metadata() -> dict[str, object]:
    return {
        "wheel_terms": list(WHEEL_TERMS),
        "wheel_abbreviation_pattern": WHEEL_ABBREVIATION_RE.pattern,
        "wheel_exclude_terms": list(WHEEL_EXCLUDE_TERMS),
        "rubber_material_terms": list(RUBBER_MATERIAL_TERMS),
        "rubber_fallback_min_parts": MIN_RUBBER_FALLBACK_PARTS,
        "rubber_fallback_max_parts": MAX_RUBBER_FALLBACK_PARTS,
    }


def basename_for_audit(path: str) -> str:
    return PurePosixPath(path.replace("\\", "/")).name.lower()
