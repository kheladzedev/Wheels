"""Regression test: A/B semantics docs reflect the 2026-05-14 spec revision.

Under the AR mock-system spec, A and B are screen-space floor-ray
points — pixels the AR client raycasts onto the floor plane to anchor
the vertical wheel plane. They are **NOT** metal-rim edge points; the
old "rim_left / rim_right" wording is forbidden when describing A/B
going forward.

This test reads a small set of authoritative docs and pins:

  - The phrase "floor-ray" (or "floor ray") appears so the new
    semantics is documented in the file at all.
  - The doc does not equate A/B with "left rim" / "right rim" /
    "rim left" / "rim right" as their *current* meaning. The literal
    label strings `rim_left` / `rim_right` (legacy training-side
    naming in `postprocess_wheels.py` and friends) may still appear
    inside an explicit "legacy / drifted" mention — the test allows
    that by only flagging the **assertive** rim-edge phrasing, not
    the literal string occurrence.

Update this test alongside any future contract revision.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DOCS_REQUIRING_NEW_SEMANTICS = (
    "docs/KEYPOINT_SPEC.md",
    "docs/AR_ML_CONTRACT.md",
    ".claude/skills/vsbl-ar-contract/SKILL.md",
    "CLAUDE.md",
)

# Phrases that assert "A/B are rim edges". We flag exact wordings; we
# do NOT flag the literal label string `rim_left` because legacy code
# still uses it (the drift is explicitly documented).
FORBIDDEN_AB_PHRASES = (
    "left rim source",
    "right rim source",
    "left rim point",
    "right rim point",
    "left rim edge",
    "right rim edge",
)


def test_authoritative_docs_describe_a_b_as_floor_ray() -> None:
    missing: list[str] = []
    for rel in DOCS_REQUIRING_NEW_SEMANTICS:
        path = ROOT / rel
        assert path.is_file(), f"missing doc: {rel}"
        text = path.read_text(encoding="utf-8").lower()
        if "floor-ray" not in text and "floor ray" not in text:
            missing.append(rel)
    assert not missing, (
        "These authoritative docs no longer describe A/B as floor-ray "
        f"points (2026-05-14 spec revision): {missing}"
    )


def test_authoritative_docs_do_not_call_a_b_rim_edges() -> None:
    hits: list[tuple[str, str]] = []
    for rel in DOCS_REQUIRING_NEW_SEMANTICS:
        text = (ROOT / rel).read_text(encoding="utf-8").lower()
        for needle in FORBIDDEN_AB_PHRASES:
            if needle in text:
                hits.append((rel, needle))
    assert not hits, (
        "Found forbidden rim-edge wording for A/B in authoritative docs "
        "(2026-05-14: A/B are floor-ray points, not rim edges): "
        f"{hits}"
    )
