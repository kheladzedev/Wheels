"""Regression tests for the first-real-plugin-batch acceptance workflow.

These tests pin the contract documented in
`docs/FIRST_PLUGIN_BATCH_ACCEPTANCE.md` and the wrapper script
`scripts/accept_first_plugin_batch.sh`:

  - The script exists and is executable.
  - The doc + script describe A/B with the post-2026-05-14 floor-ray
    semantics, not the legacy rim-edge wording.
  - The doc explicitly forbids training before a human preview.
  - The README has an entry pointing at this workflow.

The intent is "if someone silently drifts the contract back to
rim-edge wording or drops the human gate, CI catches it".
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DOC = ROOT / "docs" / "FIRST_PLUGIN_BATCH_ACCEPTANCE.md"
SCRIPT = ROOT / "scripts" / "accept_first_plugin_batch.sh"
README = ROOT / "README.md"

# Phrases asserting the obsolete "A/B are rim edges" meaning.
# We do NOT flag the literal token `rim_left` / `rim_right` here —
# legacy code still uses those strings and other docs may reference
# the drift explicitly. We only flag *assertive* rim-edge phrasing.
FORBIDDEN_AB_PHRASES = (
    "left rim source",
    "right rim source",
    "left rim point",
    "right rim point",
    "left rim edge",
    "right rim edge",
)


def test_accept_script_exists_and_is_executable() -> None:
    assert SCRIPT.is_file(), f"missing script: {SCRIPT}"
    mode = SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, (
        "scripts/accept_first_plugin_batch.sh is not executable for the "
        "owner — chmod +x it before committing."
    )


def test_acceptance_doc_exists() -> None:
    assert DOC.is_file(), f"missing doc: {DOC}"


def test_acceptance_doc_describes_a_b_as_floor_ray() -> None:
    text = DOC.read_text(encoding="utf-8").lower()
    assert "floor-ray" in text or "floor ray" in text, (
        "docs/FIRST_PLUGIN_BATCH_ACCEPTANCE.md must describe A/B as "
        "floor-ray points per the 2026-05-14 spec revision."
    )


def test_acceptance_doc_does_not_call_a_b_rim_edges() -> None:
    text = DOC.read_text(encoding="utf-8").lower()
    hits = [needle for needle in FORBIDDEN_AB_PHRASES if needle in text]
    assert not hits, (
        "Found forbidden rim-edge wording for A/B in "
        f"FIRST_PLUGIN_BATCH_ACCEPTANCE.md (A/B are floor-ray points, "
        f"not rim edges): {hits}"
    )


def test_acceptance_doc_blocks_training_before_human_preview() -> None:
    text = DOC.read_text(encoding="utf-8").lower()
    assert "do not train" in text, (
        "docs/FIRST_PLUGIN_BATCH_ACCEPTANCE.md must explicitly forbid "
        "training before a human has inspected the previews."
    )
    assert "preview" in text and (
        "manual" in text or "human" in text or "inspect" in text
    ), (
        "The 'no training before preview' rule must reference the manual "
        "/ human inspection step explicitly."
    )


def test_script_runs_no_training_or_inference() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    # The acceptance script must not invoke training or inference.
    forbidden = ("train_yolo.py", "infer_image.py", "infer_batch.py")
    hits = [f for f in forbidden if f in text]
    assert not hits, (
        "scripts/accept_first_plugin_batch.sh must not invoke training "
        f"or inference scripts. Found: {hits}"
    )


def test_script_chains_validators_and_previews() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    required = (
        "check_keypoint_incoming.py",
        "preview_keypoint_annotations.py",
        "convert_keypoint_incoming_to_yolo_pose.py",
        "check_yolo_pose_dataset.py",
        "preview_yolo_pose_labels.py",
        "--fail-on-quality-gate",
    )
    missing = [s for s in required if s not in text]
    assert not missing, (
        "scripts/accept_first_plugin_batch.sh is missing required steps "
        f"of the acceptance chain: {missing}"
    )


def test_readme_links_to_acceptance_doc() -> None:
    text = README.read_text(encoding="utf-8")
    assert "FIRST_PLUGIN_BATCH_ACCEPTANCE.md" in text, (
        "README.md must point at docs/FIRST_PLUGIN_BATCH_ACCEPTANCE.md "
        "so contributors find the workflow when the first real batch "
        "lands."
    )
    assert "accept_first_plugin_batch.sh" in text, (
        "README.md must mention the acceptance wrapper script."
    )


def test_readme_warns_against_training_before_preview() -> None:
    text = README.read_text(encoding="utf-8").lower()
    # The README section must reproduce the no-training-before-preview
    # gate, not just defer to the doc.
    assert "do not train" in text, (
        "README.md must reproduce the 'do not train before human "
        "preview' warning in the acceptance section."
    )
