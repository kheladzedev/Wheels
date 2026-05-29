"""Tests for the synthetic frames-manifest generator
(``scripts/make_synthetic_eval3d_manifest.py``).

The generator builds a frames manifest from the harness forward model so
``src/eval3d_report.py`` can be smoke-run end-to-end before real UE data
exists, and so the manifest shape is documented by a working producer.
A clean synthetic batch must pass acceptance; this validates plumbing,
NOT model quality.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import make_synthetic_eval3d_manifest as gen  # noqa: E402
import eval3d_report as r  # noqa: E402


def test_build_manifest_shape():
    man = gen.build_manifest(n_scenes=3, frames_per_scene=5, seed=7)
    assert man["units"] == "cm"
    assert len(man["scenes"]) == 3
    for scene in man["scenes"].values():
        assert "gt_disc_position" in scene
        assert len(scene["frames"]) == 5
        fr = scene["frames"][0]
        assert set(fr) >= {"frame_id", "ground", "points"}
        assert set(fr["ground"]) == {"delta_z", "roll", "pitch", "fov"}
        assert set(fr["points"]) == {"a", "b", "c_disc_bottom"}


def test_clean_synthetic_manifest_passes_acceptance():
    man = gen.build_manifest(n_scenes=4, frames_per_scene=6, seed=3)
    rep = r.run_report(man, rng=np.random.default_rng(0))
    assert rep["acceptance"]["pass_accept"] is True
    assert rep["sigma_cm"]["max"] < 0.5  # cm; synthetic round-trip is near-exact
