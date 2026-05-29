"""Tests for the val-set driver of the 3D-eval harness
(``src/eval3d_report.py``).

The driver reads a *frames manifest* — per ``scene_id``, a list of
frames carrying UE ``Ground`` metadata (intrinsics + pose) and the
model's predicted screen-space ``a`` / ``b`` / ``c_disc_bottom`` — runs
``eval3d_floorray.simulate_scene`` per scene, and aggregates disc-height
sigma across scenes against the acceptance budget (< 3 cm, target
< 1 cm; ``docs/AR_REPLAY_METRIC_PLAN.md`` §9).

These tests build the manifest from the harness forward model so the
round-trip is exact: a clean manifest must pass acceptance with ~0
sigma, a drifted one (A/B lifted off the floor) must fail. This pins
the *plumbing*; running on real model predictions is gated on a clean
UE export (``docs/EXPORT_PARITY_AUDIT.md``), which is the upstream
blocker.
"""

from __future__ import annotations

import json

import numpy as np

import eval3d_floorray as g
import eval3d_report as r


IMG_W, IMG_H = 1920, 1080
FOV = 60.0


def _ground_for(delta_z: float, wheel_depth: float, wheel_height: float):
    """UE Ground meta whose pitch aims the camera at the wheel center."""
    pitch = float(np.degrees(np.arctan2(delta_z - wheel_height, wheel_depth)))
    return {"delta_z": float(delta_z), "roll": 0.0, "pitch": pitch, "fov": FOV}


def _make_manifest(disc_height_cm: float, ab_lift_cm: float = 0.0):
    """One scene, several frames, in centimetres. ``ab_lift_cm`` lifts the
    A/B screen sources off the floor to simulate wheel-attraction drift."""
    depth = 200.0  # wheel sits 200 cm in front of the camera column
    a_src = np.array([-8.0, depth, ab_lift_cm])
    b_src = np.array([8.0, depth, ab_lift_cm])
    disc = np.array([0.0, depth, disc_height_cm])
    frames = []
    for dz in (150.0, 170.0, 190.0, 160.0, 180.0):
        ground = _ground_for(dz, depth, disc_height_cm)
        cam = g.camera_from_ue_ground(ground, IMG_W, IMG_H)
        frames.append(
            {
                "frame_id": f"f_{int(dz)}",
                "ground": ground,
                "points": {
                    "a": g.project(cam, a_src[None])[0].tolist(),
                    "b": g.project(cam, b_src[None])[0].tolist(),
                    "c_disc_bottom": g.project(cam, disc[None])[0].tolist(),
                },
            }
        )
    return {
        "units": "cm",
        "image_size": [IMG_W, IMG_H],
        "scenes": {"scene_0001": {"gt_disc_position": disc.tolist(), "frames": frames}},
    }


def test_clean_manifest_recovers_height_and_passes_acceptance():
    manifest = _make_manifest(disc_height_cm=30.0)
    rep = r.run_report(manifest, rng=np.random.default_rng(0))
    scene = rep["per_scene"]["scene_0001"]
    assert (
        scene["disc_height_mean"] == 30.0
        or abs(scene["disc_height_mean"] - 30.0) < 0.05
    )
    assert scene["disc_height_sigma"] < 0.05
    assert scene["height_error"] < 0.05
    assert rep["acceptance"]["pass_accept"] is True
    assert rep["acceptance"]["pass_target"] is True
    assert rep["n_scenes"] == 1
    assert rep["n_scenes_with_gt"] == 1


def test_ab_drift_manifest_fails_acceptance():
    manifest = _make_manifest(disc_height_cm=30.0, ab_lift_cm=12.0)
    rep = r.run_report(manifest, rng=np.random.default_rng(0))
    scene = rep["per_scene"]["scene_0001"]
    # drift pushes recovered disc height well off GT
    assert scene["height_error"] > 3.0
    assert rep["acceptance"]["pass_target"] is False


def test_report_aggregates_sigma_across_scenes():
    manifest = _make_manifest(disc_height_cm=25.0)
    # add a second clean scene with a different height
    second = _make_manifest(disc_height_cm=40.0)["scenes"]["scene_0001"]
    manifest["scenes"]["scene_0002"] = second
    rep = r.run_report(manifest, rng=np.random.default_rng(0))
    assert rep["n_scenes"] == 2
    assert set(rep["per_scene"]) == {"scene_0001", "scene_0002"}
    assert rep["sigma_cm"]["median"] < 0.05
    assert rep["sigma_cm"]["max"] < 0.05


def test_load_manifest_roundtrips_from_disk(tmp_path):
    manifest = _make_manifest(disc_height_cm=30.0)
    path = tmp_path / "frames_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = r.load_manifest(path)
    assert set(loaded["scenes"]) == {"scene_0001"}
    rep = r.run_report(loaded, rng=np.random.default_rng(0))
    assert rep["acceptance"]["pass_accept"] is True


def test_single_frame_scene_is_not_counted_as_passing_sigma():
    # A 1-frame scene has std == 0 by construction; it must NOT silently
    # pass the sigma gate. With no GT and only non-estimable scenes the
    # report must refuse to claim acceptance.
    manifest = _make_manifest(disc_height_cm=30.0)
    manifest["scenes"]["scene_0001"]["frames"] = manifest["scenes"]["scene_0001"][
        "frames"
    ][:1]
    del manifest["scenes"]["scene_0001"]["gt_disc_position"]
    rep = r.run_report(manifest, rng=np.random.default_rng(0))
    assert rep["per_scene"]["scene_0001"]["sigma_estimable"] is False
    assert rep["n_sigma_estimable"] == 0
    assert rep["acceptance"]["pass_accept"] is False


def test_report_carries_manifest_provenance_and_gate_status():
    manifest = _make_manifest(disc_height_cm=30.0)
    manifest["provenance"] = "synthetic-roundtrip; plumbing only"
    rep = r.run_report(manifest, rng=np.random.default_rng(0))
    assert rep["provenance"] == "synthetic-roundtrip; plumbing only"
    # synthetic / untrusted source must not present as a production gate
    assert rep["gate_status"] == "informational"


def test_real_geometry_with_gt_points_is_not_a_gate():
    # Hardened invariant: real-capture geometry but ground-truth 2D points
    # must NOT present as a model gate, even if mislabeled source="real".
    manifest = _make_manifest(disc_height_cm=30.0)
    manifest["source"] = "real"
    manifest["points_source"] = "ue_ground_truth"
    rep = r.run_report(manifest, rng=np.random.default_rng(0))
    assert rep["gate_status"] == "informational"
    # absent points_source on a source="real" manifest is "unknown", not
    # trusted — must NOT present as a gate either.
    manifest.pop("points_source", None)
    rep_absent = r.run_report(manifest, rng=np.random.default_rng(0))
    assert rep_absent["gate_status"] == "informational"
    # only an explicit model_prediction on a real capture IS a gate
    manifest["points_source"] = "model_prediction"
    rep2 = r.run_report(manifest, rng=np.random.default_rng(0))
    assert rep2["gate_status"] == "gate"


def test_scene_without_gt_reports_sigma_only():
    manifest = _make_manifest(disc_height_cm=30.0)
    del manifest["scenes"]["scene_0001"]["gt_disc_position"]
    rep = r.run_report(manifest, rng=np.random.default_rng(0))
    scene = rep["per_scene"]["scene_0001"]
    assert "disc_height_sigma" in scene
    assert "height_error" not in scene
    assert rep["n_scenes_with_gt"] == 0
    # acceptance on sigma still computable without GT
    assert rep["acceptance"]["pass_accept"] is True
