"""Tests for scripts/make_eval3d_manifest_from_model_predictions.py.

Exercises the PURE assembly helpers only — no GPU, no real weights, no
real images required.  The model.predict call is monkeypatched with a
fake ``confirmed_wheels`` list injected directly into
``match_predictions_to_actors`` and ``extract_points_from_confirmed_wheel``.

Assertions:
  - points_source == "model_prediction" in every assembled manifest.
  - source == "real" so the gate can flip green.
  - Per-wheel a / b / c_disc_bottom keys present in every frame.
  - Camera pose / intrinsics carried through (pose dict round-trips intact).
  - promotion_gate_3d.evaluate_3d_acceptance returns "insufficient_evidence"
    (NOT "gate") until the geometry actually passes — i.e. the gate logic
    remains the arbiter; the manifest alone cannot self-certify.
  - When geometry IS correct (sigma < accept_cm, n_est >= 1) AND
    gate_status == "gate" the gate returns ok=True.
  - assemble_manifest filters out scenes with fewer than min_frames frames.
  - match_predictions_to_actors: correct nearest-neighbour assignment.
  - match_predictions_to_actors: handles zero predictions gracefully.
  - extract_points_from_confirmed_wheel: returns None on incomplete points.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make src/ importable (mirrors conftest.py).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import make_eval3d_manifest_from_model_predictions as mp  # noqa: E402
import eval3d_report as r  # noqa: E402
import promotion_gate_3d as pg  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_POSE = {
    "location": [797.0, 49.0, 200.0],
    "rotation": [0.0, -8.5, -176.5],
    "fov": 75,
}
_IMG_SIZE = [1280, 720]


def _confirmed_wheel(
    cx: float = 640.0,
    cy: float = 360.0,
    half: float = 80.0,
) -> dict:
    """A minimal confirmed-schema wheel dict centred at (cx, cy)."""
    return {
        "bbox_xyxy": [cx - half, cy - half, cx + half, cy + half],
        "confidence": 0.91,
        "points": {
            "a": [cx - 30.0, cy + 20.0],
            "b": [cx + 30.0, cy + 20.0],
            "c_disc_bottom": [cx, cy - 10.0],
        },
    }


def _make_scene_with_frames(n_frames: int = 3) -> dict:
    """Build a scene dict as _run_inference_on_scenes would produce it,
    with ``frames`` already populated (mock inference done)."""
    frames = []
    for i in range(n_frames):
        frames.append(
            {
                "frame_id": f"ActorA__frame_{i:04d}",
                "image_size": _IMG_SIZE,
                "pose": _POSE,
                "points": {
                    "a": [610.0 + i, 380.0],
                    "b": [670.0 + i, 380.0],
                    "c_disc_bottom": [640.0 + i, 350.0],
                },
            }
        )
    return {
        "frames": frames,
        "_gt_world": [170.0, -110.0, 33.0],
        "img_w": 1280,
        "img_h": 720,
    }


# ---------------------------------------------------------------------------
# assemble_manifest — pure, schema-level tests
# ---------------------------------------------------------------------------


def test_points_source_is_model_prediction():
    scenes = {"ActorA": _make_scene_with_frames(3)}
    manifest = mp.assemble_manifest(
        scenes,
        dataset_name="WheelsDataset_v0_2",
        weights_path="runs/pose/champion/weights/best.pt",
    )
    assert manifest["points_source"] == "model_prediction"


def test_source_is_real():
    scenes = {"ActorA": _make_scene_with_frames(3)}
    manifest = mp.assemble_manifest(
        scenes,
        dataset_name="WheelsDataset_v0_2",
        weights_path="runs/pose/champion/weights/best.pt",
    )
    assert manifest["source"] == "real"


def test_per_wheel_abc_keys_present():
    scenes = {"ActorA": _make_scene_with_frames(3)}
    manifest = mp.assemble_manifest(
        scenes,
        dataset_name="WheelsDataset_v0_2",
        weights_path="runs/pose/champion/weights/best.pt",
    )
    for scene in manifest["scenes"].values():
        for frame in scene["frames"]:
            pts = frame["points"]
            assert "a" in pts
            assert "b" in pts
            assert "c_disc_bottom" in pts


def test_pose_carried_through():
    scenes = {"ActorA": _make_scene_with_frames(3)}
    manifest = mp.assemble_manifest(
        scenes,
        dataset_name="WheelsDataset_v0_2",
        weights_path="runs/pose/champion/weights/best.pt",
    )
    for scene in manifest["scenes"].values():
        for frame in scene["frames"]:
            assert frame["pose"] == _POSE
            assert frame["image_size"] == _IMG_SIZE


def test_min_frames_filter_drops_short_scenes():
    scenes = {
        "ActorA": _make_scene_with_frames(3),
        "ActorB": _make_scene_with_frames(1),  # below default min_frames=2
    }
    manifest = mp.assemble_manifest(
        scenes,
        dataset_name="WheelsDataset_v0_2",
        weights_path="runs/pose/champion/weights/best.pt",
        min_frames=2,
    )
    assert "ActorA" in manifest["scenes"]
    assert "ActorB" not in manifest["scenes"]


def test_gt_disc_position_included_when_present():
    scenes = {"ActorA": _make_scene_with_frames(3)}
    manifest = mp.assemble_manifest(
        scenes,
        dataset_name="WheelsDataset_v0_2",
        weights_path="runs/pose/champion/weights/best.pt",
    )
    assert manifest["scenes"]["ActorA"]["gt_disc_position"] == [170.0, -110.0, 33.0]


def test_gt_disc_position_absent_when_none():
    scenes = {"ActorA": _make_scene_with_frames(3)}
    scenes["ActorA"]["_gt_world"] = None
    manifest = mp.assemble_manifest(
        scenes,
        dataset_name="WheelsDataset_v0_2",
        weights_path="runs/pose/champion/weights/best.pt",
    )
    assert "gt_disc_position" not in manifest["scenes"]["ActorA"]


def test_manifest_units_cm():
    scenes = {"ActorA": _make_scene_with_frames(3)}
    manifest = mp.assemble_manifest(
        scenes,
        dataset_name="WheelsDataset_v0_2",
        weights_path="runs/pose/champion/weights/best.pt",
    )
    assert manifest["units"] == "cm"


# ---------------------------------------------------------------------------
# run_report integration: gate_status propagates correctly
# ---------------------------------------------------------------------------


def test_run_report_sets_gate_status_gate():
    """A manifest with source='real' and points_source='model_prediction'
    must get gate_status='gate' from eval3d_report.run_report."""
    scenes = {"ActorA": _make_scene_with_frames(3)}
    manifest = mp.assemble_manifest(
        scenes,
        dataset_name="WheelsDataset_v0_2",
        weights_path="runs/pose/champion/weights/best.pt",
    )
    # Swap in ground= so eval3d_report can build the camera without pose infra.
    # We just need gate_status to be checked, not a meaningful geometry pass.
    import eval3d_floorray as g

    depth = 200.0
    disc_h = 30.0
    half = 8.0
    a_src = np.array([-half, depth, 0.0])
    b_src = np.array([half, depth, 0.0])
    disc = np.array([0.0, depth, disc_h])
    frames_geo = []
    for dz in (150.0, 170.0, 185.0):
        pitch = float(np.degrees(np.arctan2(dz - disc_h, depth)))
        ground = {"delta_z": dz, "roll": 0.0, "pitch": pitch, "fov": 60.0}
        cam = g.camera_from_ue_ground(ground, 1920, 1080)
        frames_geo.append(
            {
                "frame_id": f"frame_{dz:.0f}",
                "ground": ground,
                "points": {
                    "a": g.project(cam, a_src[None])[0].tolist(),
                    "b": g.project(cam, b_src[None])[0].tolist(),
                    "c_disc_bottom": g.project(cam, disc[None])[0].tolist(),
                },
            }
        )
    # Inject a valid-geometry manifest with our provenance fields.
    geo_manifest = {
        "units": "cm",
        "image_size": [1920, 1080],
        "source": "real",
        "geometry_source": "real_ue_WheelsDataset_v0_2",
        "points_source": "model_prediction",
        "scenes": {
            "ActorA": {
                "gt_disc_position": disc.tolist(),
                "frames": frames_geo,
            }
        },
    }
    report = r.run_report(geo_manifest, rng=np.random.default_rng(0))
    assert report["gate_status"] == "gate", (
        f"Expected gate_status='gate', got {report['gate_status']!r}. "
        "Check source/points_source fields."
    )


def test_promotion_gate_insufficient_evidence_without_geometry():
    """A report from a model-prediction manifest that has NOT passed geometry
    (empty scenes -> no sigma-estimable scenes) returns insufficient_evidence
    because n_sigma_estimable=0, confirming the gate is not self-certified."""
    report = {
        "units": "cm",
        "source": "real",
        "gate_status": "gate",
        "n_sigma_estimable": 0,  # no scenes passed geometry
        "sigma_cm": {"median": float("inf"), "p95": float("inf"), "max": float("inf")},
        "acceptance": {
            "sigma_accept_cm": 3.0,
            "sigma_target_cm": 1.0,
            "pass_accept": False,
            "pass_target": False,
        },
    }
    item = pg.evaluate_3d_acceptance(report)
    assert item.ok is False
    assert item.severity in {"insufficient_evidence", "production_fail"}


def test_promotion_gate_passes_when_geometry_correct():
    """When gate_status='gate' AND geometry passes, the promotion gate is green."""
    report = {
        "units": "cm",
        "source": "real",
        "gate_status": "gate",
        "n_sigma_estimable": 4,
        "sigma_cm": {"median": 1.2, "p95": 1.8, "max": 2.1},
        "acceptance": {
            "sigma_accept_cm": 3.0,
            "sigma_target_cm": 1.0,
            "pass_accept": True,
            "pass_target": False,
        },
    }
    item = pg.evaluate_3d_acceptance(report)
    assert item.ok is True
    assert item.severity == "pass"


def test_promotion_gate_informational_without_model_prediction():
    """A real-geometry manifest with points_source='ue_ground_truth' must
    remain informational — the gate must NOT flip green."""
    report = {
        "units": "cm",
        "source": "real_geometry_gt2d",
        "gate_status": "informational",
        "n_sigma_estimable": 6,
        "sigma_cm": {"median": 0.5, "p95": 0.8, "max": 1.0},
        "acceptance": {
            "sigma_accept_cm": 3.0,
            "sigma_target_cm": 1.0,
            "pass_accept": True,
            "pass_target": True,
        },
    }
    item = pg.evaluate_3d_acceptance(report)
    assert item.ok is False
    assert item.severity == "insufficient_evidence"


# ---------------------------------------------------------------------------
# match_predictions_to_actors
# ---------------------------------------------------------------------------


def test_match_nearest_neighbour():
    pred_wheels = [
        _confirmed_wheel(cx=200.0, cy=300.0),
        _confirmed_wheel(cx=800.0, cy=300.0),
    ]
    gt_actors = [
        {"actor": "ActorA", "kp_image_center": (210.0, 295.0)},
        {"actor": "ActorB", "kp_image_center": (790.0, 305.0)},
    ]
    result = mp.match_predictions_to_actors(pred_wheels, gt_actors)
    assert result["ActorA"] is pred_wheels[0]
    assert result["ActorB"] is pred_wheels[1]


def test_match_no_predictions_returns_none_for_all():
    gt_actors = [
        {"actor": "ActorA", "kp_image_center": (640.0, 360.0)},
    ]
    result = mp.match_predictions_to_actors([], gt_actors)
    assert result["ActorA"] is None


def test_match_one_prediction_two_actors_one_gets_none():
    """Only one prediction: the closer actor gets it; the farther gets None."""
    pred_wheels = [_confirmed_wheel(cx=100.0, cy=100.0)]
    gt_actors = [
        {"actor": "Close", "kp_image_center": (105.0, 100.0)},
        {"actor": "Far", "kp_image_center": (900.0, 500.0)},
    ]
    result = mp.match_predictions_to_actors(pred_wheels, gt_actors)
    assert result["Close"] is pred_wheels[0]
    assert result["Far"] is None


def test_strict_match_rejects_prediction_count_mismatch():
    pred_wheels = [_confirmed_wheel(cx=100.0, cy=100.0)]
    gt_actors = [
        {"actor": "Close", "kp_image_center": (105.0, 100.0)},
        {"actor": "Far", "kp_image_center": (900.0, 500.0)},
    ]

    result = mp.match_predictions_to_actors(
        pred_wheels,
        gt_actors,
        require_exact_count=True,
    )

    assert result is None


def test_strict_match_rejects_duplicate_nearest_actor():
    pred_wheels = [
        _confirmed_wheel(cx=100.0, cy=100.0),
        _confirmed_wheel(cx=120.0, cy=100.0),
    ]
    gt_actors = [
        {"actor": "Close", "kp_image_center": (105.0, 100.0)},
        {"actor": "Far", "kp_image_center": (900.0, 500.0)},
    ]

    result = mp.match_predictions_to_actors(
        pred_wheels,
        gt_actors,
        require_exact_count=True,
        reject_duplicate_nearest=True,
    )

    assert result is None


def test_match_empty_actors():
    pred_wheels = [_confirmed_wheel()]
    result = mp.match_predictions_to_actors(pred_wheels, [])
    assert result == {}


# ---------------------------------------------------------------------------
# extract_points_from_confirmed_wheel
# ---------------------------------------------------------------------------


def test_extract_points_all_keys_present():
    wheel = _confirmed_wheel()
    pts = mp.extract_points_from_confirmed_wheel(wheel)
    assert pts is not None
    assert set(pts) == {"a", "b", "c_disc_bottom"}
    assert isinstance(pts["a"], list) and len(pts["a"]) == 2


def test_extract_points_missing_key_returns_none():
    wheel = _confirmed_wheel()
    del wheel["points"]["c_disc_bottom"]
    pts = mp.extract_points_from_confirmed_wheel(wheel)
    assert pts is None


def test_extract_points_empty_points_returns_none():
    wheel = {"bbox_xyxy": [0, 0, 100, 100], "confidence": 0.9, "points": {}}
    pts = mp.extract_points_from_confirmed_wheel(wheel)
    assert pts is None
