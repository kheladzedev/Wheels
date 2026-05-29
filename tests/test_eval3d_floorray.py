"""Tests for the ML-side 3D-eval harness (``src/eval3d_floorray.py``).

The harness is a deterministic, numpy-only replay of the AR client's 3D
pipeline (``docs/AR_ML_CONTRACT.md`` AR-side, ``docs/KEYPOINT_SPEC.md``
"Why three points"): raycast the predicted screen-space A/B onto the
floor, RANSAC a vertical wheel plane through the two floor anchors, then
raycast C onto that plane to recover the disc-bottom 3D position. It does
NOT change the frozen 2D ML output contract — it only *measures* the
downstream 3D quality of 2D predictions, so we can score disc-height
sigma (acceptance: <3 cm / <1 cm) on a val set with known camera
intrinsics + floor.

Correctness is pinned by round-trip: build a known 3D scene, forward
project to pixels, run the harness, and require it to recover the scene
to ~0 error. A separate group pins the *sensitivity* of the metric —
when A/B drift off the floor onto the rim (the wheel-attraction failure
mode in ``docs/AR_REPLAY_METRIC_PLAN.md`` §1), the recovered disc height
must degrade well beyond the clean case. A metric that does not move on
that failure is useless.
"""

from __future__ import annotations

import numpy as np
import pytest

import eval3d_floorray as g


FLOOR_NORMAL = np.array([0.0, 0.0, 1.0])
FLOOR_OFFSET = 0.0


# ---------------------------------------------------------------------------
# Scene-building helpers (pure forward model — the trusted reference).
# ---------------------------------------------------------------------------


def _camera(eye, target, fov_deg=60.0, img_w=640, img_h=480, up=(0.0, 0.0, 1.0)):
    K = g.intrinsics_from_fov(fov_deg, img_w, img_h)
    R, C = g.look_at(
        np.asarray(eye, float), np.asarray(target, float), np.asarray(up, float)
    )
    return g.Camera(K=K, R=R, C=C)


# ---------------------------------------------------------------------------
# 1. Intrinsics
# ---------------------------------------------------------------------------


def test_intrinsics_principal_point_at_image_center():
    K = g.intrinsics_from_fov(60.0, 640, 480)
    assert K[0, 2] == pytest.approx(320.0)
    assert K[1, 2] == pytest.approx(240.0)
    assert K[2, 2] == pytest.approx(1.0)


def test_intrinsics_focal_matches_horizontal_fov():
    # fx = (w/2) / tan(hfov/2). For hfov=90deg, tan(45)=1 => fx = w/2.
    K = g.intrinsics_from_fov(90.0, 640, 480, fov_axis="horizontal")
    assert K[0, 0] == pytest.approx(320.0)
    # square pixels by default
    assert K[1, 1] == pytest.approx(K[0, 0])


# ---------------------------------------------------------------------------
# 2. Projection / unprojection round-trip
# ---------------------------------------------------------------------------


def test_project_unproject_roundtrip_recovers_direction():
    cam = _camera(eye=(0.0, -3.0, 1.5), target=(0.0, 0.0, 0.3))
    pts = np.array(
        [
            [0.0, 0.0, 0.3],
            [0.5, 0.2, 0.6],
            [-0.4, 0.1, 0.0],
        ]
    )
    uv = g.project(cam, pts)
    for p, (u, v) in zip(pts, uv):
        origin, direction = g.pixel_to_ray(cam, np.array([u, v]))
        # The original point must lie on the recovered ray.
        to_pt = p - origin
        to_pt = to_pt / np.linalg.norm(to_pt)
        assert np.dot(to_pt, direction) == pytest.approx(1.0, abs=1e-6)


def test_center_pixel_floor_hit_is_directly_below_for_nadir_camera():
    # Camera 2 m up looking straight down; up-hint sideways to avoid degeneracy.
    cam = _camera(eye=(0.3, -0.2, 2.0), target=(0.3, -0.2, 0.0), up=(0.0, 1.0, 0.0))
    origin, direction = g.pixel_to_ray(cam, np.array([320.0, 240.0]))
    hit = g.ray_plane_intersect(origin, direction, FLOOR_NORMAL, FLOOR_OFFSET)
    assert hit is not None
    assert hit[0] == pytest.approx(0.3, abs=1e-6)
    assert hit[1] == pytest.approx(-0.2, abs=1e-6)
    assert hit[2] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. Ray / plane intersection edge cases
# ---------------------------------------------------------------------------


def test_ray_parallel_to_plane_misses():
    origin = np.array([0.0, 0.0, 1.0])
    direction = np.array([1.0, 0.0, 0.0])  # horizontal, never meets z=0
    assert g.ray_plane_intersect(origin, direction, FLOOR_NORMAL, FLOOR_OFFSET) is None


def test_ray_pointing_away_from_plane_misses():
    origin = np.array([0.0, 0.0, 1.0])
    direction = np.array([0.0, 0.0, 1.0])  # up, away from floor
    assert g.ray_plane_intersect(origin, direction, FLOOR_NORMAL, FLOOR_OFFSET) is None


# ---------------------------------------------------------------------------
# 4. Vertical plane through floor anchors
# ---------------------------------------------------------------------------


def test_vertical_plane_is_perpendicular_to_floor_and_contains_anchors():
    a = np.array([1.0, 2.0, 0.0])
    b = np.array([3.0, 2.5, 0.0])
    n, d = g.vertical_plane_through(a, b)
    # vertical => normal horizontal (no z component)
    assert n[2] == pytest.approx(0.0, abs=1e-9)
    assert np.linalg.norm(n) == pytest.approx(1.0)
    # both anchors lie on the plane n.X = d
    assert np.dot(n, a) == pytest.approx(d, abs=1e-9)
    assert np.dot(n, b) == pytest.approx(d, abs=1e-9)


def test_ransac_vertical_plane_rejects_outliers():
    rng = np.random.default_rng(0)
    # A clean vertical plane: the line y = 2 on the floor (xy), points vary in x.
    base = np.array([[x, 2.0, 0.0] for x in np.linspace(-1, 1, 12)])
    outliers = np.array([[0.0, 2.8, 0.0], [0.5, 1.0, 0.0], [-0.7, 3.0, 0.0]])
    pts = np.vstack([base, outliers])
    fit = g.fit_vertical_plane_ransac(pts, threshold=0.05, iters=200, rng=rng)
    # The clean points are inliers, the three planted ones are not.
    assert fit["inliers"][: len(base)].all()
    assert not fit["inliers"][len(base) :].any()
    # Recovered plane should be ~ the y=2 plane => normal ~ (0,1,0), offset ~2.
    n = fit["normal"] * np.sign(fit["normal"][1])
    assert n[1] == pytest.approx(1.0, abs=1e-3)
    assert abs(fit["offset"]) == pytest.approx(2.0, abs=1e-2)


# ---------------------------------------------------------------------------
# 5. Full forward-simulation round-trip — the core correctness proof.
# ---------------------------------------------------------------------------


def _build_scene(disc_height, ab_height=0.0, fov_deg=60.0):
    """Construct frames of one wheel viewed from several camera poses.

    The wheel's base line on the floor runs through A=(-0.15, 0, 0) and
    B=(0.15, 0, 0). The disc bottom sits on the vertical plane above the
    base midpoint at height ``disc_height``. ``ab_height`` lifts the A/B
    *screen* sources off the floor (simulating wheel-attraction drift):
    at 0 they sit on the floor (correct), >0 they ride up the rim.
    """
    a_src = np.array([-0.15, 0.0, ab_height])
    b_src = np.array([0.15, 0.0, ab_height])
    disc_world = np.array([0.0, 0.0, disc_height])
    eyes = [
        (0.0, -2.0, 1.4),
        (0.8, -1.8, 1.5),
        (-0.7, -2.1, 1.3),
        (0.4, -2.4, 1.6),
        (-0.3, -1.9, 1.45),
    ]
    frames = []
    for eye in eyes:
        cam = _camera(eye=eye, target=(0.0, 0.0, 0.25), fov_deg=fov_deg)
        a_uv = g.project(cam, a_src[None])[0]
        b_uv = g.project(cam, b_src[None])[0]
        c_uv = g.project(cam, disc_world[None])[0]
        frames.append({"camera": cam, "a": a_uv, "b": b_uv, "c": c_uv})
    return frames, disc_world


def test_recovers_known_disc_height_with_negligible_error():
    frames, disc_world = _build_scene(disc_height=0.30, ab_height=0.0)
    rng = np.random.default_rng(1)
    res = g.simulate_scene(frames, gt_disc_position=disc_world, rng=rng)
    assert res["disc_height_mean"] == pytest.approx(0.30, abs=1e-4)
    assert res["disc_height_sigma"] == pytest.approx(0.0, abs=1e-4)
    assert res["height_error"] == pytest.approx(0.0, abs=1e-4)
    assert res["position_error"] == pytest.approx(0.0, abs=1e-4)


def test_multiframe_recovery_is_stable_across_poses():
    frames, _ = _build_scene(disc_height=0.22, ab_height=0.0)
    rng = np.random.default_rng(2)
    res = g.simulate_scene(frames, rng=rng)
    heights = np.array(res["disc_heights"])
    assert heights.std() == pytest.approx(0.0, abs=1e-4)
    assert np.allclose(heights, 0.22, atol=1e-4)


# ---------------------------------------------------------------------------
# 6. Metric sensitivity — must catch the failure modes it exists for.
# ---------------------------------------------------------------------------


def test_ab_drift_onto_rim_inflates_disc_height_error():
    rng = np.random.default_rng(3)
    clean, disc_world = _build_scene(disc_height=0.30, ab_height=0.0)
    drifted, _ = _build_scene(disc_height=0.30, ab_height=0.18)  # A/B ride up rim

    clean_res = g.simulate_scene(clean, gt_disc_position=disc_world, rng=rng)
    drift_res = g.simulate_scene(drifted, gt_disc_position=disc_world, rng=rng)

    # Clean recovery is essentially exact; drift must blow the error up.
    assert clean_res["height_error"] < 1e-3
    assert drift_res["height_error"] > 10 * clean_res["height_error"] + 0.02


def test_pixel_noise_increases_sigma():
    frames, _ = _build_scene(disc_height=0.30, ab_height=0.0)
    rng = np.random.default_rng(4)
    noisy = []
    for fr in frames:
        nf = dict(fr)
        nf["c"] = fr["c"] + rng.normal(0.0, 3.0, size=2)  # 3 px jitter on C
        noisy.append(nf)
    clean_res = g.simulate_scene(frames, rng=np.random.default_rng(4))
    noisy_res = g.simulate_scene(noisy, rng=np.random.default_rng(4))
    assert noisy_res["disc_height_sigma"] > clean_res["disc_height_sigma"]
    # bound the magnitude: 3 px jitter at this ~2 m geometry is sub-cm to
    # cm scale, not metres. A dead metric (~0) or an exploding one both fail.
    assert 1e-4 < noisy_res["disc_height_sigma"] < 5e-2


# ---------------------------------------------------------------------------
# 7. UE ground-meta adapter (FOV -> K is exact; pose convention documented).
# ---------------------------------------------------------------------------


def test_ue_ground_meta_intrinsics_match_fov():
    meta = {"delta_z": 170.0, "roll": 0.0, "pitch": 61.77, "fov": 54.66}
    cam = g.camera_from_ue_ground(meta, img_w=1920, img_h=1080)
    K_ref = g.intrinsics_from_fov(54.66, 1920, 1080)
    assert np.allclose(cam.K, K_ref)


def test_ue_camera_height_equals_delta_z():
    meta = {"delta_z": 170.0, "roll": 0.0, "pitch": 90.0, "fov": 60.0}
    cam = g.camera_from_ue_ground(meta, img_w=1920, img_h=1080)
    # delta_z is the camera height above the floor in the harness world.
    assert cam.C[2] == pytest.approx(170.0)


def test_intrinsics_from_ue_match_absolute_focal_formula():
    # Pin the adapter K independently of intrinsics_from_fov: fx must equal
    # (w/2)/tan(fov/2). Guards against a wrong fov-axis / arg-swap.
    meta = {"delta_z": 170.0, "roll": 0.0, "pitch": 60.0, "fov": 90.0}
    cam = g.camera_from_ue_ground(meta, img_w=1920, img_h=1080)
    assert cam.K[0, 0] == pytest.approx((1920 / 2) / np.tan(np.radians(90) / 2))
    assert cam.K[0, 2] == pytest.approx(960.0)
    assert cam.K[1, 2] == pytest.approx(540.0)


def test_ue_pose_pitch_convention_is_independent_of_roundtrip():
    # A round-trip test cancels any convention sign error. Pin the camera
    # forward axis (R[2], world->camera z-row) directly against pitch.
    nadir = g.camera_from_ue_ground(
        {"delta_z": 170.0, "roll": 0.0, "pitch": 90.0, "fov": 60.0}, 1920, 1080
    )
    assert np.allclose(nadir.R[2], [0.0, 0.0, -1.0], atol=1e-6)  # straight down
    horizon = g.camera_from_ue_ground(
        {"delta_z": 170.0, "roll": 0.0, "pitch": 0.0, "fov": 60.0}, 1920, 1080
    )
    assert np.allclose(horizon.R[2], [0.0, 1.0, 0.0], atol=1e-6)  # along +y


def test_ue_yaw_rotates_camera_heading():
    cam = g.camera_from_ue_ground(
        {"delta_z": 170.0, "roll": 0.0, "pitch": 0.0, "fov": 60.0, "yaw": 90.0},
        1920,
        1080,
    )
    # yaw=90 at the horizon points the camera along +x.
    assert np.allclose(cam.R[2], [1.0, 0.0, 0.0], atol=1e-6)


def test_degenerate_coincident_anchors_raise():
    pts = np.array([[1.0, 2.0, 0.0]] * 6)  # all identical -> no plane
    with pytest.raises(ValueError):
        g.fit_vertical_plane_ransac(pts, threshold=0.05, rng=np.random.default_rng(0))


def test_ransac_threshold_must_be_positive():
    pts = np.array([[x, 2.0, 0.0] for x in np.linspace(-1, 1, 6)])
    with pytest.raises(ValueError):
        g.fit_vertical_plane_ransac(pts, threshold=0.0, rng=np.random.default_rng(0))
