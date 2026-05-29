"""Certify the UE-pose -> harness-camera convention by reprojection.

The fixture below is a verbatim slice of a real MCP export frame
(``WheelsDataset_v0_2/annotations/000001.json``): the full camera pose
plus paired world/image keypoints. If ``camera_from_ue_pose`` ever drifts
from the certified convention, the reprojection error explodes and these
tests fail — that is the guard the live ``EXPORT_PARITY_AUDIT`` asked for
("confirm the sign/zero against one clean UE export frame").
"""

from __future__ import annotations

import numpy as np

import camera_from_ue_pose as cp
import eval3d_floorray as g


# --- real frame 000001 fixture (cm world, px image) -------------------------
IMG_W, IMG_H, FOV = 1280, 720, 75.0
LOCATION = [797.2269888967276, 49.175199502883835, 200.00000009313226]
ROTATION = [0.0, -8.54400131732797, -176.47030743917554]  # roll, pitch, yaw
WORLD_TO_IMAGE = {
    # UE world (x, y, z) cm  ->  image (u, v) px
    (170.0, 110.0, 33.0): (510.597, 454.707),
    (170.0, 97.5, 28.0): (527.116, 460.763),
    (170.0, 97.5, 93.0): (525.392, 377.478),
    (170.0, 122.5, 28.0): (494.338, 461.300),
    (170.0, 122.5, 93.0): (492.107, 377.823),
    (170.0, -110.0, 33.0): (793.464, 450.198),
}


def test_real_frame_reprojection_is_subpixel():
    cam = cp.camera_from_ue_pose(LOCATION, ROTATION, FOV, IMG_W, IMG_H)
    errs = []
    for world, image in WORLD_TO_IMAGE.items():
        proj = g.project(cam, cp.ue_world_to_rh(world)[None])[0]
        errs.append(np.linalg.norm(proj - np.array(image, float)))
    errs = np.array(errs)
    # exported fixture is rounded to 3 dp, so allow a few thousandths of a px
    assert errs.max() < 0.01, f"max reproj {errs.max():.5f}px — convention drifted"


def test_reprojection_errors_helper_matches_manual():
    cam = cp.camera_from_ue_pose(LOCATION, ROTATION, FOV, IMG_W, IMG_H)
    kw = {f"p{i}": list(w) for i, w in enumerate(WORLD_TO_IMAGE)}
    ki = {f"p{i}": list(v) for i, v in enumerate(WORLD_TO_IMAGE.values())}
    errs = cp.reprojection_errors(cam, kw, ki)
    assert len(errs) == len(WORLD_TO_IMAGE)
    assert errs.max() < 0.01


def test_camera_center_is_y_negated_location():
    cam = cp.camera_from_ue_pose(LOCATION, ROTATION, FOV, IMG_W, IMG_H)
    assert np.allclose(cam.C, np.array(LOCATION) * np.array([1, -1, 1]))


def test_fov_is_horizontal_not_vertical():
    # A vertical-FOV intrinsic would not reproduce the export; assert the
    # certified (horizontal) build beats the vertical one by a wide margin.
    cam_h = cp.camera_from_ue_pose(LOCATION, ROTATION, FOV, IMG_W, IMG_H)
    f_v = (IMG_H / 2.0) / np.tan(np.radians(FOV) / 2.0)
    K_v = np.array([[f_v, 0, IMG_W / 2.0], [0, f_v, IMG_H / 2.0], [0, 0, 1.0]])
    cam_v = g.Camera(K=K_v, R=cam_h.R, C=cam_h.C)
    worlds = np.array([cp.ue_world_to_rh(w) for w in WORLD_TO_IMAGE])
    images = np.array(list(WORLD_TO_IMAGE.values()), float)
    err_h = np.linalg.norm(g.project(cam_h, worlds) - images, axis=1).max()
    err_v = np.linalg.norm(g.project(cam_v, worlds) - images, axis=1).max()
    assert err_h < 0.01 < err_v


def test_synthetic_roundtrip_arbitrary_pose():
    # Independent of the fixture: project known world points through the
    # built camera and confirm an exact round-trip for a generic pose.
    loc = [300.0, -120.0, 175.0]
    rot = [0.0, -20.0, 90.0]
    cam = cp.camera_from_ue_pose(loc, rot, 60.0, 1920, 1080)
    pts_ue = np.array([[10.0, 20.0, 30.0], [-40.0, 5.0, 12.0], [0.0, -60.0, 50.0]])
    pts_rh = pts_ue * np.array([1, -1, 1])
    px = g.project(cam, pts_rh)
    # back-project each pixel; the ray must pass through the original point
    for p_rh, uv in zip(pts_rh, px):
        o, d = g.pixel_to_ray(cam, uv)
        t = np.dot(p_rh - o, d)
        closest = o + t * d
        assert np.linalg.norm(closest - p_rh) < 1e-6
