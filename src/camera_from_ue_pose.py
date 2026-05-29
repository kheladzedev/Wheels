"""Build a 3D-eval :class:`Camera` from a *full* Unreal camera pose.

The shipped adapter ``eval3d_floorray.camera_from_ue_ground`` only knows
the four ``Ground`` scalars (``DeltaZ``/``Roll``/``Pitch``/``FOV``) and
has to assume yaw 0 + camera-on-the-origin-column; its docstring flags
the Roll/Pitch sign/zero convention as *unverified* against a real
export (``docs/EXPORT_PARITY_AUDIT.md``).

The MCP renderer (``WheelsDataset_v0_2``) instead exports the full camera
pose per frame — world ``location`` (cm), ``rotation`` ``[roll, pitch,
yaw]`` (deg), and ``fov`` (deg) — *plus*, in the rich annotation, paired
``keypoints_world`` / ``keypoints_image``. That pairing lets the UE→OpenCV
convention be **certified by reprojection**, which it now is: this
construction reproduces every exported image keypoint to < 1e-3 px across
the whole orbit (``scripts/certify_ue_export_parity.py``).

Certified convention (do not change without re-running the parity check):

  - **FOV is horizontal.** ``intrinsics_from_fov`` on the image width.
  - **UE world is left-handed** (X fwd, Y right, Z up). The harness world
    is right-handed with floor ``z = 0`` and world-up ``+z``; the bridge
    is a single **Y-negation** (``ue_world_to_rh``) applied to *both* the
    camera center and every world point. Disc *height* is the z-coord and
    is invariant under that flip, so the eval budget maps 1:1.
  - **Forward** from the UE rotator:
    ``F = (cos(yaw)cos(pitch), sin(yaw)cos(pitch), sin(pitch))`` in UE
    world, then Y-negated. Orientation is fixed with the harness'
    :func:`eval3d_floorray.look_at`; ``roll`` rotates about the optical
    axis (sign matches ``camera_from_ue_ground``).

This is *measurement-only* plumbing, exactly like the rest of the
harness: it does not run inference, does not change the frozen 2D output
contract, and does not train.
"""

from __future__ import annotations

import numpy as np

import eval3d_floorray as g

# UE (left-handed, +Z up) -> harness (right-handed, +Z up): negate Y.
_UE_TO_RH = np.array([1.0, -1.0, 1.0])


def ue_world_to_rh(point) -> np.ndarray:
    """Map a UE world point (cm, left-handed) into the harness' RH frame."""
    return np.asarray(point, float) * _UE_TO_RH


def ue_forward(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """UE world-space forward unit vector for a ``[roll, pitch, yaw]`` rotator.

    Roll does not change the optical axis; it is applied about that axis
    when the camera is assembled.
    """
    p = np.radians(pitch_deg)
    y = np.radians(yaw_deg)
    return np.array([np.cos(y) * np.cos(p), np.sin(y) * np.cos(p), np.sin(p)])


def camera_from_ue_pose(
    location,
    rotation,
    fov_deg: float,
    img_w: int,
    img_h: int,
) -> g.Camera:
    """Build a harness :class:`~eval3d_floorray.Camera` from a full UE pose.

    ``location`` is the UE world camera position ``[x, y, z]`` in cm;
    ``rotation`` is ``[roll, pitch, yaw]`` in degrees (the order the MCP
    rich annotation exports). ``fov_deg`` is the **horizontal** FOV.

    Reprojection of the exported ``keypoints_world`` through the returned
    camera matches the exported ``keypoints_image`` to < 1e-3 px (parity
    certified, see module docstring).
    """
    roll, pitch, yaw = (float(rotation[0]), float(rotation[1]), float(rotation[2]))
    forward = ue_forward(roll, pitch, yaw) * _UE_TO_RH
    center = ue_world_to_rh(location)
    R, C = g.look_at(center, center + forward, g.WORLD_UP)
    if abs(roll) > g.EPS:
        cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
        roll_cam = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
        R = roll_cam @ R
    K = g.intrinsics_from_fov(fov_deg, img_w, img_h, fov_axis="horizontal")
    return g.Camera(K=K, R=R, C=C)


def reprojection_errors(
    camera: g.Camera,
    keypoints_world: dict,
    keypoints_image: dict,
    *,
    visibility: dict | None = None,
) -> np.ndarray:
    """Per-keypoint reprojection error (px) for one wheel.

    Projects each UE world keypoint (after the RH flip) and compares to
    the exported image keypoint. Only keys present in both dicts (and
    visible, when ``visibility`` is given) are scored.
    """
    errs: list[float] = []
    for name, world in keypoints_world.items():
        if name not in keypoints_image:
            continue
        if visibility is not None and not visibility.get(name, True):
            continue
        proj = g.project(camera, ue_world_to_rh(world)[None])[0]
        errs.append(
            float(np.linalg.norm(proj - np.asarray(keypoints_image[name], float)))
        )
    return np.asarray(errs, float)
