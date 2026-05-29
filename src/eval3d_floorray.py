"""ML-side 3D-eval harness for the floor-ray wheel-pose contract.

A deterministic, numpy-only replay of the *AR-side* 3D pipeline
(``docs/AR_ML_CONTRACT.md``, ``docs/KEYPOINT_SPEC.md`` "Why three
points"). The AR client owns this geometry on device; this module
reproduces it offline so the ML side can **measure** how well its 2D
A/B/C predictions reconstruct in 3D — without changing the frozen 2D
output contract.

Pipeline per wheel (one "scene" = K frames of the same wheel):

  1. raycast the predicted screen-space ``a`` / ``b`` onto the floor
     plane (z = 0) using per-frame camera intrinsics + pose → two floor
     anchors per frame;
  2. RANSAC a *vertical* wheel plane through all floor anchors (normal
     horizontal, plane perpendicular to the floor);
  3. raycast ``c_disc_bottom`` onto that recovered vertical plane per
     frame → 3D disc-bottom positions;
  4. average across inlier frames → final disc-bottom; report the
     cross-frame disc-height sigma and, when a 3D ground-truth disc
     position is supplied, the disc-height / position error.

Acceptance target (3D error budget still open,
``docs/OPEN_QUESTIONS_AR_SPEC.md`` §9): disc-height sigma < 3 cm,
eventually < 1 cm. This harness emits
that number; it does not set the gate.

Conventions (internal, self-consistent — the UE adapter maps into it):

  - World: right-handed, floor = plane z = 0, world-up = +z. Disc
    height above the floor is therefore the z-coordinate of the
    recovered point. Units follow the caller's GT (UE export is cm, so
    the cm thresholds map 1:1).
  - Image: pixels, top-left origin, +x right, +y down
    (``docs/KEYPOINT_SPEC.md``).
  - Camera: OpenCV pinhole — camera +z forward, +x right, +y down;
    ``X_cam = R @ (X_world - C)`` with ``R`` world->camera and ``C`` the
    camera center in world.

What this module is NOT: it is not inference, not a contract change,
and not a training loss. The 2D ML output stays exactly as
``docs/AR_ML_CONTRACT.md`` specifies; this only scores it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-9
WORLD_UP = np.array([0.0, 0.0, 1.0])
FLOOR_NORMAL = np.array([0.0, 0.0, 1.0])
FLOOR_OFFSET = 0.0


@dataclass(frozen=True)
class Camera:
    """Pinhole camera. ``K`` 3x3 intrinsics, ``R`` 3x3 world->camera,
    ``C`` camera center in world (3,)."""

    K: np.ndarray
    R: np.ndarray
    C: np.ndarray


# ---------------------------------------------------------------------------
# Intrinsics & pose
# ---------------------------------------------------------------------------


def intrinsics_from_fov(
    fov_deg: float, img_w: int, img_h: int, fov_axis: str = "horizontal"
) -> np.ndarray:
    """Build a pinhole K from a single field-of-view angle.

    Square pixels (fx == fy); principal point at the image center. The
    UE ``Ground`` metadata reports a single ``FOV`` — by UE convention
    that is the horizontal FOV, which is the default here.
    """
    if fov_axis == "horizontal":
        dim = img_w
    elif fov_axis == "vertical":
        dim = img_h
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"fov_axis must be horizontal|vertical, got {fov_axis!r}")
    f = (dim / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    return np.array(
        [
            [f, 0.0, img_w / 2.0],
            [0.0, f, img_h / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )


def look_at(
    eye: np.ndarray, target: np.ndarray, up: np.ndarray = WORLD_UP
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(R, C)`` for a camera at ``eye`` looking at ``target``.

    ``R`` is world->camera (rows are the camera axes in world); ``C`` is
    the camera center. OpenCV axis convention: +z forward, +x right,
    +y down.
    """
    eye = np.asarray(eye, float)
    z = target - eye
    nz = np.linalg.norm(z)
    if nz < EPS:
        raise ValueError("eye and target coincide")
    z = z / nz
    x = np.cross(z, up)
    nx = np.linalg.norm(x)
    if nx < EPS:
        raise ValueError("view direction parallel to up hint; pick another up")
    x = x / nx
    y = np.cross(z, x)
    R = np.stack([x, y, z])  # rows = camera axes in world
    return R, eye


# ---------------------------------------------------------------------------
# Projection / unprojection
# ---------------------------------------------------------------------------


def project(cam: Camera, pts_world: np.ndarray) -> np.ndarray:
    """Project (N,3) world points to (N,2) pixels."""
    pts_world = np.atleast_2d(np.asarray(pts_world, float))
    xc = (cam.R @ (pts_world - cam.C).T).T  # (N,3) camera coords
    z = xc[:, 2]
    fx, fy = cam.K[0, 0], cam.K[1, 1]
    cx, cy = cam.K[0, 2], cam.K[1, 2]
    u = fx * xc[:, 0] / z + cx
    v = fy * xc[:, 1] / z + cy
    return np.stack([u, v], axis=-1)


def pixel_to_ray(cam: Camera, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a pixel to a world-space ray ``(origin, direction)``.

    ``origin`` is the camera center; ``direction`` is unit-length and
    points into the scene.
    """
    u, v = float(uv[0]), float(uv[1])
    fx, fy = cam.K[0, 0], cam.K[1, 1]
    cx, cy = cam.K[0, 2], cam.K[1, 2]
    dir_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    dir_world = cam.R.T @ dir_cam
    dir_world = dir_world / np.linalg.norm(dir_world)
    return cam.C.copy(), dir_world


def ray_plane_intersect(
    origin: np.ndarray, direction: np.ndarray, normal: np.ndarray, offset: float
) -> np.ndarray | None:
    """Intersect ray ``origin + s*direction`` with plane ``n.X = offset``.

    Returns the 3D hit, or ``None`` when the ray is parallel to the
    plane or the intersection lies at/behind the origin.
    """
    denom = float(np.dot(normal, direction))
    if abs(denom) < EPS:
        return None
    s = (offset - float(np.dot(normal, origin))) / denom
    if s <= EPS:
        return None
    return origin + s * direction


# ---------------------------------------------------------------------------
# Vertical wheel plane
# ---------------------------------------------------------------------------


def vertical_plane_through(
    a_floor: np.ndarray, b_floor: np.ndarray
) -> tuple[np.ndarray, float]:
    """Vertical plane (perpendicular to the floor) through two floor
    anchors. Returns ``(normal, offset)`` with ``normal`` horizontal and
    unit-length, ``offset = normal . a_floor``.
    """
    base = np.asarray(b_floor, float) - np.asarray(a_floor, float)
    n = np.cross(WORLD_UP, base)
    nn = np.linalg.norm(n)
    if nn < EPS:
        raise ValueError("floor anchors coincide; cannot define a wheel plane")
    n = n / nn
    return n, float(np.dot(n, a_floor))


def _fit_vertical_plane_lsq(pts_xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Total-least-squares vertical plane (a line in the xy-floor) through
    points. Returns horizontal unit normal (3,) and offset."""
    centroid = pts_xy.mean(axis=0)
    centered = pts_xy - centroid
    # smallest-variance direction = plane normal in the floor (xy)
    _, _, vt = np.linalg.svd(centered)
    n2 = vt[-1]
    n2 = n2 / np.linalg.norm(n2)
    n = np.array([n2[0], n2[1], 0.0])
    return n, float(np.dot(n2, centroid))


def fit_vertical_plane_ransac(
    pts: np.ndarray,
    threshold: float = 0.02,
    iters: int = 200,
    rng: np.random.Generator | None = None,
) -> dict:
    """RANSAC a vertical plane to floor anchor points ``pts`` (M,3).

    A vertical plane intersects the floor in a line, so the fit lives in
    the xy-projection. Returns ``{normal, offset, inliers}`` where
    ``inliers`` is a bool mask over the input rows.
    """
    if rng is None:  # pragma: no cover - tests always pass an rng
        rng = np.random.default_rng()
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    pts = np.atleast_2d(np.asarray(pts, float))
    m = len(pts)
    if m < 2:
        raise ValueError("need >= 2 floor anchors for a wheel plane")
    xy = pts[:, :2]
    # Coincident anchors define no plane — RANSAC would otherwise fall
    # through to an LSQ on a single distinct point and return an
    # arbitrary normal with a clean-looking (but meaningless) metric.
    if np.ptp(xy, axis=0).max() < EPS:
        raise ValueError("floor anchors are coincident; no wheel plane is defined")

    best_inliers = None
    best_count = -1
    for _ in range(iters):
        i, j = rng.choice(m, size=2, replace=False)
        if np.linalg.norm(xy[i] - xy[j]) < EPS:
            continue
        n, off = vertical_plane_through(pts[i], pts[j])
        dist = np.abs(xy @ n[:2] - off)
        inliers = dist < threshold
        c = int(inliers.sum())
        if c > best_count:
            best_count = c
            best_inliers = inliers

    if best_inliers is None or best_inliers.sum() < 2:
        # degenerate sampling fallback: fit all points
        best_inliers = np.ones(m, dtype=bool)

    n, off = _fit_vertical_plane_lsq(xy[best_inliers])
    # refine inliers against the refitted plane
    dist = np.abs(xy @ n[:2] - off)
    inliers = dist < threshold
    if inliers.sum() < 2:
        inliers = best_inliers
    return {"normal": n, "offset": off, "inliers": inliers}


# ---------------------------------------------------------------------------
# Scene simulation
# ---------------------------------------------------------------------------


def simulate_scene(
    frames: list[dict],
    gt_disc_position: np.ndarray | None = None,
    ransac_threshold: float = 0.02,
    ransac_iters: int = 200,
    rng: np.random.Generator | None = None,
) -> dict:
    """Replay the AR 3D pipeline over ``frames`` of one wheel.

    Each frame is a dict ``{"camera": Camera, "a": (2,), "b": (2,),
    "c": (2,)}`` of predicted screen-space points.

    Returns a dict with the recovered plane, per-frame disc points and
    heights, the mean disc-bottom, ``disc_height_mean`` /
    ``disc_height_sigma``, and (when ``gt_disc_position`` is given)
    ``height_error`` / ``position_error``.
    """
    if rng is None:  # pragma: no cover
        rng = np.random.default_rng()

    # 1. floor anchors per frame (a, b raycast onto the floor)
    anchors = []  # flat list of 3D floor points
    frame_anchor_idx = []  # (ia, ib) into anchors, or None if a miss
    for fr in frames:
        cam = fr["camera"]
        oa, da = pixel_to_ray(cam, np.asarray(fr["a"], float))
        ob, db = pixel_to_ray(cam, np.asarray(fr["b"], float))
        ha = ray_plane_intersect(oa, da, FLOOR_NORMAL, FLOOR_OFFSET)
        hb = ray_plane_intersect(ob, db, FLOOR_NORMAL, FLOOR_OFFSET)
        if ha is None or hb is None:
            frame_anchor_idx.append(None)
            continue
        ia, ib = len(anchors), len(anchors) + 1
        anchors.extend([ha, hb])
        frame_anchor_idx.append((ia, ib))

    if len(anchors) < 2:
        raise ValueError("scene has fewer than 2 valid floor anchors")
    anchors = np.array(anchors)

    # 2. RANSAC vertical wheel plane
    fit = fit_vertical_plane_ransac(
        anchors, threshold=ransac_threshold, iters=ransac_iters, rng=rng
    )
    n, off, anchor_inliers = fit["normal"], fit["offset"], fit["inliers"]

    # 3. raycast C onto the recovered plane, per frame; keep inlier frames
    disc_points = []
    for fr, idx in zip(frames, frame_anchor_idx):
        if idx is None:
            continue
        ia, ib = idx
        if not (anchor_inliers[ia] and anchor_inliers[ib]):
            continue
        oc, dc = pixel_to_ray(fr["camera"], np.asarray(fr["c"], float))
        hit = ray_plane_intersect(oc, dc, n, off)
        if hit is not None:
            disc_points.append(hit)

    # fallback: if inlier filtering left nothing, use every C that hits
    if not disc_points:
        for fr, idx in zip(frames, frame_anchor_idx):
            if idx is None:
                continue
            oc, dc = pixel_to_ray(fr["camera"], np.asarray(fr["c"], float))
            hit = ray_plane_intersect(oc, dc, n, off)
            if hit is not None:
                disc_points.append(hit)

    if not disc_points:
        raise ValueError("no C ray intersected the recovered wheel plane")

    disc_points = np.array(disc_points)
    heights = disc_points[:, 2]
    mean_point = disc_points.mean(axis=0)

    n_frames = int(len(disc_points))
    out = {
        "plane_normal": n,
        "plane_offset": off,
        "anchor_inliers": anchor_inliers,
        "n_inlier_frames": n_frames,
        "disc_points": disc_points,
        "disc_heights": heights.tolist(),
        "disc_bottom": mean_point,
        "disc_height_mean": float(heights.mean()),
        # population std is identically 0 for a single frame — that is an
        # absence of evidence, not a perfect reconstruction. Flag it so the
        # driver does not count a 1-frame scene as passing the sigma gate.
        "disc_height_sigma": float(heights.std()),
        "sigma_estimable": n_frames >= 2,
    }
    if gt_disc_position is not None:
        gt = np.asarray(gt_disc_position, float)
        out["height_error"] = float(abs(heights.mean() - gt[2]))
        out["position_error"] = float(np.linalg.norm(mean_point - gt))
    return out


# ---------------------------------------------------------------------------
# UE ground-meta adapter
# ---------------------------------------------------------------------------


def camera_from_ue_ground(
    ground_meta: dict,
    img_w: int,
    img_h: int,
    yaw_deg: float = 0.0,
    position_xy: tuple[float, float] = (0.0, 0.0),
) -> Camera:
    """Build a :class:`Camera` from UE ``Ground`` metadata
    (``DeltaZ``/``Roll``/``Pitch``/``FOV`` — see
    ``scripts/inspect_unreal_export.py``).

    Intrinsics from ``FOV`` (horizontal) are exact. The pose mapping
    uses this module's documented convention: the camera sits at height
    ``DeltaZ`` on the world-up axis above the origin, ``pitch`` is the
    downward tilt from the horizon (90 deg = straight down at the floor)
    and ``roll`` rotates about the optical axis.

    The UE ``Ground`` meta carries no heading (``yaw``) or camera xy
    translation, so by default the camera faces along +world_y from the
    origin column. Real walk-around captures vary both; ``yaw`` and
    ``position_xy`` are accepted (read from ``ground_meta`` if present,
    else the kwargs) so wiring a richer export is a one-call change. The
    default (yaw 0, xy origin) is exactly what the synthetic generator
    assumes.

    NOTE: the sign/zero of UE's Roll/Pitch relative to this convention
    must be confirmed against one clean UE export frame before the pose
    is trusted for real GT (the export is the current upstream blocker,
    ``docs/EXPORT_PARITY_AUDIT.md``). ``FOV`` -> ``K`` and the camera
    height are the parts safe to rely on today.
    """
    fov = float(ground_meta["fov"])
    delta_z = float(ground_meta["delta_z"])
    roll = float(ground_meta.get("roll", 0.0))
    pitch = float(ground_meta.get("pitch", 0.0))
    yaw = float(ground_meta.get("yaw", yaw_deg))
    pos_x, pos_y = ground_meta.get("position_xy", position_xy)

    K = intrinsics_from_fov(fov, img_w, img_h)

    eye = np.array([float(pos_x), float(pos_y), delta_z])
    # pitch: downward tilt from the horizon (90deg = straight down);
    # yaw: heading about world-up (0 = +y).
    p = np.radians(pitch)
    y = np.radians(yaw)
    forward = np.array([np.sin(y) * np.cos(p), np.cos(y) * np.cos(p), -np.sin(p)])
    target = eye + forward
    up_hint = WORLD_UP if abs(np.sin(p)) < 0.999 else np.array([0.0, 1.0, 0.0])
    R, C = look_at(eye, target, up_hint)

    if abs(roll) > EPS:
        cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
        roll_cam = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
        R = roll_cam @ R
    return Camera(K=K, R=R, C=C)
