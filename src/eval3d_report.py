"""Val-set driver for the 3D-eval harness (``src/eval3d_floorray.py``).

Reads a *frames manifest* — per ``scene_id``, a list of frames carrying
UE ``Ground`` metadata (intrinsics + camera pose) and the model's
predicted screen-space ``a`` / ``b`` / ``c_disc_bottom`` — replays the
AR 3D pipeline per scene, and aggregates disc-height sigma across scenes
against the acceptance budget (< 3 cm, target < 1 cm; the 3D error
budget is still open, ``docs/OPEN_QUESTIONS_AR_SPEC.md`` §9).

Manifest shape (JSON), units are the caller's (UE export is cm)::

    {
      "units": "cm",
      "image_size": [W, H],
      "scenes": {
        "<scene_id>": {
          "gt_disc_position": [x, y, z],          # optional
          "frames": [
            {
              "frame_id": "...",
              "ground": {"delta_z":.., "roll":.., "pitch":.., "fov":..},
              "points": {"a":[x,y], "b":[x,y], "c_disc_bottom":[x,y]}
            }
          ]
        }
      }
    }

This is *measurement* only — it does not run inference, does not change
the frozen 2D ML output contract, and does not train. It scores 2D
predictions in 3D. Running it on real predictions is gated on a clean
UE export with intrinsics + pose (``docs/EXPORT_PARITY_AUDIT.md``);
until then it validates the plumbing on synthetic round-trip frames.

Usage::

    python src/eval3d_report.py \\
        --manifest outputs/eval3d/frames_manifest.json \\
        --out outputs/eval3d/disc_height_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import camera_from_ue_pose as cp
import eval3d_floorray as g


SIGMA_ACCEPT_CM = 3.0
SIGMA_TARGET_CM = 1.0


def load_manifest(path: str | Path) -> dict:
    """Load + minimally validate a frames manifest."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "scenes" not in data or not isinstance(data["scenes"], dict):
        raise ValueError("manifest missing a 'scenes' object")
    return data


def _build_camera(frame: dict, image_size: tuple[int, int]) -> g.Camera:
    """Build the per-frame camera from whatever pose the manifest carries.

    Three accepted shapes, most-specific first:

      - ``frame["camera"]`` — explicit ``{K, R, C}`` matrices (already in
        the harness' OpenCV convention; used as-is).
      - ``frame["pose"]`` — full UE pose ``{location, rotation, fov}``
        (the MCP ``WheelsDataset_v0_2`` export); built via the parity-
        certified :func:`camera_from_ue_pose.camera_from_ue_pose`.
      - ``frame["ground"]`` — the legacy four ``Ground`` scalars; built
        via :func:`eval3d_floorray.camera_from_ue_ground` (yaw/xy assumed).
    """
    if "camera" in frame:
        cam = frame["camera"]
        return g.Camera(
            K=np.asarray(cam["K"], float),
            R=np.asarray(cam["R"], float),
            C=np.asarray(cam["C"], float),
        )
    if "pose" in frame:
        pose = frame["pose"]
        w, h = frame.get("image_size", image_size)
        return cp.camera_from_ue_pose(
            pose["location"], pose["rotation"], pose["fov"], int(w), int(h)
        )
    return g.camera_from_ue_ground(frame["ground"], image_size[0], image_size[1])


def run_scene(
    scene: dict,
    image_size: tuple[int, int],
    gt_disc_position: np.ndarray | None,
    ransac_threshold: float,
    rng: np.random.Generator,
) -> dict:
    """Replay one scene's frames through the harness."""
    frames = []
    for fr in scene["frames"]:
        cam = _build_camera(fr, image_size)
        pts = fr["points"]
        frames.append(
            {
                "camera": cam,
                "a": np.asarray(pts["a"], float),
                "b": np.asarray(pts["b"], float),
                "c": np.asarray(pts["c_disc_bottom"], float),
            }
        )
    res = g.simulate_scene(
        frames,
        gt_disc_position=gt_disc_position,
        ransac_threshold=ransac_threshold,
        rng=rng,
    )
    out = {
        "disc_height_mean": res["disc_height_mean"],
        "disc_height_sigma": res["disc_height_sigma"],
        "sigma_estimable": res["sigma_estimable"],
        "n_inlier_frames": res["n_inlier_frames"],
        "n_frames": len(frames),
    }
    if "height_error" in res:
        out["height_error"] = res["height_error"]
        out["position_error"] = res["position_error"]
    return out


def run_report(
    manifest: dict,
    ransac_threshold: float = 2.0,
    sigma_accept_cm: float = SIGMA_ACCEPT_CM,
    sigma_target_cm: float = SIGMA_TARGET_CM,
    rng: np.random.Generator | None = None,
) -> dict:
    """Run every scene and aggregate the disc-height metrics.

    ``ransac_threshold`` is in manifest units (cm by default). Acceptance
    is computed on the **median** scene sigma so a single bad scene does
    not by itself fail the batch (the per-scene table still surfaces it).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    image_size = tuple(manifest.get("image_size", [1920, 1080]))
    units = manifest.get("units", "cm")

    per_scene: dict[str, dict] = {}
    sigmas: list[float] = []  # only sigma-estimable scenes (>= 2 inlier frames)
    height_errors: list[float] = []
    n_with_gt = 0

    for scene_id, scene in manifest["scenes"].items():
        gt = scene.get("gt_disc_position")
        gt_arr = np.asarray(gt, float) if gt is not None else None
        res = run_scene(scene, image_size, gt_arr, ransac_threshold, rng)
        per_scene[scene_id] = res
        if res["sigma_estimable"]:
            sigmas.append(res["disc_height_sigma"])
        if "height_error" in res:
            n_with_gt += 1
            height_errors.append(res["height_error"])

    # A single-frame scene has std==0 by construction (no evidence, not a
    # perfect reconstruction) — excluded from the sigma stat above.
    if sigmas:
        sigmas_arr = np.array(sigmas)
        sigma_stats = {
            "median": float(np.median(sigmas_arr)),
            "p95": float(np.percentile(sigmas_arr, 95)),
            "max": float(sigmas_arr.max()),
        }
    else:
        inf = float("inf")
        sigma_stats = {"median": inf, "p95": inf, "max": inf}

    # Acceptance gates on cross-frame sigma AND, when 3D GT is available,
    # on the disc-height error vs GT. Sigma alone is blind to a *systematic*
    # A/B-drift bias: every frame is wrong the same way, so heights stay
    # consistent (low sigma) while the reconstruction is off (high error).
    # The error gate is what catches the wheel-attraction failure mode.
    median_err = float(np.median(height_errors)) if height_errors else None
    have_sigma = len(sigmas) > 0
    pass_accept = have_sigma and sigma_stats["median"] < sigma_accept_cm
    pass_target = have_sigma and sigma_stats["median"] < sigma_target_cm
    if median_err is not None:
        pass_accept = pass_accept and median_err < sigma_accept_cm
        pass_target = pass_target and median_err < sigma_target_cm

    # Provenance / gate maturity: a green report only counts as a real gate
    # when the source is a trusted real capture WHOSE 2D POINTS ARE MODEL
    # PREDICTIONS. ``source == "real"`` asserts both. Everything else —
    # synthetic round-trip, OR real geometry replayed on ground-truth 2D
    # (``points_source != "model_prediction"``) — is informational: it
    # validates the plumbing / geometry, never model quality (repo rule,
    # docs/EVAL3D_AND_3D_LOSS_STATUS.md; [[feedback_stop_hook_impossible_goal]]).
    source = manifest.get("source", "synthetic")
    geometry_source = manifest.get("geometry_source", source)
    points_source = manifest.get("points_source")
    # A real *model* gate needs BOTH: real-capture geometry (source ==
    # "real") AND 2D points that are explicitly model predictions. An
    # ABSENT ``points_source`` is "unknown", NOT trusted — there are no
    # legacy real manifests to grandfather, so anything short of an
    # explicit "model_prediction" stays informational. This slams the door
    # on a real-geometry/GT-2D manifest (or a half-filled one) being
    # mislabeled source="real" and sneaking past the promotion gate.
    points_ok = points_source == "model_prediction"
    gate_status = "gate" if (source == "real" and points_ok) else "informational"

    report = {
        "units": units,
        "provenance": manifest.get("provenance"),
        "source": source,
        "geometry_source": geometry_source,
        "points_source": points_source,
        "ab_contract": manifest.get("ab_contract"),
        "gate_status": gate_status,
        "n_scenes": len(per_scene),
        "n_scenes_with_gt": n_with_gt,
        "n_sigma_estimable": len(sigmas),
        "per_scene": per_scene,
        "sigma_cm": sigma_stats,
        "acceptance": {
            "sigma_accept_cm": sigma_accept_cm,
            "sigma_target_cm": sigma_target_cm,
            "gated_on_gt_error": median_err is not None,
            "pass_accept": bool(pass_accept),
            "pass_target": bool(pass_target),
        },
    }
    if height_errors:
        he = np.array(height_errors)
        report["height_error_cm"] = {
            "median": float(np.median(he)),
            "p95": float(np.percentile(he, 95)),
            "max": float(he.max()),
        }
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument(
        "--out", type=Path, default=Path("outputs/eval3d/disc_height_report.json")
    )
    p.add_argument(
        "--ransac-threshold",
        type=float,
        default=2.0,
        help="Floor-anchor inlier threshold in manifest units (cm).",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    report = run_report(
        manifest,
        ransac_threshold=args.ransac_threshold,
        rng=np.random.default_rng(args.seed),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    acc = report["acceptance"]
    if report["gate_status"] == "gate":
        tag = ""
    else:
        why = report.get("points_source") or report.get("source") or "unverified"
        tag = f"  [informational ({why}) — validates plumbing/geometry, NOT model quality]"
    print(
        f"[eval3d] scenes={report['n_scenes']} "
        f"sigma_estimable={report['n_sigma_estimable']} "
        f"median_sigma={report['sigma_cm']['median']:.3f}{report['units']} "
        f"accept(<{acc['sigma_accept_cm']})={acc['pass_accept']} "
        f"target(<{acc['sigma_target_cm']})={acc['pass_target']} -> {args.out}{tag}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
