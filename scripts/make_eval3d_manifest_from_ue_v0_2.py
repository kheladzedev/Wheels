"""Build a 3D-eval frames manifest from a real MCP ``WheelsDataset`` export.

This is the **real-geometry** counterpart to
``scripts/make_synthetic_eval3d_manifest.py``. It consumes the MCP
renderer's *rich* annotations (``annotations/<frame>.json`` — full camera
pose + per-actor 3D ``keypoints_world`` + 2D ``keypoints_image`` +
visibility) and emits the manifest ``src/eval3d_report.py`` reads.

Scene structure: the MCP capture is a turntable — a handful of static
``WheelMarker`` actors viewed by an orbiting camera — so every actor is a
genuine multi-view *scene* (group by ``actor`` / ``stencil_id``). That is
exactly what the cross-frame disc-height sigma needs.

PROVENANCE — read this before trusting a green number:

  - ``geometry_source = "real_ue_<dataset>"`` — the camera pose is REAL
    and parity-certified (reprojection < 1e-3 px,
    ``scripts/certify_ue_export_parity.py``).
  - ``points_source = "ue_ground_truth"`` — the A/B/C fed in are the
    **export's ground-truth** keypoints, NOT model predictions. So a
    green disc-height number here measures the *harness geometry on real
    data*, never model quality. ``source`` is deliberately NOT ``"real"``,
    so ``eval3d_report`` keeps ``gate_status = "informational"`` and
    ``promotion_gate_3d`` returns insufficient_evidence (load-bearing
    invariant preserved).
  - ``ab_contract = "rim_spheres_not_floor_ray"`` — CRITICAL: this export
    maps ``a`` = SphereLeft and ``b`` = SphereRight, which sit on the rim
    (world z ~= 28 cm), NOT on the floor (z = 0) as the 2026-05-14
    floor-ray contract (``docs/KEYPOINT_SPEC.md``) requires. Feeding them
    as floor-ray points produces a *systematic* A/B drift: cross-frame
    sigma stays small while the disc-height error blows up. That is the
    failure mode the GT-error gate exists to catch, and running this
    manifest demonstrates it on real data.

To turn this into a true model-promotion gate you need (a) an export that
emits floor-contact A/B (or an AR-side footprint projection), and (b)
``points_source = "model_prediction"`` from running inference on the
exported images. Neither is done here; both are the remaining blockers.

Usage::

    python scripts/make_eval3d_manifest_from_ue_v0_2.py \\
        --dataset-root /path/to/WheelsDataset_v0_2 \\
        --out outputs/eval3d/frames_manifest_ue_v0_2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import camera_from_ue_pose as cp  # noqa: E402

# rich-annotation keypoint names mapped to the AR contract keys, matching
# the plugin export's own a/b/c assignment.
A_NAME, B_NAME, C_NAME = "SphereLeft", "SphereRight", "Center"
REQUIRED = (A_NAME, B_NAME, C_NAME)


def _visible_in_image(kp_image: dict, vis: dict, w: int, h: int) -> bool:
    for name in REQUIRED:
        if name not in kp_image or not vis.get(name, True):
            return False
        x, y = kp_image[name]
        if not (0.0 <= x < w and 0.0 <= y < h):
            return False
    return True


def build_manifest(
    dataset_root: Path,
    *,
    max_frames_per_scene: int | None = None,
    min_frames: int = 2,
) -> dict:
    ann_dir = dataset_root / "annotations"
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"{ann_dir} not found (expected rich annotations)")

    scenes: dict[str, dict] = {}
    img_w = img_h = None
    kept_frames = dropped_frames = 0

    for ann_path in sorted(ann_dir.glob("*.json")):
        d = json.loads(ann_path.read_text(encoding="utf-8"))
        w, h = int(d["image_width"]), int(d["image_height"])
        img_w, img_h = w, h
        cam = d["camera"]
        pose = {
            "location": cam["location"],
            "rotation": cam["rotation"],
            "fov": cam["fov"],
        }
        frame_used = False
        for wheel in d["wheels"]:
            ki = wheel["keypoints_image"]
            kw = wheel["keypoints_world"]
            vis = wheel.get("visibility", {})
            if not _visible_in_image(ki, vis, w, h):
                continue
            actor = wheel.get("actor") or f"stencil_{wheel.get('stencil_id')}"
            scene = scenes.setdefault(actor, {"frames": [], "_gt_world": None})
            if (
                max_frames_per_scene is not None
                and len(scene["frames"]) >= max_frames_per_scene
            ):
                continue
            scene["frames"].append(
                {
                    "frame_id": f"{actor}__{d.get('frame', ann_path.stem)}",
                    "image_size": [w, h],
                    "pose": pose,
                    "points": {
                        "a": list(ki[A_NAME]),
                        "b": list(ki[B_NAME]),
                        "c_disc_bottom": list(ki[C_NAME]),
                    },
                }
            )
            # GT for what C's pixel reconstructs to. NOTE: v0_2 maps C to
            # the wheel-marker *Center* (hub, world z~=33 cm), NOT the lowest
            # visible rim/disc point the floor-ray contract calls
            # "c_disc_bottom" — a second contract drift on top of the rim
            # A/B. So this GT (and the reported "disc-height error") tracks
            # the hub-center height, self-consistently with the C pixel.
            # The actors are static across the turntable, so set it ONCE per
            # actor (don't let the last frame silently win).
            if scene["_gt_world"] is None:
                scene["_gt_world"] = cp.ue_world_to_rh(kw[C_NAME]).tolist()
            frame_used = True
        kept_frames += int(frame_used)
        dropped_frames += int(not frame_used)

    out_scenes: dict[str, dict] = {}
    for actor, scene in sorted(scenes.items()):
        if len(scene["frames"]) < min_frames:
            continue
        out_scenes[actor] = {
            "gt_disc_position": scene["_gt_world"],
            "frames": scene["frames"],
        }

    return {
        "units": "cm",
        "image_size": [img_w or 1280, img_h or 720],
        "source": "real_geometry_gt2d",
        "geometry_source": f"real_ue_{dataset_root.name}",
        "points_source": "ue_ground_truth",
        "ab_contract": "rim_spheres_not_floor_ray",
        "c_contract": "center_hub_not_lowest_rim",
        "provenance": (
            f"Real MCP export {dataset_root.name}: parity-certified camera "
            "pose + ground-truth 2D (NOT model predictions). a/b are rim "
            "spheres (z~=28cm), not floor-ray points, AND c is the hub center "
            "(z~=33cm), not the lowest rim point — both drift from the "
            "2026-05-14 floor-ray contract. Expect low sigma but high "
            "disc-height error (systematic drift). Informational only — "
            "never a model gate."
        ),
        "scenes": out_scenes,
        "_stats": {
            "frames_with_visible_wheels": kept_frames,
            "frames_dropped": dropped_frames,
            "n_scenes": len(out_scenes),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset-root", required=True, type=Path)
    p.add_argument(
        "--out", type=Path, default=Path("outputs/eval3d/frames_manifest_ue_v0_2.json")
    )
    p.add_argument("--max-frames-per-scene", type=int, default=None)
    p.add_argument("--min-frames", type=int, default=2)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_manifest(
        args.dataset_root,
        max_frames_per_scene=args.max_frames_per_scene,
        min_frames=args.min_frames,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    stats = manifest["_stats"]
    print(
        f"[ue-manifest] scenes={stats['n_scenes']} "
        f"frames_with_wheels={stats['frames_with_visible_wheels']} "
        f"-> {args.out}  [informational: GT-2D, a/b rim-drift, NOT a model gate]"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
