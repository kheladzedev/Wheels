"""Generate a synthetic frames manifest for the 3D-eval harness.

Produces the manifest consumed by ``src/eval3d_report.py`` using the
trusted numpy forward model in ``src/eval3d_floorray.py``: for each
scene, a wheel is placed in front of the camera column and viewed from
several poses; the GT 3D points are projected to pixels and written as
the (perfect) predicted A/B/C, alongside the UE-style ``Ground`` meta.

This is a **smoke / fixture** producer. A clean synthetic manifest
exercises the harness end-to-end and documents the manifest shape; it
proves the plumbing, never model accuracy. Replace the projected
"predictions" with real model output + a clean UE export to score an
actual model (``docs/EXPORT_PARITY_AUDIT.md`` is the upstream blocker).

Usage::

    python scripts/make_synthetic_eval3d_manifest.py \\
        --out outputs/eval3d/frames_manifest.json --n-scenes 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import eval3d_floorray as g  # noqa: E402


def _ground_aimed_at(delta_z: float, depth: float, height: float, fov: float) -> dict:
    """UE Ground meta whose pitch points the camera at the wheel center."""
    pitch = float(np.degrees(np.arctan2(delta_z - height, depth)))
    return {"delta_z": float(delta_z), "roll": 0.0, "pitch": pitch, "fov": float(fov)}


def build_manifest(
    n_scenes: int = 8,
    frames_per_scene: int = 6,
    img_w: int = 1920,
    img_h: int = 1080,
    fov: float = 55.0,
    seed: int = 0,
) -> dict:
    """Build a clean synthetic manifest (units: cm)."""
    rng = np.random.default_rng(seed)
    scenes: dict[str, dict] = {}
    for s in range(n_scenes):
        depth = float(rng.uniform(150.0, 260.0))  # cm in front of the camera
        disc_h = float(rng.uniform(18.0, 45.0))  # disc-bottom height, cm
        half = float(rng.uniform(6.0, 12.0))  # half A/B separation, cm
        a_floor = np.array([-half, depth, 0.0])
        b_floor = np.array([half, depth, 0.0])
        disc = np.array([0.0, depth, disc_h])

        frames = []
        for k in range(frames_per_scene):
            dz = float(rng.uniform(130.0, 200.0))  # camera height, cm
            ground = _ground_aimed_at(dz, depth, disc_h, fov)
            cam = g.camera_from_ue_ground(ground, img_w, img_h)
            frames.append(
                {
                    "frame_id": f"scene{s:04d}_f{k:02d}",
                    "ground": ground,
                    "points": {
                        "a": g.project(cam, a_floor[None])[0].tolist(),
                        "b": g.project(cam, b_floor[None])[0].tolist(),
                        "c_disc_bottom": g.project(cam, disc[None])[0].tolist(),
                    },
                }
            )
        scenes[f"scene_{s:04d}"] = {
            "gt_disc_position": disc.tolist(),
            "frames": frames,
        }

    return {
        "units": "cm",
        "image_size": [img_w, img_h],
        "provenance": "synthetic-roundtrip; validates plumbing only, not model quality",
        "scenes": scenes,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out", type=Path, default=Path("outputs/eval3d/frames_manifest.json")
    )
    p.add_argument("--n-scenes", type=int, default=8)
    p.add_argument("--frames-per-scene", type=int, default=6)
    p.add_argument("--img-w", type=int, default=1920)
    p.add_argument("--img-h", type=int, default=1080)
    p.add_argument("--fov", type=float, default=55.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    man = build_manifest(
        n_scenes=args.n_scenes,
        frames_per_scene=args.frames_per_scene,
        img_w=args.img_w,
        img_h=args.img_h,
        fov=args.fov,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"[synthetic-manifest] {args.n_scenes} scenes -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
