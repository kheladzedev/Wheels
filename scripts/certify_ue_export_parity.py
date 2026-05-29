"""Certify a UE/MCP export's camera convention by reprojection.

Resolves the standing ``docs/EXPORT_PARITY_AUDIT.md`` blocker: "the UE
Roll/Pitch sign/zero convention must be confirmed against one clean UE
export frame before the pose is trusted". For every rich annotation it
builds the camera with the parity-certified
:func:`camera_from_ue_pose.camera_from_ue_pose`, reprojects each exported
3D ``keypoints_world`` and compares to the exported 2D ``keypoints_image``.

``certified`` is True when the worst reprojection across the whole export
is under ``--tol-px`` (default 1.0 px). The report is written in the same
shape the production gate's ``certification_gate`` reads (``certified`` /
``status`` / ``reason``), so it can feed CI directly.

Usage::

    python scripts/certify_ue_export_parity.py \\
        --dataset-root /path/to/WheelsDataset_v0_2 \\
        --out outputs/eval3d/export_parity_v0_2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import camera_from_ue_pose as cp  # noqa: E402


def certify(dataset_root: Path, tol_px: float = 1.0) -> dict:
    ann_dir = dataset_root / "annotations"
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"{ann_dir} not found (expected rich annotations)")

    all_errs: list[float] = []
    n_frames = n_wheels = n_points = 0
    worst = {"err": -1.0, "frame": None}

    for ann_path in sorted(ann_dir.glob("*.json")):
        d = json.loads(ann_path.read_text(encoding="utf-8"))
        w, h = int(d["image_width"]), int(d["image_height"])
        cam = cp.camera_from_ue_pose(
            d["camera"]["location"], d["camera"]["rotation"], d["camera"]["fov"], w, h
        )
        n_frames += 1
        for wheel in d["wheels"]:
            errs = cp.reprojection_errors(
                cam,
                wheel["keypoints_world"],
                wheel["keypoints_image"],
                visibility=wheel.get("visibility"),
            )
            if errs.size == 0:
                continue
            n_wheels += 1
            n_points += int(errs.size)
            all_errs.extend(errs.tolist())
            if float(errs.max()) > worst["err"]:
                worst = {"err": float(errs.max()), "frame": ann_path.stem}

    arr = np.asarray(all_errs, float)
    if arr.size == 0:
        return {
            "certified": False,
            "status": "no_points",
            "reason": "no paired world/image keypoints found",
            "dataset_root": str(dataset_root),
        }

    max_err = float(arr.max())
    certified = max_err < tol_px
    return {
        "certified": certified,
        "status": "certified" if certified else "drift",
        "reason": (
            f"max reproj {max_err:.4f}px over {n_points} points / {n_frames} "
            f"frames (tol {tol_px}px)"
        ),
        "dataset_root": str(dataset_root),
        "tol_px": tol_px,
        "fov_axis": "horizontal",
        "world_handedness": "left (UE) -> right via Y-negation",
        "n_frames": n_frames,
        "n_wheels_scored": n_wheels,
        "n_points": n_points,
        "reproj_px": {
            "mean": float(arr.mean()),
            "p95": float(np.percentile(arr, 95)),
            "max": max_err,
        },
        "worst_frame": worst,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset-root", required=True, type=Path)
    p.add_argument(
        "--out", type=Path, default=Path("outputs/eval3d/export_parity_v0_2.json")
    )
    p.add_argument("--tol-px", type=float, default=1.0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = certify(args.dataset_root, tol_px=args.tol_px)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    rp = report.get("reproj_px", {})
    print(
        f"[export-parity] certified={report['certified']} "
        f"status={report['status']} "
        f"max={rp.get('max', float('nan')):.4f}px -> {args.out}"
    )
    return 0 if report["certified"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
