"""Smoke test: PT vs ONNX export must agree on the same image.

Runs only when both ``best.pt`` and ``best.onnx`` exist for the
``wheel_baseline_v1`` run. On a fresh clone (no trained weights) the
test is skipped, so the suite stays green for new contributors.

Uses the pure helpers from ``src/export_model.py`` so we exercise the
same matching / tolerance logic the export CLI uses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PT_PATH = REPO_ROOT / "runs/pose/wheel_baseline_v1/weights/best.pt"
ONNX_PATH = REPO_ROOT / "runs/pose/wheel_baseline_v1/weights/best.onnx"
POSE_VAL_DIR = REPO_ROOT / "data/wheel_pose_dataset/images/val"
LEGACY_VAL_DIR = REPO_ROOT / "data/wheel_dataset/images/val"


def _pick_sample() -> Path | None:
    for root in (POSE_VAL_DIR, LEGACY_VAL_DIR):
        if root.is_dir():
            imgs = sorted(
                p
                for p in root.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            )
            if imgs:
                return imgs[0]
    return None


@pytest.mark.skipif(
    not PT_PATH.exists() or not ONNX_PATH.exists(),
    reason="baseline weights not in this checkout — run training + export first",
)
def test_onnx_drift_within_tolerance() -> None:
    sample = _pick_sample()
    if sample is None:
        pytest.skip("no sample image available under data/.../images/val")

    ultralytics = pytest.importorskip("ultralytics")
    YOLO = ultralytics.YOLO

    from src.export_model import (
        DEFAULT_BBOX_ATOL,
        DEFAULT_CONF_ATOL,
        DEFAULT_KP_ATOL,
        compare_detections,
        infer_one,
    )

    pt_model = YOLO(str(PT_PATH))
    onnx_model = YOLO(str(ONNX_PATH))

    pt_result = infer_one(pt_model, sample, device="cpu")
    onnx_result = infer_one(onnx_model, sample, device="cpu")

    report = compare_detections(
        pt_result,
        onnx_result,
        bbox_atol=DEFAULT_BBOX_ATOL,
        kp_atol=DEFAULT_KP_ATOL,
        conf_atol=DEFAULT_CONF_ATOL,
    )

    assert report["matched"], (
        f"PT vs ONNX drift exceeded tolerance on {sample.name}:\n"
        + "\n".join(report["failures"])
        + f"\n(max bbox {report['max_bbox_drift_px']:.3f}px, "
        f"max kp {report['max_kp_drift_px']:.3f}px, "
        f"max conf {report['max_conf_drift']:.3f})"
    )
