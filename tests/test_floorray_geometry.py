"""Floor-ray geometry tests for synthetic generators + converter roundtrip.

Pin the 2026-05-13 AR contract on the synthetic visual-smoke path:

  - A/B are floor points near the wheel's ground footprint.
  - A/B sit in the lower band of the bbox: rel_y_{a,b} >= 0.85.
  - A/B are horizontally separated: |B_x - A_x| / bbox_w >= 0.50.
  - C is the lowest visible point of the metal rim — above A/B in
    image y: c_y < min(a_y, b_y).
  - The conversion to YOLO-pose normalized labels and back preserves
    these properties (no off-by-one or unit confusion in the converter).

Covers:
  1. ``create_sample_keypoint_incoming.py`` direct output.
  2. ``auto_draft_keypoint_annotations.py`` heuristic output.
  3. ``convert_keypoint_incoming_to_yolo_pose.py`` roundtrip — the
     normalized YOLO-pose label, when decoded back to pixels, still
     satisfies the geometry.

Audit thresholds match ``scripts/audit_geometry.py`` so the synthetic
smoke is held to the same bar as real inference output.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

REL_Y_AB_MIN = 0.85
AB_SEP_RATIO_MIN = 0.50


def _wheel_geometry(wheel: dict) -> dict:
    x1, y1, x2, y2 = wheel["bbox_xyxy"]
    bw = float(x2 - x1)
    bh = float(y2 - y1)
    pts = wheel["points"]
    ax, ay = pts["a"]
    bx, by = pts["b"]
    cx, cy = pts["c_disc_bottom"]
    return {
        "bw": bw,
        "bh": bh,
        "rel_y_a": (ay - y1) / bh,
        "rel_y_b": (by - y1) / bh,
        "rel_y_c": (cy - y1) / bh,
        "rel_x_c": (cx - x1) / bw,
        "ab_sep_ratio": abs(bx - ax) / bw,
        "c_ab_order_ok": cy < min(ay, by),
        "a_inside_bbox": x1 <= ax <= x2 and y1 <= ay <= y2,
        "b_inside_bbox": x1 <= bx <= x2 and y1 <= by <= y2,
        "c_inside_bbox": x1 <= cx <= x2 and y1 <= cy <= y2,
    }


def _check_floorray(geom: dict) -> None:
    assert geom["rel_y_a"] >= REL_Y_AB_MIN, f"A too high: rel_y_a={geom['rel_y_a']:.3f}"
    assert geom["rel_y_b"] >= REL_Y_AB_MIN, f"B too high: rel_y_b={geom['rel_y_b']:.3f}"
    assert geom["ab_sep_ratio"] >= AB_SEP_RATIO_MIN, (
        f"AB too close: ab_sep_ratio={geom['ab_sep_ratio']:.3f}"
    )
    assert geom["c_ab_order_ok"], (
        f"C below A/B: rel_y_c={geom['rel_y_c']:.3f} >= "
        f"min(rel_y_a, rel_y_b)={min(geom['rel_y_a'], geom['rel_y_b']):.3f}"
    )
    assert geom["a_inside_bbox"], "A outside bbox"
    assert geom["b_inside_bbox"], "B outside bbox"
    assert geom["c_inside_bbox"], "C outside bbox"


# ---------------------------------------------------------------------------
# 1. Synthetic generator
# ---------------------------------------------------------------------------


def test_generator_emits_floorray_geometry_for_every_wheel():
    """Every wheel from the plugin smoke generator must satisfy the contract."""
    from create_sample_keypoint_incoming import generate_one

    rng = random.Random(42)
    n_frames = 10
    seen_wheels = 0
    for _ in range(n_frames):
        _, wheels = generate_one(rng, img_w=640, img_h=480)
        assert wheels, "generator produced zero wheels — synthesis broke"
        for w in wheels:
            _check_floorray(_wheel_geometry(w))
            seen_wheels += 1
    assert seen_wheels >= 10, f"too few wheels exercised: {seen_wheels}"


def test_generator_uses_floorray_constants_not_rim_midline():
    """Anti-regression: A/B must NOT land on the wheel's horizontal midline.

    Under legacy rim semantics A and B were placed at (cx ± rim_radius, cy),
    i.e. rel_y == 0.5. A test that A != midline catches a regression to
    that shape without depending on the exact new constants.
    """
    from create_sample_keypoint_incoming import generate_one

    rng = random.Random(0)
    _, wheels = generate_one(rng, img_w=640, img_h=480)
    for w in wheels:
        x1, y1, x2, y2 = w["bbox_xyxy"]
        midline_y = 0.5 * (y1 + y2)
        ay = w["points"]["a"][1]
        by = w["points"]["b"][1]
        # Any reasonable floor-ray placement is at least 30% of bbox
        # height below the midline. Legacy rim sat ON the midline.
        assert ay - midline_y > 0.30 * (y2 - y1), (
            f"A still near midline: ay={ay:.1f} midline={midline_y:.1f}"
        )
        assert by - midline_y > 0.30 * (y2 - y1), (
            f"B still near midline: by={by:.1f} midline={midline_y:.1f}"
        )


# ---------------------------------------------------------------------------
# 2. Auto-draft heuristic
# ---------------------------------------------------------------------------


def test_auto_draft_heuristic_emits_floorray_geometry():
    from auto_draft_keypoint_annotations import _draft_two_wheels

    wheels = _draft_two_wheels(image_w=1280, image_h=720)
    assert len(wheels) == 2
    for w in wheels:
        _check_floorray(_wheel_geometry(w))


# ---------------------------------------------------------------------------
# 3. Converter roundtrip
# ---------------------------------------------------------------------------


def _decode_yolo_pose_label(label_path: Path, image_w: int, image_h: int) -> list[dict]:
    """Decode a YOLO-pose label file back to pixel-space wheels.

    Label format (per line):
      class cx cy w h kpx_a kpy_a v_a kpx_b kpy_b v_b kpx_c kpy_c v_c
    """
    wheels: list[dict] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) != 1 + 4 + 3 * 3:
            continue
        _, cx, cy, w, h = (float(x) for x in parts[:5])
        x1 = (cx - 0.5 * w) * image_w
        y1 = (cy - 0.5 * h) * image_h
        x2 = (cx + 0.5 * w) * image_w
        y2 = (cy + 0.5 * h) * image_h

        def _kp(i: int) -> list[float]:
            kx = float(parts[5 + i * 3]) * image_w
            ky = float(parts[5 + i * 3 + 1]) * image_h
            return [kx, ky]

        wheels.append(
            {
                "bbox_xyxy": [x1, y1, x2, y2],
                "points": {
                    "a": _kp(0),
                    "b": _kp(1),
                    "c_disc_bottom": _kp(2),
                },
            }
        )
    return wheels


def test_converter_roundtrip_preserves_floorray_geometry(tmp_path: Path):
    """Generate → convert → decode YOLO-pose labels → re-check geometry.

    The converter normalises pixel coordinates to [0, 1] for YOLO-pose.
    Any off-by-one in normalisation would shift A/B/C relative to the
    bbox; this test catches that.
    """
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[1]
    src = tmp_path / "incoming"
    dst = tmp_path / "yolo_pose"

    # 1. Generate a tiny incoming batch (5 frames).
    subprocess.run(
        [
            sys.executable,
            str(repo / "src" / "create_sample_keypoint_incoming.py"),
            "--output-root",
            str(src),
            "--count",
            "5",
            "--seed",
            "7",
            "--overwrite",
        ],
        check=True,
        capture_output=True,
    )

    # 2. Convert to YOLO-pose (small batch fails the default val-min gate
    #    of 2, but for the roundtrip we just need the labels — set a low
    #    split ratio so all 5 land in train and val is allowed to be empty
    #    via the script's own quality gate. Easier: bypass --fail-on-quality-gate.)
    subprocess.run(
        [
            sys.executable,
            str(repo / "src" / "convert_keypoint_incoming_to_yolo_pose.py"),
            "--source-root",
            str(src),
            "--dataset-root",
            str(dst),
            "--source-name",
            "rt",
            "--overwrite",
        ],
        check=True,
        capture_output=True,
    )

    # 3. Decode and re-check every wheel across train + val.
    import cv2

    total_checked = 0
    for split in ("train", "val"):
        img_dir = dst / "images" / split
        lbl_dir = dst / "labels" / split
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            img = cv2.imread(str(img_path))
            assert img is not None, f"cannot read {img_path}"
            h, w = img.shape[:2]
            label_path = lbl_dir / f"{img_path.stem}.txt"
            for wheel in _decode_yolo_pose_label(label_path, w, h):
                _check_floorray(_wheel_geometry(wheel))
                total_checked += 1
    assert total_checked >= 5, f"too few wheels decoded: {total_checked}"


# ---------------------------------------------------------------------------
# 4. Per-frame contract on disk (incoming JSON)
# ---------------------------------------------------------------------------


def test_generator_writes_geometry_matching_floorray_on_disk(tmp_path: Path):
    """Read back the JSON files the CLI wrote; geometry must hold."""
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[1]
    src = tmp_path / "incoming"
    subprocess.run(
        [
            sys.executable,
            str(repo / "src" / "create_sample_keypoint_incoming.py"),
            "--output-root",
            str(src),
            "--count",
            "8",
            "--seed",
            "11",
            "--overwrite",
        ],
        check=True,
        capture_output=True,
    )

    annos = list((src / "annotations").glob("*.json"))
    assert annos, "no annotations written"

    wheels_seen = 0
    for j in annos:
        payload = json.loads(j.read_text(encoding="utf-8"))
        for w in payload["wheels"]:
            _check_floorray(_wheel_geometry(w))
            wheels_seen += 1
    assert wheels_seen >= 5


# ---------------------------------------------------------------------------
# 5. Audit-thresholds source-of-truth alignment
# ---------------------------------------------------------------------------


def test_generator_thresholds_at_least_as_strict_as_inference_audit():
    """Generator output must be at least as strict as the inference audit.

    ``scripts/audit_geometry.py`` is lenient because real inference has
    noise; the synthetic generator we control here must clear that bar
    with margin. A future relax of the generator must not drop below
    the audit threshold, otherwise our own smoke would fail the
    auditor we run on real outputs.
    """
    import importlib.util

    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "audit_geometry", repo / "scripts" / "audit_geometry.py"
    )
    audit = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(audit)
    assert REL_Y_AB_MIN >= audit.REL_Y_AB_MIN, (
        f"generator threshold {REL_Y_AB_MIN} weaker than audit {audit.REL_Y_AB_MIN}"
    )
    assert AB_SEP_RATIO_MIN >= audit.AB_SEP_RATIO_MIN, (
        f"generator threshold {AB_SEP_RATIO_MIN} weaker than audit "
        f"{audit.AB_SEP_RATIO_MIN}"
    )
