"""Foundation-model auto-annotator for the plugin keypoint format.

Pipeline (foundation-model pre-label, no training):

  1. A COCO-pretrained YOLO detector (the baseline ``yolo11n.pt`` shipped
     with the repo) finds vehicles — classes ``car``/``bus``/``truck``.
  2. For each vehicle bbox, six point prompts are sampled along the
     bottom band of the box and fed to SAM 2; SAM 2 returns several mask
     hypotheses per point.
  3. Mask hypotheses are filtered (area, aspect, compactness, centroid
     in the lower half of the vehicle) and deduplicated by IoU. What
     survives is a set of wheel candidates per vehicle.
  4. A pure-Python geometric postprocess on each surviving mask derives
     the three keypoints (A, B, C) under the 2026-05-14 spec semantics.

Why this stack and not YOLO-World with a "wheel" text prompt: empirically
YOLO-World v2 (s through l checkpoints) does not detect "wheel" /
"tire" / "rim" as fine-grained concepts on car photos — CLIP's
text-image space treats those as parts inside "car", not as standalone
detectable objects. Car / truck / bus are reliably detected, and SAM 2
with point prompts cleanly segments individual wheels inside a vehicle
bbox. This is also the route the 2026 SAM2Auto / SMART-OD papers take
when ground-truth fine-grained classes are missing from the open
vocabulary.

Output is plugin-compatible (`docs/KEYPOINT_DATASET_FORMAT.md`) and
**deliberately marked as draft**. Top-level: ``"_draft": true``,
``"_warning": "NOT_GROUND_TRUTH_REQUIRES_HUMAN_REVIEW"``. Per-wheel,
when the heuristic looks shaky: ``"_needs_review": true`` plus a list
``"_review_reasons"``. The validator (``check_keypoint_incoming.py``)
ignores extra top-level and extra wheel-level keys; only the keys
inside ``points`` are strict, so the schema stays clean.

Keypoint heuristic (under the 2026-05-14 spec):

  - **A (left floor-ray)** — leftmost mask pixel within the bottom 10%
    of the bbox vertically. The mask's lower edge approximates where
    the tyre meets the floor; the AR side will raycast this onto the
    floor plane.
  - **B (right floor-ray)** — rightmost mask pixel within the same
    bottom band.
  - **C (disc bottom)** — lowest mask pixel along the bbox's vertical
    centreline, then shifted up by 8% of the bbox height. This is a
    rough offset between the tyre's lowest point (what the mask sees)
    and the metal disc's lowest point (what C must name). Clamped to
    stay inside the mask.

A/B and C are heuristics, not learned outputs. Every emitted wheel
carries ``_draft`` provenance; anything shaky gets ``_needs_review``.
Use ``manual_keypoint_annotator.py`` to hand-correct the flagged ones
before training.

Drop rules (under the spec's "occluded wheels are dropped" clause,
`docs/KEYPOINT_SPEC.md`):

  - Mask area below ``HARD_MIN_MASK_PX`` (degenerate prediction).
  - Pseudo-confidence below ``--drop-conf`` (default 0.20).
  - Any of A/B/C cannot be derived from the mask (also degenerate).

Anything between the drop threshold and ``--review-conf`` (default
0.50), or with a small/edge-touching mask, is **kept** with
``_needs_review = true``. The user wants maximum recall for the
manual pass; the spec wants strict occluded-drop. The compromise:
drop only obvious garbage; flag everything ambiguous.

Usage::

    python src/auto_annotate_wheels.py \\
        --images-dir   data/manual_real/images \\
        --output-root  data/incoming/manual_real_auto \\
        --device       mps \\
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ANNOTATION_METHOD = "coco_vehicle+sam2_grid_prompts"

# Baseline COCO detector (shipped with the repo). Classes 2/5/7 = car /
# bus / truck in the standard COCO ordering Ultralytics uses.
DEFAULT_DETECTOR_WEIGHTS = "yolo11n.pt"
DEFAULT_SAM_WEIGHTS = "sam2.1_b.pt"
COCO_VEHICLE_CLASSES: frozenset[int] = frozenset({2, 5, 7})

IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

# Pseudo-confidence floors. Hard floor: drop. Soft floor: keep + flag.
DEFAULT_DROP_CONF = 0.20
DEFAULT_REVIEW_CONF = 0.50

# Mask sanity (in pixels). Smaller than HARD_MIN is dropped; smaller than
# SOFT_MIN is flagged. The hard floor was raised from 80 → 800 after FP
# analysis on real_000 showed sub-500 px masks clinging to badges /
# distant reflected wheels — a real wheel on a 1280-px photo is rarely
# below ~40 × 40 px.
HARD_MIN_MASK_PX = 1500
SOFT_MIN_MASK_PX = 2500

# Hard minimum bbox side, in pixels. Reflections and headlight LEDs
# generate confident-looking SAM masks at 15–25 px on a side; we cut them
# here independently of pixel count (a thin sliver could still pass the
# area check). Raised 2026-05-13 from 28 → 40 after analysing the
# user-flagged FPs on auto-show stands (real_019) and BYD intake grilles.
MIN_BBOX_SIDE_PX = 40

# Edge proximity (pixels) for "bbox touches edge" / "mask touches edge".
EDGE_TOLERANCE_PX = 2

# Aspect-ratio sanity for a wheel bbox. Outside the HARD band the
# candidate is rejected outright at the proposal stage; outside the SOFT
# (wider) band it earns a review flag. Tightened on 2026-05-13: real
# wheels in our test set never go beyond 1.65:1 even at 3/4 view, so
# 1.8 leaves room without admitting bumper slivers.
ASPECT_HARD_LO = 0.55
ASPECT_HARD_HI = 1.8
ASPECT_MIN = 0.5
ASPECT_MAX = 2.0

# Mask must be roughly circular (4πA/P²). 1.0 = perfect circle, 0.785 =
# square, ~0 = thin line. Headlight ovals come in around 0.45-0.55;
# bumper slivers below 0.30; clean wheel masks above 0.65 in our data.
# 0.62 was settled on 2026-05-13 as the precision/recall sweet spot:
# 0.68 enforced the spec's strict occluded-drop rule but dropped ~60 %
# of legit wheels (SAM 2 masks are rarely closed circles in three-quarter
# views). 0.62 still rejects half-moons (~0.45-0.55) and headlight
# ovals while keeping side-on and 3/4-view wheels. Half-moons that
# slip through are caught downstream by the human review pass.
MIN_CIRCULARITY = 0.62
SOFT_CIRCULARITY = 0.75

# Mean luminance of the mask interior (Y from BGR). Tyre rubber and
# disc shadow keep wheels dark — Y < ~120 on every wheel we sampled.
# Headlights (chromed, lit) sit Y > 150, bumpers / grilles Y > 130.
MAX_TIRE_BRIGHTNESS = 130
SOFT_TIRE_BRIGHTNESS = 110

# Vertical band (fraction of bbox height) used to locate A/B at the
# tyre-to-floor contact. Bottom 10% is wide enough to survive small mask
# jitter and narrow enough to stay below the disc.
AB_BAND_FRACTION = 0.10

# How far above the mask's bottom the disc-bottom sits, as a fraction of
# bbox height. Derived from the standard tyre-to-rim radius ratio: rim
# radius ≈ 0.65 × tyre radius, so the rim's lowest point sits at
# 0.5*h - 0.325*h = 0.175*h above the tyre's lowest point. Earlier
# values (0.08) placed C on the rubber sidewall and violated the spec
# (KEYPOINT_SPEC.md §"the three keypoints": "C: lowest visible point of
# the metal rim / disc").
C_OFFSET_FRACTION = 0.175

# Wheel-proposal grid inside the vehicle bbox.
DEFAULT_PROMPTS_PER_VEHICLE = 6
DEFAULT_MAX_WHEELS_PER_VEHICLE = 4
WHEEL_AREA_MIN_FRACTION_OF_VEHICLE = 0.0015
# Tightened 2026-05-13 from 0.20 → 0.13 after a Lada Niva case where SAM 2
# segmented an entire neighbouring car body (29% of its prompt-source
# vehicle bbox) and passed the old threshold.
WHEEL_AREA_MAX_FRACTION_OF_VEHICLE = 0.13
# Mask centroid must sit in the bottom 45% of the vehicle bbox — wheels
# never live in the upper half of a car. Previously this was only a
# scoring term; on 2026-05-13 it was promoted to a hard filter.
CENTROID_MIN_FRACTION = 0.55
# Same idea but applied to the full image, not the vehicle bbox: real
# wheels are physically near the road, which translates to Y > ~50% of
# the frame on almost every realistic crop. This kills "upstairs"
# FPs — wheels of background cars whose vehicle bbox happens to be
# entirely in the upper part of the frame (auto-show stands, reflected
# vehicles, distant parked cars).
IMAGE_CENTROID_MIN_FRACTION = 0.45
DEDUP_IOU_THRESHOLD = 0.4

WARNING_STRING = "NOT_GROUND_TRUTH_REQUIRES_HUMAN_REVIEW"


# ---------------------------------------------------------------------------
# Pure helpers — covered by tests/test_auto_annotate_wheels.py. No OpenCV
# model loading here; only numpy / pure Python.
# ---------------------------------------------------------------------------


def bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    """Tight axis-aligned bbox of the True pixels in a binary mask."""
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def keypoints_from_mask(
    mask: np.ndarray,
    bbox: list[int],
    *,
    ab_band_fraction: float = AB_BAND_FRACTION,
    c_offset_fraction: float = C_OFFSET_FRACTION,
) -> dict[str, list[float]] | None:
    """Derive A/B/C keypoints from a tyre mask + bbox.

    Returns ``{"a": [x, y], "b": [x, y], "c_disc_bottom": [x, y]}`` or
    ``None`` if the geometry is degenerate.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")

    x1, y1, x2, y2 = (int(v) for v in bbox)
    height = y2 - y1
    width = x2 - x1
    if height <= 0 or width <= 0:
        return None

    band_h = max(1, int(round(height * ab_band_fraction)))
    band_y_start = max(y1, y2 - band_h)
    band = mask[band_y_start:y2, x1:x2]
    if band.size == 0:
        return None
    band_ys, band_xs = np.where(band)
    if band_ys.size == 0:
        return None

    a_local_x = int(band_xs.min())
    b_local_x = int(band_xs.max())
    a_local_y = int(band_ys[band_xs == a_local_x].max())
    b_local_y = int(band_ys[band_xs == b_local_x].max())

    a_x = float(x1 + a_local_x)
    a_y = float(band_y_start + a_local_y)
    b_x = float(x1 + b_local_x)
    b_y = float(band_y_start + b_local_y)

    centre_x = (x1 + x2) // 2
    col_half = max(1, width // 20)
    col_x1 = max(x1, centre_x - col_half)
    col_x2 = min(x2, centre_x + col_half + 1)
    column = mask[y1:y2, col_x1:col_x2]
    col_ys, _ = np.where(column)
    if col_ys.size == 0:
        return None
    c_local_y = int(col_ys.max())
    c_y_tyre = y1 + c_local_y
    c_y = float(c_y_tyre - c_offset_fraction * height)
    c_y = max(float(y1), min(float(y2 - 1), c_y))
    c_x = float(centre_x)

    return {
        "a": [a_x, a_y],
        "b": [b_x, b_y],
        "c_disc_bottom": [c_x, c_y],
    }


def mask_circularity(mask: np.ndarray) -> float:
    """Isoperimetric quotient ``4πA / P²`` for the largest contour in mask.

    ``1.0`` is a perfect circle, ``π/4 ≈ 0.785`` is a square, thin slivers
    approach ``0``. Returns ``0.0`` for empty masks.

    The perimeter is computed on the largest external contour rather than
    on the full mask boundary; this avoids tiny holes / noise pumping
    perimeter up and crashing the score on otherwise-clean disks.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    if int(mask.sum()) == 0:
        return 0.0
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    perimeter = float(cv2.arcLength(largest, closed=True))
    if perimeter <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter * perimeter))


def tire_darkness(image_bgr: np.ndarray, mask: np.ndarray) -> float:
    """Mean luminance (BT.601 Y) of the BGR image inside the mask.

    Returns ``255.0`` (max bright, treat as non-wheel) when the mask is
    empty so callers using a ``< MAX_TIRE_BRIGHTNESS`` predicate fail
    safe.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"image_bgr must be HxWx3, got shape {image_bgr.shape}")
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    if image_bgr.shape[:2] != mask.shape:
        raise ValueError(
            f"shape mismatch: image {image_bgr.shape[:2]} vs mask {mask.shape}"
        )
    sel = mask.astype(bool)
    if not sel.any():
        return 255.0
    b = image_bgr[..., 0][sel].astype(np.float32)
    g = image_bgr[..., 1][sel].astype(np.float32)
    r = image_bgr[..., 2][sel].astype(np.float32)
    # BT.601 luma.
    y = 0.114 * b + 0.587 * g + 0.299 * r
    return float(y.mean())


def bbox_touches_edge(
    bbox: list[int], image_shape: tuple[int, int], tol: int = EDGE_TOLERANCE_PX
) -> bool:
    h, w = image_shape
    x1, y1, x2, y2 = bbox
    return x1 <= tol or y1 <= tol or x2 >= w - tol or y2 >= h - tol


def mask_touches_edge(mask: np.ndarray, tol: int = EDGE_TOLERANCE_PX) -> bool:
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    if mask.shape[0] == 0 or mask.shape[1] == 0:
        return False
    top = mask[:tol, :].any()
    bottom = mask[-tol:, :].any()
    left = mask[:, :tol].any()
    right = mask[:, -tol:].any()
    return bool(top or bottom or left or right)


def aspect_outside(
    bbox: list[int], lo: float = ASPECT_MIN, hi: float = ASPECT_MAX
) -> bool:
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    w = x2 - x1
    if h <= 0 or w <= 0:
        return True
    ratio = w / h
    return ratio < lo or ratio > hi


def iou(b1: list[int], b2: list[int]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = max(1, a1 + a2 - inter)
    return inter / union


def review_reasons(
    *,
    detector_conf: float,
    mask: np.ndarray,
    bbox: list[int],
    image_shape: tuple[int, int],
    image_bgr: np.ndarray | None = None,
    review_conf: float = DEFAULT_REVIEW_CONF,
    soft_min_mask_px: int = SOFT_MIN_MASK_PX,
    soft_circularity: float = SOFT_CIRCULARITY,
    soft_brightness: float = SOFT_TIRE_BRIGHTNESS,
) -> list[str]:
    """Soft flags: keep the wheel but mark it for human review."""
    reasons: list[str] = []
    if detector_conf < review_conf:
        reasons.append("low_detector_conf")
    if int(mask.sum()) < soft_min_mask_px:
        reasons.append("mask_small")
    if mask_touches_edge(mask):
        reasons.append("mask_touches_edge")
    if bbox_touches_edge(bbox, image_shape):
        reasons.append("bbox_touches_edge")
    if aspect_outside(bbox):
        reasons.append("extreme_aspect")
    # Width/height of the proposed wheel bbox; helps flag near-miss FPs
    # that just barely cleared the hard MIN_BBOX_SIDE_PX gate.
    bb_w = bbox[2] - bbox[0]
    bb_h = bbox[3] - bbox[1]
    if min(bb_w, bb_h) < MIN_BBOX_SIDE_PX * 1.5:
        reasons.append("small_bbox")
    if mask_circularity(mask) < soft_circularity:
        reasons.append("low_circularity")
    if image_bgr is not None and tire_darkness(image_bgr, mask) > soft_brightness:
        reasons.append("light_mask")
    return reasons


def should_drop(
    *,
    detector_conf: float,
    mask: np.ndarray,
    bbox: list[int] | None,
    drop_conf: float = DEFAULT_DROP_CONF,
    hard_min_mask_px: int = HARD_MIN_MASK_PX,
) -> bool:
    """Hard floor: silently exclude the wheel from the bundle."""
    if detector_conf < drop_conf:
        return True
    if bbox is None:
        return True
    if int(mask.sum()) < hard_min_mask_px:
        return True
    return False


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def list_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _ensure_bundle_dirs(output_root: Path) -> tuple[Path, Path, Path]:
    images_dir = output_root / "images"
    annos_dir = output_root / "annotations"
    meta_dir = output_root / "metadata"
    images_dir.mkdir(parents=True, exist_ok=True)
    annos_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, annos_dir, meta_dir


def write_annotation(
    annos_dir: Path, image_path: Path, wheels: list[dict], *, draft: bool
) -> Path:
    stem = image_path.stem
    payload: dict = {
        "frame_id": stem,
        "image": image_path.name,
        "wheels": wheels,
    }
    if draft:
        payload["_draft"] = True
        payload["_warning"] = WARNING_STRING
        payload["_annotation_method"] = ANNOTATION_METHOD
    anno_path = annos_dir / f"{stem}.json"
    anno_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return anno_path


def write_source_info(meta_dir: Path, info: dict) -> Path:
    path = meta_dir / "source_info.json"
    path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Pipeline (depends on ultralytics — kept out of pure helpers above so the
# test suite never has to load weights).
# ---------------------------------------------------------------------------


@dataclass
class ImageStats:
    name: str
    vehicles: int
    wheels_kept: int
    wheels_flagged: int
    wheels_dropped: int


def _load_models(detector_weights: str, sam_weights: str, device: str):
    from ultralytics import SAM, YOLO

    detector = YOLO(detector_weights)
    sam = SAM(sam_weights)
    return detector, sam, device


def _predict_vehicles(
    detector, image: np.ndarray, conf: float, device: str
) -> list[tuple[list[int], float, int]]:
    """Return list of (xyxy_int, conf, class_id) for COCO car/bus/truck."""
    results = detector.predict(image, conf=conf, device=device, verbose=False)
    out: list[tuple[list[int], float, int]] = []
    for r in results:
        if r.boxes is None:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        for box, c, k in zip(xyxy, confs, cls):
            if int(k) not in COCO_VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            out.append(([x1, y1, x2, y2], float(c), int(k)))
    return out


def _sam_masks_for_point(
    sam, image: np.ndarray, point: tuple[float, float], device: str
) -> list[np.ndarray]:
    """SAM 2 with a single positive point prompt; return all returned masks."""
    results = sam.predict(
        image,
        points=[[float(point[0]), float(point[1])]],
        labels=[1],
        device=device,
        verbose=False,
    )
    out: list[np.ndarray] = []
    for r in results:
        if r.masks is None or r.masks.data is None:
            continue
        arr = r.masks.data.cpu().numpy()
        for m in arr:
            out.append((m > 0.5).astype(np.uint8))
    return out


def _score_candidate(
    mask: np.ndarray, bbox: list[int], vehicle_bbox: list[int]
) -> float:
    """Heuristic 0..1 score: prefer compact, square-ish, lower-half masks.

    The score doubles as a pseudo-confidence written into ``_detector_conf``
    so the drop/review thresholds work uniformly downstream.
    """
    vx1, vy1, vx2, vy2 = vehicle_bbox
    vh = max(1, vy2 - vy1)
    vw = max(1, vx2 - vx1)
    bb_w = bbox[2] - bbox[0]
    bb_h = bbox[3] - bbox[1]
    if bb_w <= 0 or bb_h <= 0:
        return 0.0
    aspect = bb_w / bb_h
    aspect_score = 1.0 - min(1.0, abs(aspect - 1.0))  # 1.0 at square, 0 at 2:1

    centroid_y = (bbox[1] + bbox[3]) / 2
    lower_half_score = max(0.0, min(1.0, (centroid_y - (vy1 + 0.5 * vh)) / (0.5 * vh)))

    area = int(mask.sum())
    area_frac = area / (vw * vh)
    if area_frac < WHEEL_AREA_MIN_FRACTION_OF_VEHICLE:
        area_score = 0.0
    elif area_frac > WHEEL_AREA_MAX_FRACTION_OF_VEHICLE:
        area_score = 0.0
    else:
        # Bell-shaped around 2-5% of vehicle area.
        target = 0.03
        area_score = max(0.0, 1.0 - abs(area_frac - target) / 0.15)

    return float(0.35 * aspect_score + 0.30 * lower_half_score + 0.35 * area_score)


def _propose_wheel_candidates(
    image: np.ndarray,
    vehicle_bbox: list[int],
    sam,
    device: str,
    *,
    n_prompts: int = DEFAULT_PROMPTS_PER_VEHICLE,
    max_per_vehicle: int = DEFAULT_MAX_WHEELS_PER_VEHICLE,
    image_shape: tuple[int, int] | None = None,
) -> list[tuple[list[int], np.ndarray, float]]:
    """Sample point prompts inside a vehicle bbox and harvest wheel masks."""
    vx1, vy1, vx2, vy2 = vehicle_bbox
    vw = vx2 - vx1
    vh = vy2 - vy1
    if vw < 40 or vh < 40:
        return []

    if image_shape is None:
        image_shape = image.shape[:2]
    h, w = image_shape

    band_y = vy1 + int(0.78 * vh)
    band_y = max(0, min(h - 1, band_y))
    x_min = vx1 + int(0.08 * vw)
    x_max = vx2 - int(0.08 * vw)
    if x_max <= x_min:
        return []
    xs = np.linspace(x_min, x_max, n_prompts)

    raw: list[tuple[list[int], np.ndarray, float]] = []
    for px in xs:
        masks = _sam_masks_for_point(sam, image, (float(px), float(band_y)), device)
        for mask in masks:
            area = int(mask.sum())
            if area < HARD_MIN_MASK_PX:
                continue
            if area > WHEEL_AREA_MAX_FRACTION_OF_VEHICLE * vw * vh:
                continue
            bb = bbox_from_mask(mask)
            if bb is None:
                continue
            bb_w = bb[2] - bb[0]
            bb_h = bb[3] - bb[1]
            # Hard min side: kills LED / badge / reflection slivers that
            # area alone lets through.
            if min(bb_w, bb_h) < MIN_BBOX_SIDE_PX:
                continue
            if aspect_outside(bb, lo=ASPECT_HARD_LO, hi=ASPECT_HARD_HI):
                continue
            # Centroid must sit in the lower part of the vehicle bbox.
            # Pre-2026-05-13 this only fed _score_candidate; now it's a
            # hard gate so a mid-height grille mask can't pass.
            centroid_y = (bb[1] + bb[3]) / 2
            if centroid_y < vy1 + CENTROID_MIN_FRACTION * vh:
                continue
            # Image-absolute lower-half gate: real wheels are near the
            # road, not the upper half of the frame. Kills FPs from
            # vehicles whose bbox sits entirely in the upper frame
            # (auto-show stand backgrounds, reflections in shop windows).
            if centroid_y < IMAGE_CENTROID_MIN_FRACTION * h:
                continue
            # Wheels are nearly circular; headlight ovals and bumper
            # slivers are not.
            if mask_circularity(mask) < MIN_CIRCULARITY:
                continue
            # Tyre rubber is dark; headlights / chrome / paint are bright.
            if tire_darkness(image, mask) > MAX_TIRE_BRIGHTNESS:
                continue
            score = _score_candidate(mask, bb, vehicle_bbox)
            if score < DEFAULT_DROP_CONF:
                continue
            raw.append((bb, mask, score))

    raw.sort(key=lambda t: -t[2])
    keep: list[tuple[list[int], np.ndarray, float]] = []
    for cand in raw:
        bb = cand[0]
        if any(iou(bb, k[0]) > DEDUP_IOU_THRESHOLD for k in keep):
            continue
        keep.append(cand)
        if len(keep) >= max_per_vehicle:
            break
    return keep


def _clip_bbox_to_image(bbox: list[int], image_shape: tuple[int, int]) -> list[int]:
    h, w = image_shape
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return [x1, y1, x2, y2]


def annotate_image(
    image: np.ndarray,
    detector,
    sam,
    *,
    device: str,
    detect_conf: float,
    drop_conf: float,
    review_conf: float,
) -> tuple[list[dict], ImageStats]:
    """Run the full pipeline on one image; return wheels + per-image stats."""
    h, w = image.shape[:2]
    vehicles = _predict_vehicles(detector, image, detect_conf, device)

    kept: list[dict] = []
    flagged = 0
    dropped = 0
    seen_bboxes: list[list[int]] = []

    for vbbox, vconf, vcls in vehicles:
        vbbox_clipped = _clip_bbox_to_image(vbbox, (h, w))
        candidates = _propose_wheel_candidates(
            image, vbbox_clipped, sam, device, image_shape=(h, w)
        )
        for bbox, mask, pseudo_conf in candidates:
            # Dedup across vehicles too — two cars partially overlapping
            # could otherwise duplicate the wheel between them.
            if any(iou(bbox, b) > DEDUP_IOU_THRESHOLD for b in seen_bboxes):
                continue

            if should_drop(
                detector_conf=pseudo_conf,
                mask=mask,
                bbox=bbox,
                drop_conf=drop_conf,
            ):
                dropped += 1
                continue

            kp = keypoints_from_mask(mask, bbox)
            if kp is None:
                dropped += 1
                continue

            reasons = review_reasons(
                detector_conf=pseudo_conf,
                mask=mask,
                bbox=bbox,
                image_shape=(h, w),
                image_bgr=image,
                review_conf=review_conf,
            )

            wheel: dict = {
                "bbox_xyxy": bbox,
                "points": kp,
                "_detector_conf": round(float(pseudo_conf), 4),
                "_vehicle_conf": round(float(vconf), 4),
                "_vehicle_class": int(vcls),
                "_mask_area_px": int(mask.sum()),
            }
            if reasons:
                wheel["_needs_review"] = True
                wheel["_review_reasons"] = reasons
                flagged += 1

            kept.append(wheel)
            seen_bboxes.append(bbox)

    return kept, ImageStats(
        name="",
        vehicles=len(vehicles),
        wheels_kept=len(kept),
        wheels_flagged=flagged,
        wheels_dropped=dropped,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Auto-annotate wheel keypoints with COCO vehicle + SAM 2"
    )
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--detector-weights", default=DEFAULT_DETECTOR_WEIGHTS)
    p.add_argument("--sam-weights", default=DEFAULT_SAM_WEIGHTS)
    p.add_argument(
        "--detect-conf",
        type=float,
        default=0.25,
        help="Vehicle detector confidence threshold",
    )
    p.add_argument(
        "--drop-conf",
        type=float,
        default=DEFAULT_DROP_CONF,
        help="Drop the wheel entirely below this pseudo-confidence",
    )
    p.add_argument(
        "--review-conf",
        type=float,
        default=DEFAULT_REVIEW_CONF,
        help="Flag _needs_review below this pseudo-confidence",
    )
    p.add_argument("--device", default="mps", help='"mps", "cuda", "cpu", or empty')
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N images (0 = all). Smoke-test helper.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    images_dir: Path = args.images_dir
    output_root: Path = args.output_root

    if not images_dir.is_dir():
        print(f"ERROR: images dir not found: {images_dir}")
        return 2

    if output_root.exists():
        if not args.overwrite:
            print(
                f"ERROR: output root already exists: {output_root}  "
                "(pass --overwrite to replace)"
            )
            return 2
        shutil.rmtree(output_root)

    images = list_images(images_dir)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        print(f"ERROR: no images in {images_dir}")
        return 1

    print(f"Annotating {len(images)} image(s) from {images_dir}")
    print(f"Output bundle root:  {output_root}")
    print(
        "Models: detector="
        f"{args.detector_weights}  sam={args.sam_weights}  device={args.device}"
    )

    images_out, annos_out, meta_out = _ensure_bundle_dirs(output_root)
    detector, sam, device = _load_models(
        args.detector_weights, args.sam_weights, args.device
    )

    started = time.time()
    total_kept = 0
    total_flagged = 0
    total_dropped = 0
    total_vehicles = 0
    images_with_zero = 0

    for i, image_path in enumerate(images, 1):
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"  [{i:>3}/{len(images)}] {image_path.name} — unreadable, skipping")
            shutil.copy2(image_path, images_out / image_path.name)
            write_annotation(annos_out, image_path, [], draft=True)
            images_with_zero += 1
            continue

        wheels, stats = annotate_image(
            img,
            detector,
            sam,
            device=device,
            detect_conf=args.detect_conf,
            drop_conf=args.drop_conf,
            review_conf=args.review_conf,
        )

        shutil.copy2(image_path, images_out / image_path.name)
        write_annotation(annos_out, image_path, wheels, draft=True)

        total_kept += stats.wheels_kept
        total_flagged += stats.wheels_flagged
        total_dropped += stats.wheels_dropped
        total_vehicles += stats.vehicles
        if stats.wheels_kept == 0:
            images_with_zero += 1

        print(
            f"  [{i:>3}/{len(images)}] {image_path.name} — "
            f"vehicles={stats.vehicles}  kept={stats.wheels_kept}  "
            f"flagged={stats.wheels_flagged}  dropped={stats.wheels_dropped}"
        )

    elapsed = time.time() - started
    info = {
        "source_name": "manual_real_auto",
        "annotation_method": ANNOTATION_METHOD,
        "_warning": WARNING_STRING,
        "detector_weights": args.detector_weights,
        "sam_weights": args.sam_weights,
        "device": args.device,
        "detect_conf": args.detect_conf,
        "drop_conf": args.drop_conf,
        "review_conf": args.review_conf,
        "image_count": len(images),
        "vehicles_seen": total_vehicles,
        "wheels_kept": total_kept,
        "wheels_flagged_review": total_flagged,
        "wheels_dropped": total_dropped,
        "images_with_zero_wheels": images_with_zero,
        "elapsed_seconds": round(elapsed, 1),
    }
    write_source_info(meta_out, info)

    print()
    print(
        f"Done. vehicles={total_vehicles}  wheels kept={total_kept}  "
        f"flagged={total_flagged}  dropped={total_dropped}  "
        f"images_with_zero={images_with_zero}  elapsed={elapsed:.1f}s"
    )
    print(f"Next: python src/check_keypoint_incoming.py --source-root {output_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
