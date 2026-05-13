"""Evaluate YOLO-pose wheel + keypoint predictions against the val split.

Measures how well a trained YOLO-pose model predicts the 3 wheel
keypoints AR consumes (point_a / rim_left, point_b / rim_right,
point_c_disc_bottom). Outputs a JSON report and an ASCII table to
stdout.

Pipeline:
  1. Load the YOLO-pose model (must have `task == "pose"`).
  2. Resolve the dataset root from configs/dataset.yaml; locate
     images/<split>/*.* and labels/<split>/*.txt.
  3. For each image, read the 14-field YOLO-pose label lines, parse
     them to GT wheels in pixel coordinates.
  4. Run model.predict() at the requested --conf / --iou / --max-det.
  5. Greedy-match preds to GT by IoU (>= 0.5). Unmatched GT = FN.
     Unmatched preds (above --conf) = FP.
  6. For each matched pair, compute per-keypoint pixel L2 error
     (skipping GT keypoints with visibility == 0) and OKS.

OKS:
    OKS = sum_i ( exp( -d_i^2 / (2 * s^2 * k_i^2) ) * delta(v_i > 0) )
          / sum_i delta(v_i > 0)
where:
    d_i  = pixel distance between predicted and GT keypoint i
    s    = sqrt(gt_bbox_w * gt_bbox_h) in pixels
    k_i  = per-keypoint sigma (default [0.10, 0.10, 0.10] — calibrated
           so a few-pixel keypoint error on a 60-px-wide wheel gives a
           non-degenerate OKS. COCO's body-part sigmas (~0.025–0.10)
           assume a person-sized scale s; on a 60px wheel s≈60 and the
           tolerance radius k·s ≈ 6px is reasonable. Override with
           `--sigma` once we have a real-data calibration target.
           Documented in the JSON output so runs are comparable.)

Usage:
    python src/eval_keypoints.py \\
        --model runs/pose/wheel_baseline/weights/best.pt \\
        --data configs/dataset.yaml \\
        --split val --device mps \\
        --conf 0.25 --iou 0.45 --max-det 20 \\
        --output outputs/eval/wheel_baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, field
from math import exp, sqrt
from pathlib import Path
from typing import Iterable, Literal, Sequence

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

KEYPOINT_NAMES = ("point_a", "point_b", "point_c_disc_bottom")
INTERNAL_KEYPOINT_NAMES = ("rim_left", "rim_right", "disc_bottom")
N_KEYPOINTS = 3
DEFAULT_SIGMAS: tuple[float, float, float] = (0.10, 0.10, 0.10)
DEFAULT_IOU_MATCH = 0.5
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_WORST_N = 10

# COCO-style bbox-area thresholds in pixels^2. Apply to GT bbox pixel area.
# Boundaries follow the COCO convention: area < 32^2 = small,
# 32^2 <= area < 96^2 = medium, area >= 96^2 = large.
BBOX_AREA_SMALL_MAX = 32**2
BBOX_AREA_MEDIUM_MAX = 96**2

BboxAreaBucket = Literal["small", "medium", "large"]
SliceAxis = Literal["bbox_area", "occlusion"]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class GTWheel:
    """Ground-truth wheel in pixel coordinates."""

    bbox_xyxy: tuple[float, float, float, float]
    keypoints_xy: list[tuple[float, float]]  # (N_KEYPOINTS, 2) in pixels
    visibilities: list[int]  # length N_KEYPOINTS, values in {0,1,2}


@dataclass
class PredWheel:
    """Predicted wheel in pixel coordinates."""

    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    keypoints_xy: list[tuple[float, float]]  # (N_KEYPOINTS, 2) in pixels


@dataclass
class MatchResult:
    """Bookkeeping for greedy IoU matching on a single image."""

    matches: list[tuple[int, int]] = field(default_factory=list)  # (pred_idx, gt_idx)
    unmatched_gt: list[int] = field(default_factory=list)
    unmatched_preds: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def box_iou(b1: Sequence[float], b2: Sequence[float]) -> float:
    """IoU of two axis-aligned boxes in xyxy pixel coordinates.

    Returns 0 if either box has non-positive area or boxes don't overlap.
    """
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    iw = x2 - x1
    ih = y2 - y1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def match_predictions_to_gt(
    preds: Sequence[PredWheel],
    gts: Sequence[GTWheel],
    iou_threshold: float = DEFAULT_IOU_MATCH,
) -> MatchResult:
    """Greedy match preds → GT by IoU.

    Predictions are sorted by confidence desc; each takes the
    highest-IoU still-unmatched GT above ``iou_threshold``.
    """
    result = MatchResult()
    n_preds = len(preds)
    n_gts = len(gts)

    if n_preds == 0:
        result.unmatched_gt = list(range(n_gts))
        return result
    if n_gts == 0:
        result.unmatched_preds = list(range(n_preds))
        return result

    order = sorted(range(n_preds), key=lambda i: preds[i].confidence, reverse=True)
    gt_taken: set[int] = set()

    for p_idx in order:
        best_gt = -1
        best_iou = iou_threshold
        for g_idx in range(n_gts):
            if g_idx in gt_taken:
                continue
            iou = box_iou(preds[p_idx].bbox_xyxy, gts[g_idx].bbox_xyxy)
            if iou >= best_iou:
                best_iou = iou
                best_gt = g_idx
        if best_gt >= 0:
            result.matches.append((p_idx, best_gt))
            gt_taken.add(best_gt)
        else:
            result.unmatched_preds.append(p_idx)

    result.unmatched_gt = [i for i in range(n_gts) if i not in gt_taken]
    return result


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------


def per_keypoint_pixel_errors(
    pred: PredWheel,
    gt: GTWheel,
) -> list[float | None]:
    """L2 pixel error for each keypoint, or None where GT visibility == 0."""
    out: list[float | None] = []
    for i in range(N_KEYPOINTS):
        if gt.visibilities[i] == 0:
            out.append(None)
            continue
        dx = pred.keypoints_xy[i][0] - gt.keypoints_xy[i][0]
        dy = pred.keypoints_xy[i][1] - gt.keypoints_xy[i][1]
        out.append(float(sqrt(dx * dx + dy * dy)))
    return out


def oks_for_match(
    pred: PredWheel,
    gt: GTWheel,
    sigmas: Sequence[float] = DEFAULT_SIGMAS,
) -> float | None:
    """OKS per the COCO-style formula. Returns None if no visible GT kps.

    s = sqrt(bbox_w * bbox_h) from GT bbox in pixels.
    """
    x1, y1, x2, y2 = gt.bbox_xyxy
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    s = sqrt(w * h)
    # Guard against degenerate bboxes: an OKS with s=0 collapses to a delta.
    # Treat as no signal rather than divide-by-zero.
    if s <= 0:
        return None

    total = 0.0
    n_visible = 0
    for i in range(N_KEYPOINTS):
        if gt.visibilities[i] == 0:
            continue
        n_visible += 1
        dx = pred.keypoints_xy[i][0] - gt.keypoints_xy[i][0]
        dy = pred.keypoints_xy[i][1] - gt.keypoints_xy[i][1]
        d2 = dx * dx + dy * dy
        k = sigmas[i]
        denom = 2.0 * (s * s) * (k * k)
        if denom <= 0:
            continue
        total += exp(-d2 / denom)
    if n_visible == 0:
        return None
    return float(total / n_visible)


# ---------------------------------------------------------------------------
# Label / dataset I/O
# ---------------------------------------------------------------------------


def parse_yolo_pose_label(
    label_path: Path,
    img_w: int,
    img_h: int,
) -> list[GTWheel]:
    """Parse a 14-field YOLO-pose .txt file into pixel-space GT wheels.

    Lines that don't have exactly 14 fields are skipped (the dataset
    validator catches those in `check_dataset.py`).
    """
    wheels: list[GTWheel] = []
    if not label_path.exists():
        return wheels
    text = label_path.read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5 + N_KEYPOINTS * 3:
            continue
        try:
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        # YOLO-normalized bbox center/size → pixel xyxy.
        cx_px = cx * img_w
        cy_px = cy * img_h
        w_px = w * img_w
        h_px = h * img_h
        x1 = cx_px - w_px / 2.0
        y1 = cy_px - h_px / 2.0
        x2 = cx_px + w_px / 2.0
        y2 = cy_px + h_px / 2.0

        kps_xy: list[tuple[float, float]] = []
        vis: list[int] = []
        kp_fields = parts[5:]
        for i in range(N_KEYPOINTS):
            try:
                kx_n = float(kp_fields[i * 3])
                ky_n = float(kp_fields[i * 3 + 1])
                v = int(float(kp_fields[i * 3 + 2]))
            except (ValueError, IndexError):
                kps_xy.append((0.0, 0.0))
                vis.append(0)
                continue
            kps_xy.append((kx_n * img_w, ky_n * img_h))
            vis.append(v)
        wheels.append(
            GTWheel(
                bbox_xyxy=(x1, y1, x2, y2),
                keypoints_xy=kps_xy,
                visibilities=vis,
            )
        )
    return wheels


def resolve_dataset_root(data_yaml: Path) -> Path:
    """Read configs/dataset.yaml and return the absolute dataset root.

    Supports both absolute and relative ``path`` entries. Relative paths
    are resolved against the yaml file's parent directory and against
    the repo root, picking the first that exists.
    """
    with data_yaml.open("r", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    raw = spec.get("path")
    if raw is None:
        raise ValueError(f"{data_yaml} has no 'path' entry")
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    candidates = [
        data_yaml.parent / p,
        Path.cwd() / p,
        data_yaml.parent.parent / p,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    # Return the first candidate so the caller error message points
    # somewhere reasonable.
    return candidates[0].resolve()


def list_split_images(dataset_root: Path, split: str) -> list[Path]:
    images_dir = dataset_root / "images" / split
    if not images_dir.is_dir():
        return []
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def read_image_size(image_path: Path) -> tuple[int, int]:
    """Return (width, height) of an image without decoding the full pixels.

    Falls back to a full cv2.imread if the cheap probe fails.
    """
    # OpenCV's imread is cheap enough for eval-time use; no extra deps.
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"OpenCV could not decode image: {image_path}")
    h, w = img.shape[:2]
    return int(w), int(h)


# ---------------------------------------------------------------------------
# Prediction extraction (mirrors src/infer_image.py)
# ---------------------------------------------------------------------------


def extract_pred_wheels(result, conf_threshold: float) -> list[PredWheel]:
    """Pull (bbox, conf, kp xy) for every detection above conf_threshold.

    Mirrors src/infer_image.py:extract_keypoints but produces PredWheel
    objects, not the AR payload shape. Detections without exactly
    N_KEYPOINTS keypoints are skipped — the model is misconfigured.
    """
    preds: list[PredWheel] = []
    if result.boxes is None:
        return preds
    for i, box in enumerate(result.boxes):
        conf = float(box.conf.item())
        if conf < conf_threshold:
            continue
        bbox = tuple(float(v) for v in box.xyxy[0].tolist())
        if len(bbox) != 4:
            continue
        if result.keypoints is None:
            continue
        xy = result.keypoints.xy[i].cpu().numpy()
        if xy.shape[0] != N_KEYPOINTS:
            # Pose model emitted the wrong number of keypoints; skip.
            continue
        kps_xy = [(float(xy[k, 0]), float(xy[k, 1])) for k in range(N_KEYPOINTS)]
        preds.append(
            PredWheel(
                bbox_xyxy=(bbox[0], bbox[1], bbox[2], bbox[3]),
                confidence=conf,
                keypoints_xy=kps_xy,
            )
        )
    return preds


# ---------------------------------------------------------------------------
# Slicing / failure-catalogue helpers
# ---------------------------------------------------------------------------


def classify_bbox_area(
    bbox_xyxy: Sequence[float],
) -> BboxAreaBucket:
    """Bucket a bbox by pixel area following COCO conventions.

    Boundaries: area < 32**2 -> small, 32**2 <= area < 96**2 -> medium,
    area >= 96**2 -> large. The thresholds are inclusive on the lower
    bound of each bucket so the buckets partition non-negative areas.
    Degenerate / negative widths or heights are clamped to zero, which
    routes them into the ``small`` bucket.
    """
    x1, y1, x2, y2 = bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2], bbox_xyxy[3]
    w = max(0.0, float(x2) - float(x1))
    h = max(0.0, float(y2) - float(y1))
    area = w * h
    if area < BBOX_AREA_SMALL_MAX:
        return "small"
    if area < BBOX_AREA_MEDIUM_MAX:
        return "medium"
    return "large"


def has_occlusion(gt: GTWheel) -> bool:
    """True iff any GT keypoint visibility is exactly 1.

    YOLO-pose visibility encoding:
      0 = missing / not annotated (e.g. cropped out of frame)
      1 = occluded but localised (annotator inferred the position)
      2 = clearly visible
    Only ``1`` counts as occluded for this slice; ``0`` means we have no
    label at all and cannot say whether the point is occluded.
    """
    return any(v == 1 for v in gt.visibilities)


@dataclass
class MatchedPair:
    """A single (pred, gt) match with bookkeeping for failure-catalog rows."""

    image_path: Path
    pred: PredWheel
    gt: GTWheel


def collect_matched_pairs(
    image_records: Sequence[tuple[Path | None, Sequence[PredWheel], Sequence[GTWheel]]],
    iou_match: float = DEFAULT_IOU_MATCH,
) -> list[MatchedPair]:
    """Flatten image records to a list of matched (pred, gt) pairs.

    Used by slicing and by the failure-catalog: both want to operate per
    pair rather than per image. ``image_path`` may be None when the
    caller does not have one (e.g. synthetic test data); the slicing
    code does not need the path but the failure catalog does.
    """
    pairs: list[MatchedPair] = []
    for image_path, preds, gts in image_records:
        m = match_predictions_to_gt(preds, gts, iou_threshold=iou_match)
        for p_idx, g_idx in m.matches:
            pairs.append(
                MatchedPair(
                    image_path=image_path if image_path is not None else Path(""),
                    pred=preds[p_idx],
                    gt=gts[g_idx],
                )
            )
    return pairs


def slice_records(
    image_records: Sequence[tuple[Sequence[PredWheel], Sequence[GTWheel]]],
    *,
    axis: SliceAxis,
    iou_match: float = DEFAULT_IOU_MATCH,
) -> dict[str, list[tuple[list[PredWheel], list[GTWheel]]]]:
    """Partition matched (pred, gt) pairs into slice buckets.

    Slicing is per-pair, not per-image: a single image whose two wheels
    span small and large area buckets contributes one matched pair to
    each bucket. Each bucket value is a list of (preds, gts) 1-element
    lists so the existing aggregator can consume them unchanged.

    For ``axis="occlusion"`` we look only at the GT side: a pair's
    occlusion-bucket is determined by the GT's keypoint visibilities,
    not the prediction's confidence.
    """
    if axis == "bbox_area":
        buckets: dict[str, list[tuple[list[PredWheel], list[GTWheel]]]] = {
            "small": [],
            "medium": [],
            "large": [],
        }
    elif axis == "occlusion":
        buckets = {"with_occlusion": [], "without_occlusion": []}
    else:
        raise ValueError(f"Unknown slice axis: {axis!r}")

    for preds, gts in image_records:
        m = match_predictions_to_gt(preds, gts, iou_threshold=iou_match)
        for p_idx, g_idx in m.matches:
            pred = preds[p_idx]
            gt = gts[g_idx]
            if axis == "bbox_area":
                key: str = classify_bbox_area(gt.bbox_xyxy)
            else:  # occlusion
                key = "with_occlusion" if has_occlusion(gt) else "without_occlusion"
            buckets[key].append(([pred], [gt]))
    return buckets


def top_n_failures(
    pairs: Sequence[MatchedPair],
    n: int,
    sigmas: Sequence[float] = DEFAULT_SIGMAS,
) -> list[dict]:
    """Rank matched pairs by mean pixel error (desc).

    Consumes ``MatchedPair`` objects as produced by
    ``collect_matched_pairs`` — the canonical pair-collector used by
    ``main()`` for the failure catalogue.

    ``score_px`` is the mean of per-keypoint pixel errors over keypoints
    with GT visibility > 0. Pairs whose GT has no visible keypoints get
    ``score_px = -inf`` so they sort to the bottom — they carry no
    signal for ranking. Returns up to ``n`` rows in worst-first order;
    if fewer pairs exist than ``n``, returns all of them (no padding).
    """
    if n <= 0 or not pairs:
        return []

    rows: list[tuple[float, dict]] = []
    for mp in pairs:
        errs = per_keypoint_pixel_errors(mp.pred, mp.gt)
        visible_errs = [e for e in errs if e is not None]
        if visible_errs:
            score = float(np.mean(visible_errs))
        else:
            score = float("-inf")
        oks = oks_for_match(mp.pred, mp.gt, sigmas=sigmas)
        rows.append(
            (
                score,
                {
                    "image": str(mp.image_path),
                    "score_px": score if score != float("-inf") else None,
                    "oks": oks,
                    "per_keypoint_px": list(errs),
                    "gt_bbox_xyxy": list(mp.gt.bbox_xyxy),
                    "pred_bbox_xyxy": list(mp.pred.bbox_xyxy),
                },
            )
        )
    rows.sort(key=lambda r: r[0], reverse=True)
    return [row for _, row in rows[:n]]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def summarize_errors(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"median": None, "mean": None, "p95": None, "p99": None, "n": 0}
    return {
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "n": int(arr.size),
    }


def _aggregate_pairs(
    matched: Sequence[tuple[PredWheel, GTWheel]],
    sigmas: Sequence[float] = DEFAULT_SIGMAS,
) -> dict:
    """Aggregate already-matched (pred, gt) pairs into per-keypoint + OKS.

    Used both for the full-set metrics and for per-slice stats. Returns
    the slice-shaped dict: ``per_keypoint_pixel_error`` + ``oks`` +
    ``n_matched``.
    """
    per_kp_errors: list[list[float]] = [[] for _ in range(N_KEYPOINTS)]
    oks_values: list[float] = []

    for pred, gt in matched:
        errs = per_keypoint_pixel_errors(pred, gt)
        for k, e in enumerate(errs):
            if e is not None:
                per_kp_errors[k].append(e)
        oks = oks_for_match(pred, gt, sigmas=sigmas)
        if oks is not None:
            oks_values.append(oks)

    per_keypoint = {
        KEYPOINT_NAMES[k]: summarize_errors(per_kp_errors[k])
        for k in range(N_KEYPOINTS)
    }
    oks_mean = float(np.mean(oks_values)) if oks_values else None
    return {
        "per_keypoint_pixel_error": per_keypoint,
        "oks": {"mean": oks_mean, "n": len(oks_values)},
        "n_matched": len(matched),
    }


def compute_metrics(
    image_records: Sequence[tuple[Sequence[PredWheel], Sequence[GTWheel]]],
    conf_threshold: float,
    iou_match: float = DEFAULT_IOU_MATCH,
    sigmas: Sequence[float] = DEFAULT_SIGMAS,
) -> dict:
    """Aggregate per-image (preds, gts) pairs into the report dict.

    Predictions are assumed to be already filtered to ``conf_threshold``;
    they're counted as-is for the FP-rate denominator. Kept as a thin
    wrapper around ``_aggregate_pairs`` so callers (and the existing
    JSON schema) stay stable while slicing reuses the same primitive.
    """
    total_gt = 0
    total_pred = 0
    total_matched = 0
    total_fn = 0
    total_fp = 0
    matched_pairs: list[tuple[PredWheel, GTWheel]] = []

    for preds, gts in image_records:
        total_gt += len(gts)
        total_pred += len(preds)
        m = match_predictions_to_gt(preds, gts, iou_threshold=iou_match)
        total_matched += len(m.matches)
        total_fn += len(m.unmatched_gt)
        total_fp += len(m.unmatched_preds)
        for p_idx, g_idx in m.matches:
            matched_pairs.append((preds[p_idx], gts[g_idx]))

    agg = _aggregate_pairs(matched_pairs, sigmas=sigmas)

    fn_rate = (total_fn / total_gt) if total_gt > 0 else None
    fp_rate = (total_fp / total_pred) if total_pred > 0 else None

    return {
        "counts": {
            "gt_wheels": total_gt,
            "pred_wheels_above_conf": total_pred,
            "matched": total_matched,
            "false_negatives": total_fn,
            "false_positives": total_fp,
        },
        "per_keypoint_pixel_error": agg["per_keypoint_pixel_error"],
        "oks": agg["oks"],
        "rates": {
            "false_negative_rate": fn_rate,
            "false_positive_rate": fp_rate,
        },
    }


def compute_sliced_metrics(
    image_records: Sequence[tuple[Sequence[PredWheel], Sequence[GTWheel]]],
    *,
    iou_match: float = DEFAULT_IOU_MATCH,
    sigmas: Sequence[float] = DEFAULT_SIGMAS,
) -> dict:
    """Return per-slice metrics for both bbox-area and occlusion axes.

    Each bucket gets ``per_keypoint_pixel_error`` + ``oks`` +
    ``n_matched`` via ``_aggregate_pairs`` on its slice of matched pairs.
    Buckets are emitted even when empty so consumers can rely on a
    stable key set.
    """
    out: dict[str, dict] = {"by_bbox_area": {}, "by_occlusion": {}}
    for axis_key, axis in (
        ("by_bbox_area", "bbox_area"),
        ("by_occlusion", "occlusion"),
    ):
        buckets = slice_records(image_records, axis=axis, iou_match=iou_match)
        for bucket_name, bucket_records in buckets.items():
            pairs: list[tuple[PredWheel, GTWheel]] = []
            for preds, gts in bucket_records:
                # slice_records emits 1-pair lists, but be defensive.
                for p, g in zip(preds, gts, strict=True):
                    pairs.append((p, g))
            out[axis_key][bucket_name] = _aggregate_pairs(pairs, sigmas=sigmas)
    return out


# ---------------------------------------------------------------------------
# Bbox mAP via Ultralytics
# ---------------------------------------------------------------------------


def compute_bbox_map(
    model: "YOLO",
    data: str | Path,
    *,
    split: str,
    conf: float,
    iou: float,
    device: str,
) -> dict:
    """Run model.val() once and pull mAP50 + mAP50-95 for boxes.

    Failure modes (missing attribute, val crashes mid-pass, etc.) are
    tolerated: we log a warning and return None in both fields so the
    overall eval script does not abort just because Ultralytics changed
    an internal attribute name.
    """
    out: dict = {"mAP50": None, "mAP50_95": None}
    try:
        results = model.val(
            data=str(data),
            split=split,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )
        box = getattr(results, "box", None)
        if box is None:
            warnings.warn("Ultralytics val() result has no .box; mAP not available.")
            return out
        map50 = getattr(box, "map50", None)
        map_ = getattr(box, "map", None)
        out["mAP50"] = float(map50) if map50 is not None else None
        out["mAP50_95"] = float(map_) if map_ is not None else None
    except Exception as exc:  # noqa: BLE001 - downstream we just want a None.
        warnings.warn(f"Ultralytics val() failed; mAP set to None: {exc!r}")
    return out


# ---------------------------------------------------------------------------
# Device handling
# ---------------------------------------------------------------------------


def resolve_device(requested: str | None) -> str:
    """Pick a device, falling back from mps→cpu if mps is unavailable.

    Ultralytics accepts strings like 'cpu', 'mps', '0' (CUDA index). We
    only special-case the macOS-default 'mps'.
    """
    if requested is None:
        requested = "mps"
    if requested != "mps":
        return requested
    try:
        import torch  # type: ignore

        mps_ok = (
            bool(getattr(torch.backends, "mps", None))
            and torch.backends.mps.is_available()
        )
    except Exception:
        mps_ok = False
    if mps_ok:
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_ascii_table(report: dict, split: str, n_images: int) -> str:
    counts = report["counts"]
    per_kp = report["per_keypoint_pixel_error"]
    oks = report["oks"]
    rates = report["rates"]

    lines: list[str] = []
    lines.append(
        f"Eval on {split} ({n_images} images, {counts['gt_wheels']} GT wheels)"
    )
    lines.append("-" * 41)
    lines.append(f"{'':<20} {'median':>7}  {'p95':>6}  {'n':>4}")

    for name in KEYPOINT_NAMES:
        s = per_kp[name]
        if s["median"] is None:
            lines.append(f"{name:<20} {'n/a':>7}  {'n/a':>6}  {0:>4}")
        else:
            lines.append(
                f"{name:<20} {s['median']:>6.1f}px {s['p95']:>5.1f}px {s['n']:>4}"
            )
    lines.append("")
    if oks["mean"] is None:
        lines.append(f"OKS (mean):  n/a  (n={oks['n']})")
    else:
        lines.append(f"OKS (mean):  {oks['mean']:.2f}  (n={oks['n']})")
    fn_rate = rates["false_negative_rate"]
    fp_rate = rates["false_positive_rate"]
    fn_str = f"{fn_rate:.2f}" if fn_rate is not None else "n/a"
    fp_str = f"{fp_rate:.2f}" if fp_rate is not None else "n/a"
    lines.append(
        f"FN rate:     {fn_str}  ({counts['false_negatives']} / {counts['gt_wheels']})"
    )
    lines.append(
        f"FP rate:     {fp_str}  ({counts['false_positives']} / {counts['pred_wheels_above_conf']})"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate YOLO-pose wheel keypoints on a dataset split"
    )
    p.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to a trained YOLO-pose checkpoint (.pt).",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=Path("configs/dataset.yaml"),
        help="Path to the YOLO dataset config (default: configs/dataset.yaml).",
    )
    p.add_argument("--split", default="val", choices=("train", "val"))
    p.add_argument(
        "--device",
        default="mps",
        help="Inference device. Defaults to mps; falls back to cpu when unavailable.",
    )
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--max-det", type=int, default=20)
    p.add_argument(
        "--iou-match",
        type=float,
        default=DEFAULT_IOU_MATCH,
        help=f"IoU threshold for GT/pred matching (default {DEFAULT_IOU_MATCH}).",
    )
    p.add_argument(
        "--sigma",
        type=str,
        default=None,
        help=(
            "Override OKS sigmas. Either a single float applied to all 3 "
            "keypoints (e.g. --sigma 0.05) or three comma-separated floats "
            f"(e.g. --sigma 0.10,0.10,0.10). Default {DEFAULT_SIGMAS}."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/eval/eval.json"),
        help="Where to write the JSON report.",
    )
    p.add_argument(
        "--worst-n",
        type=int,
        default=DEFAULT_WORST_N,
        help=(
            "Number of worst-error matched pairs to record in the failure "
            f"catalogue (default {DEFAULT_WORST_N}). Set to 0 to skip."
        ),
    )
    return p.parse_args()


def parse_sigmas(raw: str | None) -> tuple[float, float, float]:
    if raw is None:
        return DEFAULT_SIGMAS
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) == 1:
        v = float(parts[0])
        return (v, v, v)
    if len(parts) == N_KEYPOINTS:
        vals = tuple(float(p) for p in parts)
        return vals  # type: ignore[return-value]
    raise ValueError(
        f"--sigma must be one float or {N_KEYPOINTS} comma-separated floats; got {raw!r}"
    )


def main() -> int:
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset config not found: {args.data}")

    dataset_root = resolve_dataset_root(args.data)
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {dataset_root}\n"
            "Hint: run\n"
            "  python src/create_sample_incoming.py --count 20 --overwrite\n"
            "  python src/convert_incoming_to_yolo.py "
            "--source-root data/incoming/manual_sample "
            "--dataset-root data/wheel_dataset --overwrite\n"
            "Or place a real batch under data/incoming/<source>/ and run the converter."
        )

    images = list_split_images(dataset_root, args.split)
    if not images:
        raise RuntimeError(
            f"No images found under {dataset_root / 'images' / args.split}."
        )

    device = resolve_device(args.device)
    if device != args.device:
        print(
            f"WARNING: requested device {args.device!r} not available, using {device!r}."
        )

    model = YOLO(str(args.model))
    if getattr(model, "task", None) != "pose":
        raise SystemExit(
            f"ERROR: model task is {getattr(model, 'task', '?')!r}, expected 'pose'. "
            "Aborting: keypoint evaluation needs a -pose model."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Track image_path alongside (preds, gts) so we can build a failure
    # catalogue. The aggregator still consumes (preds, gts) only.
    image_records_with_paths: list[tuple[Path, list[PredWheel], list[GTWheel]]] = []
    for img_path in images:
        img_w, img_h = read_image_size(img_path)
        label_path = dataset_root / "labels" / args.split / f"{img_path.stem}.txt"
        gts = parse_yolo_pose_label(label_path, img_w, img_h)

        # One image at a time keeps memory bounded and matches the pattern
        # used by src/infer_image.py.
        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=device,
            verbose=False,
        )
        result = results[0]
        preds = extract_pred_wheels(result, conf_threshold=args.conf)
        image_records_with_paths.append((img_path, preds, gts))

    image_records: list[tuple[list[PredWheel], list[GTWheel]]] = [
        (preds, gts) for _, preds, gts in image_records_with_paths
    ]

    sigmas = parse_sigmas(args.sigma)
    metrics = compute_metrics(
        image_records,
        conf_threshold=args.conf,
        iou_match=args.iou_match,
        sigmas=sigmas,
    )
    metrics["counts"]["images"] = len(images)

    # Bbox mAP via a separate Ultralytics val() pass. Failure is tolerated:
    # mAP returns {"mAP50": None, "mAP50_95": None} with a warning so we
    # still emit the rest of the report.
    metrics_bbox = compute_bbox_map(
        model,
        data=args.data,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        device=device,
    )

    sliced = compute_sliced_metrics(
        image_records,
        iou_match=args.iou_match,
        sigmas=sigmas,
    )

    # Failure catalogue: flatten all matched pairs once with their image
    # path via the shared collect_matched_pairs helper, then rank by
    # mean pixel error desc.
    matched_pairs = collect_matched_pairs(
        image_records_with_paths, iou_match=args.iou_match
    )
    failure_samples = top_n_failures(matched_pairs, n=args.worst_n, sigmas=sigmas)

    report = {
        "model": str(args.model),
        "data": str(args.data),
        "split": args.split,
        "device": device,
        "thresholds": {
            "conf": args.conf,
            "iou_match": args.iou_match,
            "iou_nms": args.iou,
            "max_det": args.max_det,
        },
        "sigmas": sigmas,
        "sigmas_note": (
            "Per-keypoint OKS sigmas. Default 0.10 picks a tolerance "
            "radius k*s ≈ 6 px on a 60 px wheel — a few-pixel error "
            "produces a usable OKS instead of saturating at 0. Revisit "
            "once we have a real-data annotation-noise floor; override "
            "with --sigma."
        ),
        "counts": metrics["counts"],
        "per_keypoint_pixel_error": metrics["per_keypoint_pixel_error"],
        "oks": metrics["oks"],
        "rates": metrics["rates"],
        "metrics_bbox": metrics_bbox,
        "slices": sliced,
        "failure_samples": failure_samples,
    }

    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(format_ascii_table(report, args.split, len(images)))
    print()
    print(f"JSON report: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
