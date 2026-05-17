"""AR-friendly postprocessing for wheel + keypoint detections.

Turns a flat list of YOLO-pose detections (each a `wheel` with 3 keypoints)
into the JSON payload the AR client consumes.

Per the AR spec (https://docs.google.com/document/d/1HwMfJYc3eWaovN183370iWYmLjTosF9UMconj-UawFg/):

  - ML returns per-frame, per-wheel keypoints in pixel coordinates.
  - AR does raycast, RANSAC, plane reconstruction, K-frame accumulation,
    and cross-frame association. ML stays out of 3D.
  - Each confirmed response carries the input's `frame_id` so AR can
    pair it back with the camera transform it saved at capture time.
    Timestamp is retained only in the legacy/debug payload.

The three keypoints are always in this order:
  index 0 = rim_left
  index 1 = rim_right
  index 2 = disc_bottom

Run with ``--demo`` to see a worked example.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Iterable

KEYPOINT_NAMES = ("rim_left", "rim_right", "disc_bottom")
N_KEYPOINTS = 3

# Mapping from the internal/training keypoint names to the AR-target naming
# described in docs/KEYPOINT_SPEC.md and docs/AR_ML_CONTRACT.md.
# Used only by the optional `--target-schema` preview output of
# `infer_image.py` — does not affect the load-bearing current contract.
INTERNAL_TO_TARGET_KP = {
    "rim_left": "point_a",
    "rim_right": "point_b",
    "disc_bottom": "point_c_disc_bottom",
}

# Mapping from internal/training keypoint names to the AR-team **confirmed**
# naming (response 2026-05-13, see docs/AR_ML_CONTRACT.md "JSON shape").
# Shorter, flat keys; used by the primary output path via `to_confirmed_schema`.
# Independent of INTERNAL_TO_TARGET_KP.
INTERNAL_TO_CONFIRMED_KP = {
    "rim_left": "a",
    "rim_right": "b",
    "disc_bottom": "c_disc_bottom",
}

CONFIRMED_POINT_KEYS = ("a", "b", "c_disc_bottom")
FLOOR_RAY_MIN_REL_Y = 0.80
DISC_BOTTOM_MIN_REL_Y = 0.50
MIN_FLOOR_RAY_WIDTH_FRACTION = 0.50


def confirmed_geometry_issues(wheel: dict) -> list[str]:
    """Return floor-ray geometry issues that make a wheel unsafe for AR.

    The confirmed schema has no uncertainty fields. A wheel that violates the
    A/B/C geometry would look "certain" to the AR side, so it must be filtered
    before the primary JSON is emitted.
    """
    issues: list[str] = []
    bbox = wheel.get("wheel_bbox")
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return ["missing wheel_bbox[4]"]
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return ["wheel_bbox contains non-numeric values"]
    if x1 >= x2 or y1 >= y2:
        return ["wheel_bbox is degenerate"]

    points: dict[str, list[float]] = {}
    for kp in wheel.get("keypoints", []):
        target_name = INTERNAL_TO_CONFIRMED_KP.get(kp.get("name"))
        if target_name is None:
            continue
        xy = kp.get("xy")
        if not (isinstance(xy, list) and len(xy) == 2):
            issues.append(f"{target_name} missing xy[2]")
            continue
        try:
            px, py = float(xy[0]), float(xy[1])
        except (TypeError, ValueError):
            issues.append(f"{target_name} contains non-numeric values")
            continue
        points[target_name] = [px, py]

    for key in CONFIRMED_POINT_KEYS:
        if key not in points:
            issues.append(f"missing points.{key}")
            continue
        px, py = points[key]
        if not (x1 <= px <= x2 and y1 <= py <= y2):
            issues.append(f"points.{key} outside bbox")
    if issues:
        return issues

    a = points["a"]
    b = points["b"]
    c = points["c_disc_bottom"]
    width = x2 - x1
    height = y2 - y1
    a_rel_y = (a[1] - y1) / height
    b_rel_y = (b[1] - y1) / height
    c_rel_y = (c[1] - y1) / height
    ab_sep = (b[0] - a[0]) / width

    if a[0] >= b[0]:
        issues.append("A is not left of B")
    if ab_sep < MIN_FLOOR_RAY_WIDTH_FRACTION:
        issues.append("A/B horizontal separation is too small")
    if min(a_rel_y, b_rel_y) < FLOOR_RAY_MIN_REL_Y:
        issues.append("A/B are not on the lower floor-ray band")
    if c_rel_y <= DISC_BOTTOM_MIN_REL_Y:
        issues.append("C is not in the lower half of the wheel bbox")
    if c[1] >= min(a[1], b[1]):
        issues.append("C is not above the A/B floor-ray line")
    return issues


def build_ar_payload(
    detections: Iterable[dict],
    conf_threshold: float | None = None,
    frame_id: str | None = None,
    timestamp: float | None = None,
) -> dict:
    """Format pose detections as the AR JSON payload.

    Each input detection is a dict with keys:
      class_name: str — currently always "wheel"
      bbox:       [x1, y1, x2, y2] in pixels
      confidence: float in [0, 1] — wheel-level detection confidence
      keypoints:  list of {"xy": [x, y], "visibility": 0|1|2, "confidence": float}
                  in canonical order (rim_left, rim_right, disc_bottom)

    If ``conf_threshold`` is set, wheels with detection confidence below it
    are dropped. Keypoint-level confidences are passed through untouched —
    AR weights them itself during RANSAC.
    """
    out_wheels: list[dict] = []
    for d in detections:
        if d.get("class_name") != "wheel":
            continue
        wheel_conf = float(d["confidence"])
        if conf_threshold is not None and wheel_conf < conf_threshold:
            continue

        kps_in = d.get("keypoints") or []
        if len(kps_in) != N_KEYPOINTS:
            # A wheel without 3 keypoints isn't usable for AR's pipeline.
            # Drop it rather than fabricate.
            continue

        kps_out: list[dict] = []
        for i, kp in enumerate(kps_in):
            xy = kp.get("xy", [0.0, 0.0])
            kps_out.append(
                {
                    "name": KEYPOINT_NAMES[i],
                    "xy": [float(xy[0]), float(xy[1])],
                    "visibility": int(kp.get("visibility", 2)),
                    "confidence": float(kp["confidence"])
                    if "confidence" in kp
                    else None,
                }
            )

        out_wheels.append(
            {
                "wheel_bbox": [float(v) for v in d["bbox"]],
                "keypoints": kps_out,
                "confidence": wheel_conf,
                "warnings": [],
            }
        )

    # Largest wheel first — convenience for AR (likely closest to camera).
    def _bbox_area(w: dict) -> float:
        x1, y1, x2, y2 = w["wheel_bbox"]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    out_wheels.sort(key=_bbox_area, reverse=True)

    payload: dict = {
        "wheels": out_wheels,
        "stats": {
            "n_wheels": len(out_wheels),
        },
    }
    if frame_id is not None:
        payload["frame_id"] = frame_id
    if timestamp is not None:
        payload["timestamp"] = timestamp
    return payload


def to_target_schema(ar_payload: dict) -> dict:
    """Convert a current AR payload to the target schema in
    docs/AR_ML_CONTRACT.md.

    Pure restructure — no new information. Used behind --target-schema in
    `infer_image.py` so the AR team can preview the target without
    breaking consumers of the current contract.

    Field changes vs current:
      - bbox: xyxy list → xywh list under `bbox_xywh`
      - keypoints: list-of-objects → dict keyed by AR-target name
                   (point_a / point_b / point_c_disc_bottom)
      - per-keypoint confidence: lifted out into a parallel dict
                   `keypoints_confidence`
      - per-keypoint visibility: lifted out into a parallel dict
                   `visibility`
      - dropped: `image`, `image_size`, `thresholds`, `stats`,
                   `warnings` (see OPEN_QUESTIONS_AR_SPEC §10)

    Pending AR-team confirmation: §3 (field names), §8 (bbox format),
    §9 (dict vs array), §10 (dropped fields). This converter encodes the
    *current target hypothesis*; it will move when the team answers.
    """
    target: dict = {"wheels": []}
    if "frame_id" in ar_payload:
        target["frame_id"] = ar_payload["frame_id"]
    if "timestamp" in ar_payload:
        target["timestamp"] = ar_payload["timestamp"]

    for w in ar_payload.get("wheels", []):
        x1, y1, x2, y2 = w["wheel_bbox"]
        bbox_xywh = [
            float(x1),
            float(y1),
            float(x2 - x1),
            float(y2 - y1),
        ]
        kp_xy: dict[str, list[float]] = {}
        kp_conf: dict[str, float | None] = {}
        kp_vis: dict[str, int] = {}
        for kp in w.get("keypoints", []):
            target_name = INTERNAL_TO_TARGET_KP.get(kp["name"], kp["name"])
            kp_xy[target_name] = [float(kp["xy"][0]), float(kp["xy"][1])]
            kp_conf[target_name] = (
                None if kp.get("confidence") is None else float(kp["confidence"])
            )
            kp_vis[target_name] = int(kp["visibility"])
        target["wheels"].append(
            {
                "bbox_xywh": bbox_xywh,
                "confidence": float(w["confidence"]),
                "keypoints": kp_xy,
                "keypoints_confidence": kp_conf,
                "visibility": kp_vis,
            }
        )
    return target


def to_confirmed_schema(ar_payload: dict) -> dict:
    """Convert a legacy AR payload (build_ar_payload output) to the
    AR-team confirmed schema (docs/AR_ML_CONTRACT.md, 2026-05-13).

    Pure restructure. No visibility, no per-keypoint confidence, no
    timestamp. Drops `image`, `image_size`, `thresholds`, `stats`,
    `warnings`. Bbox stays xyxy (already xyxy in legacy).

    Wheels whose any internal keypoint had `visibility < 2` are
    SKIPPED (the confirmed schema represents only fully-visible
    wheels). If a wheel had three valid keypoints with visibility == 2
    in the legacy payload, all three are emitted as [x, y].
    """
    confirmed: dict = {}
    if "frame_id" in ar_payload:
        confirmed["frame_id"] = ar_payload["frame_id"]
    confirmed["wheels"] = []

    for w in ar_payload.get("wheels", []):
        kps = w.get("keypoints", [])
        # Confirmed AR JSON has no visibility field, so only fully visible
        # wheels can be emitted without hiding uncertainty from the AR side.
        if any(int(kp.get("visibility", 0)) < 2 for kp in kps):
            continue
        if confirmed_geometry_issues(w):
            continue

        x1, y1, x2, y2 = w["wheel_bbox"]
        bbox_xyxy = [float(x1), float(y1), float(x2), float(y2)]

        points: dict[str, list[float]] = {}
        for kp in kps:
            target_name = INTERNAL_TO_CONFIRMED_KP.get(kp["name"])
            if target_name is None:
                # Unknown internal name — skip; the confirmed schema is
                # a closed set {a, b, c_disc_bottom}.
                continue
            points[target_name] = [float(kp["xy"][0]), float(kp["xy"][1])]

        confirmed["wheels"].append(
            {
                "bbox_xyxy": bbox_xyxy,
                "confidence": float(w["confidence"]),
                "points": points,
            }
        )
    return confirmed


def _demo() -> int:
    detections = [
        {
            "class_name": "wheel",
            "bbox": [100, 200, 200, 300],
            "confidence": 0.93,
            "keypoints": [
                {"xy": [150, 210], "visibility": 2, "confidence": 0.95},
                {"xy": [150, 290], "visibility": 2, "confidence": 0.92},
                {"xy": [150, 295], "visibility": 2, "confidence": 0.88},
            ],
        },
        {
            "class_name": "wheel",
            "bbox": [300, 200, 380, 280],
            "confidence": 0.88,
            "keypoints": [
                {"xy": [340, 208], "visibility": 2, "confidence": 0.90},
                {"xy": [340, 272], "visibility": 1, "confidence": 0.55},  # occluded
                {"xy": [340, 275], "visibility": 0, "confidence": 0.10},  # hidden
            ],
        },
    ]
    payload = build_ar_payload(
        detections,
        conf_threshold=0.25,
        frame_id="demo-frame-0001",
        timestamp=1736_700_000.0,
    )
    print(json.dumps(payload, indent=2))
    print("--- confirmed schema ---")
    confirmed = to_confirmed_schema(payload)
    print(json.dumps(confirmed, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Postprocess wheel pose detections")
    p.add_argument(
        "--demo",
        action="store_true",
        help="Run a worked example and print the AR payload.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.demo:
        sys.exit(_demo())
    print(
        "Nothing to do. Pass --demo to see a worked example, "
        "or import build_ar_payload() from another module."
    )
    sys.exit(0)
