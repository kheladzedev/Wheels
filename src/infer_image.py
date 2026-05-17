"""Run YOLO-pose inference on a single image and emit the AR JSON payload.

JSON artifacts per image:
  - <stem>.json         — **AR-team confirmed schema (PRIMARY output)**
                          {frame_id, wheels[].{bbox_xyxy, confidence,
                          points.{a,b,c_disc_bottom}}}. Per
                          docs/AR_ML_CONTRACT.md (response 2026-05-13).
  - <stem>_legacy.json  — Intermediate legacy payload with image / image_size
                          / thresholds / stats / per-keypoint visibility +
                          confidence. Kept for debug and any tool still
                          reading the pre-confirmation shape.
  - <stem>_raw.json     — Flat list of post-threshold YOLO-pose detections.

Plus one or two visualizations depending on --viz-mode (default: final):
  - <stem>_raw_pred.jpg   — wheel bboxes with keypoints overlaid (ML debug).
  - <stem>_final_pred.jpg — AR-final view rendered from the confirmed JSON:
                             wheel bboxes + named A/B/C keypoints. AR
                             consumes <stem>.json, not the JPG overlay.

Confidence filtering is enforced in three places (defence in depth):
  1. model.predict(conf=...) — YOLO's own filter.
  2. Manual re-filter in this script.
  3. Final assert: no kept detection has conf < args.conf.

Usage:
    python src/infer_image.py --image data/samples/car.jpg
    python src/infer_image.py --image data/samples/car.jpg \\
        --model runs/pose/wheel_baseline/weights/best.pt \\
        --device mps --conf 0.25 --iou 0.45 --max-det 20 --viz-mode both \\
        --frame-id frame_001 --timestamp 1736700000.0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from postprocess_wheels import (
    N_KEYPOINTS,
    build_ar_payload,
    to_confirmed_schema,
    to_target_schema,
)

WHEEL_CLASS_NAMES = {"wheel"}

# BGR colors. Keypoint colours match the AR mock spec board: A=red, B=blue,
# C=green. Internally the training names are still rim_left/rim_right/
# disc_bottom for backward compatibility, but overlays should show the
# AR-facing A/B/C contract.
COLOR_BBOX = (255, 128, 0)
COLOR_KP = (
    (0, 0, 255),  # A / rim_left — red
    (255, 0, 0),  # B / rim_right — blue
    (0, 255, 0),  # C / disc_bottom — green
)
DISPLAY_KP_NAMES = ("A", "B", "C")
COLOR_LABEL = (255, 255, 255)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO-pose inference for AR wheel fitting")
    p.add_argument("--image", required=True, type=Path, help="Path to input image")
    p.add_argument(
        "--model", default="yolo11n-pose.pt", help="Path to YOLO-pose weights"
    )
    p.add_argument(
        "--out-dir", default=Path("outputs"), type=Path, help="Where to save artifacts"
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold. See README → Inference thresholds.",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (lower = stricter dedup).",
    )
    p.add_argument(
        "--max-det",
        type=int,
        default=20,
        help="Hard cap on detections kept after thresholding/NMS.",
    )
    p.add_argument(
        "--viz-mode",
        choices=("raw", "final", "both"),
        default="final",
        help="raw=bboxes+keypoints debug view, final=AR view, both=save both.",
    )
    p.add_argument(
        "--frame-id",
        default=None,
        help="Frame ID echoed back in the AR JSON so AR can match the "
        "response to the saved camera transform. Defaults to the image stem.",
    )
    p.add_argument(
        "--timestamp",
        type=float,
        default=None,
        help="Capture timestamp (Unix seconds) written only to legacy/debug JSON. "
        "Defaults to time.time() at inference start.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Inference device: 'mps' for Apple Silicon, 'cpu', or '0' for CUDA. "
        "If omitted, Ultralytics picks the best available.",
    )
    p.add_argument(
        "--target-schema",
        action="store_true",
        help=(
            "DEPRECATED preview path. Additionally write <stem>_target.json "
            "in the pre-confirmation target schema "
            "(point_a/point_b/point_c_disc_bottom + bbox_xywh + parallel "
            "dicts). Kept only for the AR team's earlier integration "
            "preview; the confirmed schema in <stem>.json supersedes it."
        ),
    )
    p.add_argument(
        "--confirmed-schema",
        action="store_true",
        help=(
            "No-op since <stem>.json is now the confirmed schema by "
            "default. Kept for backward compatibility with older scripts "
            "that passed this flag."
        ),
    )
    return p.parse_args()


def determine_frame_id(arg_frame_id: str | None, image_path: Path) -> str:
    """Confirmed schema requires `frame_id`. When the caller (AR client)
    didn't pass one — single-image inference, debug runs — derive it
    from the image stem so the contract stays satisfied without
    fabricating opaque IDs.
    """
    if arg_frame_id is not None and arg_frame_id != "":
        return arg_frame_id
    return image_path.stem


def extract_keypoints(box_idx: int, result) -> list[dict]:
    """Return 3 keypoints for the i-th detection from a YOLO-pose Results object.

    Each entry: {"xy": [x, y], "visibility": int, "confidence": float}.
    Visibility is inferred from per-keypoint confidence (Ultralytics doesn't
    emit a separate visibility flag at inference time — only confidence).
    """
    if result.keypoints is None:
        return []
    xy = result.keypoints.xy[box_idx].cpu().numpy()  # (K, 2)
    if result.keypoints.conf is not None:
        conf = result.keypoints.conf[box_idx].cpu().numpy()  # (K,)
    else:
        # Some pose models omit conf at inference time; fall back to 1.0
        # which means "fully present" — AR can still RANSAC them.
        conf = [1.0] * xy.shape[0]

    kps: list[dict] = []
    for i in range(xy.shape[0]):
        c = float(conf[i])
        # Heuristic: kp with confidence < 0.15 is effectively missing.
        # The exact threshold is informational — AR uses confidence directly.
        vis = 2 if c >= 0.5 else (1 if c >= 0.15 else 0)
        kps.append(
            {
                "xy": [float(xy[i, 0]), float(xy[i, 1])],
                "visibility": vis,
                "confidence": c,
            }
        )
    return kps


def draw_raw_overlay(image_bgr, detections: list[dict]):
    img = image_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in d["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_BBOX, 2)
        label = f"{d['class_name']} {d['confidence']:.3f}"
        cv2.putText(
            img,
            label,
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_BBOX,
            1,
            cv2.LINE_AA,
        )
        for i, kp in enumerate(d.get("keypoints", [])):
            if kp["visibility"] == 0:
                continue
            kx, ky = (int(round(v)) for v in kp["xy"])
            cv2.circle(img, (kx, ky), 4, COLOR_KP[i % len(COLOR_KP)], -1)
    return img


def draw_final_overlay(image_bgr, wheels: list[dict]):
    """Render confirmed-schema bbox + keypoints with non-overlapping labels.

    Anchors each keypoint's label at a distinct offset (top / right /
    bottom of the dot) so small wheels don't get a wall of overlapping
    text. Labels are drawn to the right of the bbox when the wheel is
    too narrow to leave room.
    """
    img = image_bgr.copy()
    w_img = img.shape[1]
    for w in wheels:
        x1, y1, x2, y2 = (int(round(v)) for v in w["bbox_xyxy"])
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_BBOX, 2)
        cv2.putText(
            img,
            f"wheel {w['confidence']:.2f}",
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_BBOX,
            1,
            cv2.LINE_AA,
        )
        # Anchor keypoint labels just to the right of the bbox, stacked
        # vertically. This avoids the labels piling up inside small wheels.
        label_x = min(x2 + 6, w_img - 120)
        label_y_start = max(y1 + 12, 14)
        for i, key in enumerate(("a", "b", "c_disc_bottom")):
            point = w["points"][key]
            kx, ky = (int(round(v)) for v in point)
            color = COLOR_KP[i % len(COLOR_KP)]
            cv2.circle(img, (kx, ky), 4, color, -1)
            display_name = DISPLAY_KP_NAMES[i] if i < len(DISPLAY_KP_NAMES) else key
            label_y = label_y_start + i * 14
            # Short leader line from keypoint to its label so the eye
            # can still associate them when the bbox is small.
            cv2.line(img, (kx, ky), (label_x - 2, label_y - 4), color, 1, cv2.LINE_AA)
            cv2.putText(
                img,
                display_name,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
    return img


def main() -> None:
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frame_id = determine_frame_id(args.frame_id, args.image)
    timestamp = args.timestamp if args.timestamp is not None else time.time()

    model = YOLO(str(args.model))
    if getattr(model, "task", None) != "pose":
        print(
            f"WARNING: model task is {getattr(model, 'task', '?')!r}, expected 'pose'. "
            "Keypoints will be empty."
        )

    results = model.predict(
        source=str(args.image),
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        verbose=False,
    )
    result = results[0]
    class_names: dict[int, str] = result.names

    model_has_wheels = any(name in WHEEL_CLASS_NAMES for name in class_names.values())

    total_boxes = 0 if result.boxes is None else len(result.boxes)
    dropped_by_conf = 0
    dropped_by_class = 0
    detections: list[dict] = []

    if result.boxes is not None:
        for i, box in enumerate(result.boxes):
            conf = float(box.conf.item())
            if conf < args.conf:
                dropped_by_conf += 1
                continue
            cls_id = int(box.cls.item())
            name = class_names.get(cls_id, str(cls_id))
            if model_has_wheels and name not in WHEEL_CLASS_NAMES:
                dropped_by_class += 1
                continue
            kps = extract_keypoints(i, result)
            if kps and len(kps) != N_KEYPOINTS:
                print(
                    f"WARNING: model emitted {len(kps)} keypoints, expected {N_KEYPOINTS}. "
                    "Dropping detection."
                )
                continue
            detections.append(
                {
                    "class_name": name,
                    "bbox": [float(v) for v in box.xyxy[0].tolist()],
                    "confidence": conf,
                    "keypoints": kps,
                }
            )

    assert all(d["confidence"] >= args.conf for d in detections), (
        f"Sub-threshold detection leaked into raw output (conf < {args.conf})"
    )
    if len(detections) > args.max_det:
        raise RuntimeError(
            f"Kept {len(detections)} detections but --max-det={args.max_det}. "
            "Bug in the filter."
        )

    img_size = [int(result.orig_shape[1]), int(result.orig_shape[0])]
    thresholds = {"conf": args.conf, "iou": args.iou, "max_det": args.max_det}

    raw_payload = {
        "image": str(args.image),
        "image_size": img_size,
        "frame_id": frame_id,
        "timestamp": timestamp,
        "thresholds": thresholds,
        "detections": detections,
    }

    # Internal intermediate payload — keeps wheel_bbox / per-kp visibility +
    # confidence / warnings / stats. Used as input to the confirmed-schema
    # converter, and persisted as <stem>_legacy.json for debug + backward
    # compatibility with anything still reading the pre-confirmation shape.
    legacy_payload = build_ar_payload(
        detections,
        conf_threshold=args.conf,
        frame_id=frame_id,
        timestamp=timestamp,
    )
    legacy_with_meta = dict(legacy_payload)
    legacy_with_meta["image"] = str(args.image)
    legacy_with_meta["image_size"] = img_size
    legacy_with_meta["thresholds"] = thresholds

    assert all(w["confidence"] >= args.conf for w in legacy_payload["wheels"]), (
        f"Sub-threshold candidate leaked into AR payload (conf < {args.conf})"
    )

    # PRIMARY output — AR-team confirmed schema, no legacy fields.
    confirmed_payload = to_confirmed_schema(legacy_payload)
    assert "frame_id" in confirmed_payload, (
        "Confirmed schema must always include frame_id (derived from image "
        "stem if not provided) — see docs/AR_ML_CONTRACT.md."
    )
    for forbidden in ("timestamp", "stats", "image", "image_size", "thresholds"):
        assert forbidden not in confirmed_payload, (
            f"Confirmed payload top-level leaked '{forbidden}'."
        )

    stem = args.image.stem
    json_path = args.out_dir / f"{stem}.json"
    legacy_json_path = args.out_dir / f"{stem}_legacy.json"
    raw_json_path = args.out_dir / f"{stem}_raw.json"
    target_json_path = args.out_dir / f"{stem}_target.json"
    raw_vis_path = args.out_dir / f"{stem}_raw_pred.jpg"
    final_vis_path = args.out_dir / f"{stem}_final_pred.jpg"

    json_path.write_text(
        json.dumps(confirmed_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    legacy_json_path.write_text(
        json.dumps(legacy_with_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    raw_json_path.write_text(
        json.dumps(raw_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if args.target_schema:
        target_payload = to_target_schema(legacy_payload)
        target_json_path.write_text(
            json.dumps(target_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    original = cv2.imread(str(args.image))
    if original is None:
        raise RuntimeError(f"OpenCV could not decode image: {args.image}")

    written_images: list[Path] = []
    if args.viz_mode in ("raw", "both"):
        cv2.imwrite(str(raw_vis_path), draw_raw_overlay(original, detections))
        written_images.append(raw_vis_path)
    if args.viz_mode in ("final", "both"):
        cv2.imwrite(
            str(final_vis_path),
            draw_final_overlay(original, confirmed_payload["wheels"]),
        )
        written_images.append(final_vis_path)

    stats = legacy_payload["stats"]
    print(
        f"Thresholds:           conf={args.conf}, iou={args.iou}, max_det={args.max_det}"
    )
    print(f"Viz mode:             {args.viz_mode}")
    print(f"Frame ID:             {frame_id}")
    print(f"Timestamp:            {timestamp}")
    print(f"Total model boxes:    {total_boxes}")
    print(f"Dropped by conf:      {dropped_by_conf}")
    print(f"Dropped by class:     {dropped_by_class}")
    print(f"Detections kept:      {len(detections)}")
    print(f"Final wheels (legacy): {stats['n_wheels']}")
    print(f"Final wheels (confirmed): {len(confirmed_payload['wheels'])}")
    if len(detections) == 0:
        print(
            "No detections above conf threshold. "
            "Try lower --conf or use better trained weights."
        )
    print(f"AR JSON (confirmed):  {json_path}")
    print(f"Legacy JSON:          {legacy_json_path}")
    print(f"Raw JSON:             {raw_json_path}")
    if args.target_schema:
        print(f"Target preview JSON:  {target_json_path}")
    if args.confirmed_schema:
        print(
            "Note: --confirmed-schema is a no-op now; "
            f"{json_path} already uses the confirmed schema."
        )
    for p in written_images:
        print(f"Image:                {p}")


if __name__ == "__main__":
    main()
