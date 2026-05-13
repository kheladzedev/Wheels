"""Render an inference JSON back onto its image.

Useful for inspecting the AR payload independent of Ultralytics' own plotter
(e.g. when reviewing what the AR client will actually receive).

Usage:
    python src/visualize_predictions.py --image data/samples/car.jpg --json outputs/car.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

COLOR_BBOX = (255, 128, 0)
COLOR_KP = (
    (0, 255, 0),  # rim_left
    (0, 200, 255),  # rim_right
    (0, 0, 255),  # disc_bottom
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize wheel pose JSON predictions")
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--json", required=True, type=Path)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: <image_stem>_vis.jpg)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    img = cv2.imread(str(args.image))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")
    with open(args.json, encoding="utf-8") as f:
        payload = json.load(f)

    for w in payload.get("wheels", []):
        bbox = w.get("wheel_bbox")
        if bbox:
            x1, y1, x2, y2 = (int(round(v)) for v in bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_BBOX, 2)
            label = f"wheel {w.get('confidence', 0):.2f}"
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

        for i, kp in enumerate(w.get("keypoints", [])):
            if kp.get("visibility", 0) == 0:
                continue
            kx, ky = (int(round(v)) for v in kp["xy"])
            color = COLOR_KP[i % len(COLOR_KP)]
            cv2.circle(img, (kx, ky), 5, color, -1)
            kp_conf = kp.get("confidence")
            tag = (
                f"{kp.get('name', f'kp{i}')} {kp_conf:.2f}"
                if kp_conf is not None
                else kp.get("name", f"kp{i}")
            )
            cv2.putText(
                img,
                tag,
                (kx + 6, ky - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )

    out_path = args.out or args.image.with_name(f"{args.image.stem}_vis.jpg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
