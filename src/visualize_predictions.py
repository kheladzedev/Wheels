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

from postprocess_wheels import assert_confirmed_schema_closed

COLOR_BBOX = (255, 128, 0)
COLOR_KP = (
    (0, 0, 255),  # A — red
    (255, 0, 0),  # B — blue
    (0, 255, 0),  # C — green
)
DISPLAY_KP_NAMES = ("A", "B", "C")


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
    p.add_argument(
        "--require-frame-id",
        action="store_true",
        help="Require the confirmed JSON to include a non-empty frame_id.",
    )
    return p.parse_args()


def validate_confirmed_payload(
    payload: dict,
    *,
    source_label: str,
    require_frame_id: bool = False,
) -> list[dict]:
    """Return wheels from a confirmed-schema payload or raise ValueError."""
    try:
        assert_confirmed_schema_closed(
            payload,
            source_label=source_label,
            require_frame_id=require_frame_id,
        )
    except AssertionError as exc:
        raise ValueError(
            f"{source_label} is not a confirmed schema prediction JSON: {exc}"
        ) from exc
    return payload["wheels"]


def main() -> None:
    args = parse_args()
    img = cv2.imread(str(args.image))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")
    with open(args.json, encoding="utf-8") as f:
        payload = json.load(f)

    wheels = validate_confirmed_payload(
        payload,
        source_label=str(args.json),
        require_frame_id=args.require_frame_id,
    )

    for w in wheels:
        x1, y1, x2, y2 = (int(round(v)) for v in w["bbox_xyxy"])
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_BBOX, 2)
        label = f"wheel {w['confidence']:.2f}"
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
        keypoints = [
            {"name": "a", "xy": w["points"]["a"]},
            {"name": "b", "xy": w["points"]["b"]},
            {"name": "c_disc_bottom", "xy": w["points"]["c_disc_bottom"]},
        ]
        for i, kp in enumerate(keypoints):
            kx, ky = (int(round(v)) for v in kp["xy"])
            color = COLOR_KP[i % len(COLOR_KP)]
            cv2.circle(img, (kx, ky), 5, color, -1)
            display_name = (
                DISPLAY_KP_NAMES[i]
                if i < len(DISPLAY_KP_NAMES)
                else kp.get("name", f"kp{i}")
            )
            cv2.putText(
                img,
                display_name,
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
