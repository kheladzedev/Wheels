"""Generate a synthetic incoming batch in the Android-plugin keypoint format.

Writes the on-disk contract documented in `docs/KEYPOINT_DATASET_FORMAT.md`:
one `images/<stem>.jpg` per frame, one `annotations/<stem>.json` per frame,
plus a single `metadata/source_info.json`. This is a smoke-test fixture for
the *format*, not a source of training data — see `create_sample_incoming.py`
for the cartoon training generator.

The two generators are intentionally independent. The legacy one produces
the interim YOLO-pose oriented JSON (`objects[].keypoints[].name/xy/visibility`).
This one produces the plugin contract (`wheels[].points.a/b/c_disc_bottom`).

Per the 2026-05-14 spec revision, A/B are floor / raycast points near the
wheel footprint, not rim edges. The synthetic geometry below matches that:
A/B sit in the lower band of the bbox, near the tyre base; C stays at the
lowest visible point of the metal disc.

Usage:
    python src/create_sample_keypoint_incoming.py --count 50 --overwrite
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# Plugin spec lists these as the allowed input image extensions, but we
# always emit JPEG — the validator accepts any of them, no need to vary.
DEFAULT_OUTPUT_ROOT = Path("data/incoming/android_plugin")

# Drawing constants. Pure BGR — synthesised images go straight to cv2.imwrite.
BACKGROUND_BGR = (180, 180, 180)
GROUND_BAND_BGR = (140, 140, 140)
CAR_BODY_BGR = (60, 60, 60)
TYRE_BGR = (20, 20, 20)
RIM_BGR = (190, 190, 195)

# Probability of drawing a 2-wheel side view vs a 4-wheel front/back view.
# Weighted toward 2 wheels so most frames look like a side profile (more
# variety for visual sanity-checking the format).
P_TWO_WHEELS = 0.7

# Rim radius as a fraction of the tyre (outer) radius.
RIM_TO_TYRE = 0.65

# A/B placement under the 2026-05-14 contract: A and B are floor / raycast
# points near the wheel footprint, NOT rim edges. The fractions below are
# expressed in tyre-radius units relative to the wheel centre — chosen so
# both points sit comfortably inside the bbox (no validator warnings) and
# clearly in the lower band, below the disc-bottom point.
A_B_X_FRACTION_OF_TYRE = 0.70
A_B_Y_FRACTION_OF_TYRE = 0.88


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a synthetic incoming batch in the plugin keypoint format"
    )
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--count", type=int, default=50)
    p.add_argument("--image-width", type=int, default=640)
    p.add_argument("--image-height", type=int, default=480)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, delete the existing output root before writing.",
    )
    return p.parse_args(argv)


def _draw_background(img: np.ndarray, ground_y: int) -> None:
    """Solid grey background with a thin darker band near the bottom for context."""
    img[:] = BACKGROUND_BGR
    cv2.rectangle(
        img,
        (0, ground_y),
        (img.shape[1] - 1, img.shape[0] - 1),
        GROUND_BAND_BGR,
        -1,
    )


def _draw_car_body(
    img: np.ndarray,
    body_x1: int,
    body_y1: int,
    body_x2: int,
    body_y2: int,
) -> None:
    """Rounded dark rectangle. Corners drawn as filled circles + an inner rect."""
    radius = max(4, (body_y2 - body_y1) // 6)
    # Inner rectangle covers the area minus the corner radius on each side.
    cv2.rectangle(
        img,
        (body_x1 + radius, body_y1),
        (body_x2 - radius, body_y2),
        CAR_BODY_BGR,
        -1,
    )
    cv2.rectangle(
        img,
        (body_x1, body_y1 + radius),
        (body_x2, body_y2 - radius),
        CAR_BODY_BGR,
        -1,
    )
    for cx, cy in (
        (body_x1 + radius, body_y1 + radius),
        (body_x2 - radius, body_y1 + radius),
        (body_x1 + radius, body_y2 - radius),
        (body_x2 - radius, body_y2 - radius),
    ):
        cv2.circle(img, (cx, cy), radius, CAR_BODY_BGR, -1, lineType=cv2.LINE_AA)


def _draw_wheel(img: np.ndarray, cx: int, cy: int, tyre_radius: int) -> dict:
    """Draw one wheel (tyre + rim) and return its annotation entry.

    Geometry follows the 2026-05-14 contract revision:
      - bbox covers the full wheel including tyre (square of side 2r).
      - `a` / `b` are floor / raycast points near the wheel footprint, in
        the lower band of the bbox, not on the rim. They stay inside the
        bbox so the validator's "point-in-bbox" check passes clean.
      - `c_disc_bottom` is the lowest visible point of the metal disc —
        bottom edge of the rim ellipse.

    Floats throughout: plugin format spec says no numpy floats, no ints —
    see docs/KEYPOINT_DATASET_FORMAT.md.
    """
    rim_radius = max(2, int(tyre_radius * RIM_TO_TYRE))
    cv2.circle(img, (cx, cy), tyre_radius, TYRE_BGR, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (cx, cy), rim_radius, RIM_BGR, -1, lineType=cv2.LINE_AA)

    cx_f = float(cx)
    cy_f = float(cy)
    r_f = float(tyre_radius)
    rim_f = float(rim_radius)
    a_b_dx = r_f * A_B_X_FRACTION_OF_TYRE
    a_b_dy = r_f * A_B_Y_FRACTION_OF_TYRE

    return {
        "bbox_xyxy": [cx_f - r_f, cy_f - r_f, cx_f + r_f, cy_f + r_f],
        "points": {
            "a": [cx_f - a_b_dx, cy_f + a_b_dy],
            "b": [cx_f + a_b_dx, cy_f + a_b_dy],
            "c_disc_bottom": [cx_f, cy_f + rim_f],
        },
    }


def _wheel_centers(
    rng: random.Random,
    img_w: int,
    img_h: int,
    n_wheels: int,
    tyre_radius: int,
) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]]:
    """Pick wheel centres + a car-body bbox above them.

    Wheels sit on a horizontal line near the bottom of the image so each
    tyre fits entirely on-canvas (no clipping → the validator's in-bounds
    rule stays satisfied). Returns (centres, body_bbox).
    """
    # Vertical position fixed so the bottom of the tyre clears the bottom of
    # the image by at least a few pixels.
    cy = img_h - tyre_radius - 8

    # Horizontal extent: pick a span wide enough to fit all wheels with margin,
    # then jitter its leftmost position.
    min_span = 2 * tyre_radius * max(1, n_wheels - 1) + 2 * tyre_radius
    span = min(img_w - 20, max(min_span + 20, int(img_w * 0.55)))
    max_x_offset = img_w - span - 10
    x_offset = rng.randint(10, max(10, max_x_offset))

    if n_wheels == 1:
        centers = [(x_offset + span // 2, cy)]
    else:
        step = (span - 2 * tyre_radius) // (n_wheels - 1)
        centers = [(x_offset + tyre_radius + step * i, cy) for i in range(n_wheels)]

    body_h = max(40, int(img_h * 0.22))
    body_y1 = max(10, cy - tyre_radius - body_h + tyre_radius // 2)
    body_y2 = cy - tyre_radius // 2
    body_x1 = max(0, centers[0][0] - tyre_radius - 4)
    body_x2 = min(img_w - 1, centers[-1][0] + tyre_radius + 4)
    return centers, (body_x1, body_y1, body_x2, body_y2)


def generate_one(
    rng: random.Random, img_w: int, img_h: int
) -> tuple[np.ndarray, list[dict]]:
    """Build one synthetic frame + its plugin-format wheel list."""
    img = np.empty((img_h, img_w, 3), dtype=np.uint8)
    ground_y = int(img_h * 0.78)
    _draw_background(img, ground_y)

    n_wheels = 2 if rng.random() < P_TWO_WHEELS else 4
    tyre_radius = rng.randint(max(14, img_h // 22), max(20, img_h // 14))
    centers, body = _wheel_centers(rng, img_w, img_h, n_wheels, tyre_radius)

    _draw_car_body(img, *body)

    wheels: list[dict] = []
    for cx, cy in centers:
        wheels.append(_draw_wheel(img, cx, cy, tyre_radius))
    return img, wheels


def _ensure_clean_output_root(root: Path, overwrite: bool) -> int | None:
    """Apply the --overwrite rule. Returns an exit code on refusal, else None."""
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            print(f"ERROR: output root already exists and is not empty: {root}")
            print(
                "Pass --overwrite to delete and regenerate, "
                "or pick a different --output-root."
            )
            return 1
        shutil.rmtree(root)
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root: Path = args.output_root

    refusal = _ensure_clean_output_root(root, args.overwrite)
    if refusal is not None:
        return refusal

    images_dir = root / "images"
    annos_dir = root / "annotations"
    meta_dir = root / "metadata"
    for d in (images_dir, annos_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    total_wheels = 0

    for i in range(args.count):
        img, wheels = generate_one(rng, args.image_width, args.image_height)
        stem = f"sample_{i:04d}"
        img_name = f"{stem}.jpg"
        ok = cv2.imwrite(str(images_dir / img_name), img)
        if not ok:
            print(f"ERROR: failed to write image: {images_dir / img_name}")
            return 1

        annotation = {
            "frame_id": stem,
            "image": img_name,
            "wheels": wheels,
        }
        (annos_dir / f"{stem}.json").write_text(
            json.dumps(annotation, indent=2), encoding="utf-8"
        )
        total_wheels += len(wheels)

    source_info = {
        "source_name": "synthetic_keypoint_sample",
        "image_count": int(args.count),
        "image_width": int(args.image_width),
        "image_height": int(args.image_height),
        "seed": int(args.seed),
        "notes": (
            "Synthetic smoke-test batch for plugin format validation. "
            "A/B follow the 2026-05-14 floor-ray semantics (lower band of "
            "the wheel bbox near the footprint, not rim edges); "
            "c_disc_bottom is the lowest visible rim point. "
            "Not real training data."
        ),
    }
    (meta_dir / "source_info.json").write_text(
        json.dumps(source_info, indent=2), encoding="utf-8"
    )

    print(f"Output root:    {root}")
    print(f"Images:         {args.count} ({args.image_width}x{args.image_height})")
    print(f"Wheels (total): {total_wheels}")
    print(f"Source info:    {meta_dir / 'source_info.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
