"""Generate a synthetic incoming batch for testing the ML pipeline.

Writes annotations in the interim JSON format documented in
docs/ANNOTATION_JSON_FORMAT.md: one `wheel` object per wheel, each with a
bbox and 3 keypoints whose literal label strings are still
(`rim_left`, `rim_right`, `disc_bottom`) for backward compatibility with
the legacy converter and `postprocess_wheels.KEYPOINT_NAMES`.

    create_sample_incoming.py -> convert_incoming_to_yolo.py -> check_dataset.py

Per the 2026-05-14 contract revision the *content* of `rim_left` and
`rim_right` has shifted: they are now floor / raycast points near the
wheel footprint, NOT left/right edges of the metal rim. `disc_bottom`
keeps its meaning — the lowest visible point of the metal rim. The
literal strings remain because the legacy converter, training labels,
and `postprocess_wheels.KEYPOINT_NAMES` index by these names; only the
geometric semantics drift.

This is NOT real training data. Cartoon cars only — purpose is to validate
the ingestion pipeline AND give the YOLO-pose head a non-degenerate
keypoint distribution. Earlier versions placed all wheels in a frame
straight-on (A.y == B.y), which trains the model into a shortcut. The
current version randomizes:

  - per-image camera yaw (car rotated around vertical → ellipse aspect)
  - per-image camera tilt (whole-image roll → ellipse rotated)
  - 1 or 2 cars per image (4 / 6 / 8 wheels depending on geometry)
  - rim style (solid disc / ring / spokes)
  - edge clipping → some keypoints with visibility=0 or 1

Usage:
    python src/create_sample_incoming.py --count 20 --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a mock incoming source batch")
    p.add_argument(
        "--output-root", type=Path, default=Path("data/incoming/manual_sample")
    )
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--image-width", type=int, default=640)
    p.add_argument("--image-height", type=int, default=480)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, delete existing non-empty output root before writing.",
    )
    return p.parse_args()


def ellipse_extremes(
    a: float, b: float, alpha: float
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Leftmost, rightmost, bottommost points of an ellipse with semi-axes
    (a, b) rotated by `alpha` radians, returned as offsets from the
    ellipse center in image coords (+y down).

    Parametric form (matching OpenCV's CW rotation under +y-down):
        x(t) = a·cos(t)·cos(α) - b·sin(t)·sin(α)
        y(t) = a·cos(t)·sin(α) + b·sin(t)·cos(α)

    x-extreme: solve ∂x/∂t = 0 → tan(t) = -b·sin(α) / (a·cos(α)).
    y-extreme: solve ∂y/∂t = 0 → tan(t) =  b·cos(α) / (a·sin(α)).
    """
    ca, sa = math.cos(alpha), math.sin(alpha)

    t_x = math.atan2(-b * sa, a * ca)
    x_e = a * math.cos(t_x) * ca - b * math.sin(t_x) * sa
    y_at_x = a * math.cos(t_x) * sa + b * math.sin(t_x) * ca
    if x_e >= 0:
        rightmost = (x_e, y_at_x)
        leftmost = (-x_e, -y_at_x)
    else:
        leftmost = (x_e, y_at_x)
        rightmost = (-x_e, -y_at_x)

    t_y = math.atan2(b * ca, a * sa)
    x_at_y = a * math.cos(t_y) * ca - b * math.sin(t_y) * sa
    y_e = a * math.cos(t_y) * sa + b * math.sin(t_y) * ca
    bottom = (x_at_y, y_e) if y_e >= 0 else (-x_at_y, -y_e)

    return leftmost, rightmost, bottom


def ellipse_bbox_half(a: float, b: float, alpha: float) -> tuple[float, float]:
    """Tight half-extent (dx, dy) of an axis-aligned bbox around a rotated ellipse."""
    ca2, sa2 = math.cos(alpha) ** 2, math.sin(alpha) ** 2
    return (
        math.sqrt(a * a * ca2 + b * b * sa2),
        math.sqrt(a * a * sa2 + b * b * ca2),
    )


def random_scene(rng: random.Random, w: int, h: int) -> np.ndarray:
    """Cartoon ground+sky background with small intensity variance."""
    sky = (rng.randint(180, 230), rng.randint(190, 235), rng.randint(200, 245))
    ground = (rng.randint(60, 130), rng.randint(60, 130), rng.randint(60, 130))
    img = np.empty((h, w, 3), dtype=np.uint8)
    horizon = rng.randint(int(h * 0.45), int(h * 0.70))
    img[:horizon, :] = sky
    img[horizon:, :] = ground
    # Coarse intensity noise to avoid solid-color shortcut
    noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).integers(
        -8, 8, size=(h, w, 1), dtype=np.int16
    )
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def draw_car_body(
    img: np.ndarray,
    rng: random.Random,
    body_x1: int,
    body_y1: int,
    body_x2: int,
    body_y2: int,
) -> tuple[int, int, int, int]:
    """Draw a cartoon car body (chassis + cabin). Returns the cabin bbox so wheels
    are placed below the chassis."""
    body_color = (
        rng.randint(40, 220),
        rng.randint(40, 220),
        rng.randint(40, 220),
    )
    cv2.rectangle(img, (body_x1, body_y1), (body_x2, body_y2), body_color, -1)

    cabin_h = rng.randint(20, 55)
    cabin_top = max(0, body_y1 - cabin_h)
    pad_l = rng.randint(20, max(21, (body_x2 - body_x1) // 4))
    pad_r = rng.randint(20, max(21, (body_x2 - body_x1) // 4))
    cabin = np.array(
        [
            [body_x1 + pad_l, body_y1],
            [body_x2 - pad_r, body_y1],
            [body_x2 - pad_r - rng.randint(0, 20), cabin_top],
            [body_x1 + pad_l + rng.randint(0, 20), cabin_top],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(img, [cabin], body_color)
    return body_x1, body_y1, body_x2, body_y2


def draw_rim(
    img: np.ndarray,
    rng: random.Random,
    center: tuple[int, int],
    rim_a: float,
    rim_b: float,
    alpha_deg: float,
    style: str,
) -> None:
    """Render a rim with one of three styles to vary visual features."""
    rim_color = (
        rng.randint(150, 230),
        rng.randint(150, 230),
        rng.randint(150, 230),
    )
    hub_color = (
        max(20, rim_color[0] - rng.randint(40, 90)),
        max(20, rim_color[1] - rng.randint(40, 90)),
        max(20, rim_color[2] - rng.randint(40, 90)),
    )
    # Outer rim ellipse
    cv2.ellipse(
        img,
        center,
        (max(2, int(rim_a)), max(2, int(rim_b))),
        alpha_deg,
        0,
        360,
        rim_color,
        -1,
        lineType=cv2.LINE_AA,
    )
    if style == "solid":
        # No further detail
        pass
    elif style == "ring":
        inner_a = max(2, int(rim_a * 0.55))
        inner_b = max(2, int(rim_b * 0.55))
        cv2.ellipse(
            img,
            center,
            (inner_a, inner_b),
            alpha_deg,
            0,
            360,
            hub_color,
            -1,
            lineType=cv2.LINE_AA,
        )
    elif style == "spokes":
        # 4 spokes through center, rotated with the rim
        ca, sa = math.cos(math.radians(alpha_deg)), math.sin(math.radians(alpha_deg))
        for spoke_deg in (0, 45, 90, 135):
            rad = math.radians(spoke_deg)
            # vector in rim-local frame (along major axis)
            vx = rim_a * math.cos(rad)
            vy = rim_b * math.sin(rad)
            # rotate by alpha into image frame
            ex = vx * ca - vy * sa
            ey = vx * sa + vy * ca
            p1 = (int(center[0] - ex), int(center[1] - ey))
            p2 = (int(center[0] + ex), int(center[1] + ey))
            cv2.line(img, p1, p2, hub_color, 2, lineType=cv2.LINE_AA)
        # Small hub at center
        cv2.circle(img, center, max(2, int(min(rim_a, rim_b) * 0.18)), hub_color, -1)


def draw_wheel(
    img: np.ndarray,
    rng: random.Random,
    cx: int,
    cy: int,
    wheel_r: int,
    yaw_rad: float,
    tilt_rad: float,
    rim_style: str,
) -> tuple[
    tuple[int, int, int, int],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    """Render one wheel as a tire ellipse + rim ellipse. Returns
    (bbox_xyxy, point_a, point_b, disc_bottom) in image coordinates.

    The two middle return values are the floor / raycast points A and B
    (still serialised under the legacy label strings `rim_left` and
    `rim_right` — the *names* drifted on 2026-05-14, not the index
    order). They sit in the lower band of the tight tire bbox, close to
    the wheel's ground footprint. `disc_bottom` keeps its old meaning:
    the lowest visible point of the rim ellipse.

    `yaw_rad` is the wheel-plane rotation around vertical (0 = facing camera).
    `tilt_rad` is an in-image rotation of the projected ellipse (camera roll).
    """
    cos_yaw = max(0.18, math.cos(yaw_rad))  # avoid degenerate edge-on wheel

    tire_a = float(wheel_r)
    tire_b = float(wheel_r) * cos_yaw
    rim_scale = rng.uniform(0.55, 0.72)
    rim_a = tire_a * rim_scale
    rim_b = tire_b * rim_scale
    alpha_deg = math.degrees(tilt_rad)

    # Tire body (dark annulus drawn as filled outer ellipse; rim drawn on top)
    cv2.ellipse(
        img,
        (cx, cy),
        (int(tire_a), max(2, int(tire_b))),
        alpha_deg,
        0,
        360,
        (rng.randint(15, 40), rng.randint(15, 40), rng.randint(15, 40)),
        -1,
        lineType=cv2.LINE_AA,
    )

    draw_rim(img, rng, (cx, cy), rim_a, rim_b, alpha_deg, rim_style)

    # Disc-bottom stays on the metal rim — bottommost point of the rim ellipse.
    _, _, disc_bottom_offset = ellipse_extremes(rim_a, rim_b, tilt_rad)
    disc_bottom = (cx + disc_bottom_offset[0], cy + disc_bottom_offset[1])

    # Tight bbox on the TIRE (full wheel visible from outside).
    half_w, half_h = ellipse_bbox_half(tire_a, tire_b, tilt_rad)

    # A / B are floor / raycast points near the wheel footprint. Place
    # them in the lower band of the tight tire bbox so they stay inside
    # the bbox the converter writes out, sit clearly below the rim
    # centerline, and remain below the disc-bottom approximation.
    a_b_dx = 0.70 * half_w
    a_b_dy = 0.88 * half_h
    point_a = (cx - a_b_dx, cy + a_b_dy)
    point_b = (cx + a_b_dx, cy + a_b_dy)

    bbox = (
        int(round(cx - half_w)),
        int(round(cy - half_h)),
        int(round(cx + half_w)),
        int(round(cy + half_h)),
    )
    return bbox, point_a, point_b, disc_bottom


def clip_visibility(
    pt: tuple[float, float], img_w: int, img_h: int
) -> tuple[tuple[float, float], int]:
    """Map a keypoint to (xy, visibility). visibility=2 if inside the image,
    0 if outside. (We do not synthesise visibility=1 — that would require a
    real occlusion model.)"""
    x, y = pt
    if 0 <= x < img_w and 0 <= y < img_h:
        return (x, y), 2
    return (0.0, 0.0), 0


def wheel_visible_fraction(
    bbox: tuple[int, int, int, int], img_w: int, img_h: int
) -> float:
    """Fraction of the bbox area that lies inside the image."""
    x1, y1, x2, y2 = bbox
    inter_x1 = max(0, x1)
    inter_y1 = max(0, y1)
    inter_x2 = min(img_w, x2)
    inter_y2 = min(img_h, y2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    total = max(1, (x2 - x1) * (y2 - y1))
    return (iw * ih) / total


def place_car(
    img: np.ndarray,
    rng: random.Random,
    img_w: int,
    img_h: int,
    yaw_rad: float,
    tilt_rad: float,
    wheels_target: int,
    x_offset: int,
    body_y_band: tuple[int, int],
    body_w_band: tuple[int, int],
) -> list[dict]:
    """Place one car at horizontal offset `x_offset` with `wheels_target`
    wheels and append valid wheel objects to a list.

    Wheels that fall <50% inside the image are skipped entirely (matching the
    annotation-guideline rule). Keypoints fully off-image are emitted with
    visibility=0 and xy=(0,0).
    """
    body_w = rng.randint(*body_w_band)
    body_h = rng.randint(int(img_h * 0.18), int(img_h * 0.28))
    body_x1 = x_offset
    body_y1 = rng.randint(*body_y_band)
    body_x2 = body_x1 + body_w
    body_y2 = body_y1 + body_h
    draw_car_body(img, rng, body_x1, body_y1, body_x2, body_y2)

    wheel_r = rng.randint(20, 38)
    wheel_cy = body_y2 + int(wheel_r * 0.35)
    rim_style = rng.choice(["solid", "ring", "spokes"])

    margin = wheel_r + 4
    xs_start = body_x1 + margin
    xs_end = body_x2 - margin
    if xs_end <= xs_start:
        return []

    if wheels_target <= 2:
        wheel_xs = [xs_start, xs_end]
    else:
        step = (xs_end - xs_start) / (wheels_target - 1)
        wheel_xs = [int(xs_start + step * i) for i in range(wheels_target)]

    objects: list[dict] = []
    for cx in wheel_xs:
        # point_a / point_b carry the floor-ray semantics; the JSON below
        # keeps the legacy literal label strings rim_left/rim_right for
        # downstream compatibility (label indices, not names, are the
        # load-bearing contract).
        bbox, point_a, point_b, disc_bottom = draw_wheel(
            img, rng, cx, wheel_cy, wheel_r, yaw_rad, tilt_rad, rim_style
        )
        if wheel_visible_fraction(bbox, img_w, img_h) < 0.5:
            continue
        # Clip bbox to image bounds
        clipped = (
            max(0, bbox[0]),
            max(0, bbox[1]),
            min(img_w - 1, bbox[2]),
            min(img_h - 1, bbox[3]),
        )
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            continue
        (ka_xy, ka_v) = clip_visibility(point_a, img_w, img_h)
        (kb_xy, kb_v) = clip_visibility(point_b, img_w, img_h)
        (kc_xy, kc_v) = clip_visibility(disc_bottom, img_w, img_h)
        objects.append(
            {
                "class_name": "wheel",
                "bbox_xyxy": list(clipped),
                "keypoints": [
                    {
                        "name": "rim_left",
                        "xy": [round(ka_xy[0], 2), round(ka_xy[1], 2)],
                        "visibility": ka_v,
                    },
                    {
                        "name": "rim_right",
                        "xy": [round(kb_xy[0], 2), round(kb_xy[1], 2)],
                        "visibility": kb_v,
                    },
                    {
                        "name": "disc_bottom",
                        "xy": [round(kc_xy[0], 2), round(kc_xy[1], 2)],
                        "visibility": kc_v,
                    },
                ],
            }
        )
    return objects


def generate_one(
    rng: random.Random, img_w: int, img_h: int
) -> tuple[np.ndarray, list[dict]]:
    """Generate one synthetic image with 1-2 cars and 2..8 wheels total."""
    img = random_scene(rng, img_w, img_h)

    # Per-image camera params shared by all wheels in the frame.
    yaw_rad = math.radians(rng.uniform(0.0, 50.0))
    tilt_rad = math.radians(rng.uniform(-12.0, 12.0))

    objects: list[dict] = []

    primary_w_band = (int(img_w * 0.45), int(img_w * 0.70))
    primary_y_band = (int(img_h * 0.42), int(img_h * 0.55))
    primary_w = rng.randint(*primary_w_band)
    primary_x = rng.randint(20, max(21, img_w - primary_w - 20))
    primary_wheels = rng.choice([2, 4])
    objects.extend(
        place_car(
            img,
            rng,
            img_w,
            img_h,
            yaw_rad,
            tilt_rad,
            primary_wheels,
            primary_x,
            primary_y_band,
            primary_w_band,
        )
    )

    # 30% chance of a second car partially in frame — exercises multi-instance
    # detection and produces some clipped wheels.
    if rng.random() < 0.30:
        sec_w_band = (int(img_w * 0.35), int(img_w * 0.55))
        sec_w = rng.randint(*sec_w_band)
        # Place the second car so it might overlap or fall partly off-screen
        sec_x = rng.choice(
            [-rng.randint(20, sec_w // 2), img_w - rng.randint(sec_w // 2, sec_w + 20)]
        )
        sec_y_band = (int(img_h * 0.40), int(img_h * 0.58))
        sec_wheels = rng.choice([2, 4])
        objects.extend(
            place_car(
                img,
                rng,
                img_w,
                img_h,
                yaw_rad,
                tilt_rad,
                sec_wheels,
                sec_x,
                sec_y_band,
                sec_w_band,
            )
        )

    return img, objects


def main() -> int:
    args = parse_args()
    root: Path = args.output_root

    if root.exists() and any(root.iterdir()):
        if not args.overwrite:
            print(f"ERROR: output root already exists and is not empty: {root}")
            print(
                "Pass --overwrite to delete and regenerate, or pick a different --output-root."
            )
            return 1
        shutil.rmtree(root)

    images_dir = root / "images"
    annos_dir = root / "annotations"
    meta_dir = root / "metadata"
    for d in (images_dir, annos_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    total_objects = 0
    per_class: dict[str, int] = {"wheel": 0}
    per_visibility_kp = {0: 0, 1: 0, 2: 0}

    for i in range(args.count):
        img, objects = generate_one(rng, args.image_width, args.image_height)
        stem = f"sample_{i:04d}"
        img_name = f"{stem}.jpg"
        cv2.imwrite(str(images_dir / img_name), img)

        annotation = {"image": img_name, "objects": objects}
        (annos_dir / f"{stem}.json").write_text(
            json.dumps(annotation, indent=2), encoding="utf-8"
        )
        total_objects += len(objects)
        for o in objects:
            per_class[o["class_name"]] = per_class.get(o["class_name"], 0) + 1
            for kp in o["keypoints"]:
                per_visibility_kp[kp["visibility"]] = (
                    per_visibility_kp.get(kp["visibility"], 0) + 1
                )

    source_info = {
        "source_name": root.name,
        "count": args.count,
        "image_size": [args.image_width, args.image_height],
        "classes": ["wheel"],
        "keypoint_names": ["rim_left", "rim_right", "disc_bottom"],
        "seed": args.seed,
        "total_objects": total_objects,
        "object_counts_by_class": per_class,
        "keypoint_visibility_counts": per_visibility_kp,
        "note": (
            "Synthetic incoming sample for pipeline testing. Camera yaw + tilt "
            "randomized per image, 1-2 cars per image, rim styles vary. "
            "Per the 2026-05-14 contract revision, label strings rim_left / "
            "rim_right are A / B floor-ray points near the wheel footprint "
            "(lower band of the tight tyre bbox), not rim edges. disc_bottom "
            "is the bottommost point of the rim ellipse. Label *strings* "
            "remain for backward compatibility with the legacy converter and "
            "postprocess_wheels.KEYPOINT_NAMES; only the geometric meaning "
            "has shifted. Not real training data — ingestion pipeline + "
            "augmentation validation only."
        ),
    }
    (meta_dir / "source_info.json").write_text(
        json.dumps(source_info, indent=2), encoding="utf-8"
    )

    print()
    print(f"Output root:        {root}")
    print(f"Images:             {args.count} ({args.image_width}x{args.image_height})")
    print(f"Annotations:        {args.count} JSON files")
    print(f"Total objects:      {total_objects}")
    print(f"Per class:          {per_class}")
    print(f"Keypoint vis. dist: {per_visibility_kp}")
    print(f"Source info:        {meta_dir / 'source_info.json'}")
    print()
    print("NOTE: synthetic for pipeline testing — not real training data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
