"""Audit bbox quality for the imported Unreal/plugin export.

Context
-------
The bboxes in ``data/incoming/android_plugin_real/annotations/`` were NOT
supplied by the plugin. They are synthesized by
``scripts/import_unreal_export.py::build_bbox_from_points`` as the
axis-aligned hull of three keypoints (Right, Left, Center) inflated by a
constant margin (``80 px`` default), then clipped to image bounds.

That is fine for plumbing, but is **not** an object-derived bbox: a YOLO
wheel detector trained on this batch will learn to localise a point
cluster, not a wheel/tire object. This script makes that fact visible
and quantifies how far the synthesized bbox is from a plausible
full-wheel bbox.

Heuristic signals (no ground truth):
  - per-edge distance from each keypoint to the nearest bbox edge
    (point-derived bbox sits exactly at ``margin`` from at least two
    edges before clipping)
  - bbox dimensions / aspect ratio / area as a fraction of the image
  - point-spread vs bbox interior (the "tightness" ratio)
  - vertical position of ``c_disc_bottom`` inside the bbox — a true
    wheel bbox has its full rim above the disc-bottom point, a
    synthesized one does not.

Outputs::

    <out-dir>/report.json
    <out-dir>/report.md
    <out-dir>/contact_sheet.jpg
    <out-dir>/samples/<frame_id>__wheel<i>.jpg  # individual crops

Usage::

    python scripts/audit_unreal_bbox_quality.py \\
        --source-root data/incoming/android_plugin_real \\
        --out-dir outputs/unreal_bbox_audit \\
        --max-samples 30
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_MARGIN_PX = 80
# A synthesized-from-points bbox sits at exactly `margin` from at least
# two edges. We round-trip-detect this with a 1 px tolerance.
POINT_DERIVED_EDGE_TOL_PX = 1.5

# Empirical lower bound for a "full wheel" bbox in 2048x2048 frames.
# A passenger-car wheel in foreground covers ≥ ~300 px on the longer
# side; rear-wheel / distant cases drop below that. Anything under
# 200 px is almost certainly a point-cluster bbox, not a wheel.
SUSPICIOUS_MAX_SIDE_PX = 200
# Aspect-ratio band. Real wheel bboxes are roughly square (0.6..1.6).
# Synthesized bboxes from 3 keypoints can be extremely thin if A/B sit
# on the same y and C also sits low, producing a wide-flat strip.
ASPECT_LOW = 0.5
ASPECT_HIGH = 2.0

COLOR_BBOX = (0, 165, 255)  # BGR orange
COLOR_A = (0, 255, 0)
COLOR_B = (0, 255, 255)
COLOR_C = (0, 0, 255)


@dataclass
class WheelStats:
    frame_id: str
    image_path: Path
    wheel_idx: int
    bbox: tuple[float, float, float, float]
    points: dict[str, tuple[float, float]]
    image_size: tuple[int, int]

    # derived
    bbox_w: float = 0.0
    bbox_h: float = 0.0
    aspect: float = 0.0
    area_frac: float = 0.0
    long_side: float = 0.0
    tightness_x: float = 0.0
    tightness_y: float = 0.0
    edges_at_margin: int = 0
    c_rel_y: float = 0.0
    flags: list[str] = field(default_factory=list)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Heuristic audit of synthesized bbox quality."
    )
    p.add_argument("--source-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--max-samples", type=int, default=30)
    p.add_argument("--margin-px", type=int, default=DEFAULT_MARGIN_PX)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def compute_stats(
    wheel: dict,
    image_w: int,
    image_h: int,
    frame_id: str,
    image_path: Path,
    wheel_idx: int,
    margin_px: int,
) -> WheelStats | None:
    bbox = wheel.get("bbox_xyxy")
    pts_raw = wheel.get("points", {})
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return None
    x1, y1, x2, y2 = (float(v) for v in bbox)
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    pts = {
        k: (float(v[0]), float(v[1]))
        for k, v in pts_raw.items()
        if isinstance(v, list) and len(v) == 2
    }

    stats = WheelStats(
        frame_id=frame_id,
        image_path=image_path,
        wheel_idx=wheel_idx,
        bbox=(x1, y1, x2, y2),
        points=pts,
        image_size=(image_w, image_h),
        bbox_w=bw,
        bbox_h=bh,
        aspect=bw / bh,
        area_frac=(bw * bh) / (image_w * image_h),
        long_side=max(bw, bh),
    )

    xs = [p[0] for p in pts.values()] or [x1]
    ys = [p[1] for p in pts.values()] or [y1]
    spread_x = max(xs) - min(xs)
    spread_y = max(ys) - min(ys)
    stats.tightness_x = (bw - spread_x) / max(bw, 1.0)
    stats.tightness_y = (bh - spread_y) / max(bh, 1.0)

    # Point-at-margin detection. For an unclipped synthesized bbox the
    # min/max-x point sits at exactly `margin` from the left/right edge
    # respectively (and same for y). Clipped edges reduce the count.
    edges_hit = 0
    if pts:
        for ex_v, edge in (
            (min(xs) - x1, "left"),
            (x2 - max(xs), "right"),
            (min(ys) - y1, "top"),
            (y2 - max(ys), "bottom"),
        ):
            if abs(ex_v - margin_px) <= POINT_DERIVED_EDGE_TOL_PX:
                edges_hit += 1
    stats.edges_at_margin = edges_hit

    cy_disc = pts.get("c_disc_bottom", (None, None))[1]
    if cy_disc is not None and bh > 0:
        stats.c_rel_y = (cy_disc - y1) / bh

    # Flags
    if stats.long_side < SUSPICIOUS_MAX_SIDE_PX:
        stats.flags.append("small_bbox")
    if stats.aspect < ASPECT_LOW or stats.aspect > ASPECT_HIGH:
        stats.flags.append("extreme_aspect")
    if stats.edges_at_margin >= 2:
        stats.flags.append("point_derived_bbox")
    if 0.4 <= stats.c_rel_y <= 0.7:
        # c_disc_bottom sits in the middle band — for a real wheel this
        # would be unusual; the disc bottom typically sits near the
        # tyre-ground contact, i.e. close to bbox bottom.
        stats.flags.append("c_disc_in_middle")
    return stats


def render_thumb(
    stats: WheelStats,
    crop_size: int = 340,
    pad_px: int = 60,
) -> np.ndarray | None:
    img = cv2.imread(str(stats.image_path))
    if img is None:
        return None
    H, W = img.shape[:2]
    x1, y1, x2, y2 = stats.bbox
    # Pad the crop so the bbox doesn't touch the edges visually.
    cx1 = int(max(0, x1 - pad_px))
    cy1 = int(max(0, y1 - pad_px))
    cx2 = int(min(W, x2 + pad_px))
    cy2 = int(min(H, y2 + pad_px))
    if cx2 - cx1 < 4 or cy2 - cy1 < 4:
        return None
    crop = img[cy1:cy2, cx1:cx2].copy()
    if crop.size == 0:
        return None

    # Draw bbox + points in original coords, then resize.
    def _pt(v):
        return (int(round(v[0] - cx1)), int(round(v[1] - cy1)))

    cv2.rectangle(crop, _pt((x1, y1)), _pt((x2, y2)), COLOR_BBOX, 3)
    for key, color, label in (
        ("a", COLOR_A, "a"),
        ("b", COLOR_B, "b"),
        ("c_disc_bottom", COLOR_C, "c"),
    ):
        if key not in stats.points:
            continue
        px, py = _pt(stats.points[key])
        cv2.circle(crop, (px, py), 7, color, -1, cv2.LINE_AA)
        cv2.circle(crop, (px, py), 9, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(
            crop,
            label,
            (px + 8, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            crop,
            label,
            (px + 8, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            1,
            cv2.LINE_AA,
        )

    flag_txt = ",".join(stats.flags) or "ok"
    header = (
        f"{stats.frame_id}#{stats.wheel_idx} {int(stats.bbox_w)}x{int(stats.bbox_h)}"
    )
    sub = f"a/r={stats.aspect:.2f} c_rel_y={stats.c_rel_y:.2f}"
    for y, text in (
        (24, header),
        (48, sub),
        (72, flag_txt),
    ):
        cv2.putText(
            crop, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA
        )
        cv2.putText(
            crop,
            text,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    ch, cw = crop.shape[:2]
    scale = crop_size / max(ch, cw)
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((crop_size, crop_size, 3), 30, dtype=np.uint8)
    oy = (crop_size - nh) // 2
    ox = (crop_size - nw) // 2
    canvas[oy : oy + nh, ox : ox + nw] = resized
    return canvas


def build_contact_sheet(
    thumbs: list[np.ndarray],
    cols: int = 6,
    cell: int = 340,
    pad: int = 8,
) -> np.ndarray:
    if not thumbs:
        return np.full((cell, cell, 3), 40, dtype=np.uint8)
    rows = math.ceil(len(thumbs) / cols)
    H = rows * cell + (rows + 1) * pad
    W = cols * cell + (cols + 1) * pad
    sheet = np.full((H, W, 3), 20, dtype=np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        y0 = pad + r * (cell + pad)
        x0 = pad + c * (cell + pad)
        sheet[y0 : y0 + cell, x0 : x0 + cell] = t
    return sheet


def pick_samples(
    wheels: list[WheelStats],
    n: int,
    seed: int,
) -> list[WheelStats]:
    """Mix of flagged + unflagged, plus extreme cases."""
    rng = random.Random(seed)
    by_flag: dict[str, list[WheelStats]] = {}
    for w in wheels:
        for f in w.flags or ["ok"]:
            by_flag.setdefault(f, []).append(w)

    picks: list[WheelStats] = []
    seen: set[tuple[str, int]] = set()

    def _key(w: WheelStats) -> tuple[str, int]:
        return (w.frame_id, w.wheel_idx)

    # Front-load the categories the audit is meant to expose.
    priority = (
        "point_derived_bbox",
        "small_bbox",
        "extreme_aspect",
        "c_disc_in_middle",
        "ok",
    )
    # Allocate roughly evenly across categories present.
    present = [c for c in priority if by_flag.get(c)]
    if not present:
        return []
    per = max(1, n // len(present))

    for cat in present:
        pool = by_flag[cat][:]
        rng.shuffle(pool)
        for w in pool:
            if len(picks) >= n:
                break
            k = _key(w)
            if k in seen:
                continue
            picks.append(w)
            seen.add(k)
            if sum(1 for p in picks if cat in (p.flags or ["ok"])) >= per:
                break
        if len(picks) >= n:
            break

    # Fill any remaining slots from the global pool.
    if len(picks) < n:
        rest = [w for w in wheels if _key(w) not in seen]
        rng.shuffle(rest)
        picks.extend(rest[: n - len(picks)])
    return picks[:n]


def aggregate(wheels: Iterable[WheelStats]) -> dict:
    ws = list(wheels)
    n = len(ws)
    if n == 0:
        return {"n_wheels": 0}
    flag_counts: dict[str, int] = {}
    for w in ws:
        if not w.flags:
            flag_counts["ok"] = flag_counts.get("ok", 0) + 1
        for f in w.flags:
            flag_counts[f] = flag_counts.get(f, 0) + 1

    def _arr(key: str) -> np.ndarray:
        return np.asarray([getattr(w, key) for w in ws], dtype=float)

    def _summary(arr: np.ndarray) -> dict:
        return {
            "min": float(arr.min()),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)),
            "p75": float(np.percentile(arr, 75)),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
        }

    return {
        "n_wheels": n,
        "flag_counts": flag_counts,
        "bbox_long_side_px": _summary(_arr("long_side")),
        "bbox_w_px": _summary(_arr("bbox_w")),
        "bbox_h_px": _summary(_arr("bbox_h")),
        "aspect_w_over_h": _summary(_arr("aspect")),
        "area_fraction_of_image": _summary(_arr("area_frac")),
        "c_disc_bottom_relative_y": _summary(_arr("c_rel_y")),
        "edges_at_margin_count": _summary(_arr("edges_at_margin")),
        "n_with_two_or_more_edges_at_margin": int(
            sum(1 for w in ws if w.edges_at_margin >= 2)
        ),
        "fraction_with_two_or_more_edges_at_margin": float(
            sum(1 for w in ws if w.edges_at_margin >= 2) / n
        ),
    }


def write_report_md(
    path: Path, summary: dict, picks: list[WheelStats], source_root: Path
) -> None:
    L = []
    L += [
        "# Unreal export — bbox quality audit",
        "",
        f"Source: `{source_root}`",
        "",
        "## bbox provenance",
        "",
        "**Synthesized by the adapter, not exported by the plugin.**",
        "",
        "The bboxes in `annotations/` were built by "
        "`scripts/import_unreal_export.py::build_bbox_from_points` "
        "(lines 107-136) as the axis-aligned hull of the three keypoints "
        "(Right, Left, Center) inflated by `margin = 80 px`, then clipped "
        "to image bounds:",
        "",
        "```python",
        "xs = [a[0], b[0], c[0]]",
        "ys = [a[1], b[1], c[1]]",
        "x1 = max(0.0, min(xs) - margin)",
        "y1 = max(0.0, min(ys) - margin)",
        "x2 = min(image_w - 1, max(xs) + margin)",
        "y2 = min(image_h - 1, max(ys) + margin)",
        "```",
        "",
        "## Why this is risky for training",
        "",
        "A YOLO bbox is supposed to enclose the *object* (the wheel: "
        "rim + tire). The bbox we have encloses **three points on/under "
        "the wheel**, plus a fixed 80 px margin. Consequences:",
        "",
        "- The bbox systematically under-covers the tire (it has no idea "
        "where the tire edge is).",
        "- The top of the rim is usually **above** the keypoint cluster, "
        "so the bbox crops the top of the wheel.",
        "- Aspect ratio is dictated by point geometry, not by wheel "
        "geometry — wide, flat, or square depending on the camera pose.",
        "- A detector trained on this will learn to localise a "
        "**keypoint hull**, not a wheel. At inference time it will "
        "produce bboxes that are too small and biased low.",
        "",
        "## Audit stats",
        "",
        f"- Wheels audited: **{summary['n_wheels']}**",
        f"- Wheels with ≥ 2 keypoints sitting at exactly `margin` from "
        f"the bbox edge (definitive point-derived signal): "
        f"**{summary['n_with_two_or_more_edges_at_margin']} "
        f"({summary['fraction_with_two_or_more_edges_at_margin'] * 100:.1f}%)**",
        "",
        "### Flag counts",
        "",
    ]
    for k, v in sorted(summary["flag_counts"].items(), key=lambda kv: -kv[1]):
        L.append(f"- `{k}`: {v}")
    L.append("")

    def _block(title: str, key: str, unit: str = "") -> None:
        s = summary[key]
        L.append(f"### {title}")
        L.append("")
        L.append("| stat | value |")
        L.append("|---|---|")
        for tag in ("min", "p25", "median", "mean", "p75", "max"):
            L.append(f"| {tag} | {s[tag]:.3f}{unit} |")
        L.append("")

    _block("bbox long side (px)", "bbox_long_side_px", " px")
    _block("bbox width (px)", "bbox_w_px", " px")
    _block("bbox height (px)", "bbox_h_px", " px")
    _block("aspect ratio (w/h)", "aspect_w_over_h")
    _block("area as fraction of image", "area_fraction_of_image")
    _block("c_disc_bottom relative y inside bbox", "c_disc_bottom_relative_y")
    _block("edges sitting at exactly margin", "edges_at_margin_count")

    L += [
        "## Examples where bbox almost certainly misses the full wheel",
        "",
    ]
    notable = [
        w
        for w in picks
        if "point_derived_bbox" in w.flags
        or "small_bbox" in w.flags
        or "extreme_aspect" in w.flags
    ][:10]
    for w in notable:
        L.append(
            f"- `{w.frame_id}` wheel[{w.wheel_idx}] — "
            f"{int(w.bbox_w)}x{int(w.bbox_h)}, aspect={w.aspect:.2f}, "
            f"edges_at_margin={w.edges_at_margin}, "
            f"c_rel_y={w.c_rel_y:.2f}, flags={w.flags}"
        )
    L.append("")

    L += [
        "## Recommendation",
        "",
        "**ACCEPT_ONLY_AS_DEBUG.**",
        "",
        "Do not train production weights on this batch in its current "
        "form. Allowed uses:",
        "",
        "- plumbing / pipeline smoke tests;",
        "- keypoint-only loss ablations where bbox is not the training   signal;",
        "- visual review.",
        "",
        "`ACCEPT_FOR_TRAINING` requires either (a) human confirmation "
        "per-frame that the synthesized bbox does in fact cover the full "
        "wheel, or (b) a real wheel bbox exported by the plugin.",
        "",
        "## Exact request to the plugin author",
        "",
        "Please export an object bbox per wheel — the same axis-aligned "
        "bounding rectangle an annotator would draw around the entire "
        "wheel silhouette (rim + tire), in image pixel coordinates of "
        "the exported JPEG. Format suggestion::",
        "",
        "```",
        "keyPoint/<frame_id>/<object_id>.txt",
        "  ...Right/Left/Center as today...",
        '  {name:"BBox",XYXY:x1,y1,x2,y2}',
        "```",
        "",
        "Notes for the plugin side:",
        "",
        "- Include the **tire**, not just the rim. The detector needs the "
        "  full wheel silhouette as the training target.",
        "- One bbox per wheel object, axis-aligned.",
        "- Coordinates in the same pixel space as Right/Left/Center "
        "  (0..2047 for 2048×2048 frames).",
        "- For wheels that are fully outside the frame or fully occluded, "
        "  omit the whole object rather than emitting a partial bbox "
        "  (per the existing drop policy for keypoints).",
        "- Until this lands, the dataset stays `ACCEPT_ONLY_AS_DEBUG`.",
        "",
        "## See also",
        "",
        "- `report.json` — full per-wheel measurements and flags.",
        "- `contact_sheet.jpg` — 30 representative samples (bbox + a/b/c).",
        "- `samples/` — individual crops for the same 30 wheels.",
        "",
    ]
    path.write_text("\n".join(L), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    src = args.source_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    images_dir = src / "images"
    annos_dir = src / "annotations"
    if not images_dir.is_dir() or not annos_dir.is_dir():
        print(f"ERROR: missing images/ or annotations/ under {src}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    all_wheels: list[WheelStats] = []
    for img_path in images:
        anno_path = annos_dir / f"{img_path.stem}.json"
        if not anno_path.is_file():
            continue
        try:
            payload = json.loads(anno_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        wheels = payload.get("wheels", [])
        if not isinstance(wheels, list):
            continue
        for i, w in enumerate(wheels):
            stats = compute_stats(
                w,
                iw,
                ih,
                frame_id=str(payload.get("frame_id", img_path.stem)),
                image_path=img_path,
                wheel_idx=i,
                margin_px=args.margin_px,
            )
            if stats is not None:
                all_wheels.append(stats)

    summary = aggregate(all_wheels)

    picks = pick_samples(all_wheels, args.max_samples, args.seed)
    thumbs: list[np.ndarray] = []
    for w in picks:
        t = render_thumb(w)
        if t is None:
            continue
        thumbs.append(t)
        sample_path = out_dir / "samples" / f"{w.frame_id}__wheel{w.wheel_idx}.jpg"
        cv2.imwrite(str(sample_path), t)

    contact = build_contact_sheet(thumbs)
    cv2.imwrite(str(out_dir / "contact_sheet.jpg"), contact)

    report = {
        "source_root": str(src),
        "bbox_provenance": {
            "source": "synthesized_by_adapter",
            "adapter": "scripts/import_unreal_export.py",
            "function": "build_bbox_from_points",
            "lines": "107-136",
            "margin_px": args.margin_px,
            "covers_full_wheel": False,
        },
        "summary": summary,
        "samples": [
            {
                "frame_id": w.frame_id,
                "wheel_idx": w.wheel_idx,
                "bbox_xyxy": list(w.bbox),
                "bbox_w": w.bbox_w,
                "bbox_h": w.bbox_h,
                "aspect": w.aspect,
                "area_frac": w.area_frac,
                "edges_at_margin": w.edges_at_margin,
                "c_disc_bottom_rel_y": w.c_rel_y,
                "flags": w.flags,
                "image": str(w.image_path),
                "points": {k: list(v) for k, v in w.points.items()},
            }
            for w in picks
        ],
        "recommendation": "ACCEPT_ONLY_AS_DEBUG",
        "training_allowed": False,
        "requires_plugin_bbox": True,
        "requires_human_preview": True,
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report_md(out_dir / "report.md", summary, picks, src)

    print(f"Audited wheels: {summary['n_wheels']}")
    print(
        f"Point-derived bbox detected: "
        f"{summary['n_with_two_or_more_edges_at_margin']} "
        f"({summary['fraction_with_two_or_more_edges_at_margin'] * 100:.1f}%)"
    )
    print(f"Outputs in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
