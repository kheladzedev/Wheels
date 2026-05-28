"""Convert NeuralData UE-output into a VSBL plugin-format incoming batch.

Bridges the legacy UE renderer (which writes per-frame
``Images/<N>.jpg`` + ``keyPoint/<N>/<0..5>.txt`` with five named rim
keypoints per wheel) to the plugin contract documented in
``docs/KEYPOINT_DATASET_FORMAT.md`` (``frame_id`` + ``image`` +
``wheels[].bbox_xyxy`` + ``wheels[].points.{a, b, c_disc_bottom}``).

Five-to-three mapping (heuristic, ``_needs_review`` always set):

    UE name        → VSBL semantic         note
    ──────────────────────────────────────────────────────────────────
    Left           → a (floor-ray)         drift: UE.Left is on the
                                            rim edge, AR.A wants the
                                            floor projection below.
    Right          → b (floor-ray)         same drift, mirrored.
    Center/.../    → c_disc_bottom         heuristic: pick the point
      Right/Left      among Right/Left/Center with the largest Y
                      (closest to the bottom of the image).
    LeftTop, RightTop  unused              upper rim edge — useful for
                                            recovering the wheel plane
                                            in 3D but not part of the
                                            AR contract.

Because A/B from UE rim are NOT the AR floor-ray points specified on
2026-05-14, every annotation is flagged ``_draft: true`` /
``_needs_review: true``. The bundle is a smoke-test for the end-to-end
ingest → preview → convert → YOLO-pose chain, not a training target.
Production-grade UE synthesis requires rewriting the UE Blueprint to
output floor projections directly.

Output bundle is plugin-format and is read transparently by
``check_keypoint_incoming.py`` and
``convert_keypoint_incoming_to_yolo_pose.py``.

Usage:
    python src/convert_ue_to_keypoint_incoming.py \
        --ue-root "/Users/codefactory/Downloads/NeuralData1 2" \
        --output-root data/incoming/ue_synthetic \
        --source-name ue_synthetic_v1 \
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2

KEYPOINT_NAMES: tuple[str, ...] = ("Right", "Left", "Center", "LeftTop", "RightTop")
WHEELS_PER_FRAME = 6
RIM_TO_TIRE_FACTOR = 1.18
MIN_BBOX_SIDE_PX = 5.0
KP_LINE_RE = re.compile(r'name:"(\w+)",XY:([\d\-\.eE+]+),([\d\-\.eE+]+)')
GOAL_RE = re.compile(
    r"DeltaZ\{([\d\-\.]+)\},Roll:([\d\-\.]+),Pitch:([\d\-\.]+),FOV:([\d\-\.]+)"
)


@dataclass
class ConversionStats:
    frames_total: int = 0
    frames_written: int = 0
    frames_skipped: int = 0
    wheels_kept: int = 0
    wheels_dropped: int = 0
    drop_reasons: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.drop_reasons is None:
            self.drop_reasons = {}

    def bump_drop(self, reason: str) -> None:
        assert self.drop_reasons is not None
        self.drop_reasons[reason] = self.drop_reasons.get(reason, 0) + 1


def parse_keypoint_file(path: Path) -> dict[str, tuple[float, float]] | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    pts: dict[str, tuple[float, float]] = {}
    for match in KP_LINE_RE.finditer(text):
        name = match.group(1)
        x = float(match.group(2))
        y = float(match.group(3))
        pts[name] = (x, y)
    if not pts:
        return None
    if all(x == 0.0 and y == 0.0 for x, y in pts.values()):
        return None
    return pts


def parse_goal_file(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    match = GOAL_RE.search(text)
    if not match:
        return None
    return {
        "delta_z": float(match.group(1)),
        "roll_deg": float(match.group(2)),
        "pitch_deg": float(match.group(3)),
        "fov_deg": float(match.group(4)),
    }


def _envelope_bbox(
    pts: dict[str, tuple[float, float]], img_w: int, img_h: int
) -> tuple[float, float, float, float] | None:
    xs = [p[0] for p in pts.values()]
    ys = [p[1] for p in pts.values()]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    rim_w = max(xs) - min(xs)
    rim_h = max(ys) - min(ys)
    half_w = rim_w * RIM_TO_TIRE_FACTOR / 2
    half_h = rim_h * RIM_TO_TIRE_FACTOR / 2
    x1 = max(0.0, cx - half_w)
    y1 = max(0.0, cy - half_h)
    x2 = min(img_w - 1.0, cx + half_w)
    y2 = min(img_h - 1.0, cy + half_h)
    if (x2 - x1) < MIN_BBOX_SIDE_PX or (y2 - y1) < MIN_BBOX_SIDE_PX:
        return None
    return x1, y1, x2, y2


def build_wheel(
    pts: dict[str, tuple[float, float]], img_w: int, img_h: int
) -> tuple[dict | None, str | None]:
    bbox = _envelope_bbox(pts, img_w, img_h)
    if bbox is None:
        return None, "bbox_too_small"
    a = pts.get("Left")
    b = pts.get("Right")
    if a is None or b is None:
        return None, "missing_left_or_right"
    bottom_candidates = [pts.get(name) for name in ("Right", "Left", "Center")]
    bottom_candidates = [p for p in bottom_candidates if p is not None]
    if not bottom_candidates:
        return None, "missing_bottom_candidates"
    c = max(bottom_candidates, key=lambda p: p[1])
    inside = lambda p: 0.0 <= p[0] < img_w and 0.0 <= p[1] < img_h  # noqa: E731
    if not (inside(a) and inside(b) and inside(c)):
        return None, "point_outside_image"
    wheel = {
        "bbox_xyxy": [round(v, 3) for v in bbox],
        "points": {
            "a": [round(a[0], 3), round(a[1], 3)],
            "b": [round(b[0], 3), round(b[1], 3)],
            "c_disc_bottom": [round(c[0], 3), round(c[1], 3)],
        },
        "_needs_review": True,
        "_review_reasons": ["ue_rim_to_floor_heuristic"],
    }
    return wheel, None


def iter_frame_ids(images_dir: Path, *, limit: int | None) -> list[int]:
    ids: list[int] = []
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            ids.append(int(path.stem))
        except ValueError:
            continue
    ids.sort()
    if limit is not None:
        ids = ids[:limit]
    return ids


def convert_frame(
    ue_root: Path,
    frame_int: int,
    out_root: Path,
    source_name: str,
    stats: ConversionStats,
) -> bool:
    img_path = ue_root / "Images" / f"{frame_int}.jpg"
    kp_dir = ue_root / "keyPoint" / str(frame_int)
    goal_path = ue_root / "Goal" / f"{frame_int}.txt"

    if not img_path.is_file():
        stats.bump_drop("image_missing")
        return False
    if not kp_dir.is_dir():
        stats.bump_drop("keypoints_missing")
        return False
    img = cv2.imread(str(img_path))
    if img is None:
        stats.bump_drop("image_unreadable")
        return False
    img_h, img_w = img.shape[:2]

    wheels: list[dict] = []
    for i in range(WHEELS_PER_FRAME):
        kp_file = kp_dir / f"{i}.txt"
        if not kp_file.is_file():
            continue
        pts = parse_keypoint_file(kp_file)
        if pts is None:
            continue
        wheel, drop_reason = build_wheel(pts, img_w, img_h)
        if wheel is None:
            stats.wheels_dropped += 1
            stats.bump_drop(drop_reason or "unknown")
            continue
        wheels.append(wheel)
        stats.wheels_kept += 1

    if not wheels:
        stats.bump_drop("no_wheels_in_frame")
        return False

    frame_id = f"{source_name}__{frame_int:06d}"
    dst_image = out_root / "images" / f"{frame_id}.jpg"
    dst_annot = out_root / "annotations" / f"{frame_id}.json"
    shutil.copy2(img_path, dst_image)

    annotation = {
        "frame_id": frame_id,
        "image": dst_image.name,
        "wheels": wheels,
        "_draft": True,
        "_warning": "UE_RIM_KEYPOINTS_MAPPED_TO_FLOOR_RAY_HEURISTIC",
        "_source_frame": frame_int,
    }
    goal = parse_goal_file(goal_path)
    if goal:
        annotation["_ue_camera"] = goal
    dst_annot.write_text(json.dumps(annotation, indent=2, ensure_ascii=False))
    return True


def write_source_info(
    out_root: Path, source_name: str, args: argparse.Namespace
) -> None:
    meta_dir = out_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "source_name": source_name,
        "annotation_method": "ue_rim_keypoints_5_to_3_heuristic",
        "_warning": "NOT_GROUND_TRUTH_REQUIRES_HUMAN_REVIEW",
        "ue_root": str(args.ue_root),
        "rim_to_tire_factor": RIM_TO_TIRE_FACTOR,
        "five_to_three_mapping": {
            "a": "UE.Left (rim) — heuristic floor-ray; AR-spec asks for floor projection",
            "b": "UE.Right (rim) — heuristic floor-ray; AR-spec asks for floor projection",
            "c_disc_bottom": "max-Y among UE.{Right, Left, Center} — best rim-bottom approx",
        },
    }
    (meta_dir / "source_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False)
    )


def write_conversion_report(out_root: Path, stats: ConversionStats) -> None:
    meta_dir = out_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "frames_total": stats.frames_total,
        "frames_written": stats.frames_written,
        "frames_skipped": stats.frames_skipped,
        "wheels_kept": stats.wheels_kept,
        "wheels_dropped": stats.wheels_dropped,
        "drop_reasons": stats.drop_reasons,
    }
    (meta_dir / "conversion_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ue-root", type=Path, required=True, help="NeuralData UE project root"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/incoming/ue_synthetic"),
        help="Plugin-format output bundle root",
    )
    parser.add_argument(
        "--source-name",
        default="ue_synthetic_v1",
        help="Bundle source label (lands in frame_id prefix + metadata)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N frames (for smoke runs)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing --output-root before writing",
    )
    args = parser.parse_args(argv)

    images_dir = args.ue_root / "Images"
    if not images_dir.is_dir():
        print(f"[err] {images_dir} not found", file=sys.stderr)
        return 2

    if args.overwrite and args.output_root.exists():
        shutil.rmtree(args.output_root)
    (args.output_root / "images").mkdir(parents=True, exist_ok=True)
    (args.output_root / "annotations").mkdir(parents=True, exist_ok=True)

    frame_ids = iter_frame_ids(images_dir, limit=args.limit)
    stats = ConversionStats(frames_total=len(frame_ids))

    for frame_int in frame_ids:
        if convert_frame(
            args.ue_root, frame_int, args.output_root, args.source_name, stats
        ):
            stats.frames_written += 1
        else:
            stats.frames_skipped += 1

    write_source_info(args.output_root, args.source_name, args)
    write_conversion_report(args.output_root, stats)

    print(
        f"[ue_synthetic] frames {stats.frames_written}/{stats.frames_total} written, "
        f"wheels kept={stats.wheels_kept} dropped={stats.wheels_dropped}"
    )
    if stats.drop_reasons:
        print(f"[ue_synthetic] drop_reasons={stats.drop_reasons}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
