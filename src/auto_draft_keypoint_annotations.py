"""Auto-draft keypoint annotations for manual_real images.

Stop-gap for sanity-checking the manual-real pipeline before any real
labels exist. Walks an --images-dir, copies each image into a
plugin-format bundle, and emits a *draft* bbox + A/B/C keypoint set
using a filename-keyword heuristic only — no training, no inference.

Output bundle is plugin-compatible (docs/KEYPOINT_DATASET_FORMAT.md)
so `check_keypoint_incoming.py` /
`convert_keypoint_incoming_to_yolo_pose.py` consume it transparently.

These annotations are NOT ground truth:

  - Every annotation JSON carries top-level "_draft": true and
    "_warning": "NOT_GROUND_TRUTH_REQUIRES_HUMAN_REVIEW".
  - metadata/source_info.json records
    "annotation_method": "auto_draft_heuristic" and the same warning.

Do not train on this bundle without explicit human review.

Heuristic (deliberately dumb so the "draft" label is honest):

  - If the image filename stem matches any of the side-view keywords
    (sboku / side / avto / auto / car / mashin / vid / wheel / koleso),
    draft two wheels in the lower third: centres at x=0.27*W and
    x=0.73*W, cy=0.82*H, radius=max(8, 0.08*min(W, H)). Under the
    2026-05-14 spec revision A and B are floor / raycast points: both
    sit in the lower band of the bbox at (cx ± 0.7*r, cy + 0.88*r),
    well below the rim centerline and below the disc-bottom point. C
    (c_disc_bottom) sits 0.65*r below the centre, on the rim's lower
    edge. Centres are clamped so bbox + points stay strictly inside
    [0, W) x [0, H).
  - Otherwise emit wheels=[].

Usage:
    python src/auto_draft_keypoint_annotations.py \\
        --images-dir   data/manual_real/images \\
        --output-root  data/incoming/manual_real_draft \\
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Lowercase substring match against the image stem. "mashin" picks up
# both "mashin" and "mashina"; "avto" picks up Cyrillic-translit
# "avtomobil".
DEFAULT_KEYWORDS: tuple[str, ...] = (
    "sboku",
    "side",
    "avto",
    "auto",
    "car",
    "mashin",
    "vid",
    "wheel",
    "koleso",
)

ANNOTATION_METHOD = "auto_draft_heuristic"
WARNING_TAG = "NOT_GROUND_TRUTH_REQUIRES_HUMAN_REVIEW"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Draft plugin-format keypoint annotations from filenames. "
            "NOT ground truth — for plumbing sanity-check only."
        )
    )
    p.add_argument("--images-dir", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, delete the existing --output-root before writing.",
    )
    p.add_argument(
        "--keywords",
        nargs="*",
        default=list(DEFAULT_KEYWORDS),
        help=(
            "Lowercase substrings; if any matches the image stem (also "
            "lowercased), the heuristic emits two drafted wheels. "
            "Otherwise the image gets wheels=[]."
        ),
    )
    return p.parse_args(argv)


def _list_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _filename_suggests_car(stem: str, keywords: list[str]) -> bool:
    needle = stem.lower()
    return any(k.lower() in needle for k in keywords)


def _draft_two_wheels(image_w: int, image_h: int) -> list[dict]:
    """Two synthetic wheels in the lower third of the frame.

    Geometry follows the 2026-05-14 contract revision:
      - bbox covers the full wheel (tyre + rim), strictly inside the
        image — the validator's "outside image" warning fires on
        >= W or >= H, so we keep a 1-2 px margin.
      - A / B are floor / raycast points near the wheel footprint, in
        the lower band of the bbox, not on the rim midline.
      - C (c_disc_bottom) is the lowest visible point of the metal
        disc — above A/B (because the rim sits above the tyre base)
        but still below the bbox centerline.
      - All three points stay strictly inside the bbox so the
        validator passes clean.
    """
    r = max(8.0, 0.08 * min(image_w, image_h))
    raw_cy = image_h * 0.82
    raw_cxs = (image_w * 0.27, image_w * 0.73)

    wheels: list[dict] = []
    for raw_cx in raw_cxs:
        # Clamp so the bbox does not touch the image edges (the
        # validator's "outside image" warning fires on >= W or >= H).
        cx = max(r + 1.0, min(image_w - r - 2.0, raw_cx))
        cy = max(r + 1.0, min(image_h - r - 2.0, raw_cy))
        wheels.append(
            {
                "bbox_xyxy": [cx - r, cy - r, cx + r, cy + r],
                "points": {
                    "a": [cx - 0.7 * r, cy + 0.88 * r],
                    "b": [cx + 0.7 * r, cy + 0.88 * r],
                    "c_disc_bottom": [cx, cy + 0.65 * r],
                },
            }
        )
    return wheels


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
    images_dir: Path = args.images_dir
    out_root: Path = args.output_root
    keywords: list[str] = list(args.keywords)

    if not images_dir.is_dir():
        print(f"ERROR: --images-dir does not exist or is not a directory: {images_dir}")
        return 2

    refusal = _ensure_clean_output_root(out_root, args.overwrite)
    if refusal is not None:
        return refusal

    out_images_dir = out_root / "images"
    out_annos_dir = out_root / "annotations"
    out_meta_dir = out_root / "metadata"
    for d in (out_images_dir, out_annos_dir, out_meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    images = _list_images(images_dir)

    n_drafted = 0
    n_empty = 0
    total_wheels = 0
    unreadable: list[str] = []

    for src_image in images:
        img = cv2.imread(str(src_image))
        if img is None:
            print(f"WARNING: unreadable image, skipping: {src_image}")
            unreadable.append(src_image.name)
            continue
        image_h, image_w = img.shape[:2]

        shutil.copy2(src_image, out_images_dir / src_image.name)

        if _filename_suggests_car(src_image.stem, keywords):
            wheels = _draft_two_wheels(image_w, image_h)
            n_drafted += 1
        else:
            wheels = []
            n_empty += 1
        total_wheels += len(wheels)

        annotation = {
            "frame_id": src_image.stem,
            "image": src_image.name,
            "wheels": wheels,
            "_draft": True,
            "_warning": WARNING_TAG,
        }
        (out_annos_dir / f"{src_image.stem}.json").write_text(
            json.dumps(annotation, indent=2), encoding="utf-8"
        )

    processed = len(images) - len(unreadable)
    source_info = {
        "source_name": "manual_real_draft",
        "annotation_method": ANNOTATION_METHOD,
        "warning": WARNING_TAG,
        "heuristic": "filename_keyword_two_wheels_lower_third",
        "keywords": keywords,
        "image_count": processed,
        "images_with_drafted_wheels": n_drafted,
        "images_with_empty_wheels": n_empty,
        "unreadable_images": unreadable,
        "notes": (
            "Auto-generated draft from auto_draft_keypoint_annotations.py. "
            "NOT real labels. Do NOT use for training without explicit "
            "human review."
        ),
    }
    (out_meta_dir / "source_info.json").write_text(
        json.dumps(source_info, indent=2), encoding="utf-8"
    )

    print(f"Images dir:                {images_dir}")
    print(f"Output root:               {out_root}")
    print(f"Images processed:          {processed}")
    print(f"  drafted wheels (image):  {n_drafted}")
    print(f"  empty wheels (image):    {n_empty}")
    print(f"Wheels (total, draft):     {total_wheels}")
    if unreadable:
        print(f"Unreadable images:         {len(unreadable)}")
    print(f"WARNING: {WARNING_TAG}")
    print(f"Source info:               {out_meta_dir / 'source_info.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
