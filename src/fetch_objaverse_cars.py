"""Fetch car-body GLBs from Objaverse LVIS categories as a fallback pool.

Objaverse mirrors many downloadable Sketchfab assets through Hugging Face,
which gives us a practical fallback when the live Sketchfab API is
temporarily rate-limited. Files are written into the same GLB pool with an
``ov_`` prefix and a JSON manifest that keeps provenance explicit.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path("data/sketchfab_cars")
DEFAULT_TARGET_TOTAL = 300
DEFAULT_MAX_MB = 45.0
DEFAULT_CATEGORIES = (
    "car_(automobile)",
    "convertible_(automobile)",
    "race_car",
    "pickup_truck",
    "truck",
    "bus_(vehicle)",
    "school_bus",
    "minivan",
    "cab_(taxi)",
    "camper_(vehicle)",
    "tow_truck",
    "trailer_truck",
    "garbage_truck",
    "motor_vehicle",
)


def _count_glbs(output_dir: Path) -> int:
    return sum(1 for _ in output_dir.glob("*.glb")) if output_dir.is_dir() else 0


def _safe_category_name(category: str) -> str:
    return category.replace("_", " ").replace("(", "").replace(")", "")


def _manifest_for_objaverse(uid: str, category: str, metadata: dict | None) -> dict:
    metadata = metadata or {}
    source_name = metadata.get("name") or uid
    return {
        "uid": f"ov_{uid}",
        "source_uid": uid,
        "source_platform": "objaverse",
        "source_dataset": "objaverse_lvis",
        "source_category": category,
        "name": f"Objaverse {_safe_category_name(category)} - {source_name}",
        "viewer_url": metadata.get("viewerUrl") or metadata.get("viewer_url"),
        "license": metadata.get("license"),
        "user": (metadata.get("user") or {}).get("username")
        if isinstance(metadata.get("user"), dict)
        else metadata.get("user"),
    }


def _candidate_uids(
    annotations: dict[str, list[str]],
    categories: list[str],
    *,
    shuffle_seed: int,
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for category in categories:
        for uid in annotations.get(category, []):
            if uid in seen:
                continue
            seen.add(uid)
            out.append((uid, category))
    rng = random.Random(shuffle_seed)
    rng.shuffle(out)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-total", type=int, default=DEFAULT_TARGET_TOTAL)
    parser.add_argument("--max-downloads", type=int, default=66)
    parser.add_argument("--max-mb", type=float, default=DEFAULT_MAX_MB)
    parser.add_argument("--download-processes", type=int, default=1)
    parser.add_argument("--shuffle-seed", type=int, default=527)
    parser.add_argument(
        "--category",
        action="append",
        default=None,
        help="Objaverse LVIS category. Can be passed multiple times.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    current = _count_glbs(args.output_dir)
    if current >= args.target_total:
        print(f"[objaverse] target already reached: {current}/{args.target_total}")
        return 0

    try:
        import objaverse
    except ImportError:
        print(
            "ERROR: objaverse is not installed. Run: ./.venv/bin/python -m pip install objaverse",
            file=sys.stderr,
        )
        return 2

    categories = args.category or list(DEFAULT_CATEGORIES)
    annotations = objaverse.load_lvis_annotations()
    candidates = _candidate_uids(
        annotations,
        categories,
        shuffle_seed=args.shuffle_seed,
    )
    needed = min(args.max_downloads, max(0, args.target_total - current))
    print(
        f"[objaverse] current={current}/{args.target_total} "
        f"needed_this_run={needed} candidates={len(candidates)}"
    )
    if args.dry_run:
        for uid, category in candidates[:needed]:
            print(f"[dry-run] {uid} {category}")
        return 0

    downloaded = 0
    skipped_existing = 0
    skipped_large = 0
    skipped_failed = 0
    max_bytes = int(args.max_mb * 1024 * 1024)

    for uid, category in candidates:
        if _count_glbs(args.output_dir) >= args.target_total or downloaded >= needed:
            break
        out_stem = f"ov_{uid}"
        out_glb = args.output_dir / f"{out_stem}.glb"
        out_manifest = args.output_dir / f"{out_stem}.json"
        if out_glb.exists():
            skipped_existing += 1
            continue

        try:
            paths = objaverse.load_objects(
                [uid],
                download_processes=max(1, int(args.download_processes)),
            )
        except Exception as exc:  # noqa: BLE001
            skipped_failed += 1
            print(f"[objaverse] {uid} failed: {exc}", file=sys.stderr)
            continue
        src_path = Path(paths.get(uid, ""))
        if not src_path.is_file():
            skipped_failed += 1
            print(f"[objaverse] {uid} no local file", file=sys.stderr)
            continue
        size = src_path.stat().st_size
        if size > max_bytes:
            skipped_large += 1
            print(
                f"[objaverse] {uid} skipped: {size / 1024 / 1024:.1f} MB > {args.max_mb:.1f} MB",
                file=sys.stderr,
            )
            continue

        metadata = {}
        try:
            metadata = objaverse.load_annotations([uid]).get(uid, {})
        except Exception as exc:  # noqa: BLE001
            print(f"[objaverse] {uid} metadata warning: {exc}", file=sys.stderr)
        shutil.copy2(src_path, out_glb)
        out_manifest.write_text(
            json.dumps(
                _manifest_for_objaverse(uid, category, metadata),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        downloaded += 1
        print(
            f"[ok] ov_{uid} {size / 1024 / 1024:6.1f} MB "
            f"{_safe_category_name(category)}"
        )

    print(
        f"[done] downloaded={downloaded} skipped_existing={skipped_existing} "
        f"skipped_large={skipped_large} skipped_failed={skipped_failed} "
        f"glbs_total={_count_glbs(args.output_dir)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
