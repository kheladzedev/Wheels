"""Build a union YOLO-pose dataset from existing YOLO-pose datasets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

SPLITS = ("train", "val")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--train-source", action="append", type=Path, default=[])
    parser.add_argument("--val-source", action="append", type=Path, default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _copy_split(source_root: Path, output_root: Path, split: str) -> dict[str, int]:
    counts = {"images": 0, "labels": 0, "skipped_missing_label": 0}
    images_dir = source_root / "images" / split
    labels_dir = source_root / "labels" / split
    if not images_dir.is_dir() or not labels_dir.is_dir():
        return counts
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file():
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            counts["skipped_missing_label"] += 1
            continue
        out_image = output_root / "images" / split / image_path.name
        out_label = output_root / "labels" / split / label_path.name
        if out_image.exists() or out_label.exists():
            raise FileExistsError(f"duplicate dataset item stem: {image_path.stem}")
        shutil.copy2(image_path, out_image)
        shutil.copy2(label_path, out_label)
        counts["images"] += 1
        counts["labels"] += 1
    return counts


def build_union(args: argparse.Namespace) -> dict:
    if args.output_root.exists() and args.overwrite:
        shutil.rmtree(args.output_root)
    for split in SPLITS:
        (args.output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (args.output_root / "metadata").mkdir(parents=True, exist_ok=True)

    report = {
        "output_root": str(args.output_root),
        "train_sources": [str(p) for p in args.train_source],
        "val_sources": [str(p) for p in args.val_source],
        "splits": {"train": [], "val": []},
        "totals": {
            "train_images": 0,
            "train_labels": 0,
            "val_images": 0,
            "val_labels": 0,
            "skipped_missing_label": 0,
        },
    }
    for split, sources in (("train", args.train_source), ("val", args.val_source)):
        for source in sources:
            counts = _copy_split(source, args.output_root, split)
            report["splits"][split].append({"source": str(source), **counts})
            report["totals"][f"{split}_images"] += counts["images"]
            report["totals"][f"{split}_labels"] += counts["labels"]
            report["totals"]["skipped_missing_label"] += counts["skipped_missing_label"]
    (args.output_root / "metadata" / "union_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_union(args)
    print(json.dumps(report["totals"], indent=2))
    print(f"Report: {args.output_root / 'metadata/union_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
