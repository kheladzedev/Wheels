"""Combine accepted YOLO-pose datasets into one provisional training set.

The Unreal acceptance pipeline writes one dataset per raw export:

    images/{train,val}
    labels/{train,val}

This utility copies multiple accepted datasets into a single dataset root while
prefixing file stems by source name. It keeps all labelled images and can cap
empty-label images so a dirty export does not drown the real wheel examples in
negative frames.

It intentionally works only on already converted YOLO-pose datasets. Raw Unreal
objects that failed acceptance stay excluded.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    root: Path


@dataclass(frozen=True)
class Item:
    source: SourceSpec
    split: str
    image_path: Path
    label_path: Path
    wheel_count: int

    @property
    def is_empty(self) -> bool:
        return self.wheel_count == 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine YOLO-pose datasets emitted by Unreal acceptance"
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help=(
            "Source dataset root. Repeat for each accepted dataset, e.g. "
            "--source unreal_0003=outputs/.../pose_dataset"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="Output combined YOLO-pose dataset root.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-empty-ratio",
        type=float,
        default=None,
        help=(
            "Optional cap on empty-label image ratio per split. All labelled "
            "images are kept; empty images are sampled down to this ratio."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    return slug or "source"


def parse_source(raw: str) -> SourceSpec:
    if "=" not in raw:
        raise ValueError(f"--source must be NAME=PATH, got: {raw}")
    name, path = raw.split("=", 1)
    name = slugify(name)
    root = Path(path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"source dataset root does not exist: {root}")
    return SourceSpec(name=name, root=root)


def ensure_source_layout(source: SourceSpec) -> None:
    missing = []
    for split in SPLITS:
        for kind in ("images", "labels"):
            path = source.root / kind / split
            if not path.is_dir():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            f"source {source.name} is missing required directories: {missing}"
        )


def read_wheel_count(label_path: Path) -> int:
    if not label_path.is_file() or label_path.stat().st_size == 0:
        return 0
    return sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())


def collect_items(source: SourceSpec, split: str) -> list[Item]:
    images_dir = source.root / "images" / split
    labels_dir = source.root / "labels" / split
    items: list[Item] = []
    for image_path in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"missing label for image: {image_path}")
        items.append(
            Item(
                source=source,
                split=split,
                image_path=image_path,
                label_path=label_path,
                wheel_count=read_wheel_count(label_path),
            )
        )
    return items


def select_items(
    items: list[Item],
    max_empty_ratio: float | None,
    rng: random.Random,
) -> tuple[list[Item], dict[str, int | float | None]]:
    labelled = [item for item in items if not item.is_empty]
    empty = [item for item in items if item.is_empty]
    if max_empty_ratio is None:
        selected_empty = empty
    else:
        if not 0.0 <= max_empty_ratio < 1.0:
            raise ValueError("--max-empty-ratio must be in [0, 1)")
        max_empty = int(len(labelled) * max_empty_ratio / (1.0 - max_empty_ratio))
        if len(empty) <= max_empty:
            selected_empty = empty
        else:
            selected_empty = sorted(
                rng.sample(empty, max_empty),
                key=lambda item: (item.source.name, item.image_path.name),
            )

    selected = sorted(
        labelled + selected_empty,
        key=lambda item: (item.source.name, item.split, item.image_path.name),
    )
    return selected, {
        "candidate_images": len(items),
        "candidate_labelled_images": len(labelled),
        "candidate_empty_images": len(empty),
        "selected_images": len(selected),
        "selected_labelled_images": len(labelled),
        "selected_empty_images": len(selected_empty),
        "max_empty_ratio": max_empty_ratio,
    }


def prepare_output(root: Path, overwrite: bool) -> None:
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output dataset root is not empty: {root}")
        shutil.rmtree(root)
    for split in SPLITS:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)


def copy_item(item: Item, out_root: Path) -> dict[str, Any]:
    out_stem = f"{item.source.name}__{item.image_path.stem}"
    out_image = out_root / "images" / item.split / f"{out_stem}{item.image_path.suffix.lower()}"
    out_label = out_root / "labels" / item.split / f"{out_stem}.txt"
    shutil.copy2(item.image_path, out_image)
    shutil.copy2(item.label_path, out_label)
    return {
        "source": item.source.name,
        "split": item.split,
        "image": str(out_image.relative_to(out_root)),
        "label": str(out_label.relative_to(out_root)),
        "wheel_count": item.wheel_count,
    }


def write_data_yaml(root: Path) -> None:
    (root / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: images/train",
                "val: images/val",
                "kpt_shape: [3, 3]",
                "names:",
                "  0: wheel",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_summary(
    root: Path,
    sources: list[SourceSpec],
    split_stats: dict[str, dict[str, int | float | None]],
    manifest: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    by_split: dict[str, dict[str, int]] = {}
    by_source: dict[str, dict[str, int]] = {}
    for row in manifest:
        split = str(row["split"])
        source = str(row["source"])
        wheels = int(row["wheel_count"])
        by_split.setdefault(split, {"images": 0, "empty_images": 0, "wheels": 0})
        by_source.setdefault(source, {"images": 0, "empty_images": 0, "wheels": 0})
        by_split[split]["images"] += 1
        by_source[source]["images"] += 1
        if wheels == 0:
            by_split[split]["empty_images"] += 1
            by_source[source]["empty_images"] += 1
        by_split[split]["wheels"] += wheels
        by_source[source]["wheels"] += wheels

    total_images = sum(v["images"] for v in by_split.values())
    total_empty = sum(v["empty_images"] for v in by_split.values())
    total_wheels = sum(v["wheels"] for v in by_split.values())
    return {
        "dataset_root": str(root),
        "sources": [{"name": source.name, "root": str(source.root)} for source in sources],
        "seed": seed,
        "total_images": total_images,
        "total_empty_images": total_empty,
        "total_wheels": total_wheels,
        "empty_image_ratio": (total_empty / total_images) if total_images else 0.0,
        "by_split": by_split,
        "by_source": by_source,
        "selection": split_stats,
        "status": "provisional_combined_not_production",
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sources = [parse_source(raw) for raw in args.source]
    for source in sources:
        ensure_source_layout(source)

    rng = random.Random(args.seed)
    out_root: Path = args.dataset_root.expanduser()
    prepare_output(out_root, args.overwrite)

    manifest: list[dict[str, Any]] = []
    split_stats: dict[str, dict[str, int | float | None]] = {}
    for split in SPLITS:
        candidates: list[Item] = []
        for source in sources:
            candidates.extend(collect_items(source, split))
        selected, stats = select_items(candidates, args.max_empty_ratio, rng)
        split_stats[split] = stats
        for item in selected:
            manifest.append(copy_item(item, out_root))

    write_data_yaml(out_root)
    summary = build_summary(out_root, sources, split_stats, manifest, args.seed)
    (out_root / "metadata" / "combined_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_root / "metadata" / "combine_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Combined dataset: {out_root}")
    print(f"Images: {summary['total_images']}")
    print(f"Wheels: {summary['total_wheels']}")
    print(f"Empty images: {summary['total_empty_images']}")
    print(f"Status: {summary['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
