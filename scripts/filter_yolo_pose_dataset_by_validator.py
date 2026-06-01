"""Build a strict-clean YOLO-pose dataset by dropping invalid label lines.

This does not relabel or move keypoints. It copies every source image and keeps
only label rows accepted by check_yolo_pose_dataset.validate_label_text(). Images
whose labels are all rejected become explicit empty-label images so downstream
training/eval can treat them as hard negatives without pretending the removed
wheel annotations were corrected.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from check_yolo_pose_dataset import IMAGE_EXTS, SPLITS, validate_label_text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        required=True,
        type=Path,
        help="Input YOLO-pose dataset root.",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="Output strict-clean YOLO-pose dataset root.",
    )
    parser.add_argument(
        "--config-out",
        type=Path,
        default=None,
        help="Optional YOLO data YAML to write for the output dataset.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help=(
            "Optional extra report path. The report is always written to "
            "<dataset-root>/metadata/strict_filter_report.json as well."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def list_images(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        return []
    return sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def prepare_output(root: Path, overwrite: bool) -> None:
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output dataset root is not empty: {root}")
        shutil.rmtree(root)
    for split in SPLITS:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (root / "metadata").mkdir(parents=True, exist_ok=True)


def ensure_source_layout(root: Path) -> None:
    missing: list[str] = []
    for split in SPLITS:
        for kind in ("images", "labels"):
            path = root / kind / split
            if not path.is_dir():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"source dataset is missing required directories: {missing}")


def _failure_reason(problem: str) -> str:
    marker = ":1: "
    if marker in problem:
        return problem.split(marker, 1)[1]
    return problem


def filter_label_text(label_path: Path, text: str) -> tuple[list[str], list[dict[str, Any]]]:
    kept: list[str] = []
    dropped: list[dict[str, Any]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        problems = validate_label_text(label_path, line + "\n")
        if problems:
            dropped.append(
                {
                    "line": lineno,
                    "text": line,
                    "reasons": [_failure_reason(problem) for problem in problems],
                }
            )
            continue
        kept.append(line)
    return kept, dropped


def _empty_split_stats() -> dict[str, Any]:
    return {
        "images": 0,
        "source_wheel_labels": 0,
        "kept_wheel_labels": 0,
        "dropped_wheel_labels": 0,
        "missing_source_labels": 0,
        "images_with_any_dropped_labels": 0,
        "images_without_valid_labels": 0,
        "drop_reasons": {},
    }


def _merge_totals(by_split: dict[str, dict[str, Any]]) -> dict[str, Any]:
    totals = _empty_split_stats()
    reasons: Counter[str] = Counter()
    for stats in by_split.values():
        for key, value in stats.items():
            if key == "drop_reasons":
                reasons.update(value)
            else:
                totals[key] += int(value)
    totals["drop_reasons"] = dict(sorted(reasons.items()))
    return totals


def copy_filtered_split(source_root: Path, output_root: Path, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    images_dir = source_root / "images" / split
    labels_dir = source_root / "labels" / split
    out_images_dir = output_root / "images" / split
    out_labels_dir = output_root / "labels" / split
    stats = _empty_split_stats()
    dropped_manifest: list[dict[str, Any]] = []
    drop_reasons: Counter[str] = Counter()

    for image_path in list_images(images_dir):
        stats["images"] += 1
        out_image = out_images_dir / image_path.name
        out_label = out_labels_dir / f"{image_path.stem}.txt"
        shutil.copy2(image_path, out_image)

        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            stats["missing_source_labels"] += 1
            stats["images_without_valid_labels"] += 1
            out_label.write_text("", encoding="utf-8")
            continue

        source_text = label_path.read_text(encoding="utf-8")
        source_lines = [line.strip() for line in source_text.splitlines() if line.strip()]
        kept, dropped = filter_label_text(label_path, source_text)
        stats["source_wheel_labels"] += len(source_lines)
        stats["kept_wheel_labels"] += len(kept)
        stats["dropped_wheel_labels"] += len(dropped)
        if dropped:
            stats["images_with_any_dropped_labels"] += 1
        if not kept:
            stats["images_without_valid_labels"] += 1
        for row in dropped:
            for reason in row["reasons"]:
                drop_reasons[reason] += 1
            dropped_manifest.append(
                {
                    "split": split,
                    "image": str((Path("images") / split / image_path.name).as_posix()),
                    "label": str((Path("labels") / split / f"{image_path.stem}.txt").as_posix()),
                    **row,
                }
            )
        out_label.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")

    stats["drop_reasons"] = dict(sorted(drop_reasons.items()))
    return stats, dropped_manifest


def write_config(config_out: Path, dataset_root: Path) -> None:
    config_out.parent.mkdir(parents=True, exist_ok=True)
    config_out.write_text(
        "\n".join(
            [
                "# Strict-clean YOLO-pose config generated by filter_yolo_pose_dataset_by_validator.py.",
                f"path: {dataset_root}",
                "train: images/train",
                "val: images/val",
                "",
                "names:",
                "  0: wheel",
                "",
                "kpt_shape: [3, 3]",
                "flip_idx: [1, 0, 2]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_filtered_dataset(
    *,
    source_root: Path,
    output_root: Path,
    overwrite: bool,
    config_out: Path | None = None,
    report_out: Path | None = None,
) -> dict[str, Any]:
    source_root = source_root.expanduser()
    output_root = output_root.expanduser()
    ensure_source_layout(source_root)
    prepare_output(output_root, overwrite=overwrite)

    by_split: dict[str, dict[str, Any]] = {}
    dropped_manifest: list[dict[str, Any]] = []
    for split in SPLITS:
        stats, dropped = copy_filtered_split(source_root, output_root, split)
        by_split[split] = stats
        dropped_manifest.extend(dropped)

    summary: dict[str, Any] = {
        "ok": True,
        "status": "strict_filtered_not_relabelled",
        "source_root": str(source_root),
        "dataset_root": str(output_root),
        "config_out": str(config_out) if config_out else None,
        "by_split": by_split,
        "totals": _merge_totals(by_split),
        "dropped_label_manifest": dropped_manifest[:500],
        "dropped_label_manifest_truncated": len(dropped_manifest) > 500,
    }

    metadata_report = output_root / "metadata" / "strict_filter_report.json"
    metadata_report.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if config_out is not None:
        write_config(config_out, output_root)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_filtered_dataset(
        source_root=args.source_root,
        output_root=args.dataset_root,
        overwrite=args.overwrite,
        config_out=args.config_out,
        report_out=args.report_out,
    )
    totals = summary["totals"]
    print(f"strict_dataset={summary['dataset_root']}")
    print(
        "labels "
        f"source={totals['source_wheel_labels']} "
        f"kept={totals['kept_wheel_labels']} "
        f"dropped={totals['dropped_wheel_labels']}"
    )
    if args.config_out is not None:
        print(f"config={args.config_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
