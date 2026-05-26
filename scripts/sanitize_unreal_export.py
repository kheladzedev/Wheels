"""Sanitize a raw Unreal wheel export before acceptance.

This is a non-training helper for plugin/export debugging. It copies a raw
Unreal export into a new folder, keeping only object ``keyPoint`` files whose
Right/Left/Center points and bbox path pass the same checks used by the
official importer.

The original export is never modified. The output remains a raw export shape::

    <out-root>/
      Images/<frame>.jpg
      keyPoint/<frame>/<object>.txt
      Ground/<frame>.txt   # copied when present
      Depth/<frame>.*      # copied when present
      Goal/<frame>.*       # copied when present
      metadata/sanitize_report.json

By default, frames with no kept wheel objects are dropped. Use
``--keep-empty-frames`` only when explicitly testing negative-frame behavior.

No training, inference, or AR runtime is run here.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import import_unreal_export as imp  # noqa: E402
import inspect_unreal_export as ix  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SIDE_DIRS = ("Ground", "Depth", "Goal")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Copy a raw Unreal export while dropping invalid wheel object txt "
            "files. Does not train or run model inference."
        )
    )
    p.add_argument("--source-root", required=True, type=Path)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--margin", type=int, default=imp.DEFAULT_MARGIN_PX)
    p.add_argument(
        "--right-left-mapping",
        choices=(
            imp.RIGHT_LEFT_MAPPING_AUTO,
            imp.RIGHT_LEFT_MAPPING_CONFIRMED,
            imp.RIGHT_LEFT_MAPPING_SCREEN_SIDES,
        ),
        default=imp.RIGHT_LEFT_MAPPING_AUTO,
    )
    p.add_argument(
        "--allow-synthetic-bbox",
        action="store_true",
        help=(
            "DEBUG ONLY: keep objects whose bbox can be synthesized from "
            "LeftTop/RightTop or A/B/C when raw BBox/WheelBBox is missing."
        ),
    )
    p.add_argument(
        "--keep-empty-frames",
        action="store_true",
        help="Copy image/sidecar files even when no valid wheel object remains.",
    )
    return p.parse_args(argv)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _prepare_out_root(out_root: Path, overwrite: bool) -> None:
    if out_root.exists() and any(out_root.iterdir()):
        if not overwrite:
            raise SystemExit(
                f"ERROR: out-root already exists and is not empty: {out_root}\n"
                "Pass --overwrite to delete and regenerate."
            )
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)


def _copy_sidecars(src: Path, out: Path, frame_id: str) -> dict[str, int]:
    copied: dict[str, int] = {}
    for dirname in SIDE_DIRS:
        src_dir = src / dirname
        if not src_dir.is_dir():
            copied[dirname] = 0
            continue
        count = 0
        for sidecar in sorted(src_dir.glob(f"{frame_id}.*")):
            if sidecar.is_file():
                _copy_file(sidecar, out / dirname / sidecar.name)
                count += 1
        copied[dirname] = count
    return copied


def _inspect_status(
    text: str,
    image_w: int,
    image_h: int,
) -> tuple[str, str, dict[str, tuple[float, float]]]:
    pts = ix.parse_keypoint_text(text)
    status, reason = ix.classify(pts, image_w, image_h)
    return status, reason, pts


def _mapping_namespace(args: argparse.Namespace) -> argparse.Namespace:
    # import_unreal_export._resolve_right_left_mapping mutates argparse fields.
    return argparse.Namespace(
        right_left_mapping=args.right_left_mapping,
        swap_right_left=False,
    )


def run(args: argparse.Namespace) -> int:
    src = args.source_root.expanduser().resolve()
    out = args.out_root.expanduser().resolve()
    images_dir = src / "Images"
    kp_root = src / "keyPoint"

    if not images_dir.is_dir():
        print(f"ERROR: missing Images directory: {images_dir}", file=sys.stderr)
        return 2
    if not kp_root.is_dir():
        print(f"ERROR: missing keyPoint directory: {kp_root}", file=sys.stderr)
        return 2

    _prepare_out_root(out, args.overwrite)

    images = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    mapping_args = _mapping_namespace(args)
    imp._resolve_right_left_mapping(src, images, kp_root, mapping_args)
    resolved_mapping = mapping_args.right_left_mapping_resolved

    report: dict[str, Any] = {
        "source_root": str(src),
        "out_root": str(out),
        "right_left_mapping_requested": mapping_args.right_left_mapping_requested,
        "right_left_mapping_resolved": resolved_mapping,
        "right_left_mapping_basis": mapping_args.right_left_mapping_basis,
        "right_left_mapping_counts": mapping_args.right_left_mapping_counts,
        "allow_synthetic_bbox": bool(args.allow_synthetic_bbox),
        "keep_empty_frames": bool(args.keep_empty_frames),
        "status": "PASS",
        "training_approved": False,
        "requires_human_preview": True,
        "frames_seen": 0,
        "frames_kept": 0,
        "images_copied": 0,
        "keypoint_files_seen": 0,
        "keypoint_files_kept": 0,
        "empty_frames_dropped": 0,
        "inspect_status_counts": {},
        "drop_counts": {},
        "bbox_source_counts": {
            imp.BBOX_SOURCE_PLUGIN: 0,
            imp.BBOX_SOURCE_SYNTHESIZED: 0,
        },
        "object_id_counts": {},
        "sidecars_copied": {name: 0 for name in SIDE_DIRS},
        "notes": [
            "Sanitizer is a debug/export-quality bridge, not production approval.",
            "Native WheelBBox/BBox is still preferred for production-candidate data.",
        ],
    }

    inspect_counts: Counter[str] = Counter()
    drop_counts: Counter[str] = Counter()
    bbox_source_counts: Counter[str] = Counter()
    object_id_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for img_path in images:
        frame_id = img_path.stem
        report["frames_seen"] += 1
        size = imp._read_image_size(img_path)
        if size is None:
            drop_counts["image_decode_failed"] += 1
            continue
        image_w, image_h = size
        kp_dir = kp_root / frame_id
        kept_objects: list[tuple[Path, str]] = []

        if kp_dir.is_dir():
            for kp_file in sorted(kp_dir.glob("*.txt"), key=lambda p: p.name):
                report["keypoint_files_seen"] += 1
                object_id = kp_file.stem
                try:
                    text = kp_file.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    drop_counts[imp.DROP_PARSE_ERROR] += 1
                    object_id_counts[object_id][imp.DROP_PARSE_ERROR] += 1
                    continue

                status, _reason, _pts = _inspect_status(text, image_w, image_h)
                inspect_counts[status] += 1

                summary = imp.ImportSummary()
                wheel = imp._try_build_wheel(
                    text,
                    image_w,
                    image_h,
                    margin=args.margin,
                    summary=summary,
                    right_left_mapping=resolved_mapping,
                    allow_synthetic_bbox=args.allow_synthetic_bbox,
                )
                if wheel is None:
                    reason = "unknown_drop"
                    for key, value in summary.drop_counts.items():
                        if value:
                            reason = key
                            drop_counts[key] += value
                            object_id_counts[object_id][key] += value
                    if reason == "unknown_drop":
                        drop_counts[reason] += 1
                        object_id_counts[object_id][reason] += 1
                    continue

                kept_objects.append((kp_file, text))
                object_id_counts[object_id]["kept"] += 1
                report["keypoint_files_kept"] += 1
                for key, value in summary.bbox_source_counts.items():
                    if value:
                        bbox_source_counts[key] += value

        if not kept_objects and not args.keep_empty_frames:
            report["empty_frames_dropped"] += 1
            continue

        report["frames_kept"] += 1
        _copy_file(img_path, out / "Images" / img_path.name)
        report["images_copied"] += 1
        sidecars = _copy_sidecars(src, out, frame_id)
        for key, value in sidecars.items():
            report["sidecars_copied"][key] += value

        out_kp_dir = out / "keyPoint" / frame_id
        out_kp_dir.mkdir(parents=True, exist_ok=True)
        for kp_file, text in kept_objects:
            (out_kp_dir / kp_file.name).write_text(text, encoding="utf-8")

    report["inspect_status_counts"] = dict(sorted(inspect_counts.items()))
    report["drop_counts"] = dict(sorted(drop_counts.items()))
    for key in (imp.BBOX_SOURCE_PLUGIN, imp.BBOX_SOURCE_SYNTHESIZED):
        report["bbox_source_counts"][key] = int(bbox_source_counts.get(key, 0))
    report["object_id_counts"] = {
        key: dict(sorted(counter.items()))
        for key, counter in sorted(
            object_id_counts.items(),
            key=lambda item: int(item[0]) if item[0].isdigit() else item[0],
        )
    }
    if report["keypoint_files_kept"] <= 0:
        report["status"] = "FAIL_NO_VALID_OBJECTS"

    metadata_dir = out / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "sanitize_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (metadata_dir / "sanitize_report.md").write_text(_render_md(report), encoding="utf-8")

    print(f"Status:              {report['status']}")
    print(f"Frames seen/kept:    {report['frames_seen']} / {report['frames_kept']}")
    print(
        "Objects seen/kept:   "
        f"{report['keypoint_files_seen']} / {report['keypoint_files_kept']}"
    )
    print(f"Dropped empty frames:{report['empty_frames_dropped']}")
    print(f"Report:              {metadata_dir / 'sanitize_report.json'}")
    return 0 if report["status"] == "PASS" else 1


def _render_md(report: dict[str, Any]) -> str:
    lines = [
        "# Sanitized Unreal Export Report",
        "",
        f"- Status: **{report['status']}**",
        f"- Source: `{report['source_root']}`",
        f"- Output: `{report['out_root']}`",
        f"- Right/Left mapping: `{report['right_left_mapping_resolved']}`",
        f"- Mapping basis: `{report['right_left_mapping_basis']}`",
        f"- Synthetic bbox allowed: `{report['allow_synthetic_bbox']}`",
        f"- Training approved: `{report['training_approved']}`",
        f"- Requires human preview: `{report['requires_human_preview']}`",
        "",
        "## Counts",
        "",
        f"- Frames seen / kept: `{report['frames_seen']}` / `{report['frames_kept']}`",
        f"- Images copied: `{report['images_copied']}`",
        f"- keyPoint files seen / kept: `{report['keypoint_files_seen']}` / `{report['keypoint_files_kept']}`",
        f"- Empty frames dropped: `{report['empty_frames_dropped']}`",
        "",
        "## Raw Inspect Status Counts",
        "",
    ]
    for key, value in report.get("inspect_status_counts", {}).items():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## Drops", ""]
    for key, value in report.get("drop_counts", {}).items():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## BBox Source Counts", ""]
    for key, value in report.get("bbox_source_counts", {}).items():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## Object ID Counts", ""]
    for object_id, counts in report.get("object_id_counts", {}).items():
        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        lines.append(f"- `{object_id}`: {summary}")
    lines += [
        "",
        "## Notes",
        "",
    ]
    for note in report.get("notes", []):
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
