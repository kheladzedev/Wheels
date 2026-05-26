"""Run the full intake gate for a raw Unreal/plugin wheel export.

This is the one-command path for batches shaped like the 0002 trial::

    <source-root>/
      Images/<frame_id>.jpg
      keyPoint/<frame_id>/<object_id>.txt
      Ground/<frame_id>.txt  # optional

The script does not decide that a dataset is production-ready. It proves
that the raw batch can move through the local ML plumbing and writes a
human-review report with counts, logs, and preview paths.

Usage::

    python scripts/accept_unreal_export.py \
        --source-root ~/Downloads/0002 \
        --source-name unreal_0002_trial \
        --overwrite

Optional smoke training::

    python scripts/accept_unreal_export.py \
        --source-root ~/Downloads/0002 \
        --source-name unreal_0002_trial \
        --overwrite \
        --smoke-train --device mps
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = Path("outputs/unreal_export_acceptance")


@dataclass
class StepResult:
    name: str
    command: list[str]
    returncode: int
    log_path: Path

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Accept/check a raw Unreal plugin export end to end."
    )
    p.add_argument("--source-root", required=True, type=Path)
    p.add_argument(
        "--source-name",
        default=None,
        help="Slug used in output filenames. Defaults to unreal_<source folder>.",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Root for generated reports, imported incoming data, previews, logs.",
    )
    p.add_argument("--preview-count", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-skip-ratio", type=float, default=0.05)
    p.add_argument("--max-warning-ratio", type=float, default=0.10)
    p.add_argument(
        "--min-usable-ratio",
        type=float,
        default=0.60,
        help="Training data-quality gate: minimum valid wheels / raw objects.",
    )
    p.add_argument(
        "--max-invalid-required-ratio",
        type=float,
        default=0.20,
        help=(
            "Training data-quality gate: maximum raw objects with partial-zero "
            "or out-of-bounds required Right/Left/Center points."
        ),
    )
    p.add_argument(
        "--max-all-zero-ratio",
        type=float,
        default=0.25,
        help="Training data-quality gate: maximum all-zero raw objects.",
    )
    p.add_argument(
        "--max-bad-geometry-ratio",
        type=float,
        default=0.15,
        help="Training data-quality gate: maximum imported drops by bad geometry.",
    )
    p.add_argument(
        "--max-bbox-fallback-ratio",
        type=float,
        default=0.10,
        help=(
            "Training data-quality gate: maximum valid wheels whose bbox was "
            "built by fallback floor-ray heuristic instead of LeftTop/RightTop."
        ),
    )
    p.add_argument(
        "--max-empty-label-image-ratio",
        type=float,
        default=0.30,
        help="Training data-quality gate: maximum converted images with empty labels.",
    )
    p.add_argument(
        "--fail-on-data-quality-gate",
        action="store_true",
        help=(
            "Exit non-zero when the ML data-quality gate fails. By default, a "
            "failed data-quality gate is reported but does not make the "
            "technical pipeline run fail."
        ),
    )
    p.add_argument("--smoke-train", action="store_true")
    p.add_argument("--smoke-model", default="yolo11n-pose.pt")
    p.add_argument("--smoke-epochs", type=int, default=1)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument(
        "--device",
        default=None,
        help="Forwarded to train_yolo.py for --smoke-train, e.g. mps, cpu, 0.",
    )
    p.add_argument(
        "--right-left-mapping",
        choices=("auto", "confirmed", "screen-sides"),
        default="auto",
        help=(
            "Forwarded to import_unreal_export.py. 'auto' selects the raw "
            "Right/Left mapping from batch x-order; 'confirmed' uses legacy "
            "0002 Right->a, Left->b; 'screen-sides' uses Left->a, Right->b."
        ),
    )
    p.add_argument(
        "--swap-right-left",
        action="store_true",
        help=(
            "Alias for --right-left-mapping screen-sides, kept for the 0003 "
            "diagnostic workflow."
        ),
    )
    p.add_argument(
        "--allow-synthetic-bbox",
        action="store_true",
        help=(
            "DEBUG ONLY: forward to import_unreal_export.py to synthesize bbox "
            "when raw BBox/WheelBBox is missing."
        ),
    )
    return p.parse_args(argv)


def slugify(raw: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw.strip()).strip("._-")
    if not slug:
        slug = "batch"
    if not slug.startswith("unreal_"):
        slug = f"unreal_{slug}"
    return slug


def _json_or_empty(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _ratio(n: int | float, d: int | float) -> float:
    return 0.0 if d <= 0 else float(n) / float(d)


def _count_empty_labels(dataset_root: Path) -> dict[str, int]:
    out = {"label_files": 0, "empty_label_files": 0}
    labels_root = dataset_root / "labels"
    if not labels_root.is_dir():
        return out
    for label_file in labels_root.glob("*/*.txt"):
        if not label_file.is_file():
            continue
        out["label_files"] += 1
        if not label_file.read_text(encoding="utf-8").strip():
            out["empty_label_files"] += 1
    return out


def _evaluate_data_quality_gate(
    inspection: dict[str, Any],
    import_report: dict[str, Any],
    conversion: dict[str, Any],
    dataset_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    counts = inspection.get("counts_by_status") or {}
    drops = import_report.get("drop_counts") or {}
    bbox_counts = import_report.get("bbox_strategy_counts") or {}

    raw_objects = int(inspection.get("n_keypoint_object_files") or 0)
    valid_wheels = int(import_report.get("valid_wheels") or 0)
    all_zero = int(counts.get("EMPTY_ALL_ZERO") or 0)
    invalid_required = int(counts.get("OUT_OF_BOUNDS") or 0) + int(
        counts.get("PARTIAL_ZERO") or 0
    )
    bad_geometry = int(drops.get("bad_floorray_geometry") or 0)
    bbox_fallback = int(bbox_counts.get("floorray") or 0)
    label_stats = _count_empty_labels(dataset_root)

    metrics = {
        "usable_ratio": _ratio(valid_wheels, raw_objects),
        "invalid_required_ratio": _ratio(invalid_required, raw_objects),
        "all_zero_ratio": _ratio(all_zero, raw_objects),
        "bad_geometry_ratio": _ratio(bad_geometry, raw_objects),
        "bbox_fallback_ratio": _ratio(bbox_fallback, valid_wheels),
        "empty_label_image_ratio": _ratio(
            label_stats["empty_label_files"], label_stats["label_files"]
        ),
    }
    thresholds = {
        "min_usable_ratio": args.min_usable_ratio,
        "max_invalid_required_ratio": args.max_invalid_required_ratio,
        "max_all_zero_ratio": args.max_all_zero_ratio,
        "max_bad_geometry_ratio": args.max_bad_geometry_ratio,
        "max_bbox_fallback_ratio": args.max_bbox_fallback_ratio,
        "max_empty_label_image_ratio": args.max_empty_label_image_ratio,
    }
    reasons: list[str] = []
    if metrics["usable_ratio"] < args.min_usable_ratio:
        reasons.append(
            f"usable_ratio={metrics['usable_ratio']:.4f} < "
            f"min_usable_ratio={args.min_usable_ratio:.4f} "
            f"({valid_wheels}/{raw_objects} raw objects usable)"
        )
    if metrics["invalid_required_ratio"] > args.max_invalid_required_ratio:
        reasons.append(
            f"invalid_required_ratio={metrics['invalid_required_ratio']:.4f} > "
            f"max_invalid_required_ratio={args.max_invalid_required_ratio:.4f} "
            f"({invalid_required}/{raw_objects} partial-zero or OOB objects)"
        )
    if metrics["all_zero_ratio"] > args.max_all_zero_ratio:
        reasons.append(
            f"all_zero_ratio={metrics['all_zero_ratio']:.4f} > "
            f"max_all_zero_ratio={args.max_all_zero_ratio:.4f} "
            f"({all_zero}/{raw_objects} all-zero objects)"
        )
    if metrics["bad_geometry_ratio"] > args.max_bad_geometry_ratio:
        reasons.append(
            f"bad_geometry_ratio={metrics['bad_geometry_ratio']:.4f} > "
            f"max_bad_geometry_ratio={args.max_bad_geometry_ratio:.4f} "
            f"({bad_geometry}/{raw_objects} geometry drops)"
        )
    if metrics["bbox_fallback_ratio"] > args.max_bbox_fallback_ratio:
        reasons.append(
            f"bbox_fallback_ratio={metrics['bbox_fallback_ratio']:.4f} > "
            f"max_bbox_fallback_ratio={args.max_bbox_fallback_ratio:.4f} "
            f"({bbox_fallback}/{valid_wheels} valid bboxes used fallback)"
        )
    if metrics["empty_label_image_ratio"] > args.max_empty_label_image_ratio:
        reasons.append(
            f"empty_label_image_ratio={metrics['empty_label_image_ratio']:.4f} > "
            f"max_empty_label_image_ratio={args.max_empty_label_image_ratio:.4f} "
            f"({label_stats['empty_label_files']}/{label_stats['label_files']} "
            "converted images have no labels)"
        )

    return {
        "passed": len(reasons) == 0,
        "metrics": metrics,
        "thresholds": thresholds,
        "reasons": reasons,
        "counts": {
            "raw_objects": raw_objects,
            "valid_wheels": valid_wheels,
            "all_zero": all_zero,
            "invalid_required": invalid_required,
            "bad_geometry": bad_geometry,
            "bbox_fallback": bbox_fallback,
            **label_stats,
        },
        "note": (
            "This gate is stricter than technical conversion. It answers whether "
            "a full batch looks clean enough to train without first fixing the "
            "exporter or reviewing a large number of bad frames."
        ),
    }


def _write_data_yaml(path: Path, dataset_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"path: {dataset_root.resolve()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: wheel",
                "kpt_shape: [3, 3]",
                "flip_idx: [1, 0, 2]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _run_step(
    name: str,
    command: list[str],
    logs_dir: Path,
    steps: list[StepResult],
) -> StepResult:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{len(steps) + 1:02d}_{name}.log"
    print(f"\n==> {name}")
    print(" ".join(command))
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    result = StepResult(
        name=name,
        command=command,
        returncode=proc.returncode,
        log_path=log_path,
    )
    steps.append(result)
    if proc.stdout:
        tail = "\n".join(proc.stdout.rstrip().splitlines()[-12:])
        print(tail)
    print(f"--> {name}: {'OK' if result.ok else 'FAIL'} (log: {log_path})")
    return result


def _paths(work_root: Path) -> dict[str, Path]:
    return {
        "inspection": work_root / "inspection",
        "incoming": work_root / "incoming",
        "dataset": work_root / "pose_dataset",
        "incoming_preview": work_root / "previews" / "incoming",
        "pose_preview": work_root / "previews" / "pose",
        "logs": work_root / "logs",
        "runs": work_root / "runs",
        "data_yaml": work_root / "config" / "pose_dataset.yaml",
    }


def _summarise(
    source_root: Path,
    source_name: str,
    work_root: Path,
    paths: dict[str, Path],
    steps: list[StepResult],
    args: argparse.Namespace,
) -> dict[str, Any]:
    inspection = _json_or_empty(paths["inspection"] / "report.json")
    import_report = _json_or_empty(paths["incoming"] / "metadata" / "import_report.json")
    conversion = _json_or_empty(
        paths["dataset"] / "metadata" / "conversion_report.json"
    )
    mapping_mode = import_report.get("mapping_mode") or "unknown"
    diagnostic_swap = bool(
        import_report.get("diagnostic_swap_right_left", args.swap_right_left)
    )
    mapping_basis = import_report.get("mapping_basis")
    mapping_requested = import_report.get("right_left_mapping_requested")
    mapping_resolved = import_report.get("right_left_mapping_resolved")
    mapping_counts = import_report.get("right_left_mapping_counts", {})
    raw_point_aliases = import_report.get("raw_point_aliases", {})

    all_required_steps_ok = all(s.ok for s in steps)
    valid_wheels = int(import_report.get("valid_wheels") or 0)
    qg = conversion.get("quality_gate") or {}
    quality_gate_passed = bool(qg.get("passed", False)) if conversion else False
    data_quality_gate = _evaluate_data_quality_gate(
        inspection, import_report, conversion, paths["dataset"], args
    )

    technical_status = (
        "PASS" if all_required_steps_ok and valid_wheels > 0 and quality_gate_passed else "FAIL"
    )
    review_status = (
        "READY_FOR_HUMAN_PREVIEW"
        if technical_status == "PASS"
        else "BLOCKED_BEFORE_HUMAN_PREVIEW"
    )
    if technical_status != "PASS":
        training_status = "NOT_APPROVED_FOR_TRAINING_TECHNICAL_GATE_FAILED"
    elif not data_quality_gate["passed"]:
        training_status = "NOT_APPROVED_FOR_TRAINING_DATA_QUALITY_GATE_FAILED"
    else:
        training_status = "NOT_APPROVED_FOR_TRAINING_UNTIL_HUMAN_PREVIEW_ACCEPTS_GEOMETRY"

    return {
        "source_root": str(source_root),
        "source_name": source_name,
        "work_root": str(work_root),
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "technical_status": technical_status,
        "review_status": review_status,
        "training_status": training_status,
        "smoke_train_requested": args.smoke_train,
        "mapping_mode": mapping_mode,
        "mapping_basis": mapping_basis,
        "right_left_mapping_requested": mapping_requested,
        "right_left_mapping_resolved": mapping_resolved,
        "right_left_mapping_counts": mapping_counts,
        "raw_point_aliases": raw_point_aliases,
        "diagnostic_swap_right_left": diagnostic_swap,
        "data_quality_gate": data_quality_gate,
        "steps": [
            {
                "name": s.name,
                "returncode": s.returncode,
                "log_path": str(s.log_path),
                "command": s.command,
            }
            for s in steps
        ],
        "raw_export": {
            "images": inspection.get("n_images"),
            "keypoint_object_files": inspection.get("n_keypoint_object_files"),
            "counts_by_status": inspection.get("counts_by_status", {}),
            "image_resolutions": inspection.get("image_resolutions", {}),
        },
        "import": {
            "mapping_mode": mapping_mode,
            "mapping_basis": mapping_basis,
            "right_left_mapping_requested": mapping_requested,
            "right_left_mapping_resolved": mapping_resolved,
            "right_left_mapping_counts": mapping_counts,
            "raw_point_aliases": raw_point_aliases,
            "diagnostic_swap_right_left": diagnostic_swap,
            "images_imported": import_report.get("images_imported"),
            "keypoint_object_files_found": import_report.get(
                "keypoint_object_files_found"
            ),
            "valid_wheels": import_report.get("valid_wheels"),
            "bbox_strategy_counts": import_report.get("bbox_strategy_counts", {}),
            "drop_counts": import_report.get("drop_counts", {}),
        },
        "conversion": {
            "converted_images": conversion.get("converted_images"),
            "train": conversion.get("train"),
            "val": conversion.get("val"),
            "wheels": conversion.get("wheels"),
            "quality_gate": conversion.get("quality_gate", {}),
            "warnings_count": conversion.get("warnings_count"),
        },
        "artifacts": {
            "inspection_report_md": str(paths["inspection"] / "report.md"),
            "status_preview_root": str(paths["inspection"] / "previews" / "by_status"),
            "incoming_root": str(paths["incoming"]),
            "incoming_preview_dir": str(paths["incoming_preview"]),
            "pose_dataset_root": str(paths["dataset"]),
            "pose_preview_dir": str(paths["pose_preview"] / "train"),
            "data_yaml": str(paths["data_yaml"]),
            "logs_dir": str(paths["logs"]),
            "smoke_runs_dir": str(paths["runs"]),
        },
    }


def _write_md(path: Path, report: dict[str, Any]) -> None:
    raw = report["raw_export"]
    imp = report["import"]
    conv = report["conversion"]
    lines = [
        "# Unreal Export Acceptance Report",
        "",
        f"- Source: `{report['source_root']}`",
        f"- Source name: `{report['source_name']}`",
        f"- Technical status: **{report['technical_status']}**",
        f"- Review status: **{report['review_status']}**",
        f"- Training status: **{report['training_status']}**",
        f"- Mapping mode: **{report.get('mapping_mode')}**",
        f"- Mapping basis: **{report.get('mapping_basis')}**",
        f"- Right/Left mapping requested: **{report.get('right_left_mapping_requested')}**",
        f"- Right/Left mapping resolved: **{report.get('right_left_mapping_resolved')}**",
        f"- Diagnostic Right/Left swap: **{report.get('diagnostic_swap_right_left')}**",
    ]
    aliases = report.get("raw_point_aliases") or {}
    if aliases:
        alias_text = ", ".join(f"{k}->{v}" for k, v in sorted(aliases.items()))
        lines.append(f"- Raw point aliases: `{alias_text}`")
    lines += [
        "",
        "## Counts",
        "",
        f"- Raw images: {raw.get('images')}",
        f"- Raw keyPoint objects: {raw.get('keypoint_object_files')}",
        f"- Imported images: {imp.get('images_imported')}",
        f"- Valid imported wheels: {imp.get('valid_wheels')}",
        f"- Converted images: {conv.get('converted_images')}",
        f"- YOLO train / val: {conv.get('train')} / {conv.get('val')}",
        f"- YOLO wheel lines: {conv.get('wheels')}",
        "",
        "## Raw Status Counts",
        "",
    ]
    for key, value in sorted((raw.get("counts_by_status") or {}).items()):
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## Import Drops", ""]
    for key, value in sorted((imp.get("drop_counts") or {}).items()):
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## BBox Strategy Counts", ""]
    for key, value in sorted((imp.get("bbox_strategy_counts") or {}).items()):
        lines.append(f"- `{key}`: {value}")
    dqg = report.get("data_quality_gate") or {}
    lines += [
        "",
        "## ML Data Quality Gate",
        "",
        f"- Passed: **{dqg.get('passed')}**",
        "",
        "### Metrics",
        "",
    ]
    for key, value in sorted((dqg.get("metrics") or {}).items()):
        lines.append(f"- `{key}`: {float(value):.4f}")
    lines += ["", "### Thresholds", ""]
    for key, value in sorted((dqg.get("thresholds") or {}).items()):
        lines.append(f"- `{key}`: {float(value):.4f}")
    reasons = dqg.get("reasons") or []
    lines += ["", "### Reasons", ""]
    if reasons:
        for reason in reasons:
            lines.append(f"- {reason}")
    else:
        lines.append("- none")
    lines += [
        "",
        "## Conversion Quality Gate",
        "",
        f"```json\n{json.dumps(conv.get('quality_gate') or {}, indent=2)}\n```",
        "",
        "## Artifacts",
        "",
    ]
    for key, value in report["artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines += [
        "",
        "## Manual Review Gate",
        "",
        "Open the incoming and YOLO-pose previews before marking a full batch "
        "ACCEPT_FOR_TRAINING. Programmatic checks only prove that the pipeline "
        "runs; they do not prove that A/B/C are semantically correct in every "
        "scene.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_reports(
    source_root: Path,
    source_name: str,
    work_root: Path,
    paths: dict[str, Path],
    steps: list[StepResult],
    args: argparse.Namespace,
) -> dict[str, Any]:
    report = _summarise(source_root, source_name, work_root, paths, steps, args)
    json_path = work_root / "acceptance_report.json"
    md_path = work_root / "acceptance_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_md(md_path, report)
    return report


def run(args: argparse.Namespace) -> int:
    source_root = args.source_root.expanduser().resolve()
    if not source_root.is_dir():
        print(f"ERROR: source-root not found: {source_root}", file=sys.stderr)
        return 2
    if not 0.0 <= args.val_ratio <= 1.0:
        print(f"ERROR: --val-ratio must be in [0, 1], got {args.val_ratio}")
        return 2

    source_name = args.source_name or slugify(source_root.name)
    source_name = slugify(source_name)
    out_root = args.out_root.expanduser()
    work_root = (REPO_ROOT / out_root / source_name).resolve()
    if work_root.exists() and any(work_root.iterdir()):
        if not args.overwrite:
            print(
                f"ERROR: output root already exists and is not empty: {work_root}\n"
                "Pass --overwrite or choose another --source-name/--out-root.",
                file=sys.stderr,
            )
            return 2
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    paths = _paths(work_root)
    for p in paths.values():
        if p.suffix:
            continue
        p.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    steps: list[StepResult] = []

    commands: list[tuple[str, list[str]]] = [
        (
            "inspect_raw",
            [
                py,
                "scripts/inspect_unreal_export.py",
                "--source-root",
                str(source_root),
                "--out-dir",
                str(paths["inspection"]),
                "--max-preview",
                str(args.preview_count),
                "--seed",
                str(args.seed),
            ],
        ),
        (
            "import_raw",
            [
                py,
                "scripts/import_unreal_export.py",
                "--source-root",
                str(source_root),
                "--out-root",
                str(paths["incoming"]),
                "--source-name",
                source_name,
                "--overwrite",
            ],
        ),
        (
            "check_incoming",
            [
                py,
                "src/check_keypoint_incoming.py",
                "--source-root",
                str(paths["incoming"]),
            ],
        ),
        (
            "preview_incoming",
            [
                py,
                "src/preview_keypoint_annotations.py",
                "--source-root",
                str(paths["incoming"]),
                "--count",
                str(args.preview_count),
                "--seed",
                str(args.seed),
                "--output-root",
                str(paths["incoming_preview"]),
            ],
        ),
        (
            "convert_pose",
            [
                py,
                "src/convert_keypoint_incoming_to_yolo_pose.py",
                "--source-root",
                str(paths["incoming"]),
                "--dataset-root",
                str(paths["dataset"]),
                "--source-name",
                source_name,
                "--val-ratio",
                str(args.val_ratio),
                "--seed",
                str(args.seed),
                "--max-skip-ratio",
                str(args.max_skip_ratio),
                "--max-warning-ratio",
                str(args.max_warning_ratio),
                "--overwrite",
                "--fail-on-quality-gate",
            ],
        ),
        (
            "check_pose",
            [
                py,
                "src/check_yolo_pose_dataset.py",
                "--dataset-root",
                str(paths["dataset"]),
            ],
        ),
        (
            "preview_pose",
            [
                py,
                "src/preview_yolo_pose_labels.py",
                "--dataset-root",
                str(paths["dataset"]),
                "--split",
                "train",
                "--count",
                str(args.preview_count),
                "--seed",
                str(args.seed),
                "--out-dir",
                str(paths["pose_preview"]),
            ],
        ),
    ]
    if args.swap_right_left:
        for name, command in commands:
            if name == "import_raw":
                command.append("--swap-right-left")
                break
    else:
        for name, command in commands:
            if name == "import_raw":
                command.extend(["--right-left-mapping", args.right_left_mapping])
                break
    if args.allow_synthetic_bbox:
        for name, command in commands:
            if name == "import_raw":
                command.append("--allow-synthetic-bbox")
                break

    failed = False
    for name, command in commands:
        result = _run_step(name, command, paths["logs"], steps)
        if not result.ok:
            failed = True
            break

    if not failed:
        _write_data_yaml(paths["data_yaml"], paths["dataset"])
        if args.smoke_train:
            smoke_cmd = [
                py,
                "src/train_yolo.py",
                "--data",
                str(paths["data_yaml"]),
                "--model",
                args.smoke_model,
                "--epochs",
                str(args.smoke_epochs),
                "--imgsz",
                str(args.imgsz),
                "--batch",
                str(args.batch),
                "--project",
                str(paths["runs"]),
                "--name",
                f"{source_name}_smoke_e{args.smoke_epochs}",
            ]
            if args.device:
                smoke_cmd.extend(["--device", args.device])
            result = _run_step("smoke_train", smoke_cmd, paths["logs"], steps)
            failed = not result.ok

    report = _write_reports(
        source_root, source_name, work_root, paths, steps, args
    )

    print()
    print(f"Acceptance report: {work_root / 'acceptance_report.md'}")
    print(f"Technical status:  {report['technical_status']}")
    print(f"Review status:     {report['review_status']}")
    print(f"Training status:   {report['training_status']}")
    data_quality_failed = not report["data_quality_gate"]["passed"]
    return 1 if (
        failed
        or report["technical_status"] != "PASS"
        or (args.fail_on_data_quality_gate and data_quality_failed)
    ) else 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
