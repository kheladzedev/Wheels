"""Package and accept a NeuralData1 Unreal capture export.

This is the safe wrapper for the local Unreal project Igor sent:

    /Users/edward/Desktop/VSBL/NeuralData1 2

The Unreal project itself is not a training dataset. Only the generated
export folders are copied into VSBL:

    Images/
    keyPoint/
    Depth/   # optional/debug
    Goal/    # optional/debug

The wrapper refuses to run acceptance on an empty capture, writes a short
capture report, and never marks a batch training-ready unless the technical
acceptance passes, the data-quality gate passes, and the caller explicitly
records that human preview was accepted.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT_ROOT = Path("/Users/edward/Desktop/VSBL/NeuralData1 2")
DEFAULT_RAW_OUT_ROOT = Path("outputs/raw_unreal_exports")
DEFAULT_ACCEPTANCE_OUT_ROOT = Path("outputs/unreal_export_acceptance_neuraldata1")
EXPORT_DIRS = ("Images", "keyPoint", "Depth", "Goal")

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import accept_unreal_export as accept  # noqa: E402


@dataclass(frozen=True)
class ExportCounts:
    images: int
    keypoint_files: int
    depth_files: int
    goal_files: int
    project_files: int
    quarantine_files: int | None
    quarantine_size: str | None

    @property
    def has_required_export(self) -> bool:
        return self.images > 0 and self.keypoint_files > 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Copy a NeuralData1 Unreal capture export into outputs/raw_unreal_exports "
            "and run the official Unreal acceptance pipeline."
        )
    )
    p.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Path to the local NeuralData1 Unreal project.",
    )
    p.add_argument(
        "--source-name",
        default=None,
        help=(
            "Acceptance slug. Defaults to neuraldata1_manual_capture_YYYYMMDD_HHMM."
        ),
    )
    p.add_argument(
        "--raw-out-root",
        type=Path,
        default=DEFAULT_RAW_OUT_ROOT,
        help="Where the copied raw export should be written.",
    )
    p.add_argument(
        "--acceptance-out-root",
        type=Path,
        default=DEFAULT_ACCEPTANCE_OUT_ROOT,
        help="Where acceptance reports/previews/logs should be written.",
    )
    p.add_argument(
        "--quarantine-root",
        type=Path,
        default=Path("/Users/edward/Downloads/NeuralData1__legacy_3d_quarantine_20260520_1450"),
        help="Optional legacy asset quarantine path recorded in the report.",
    )
    p.add_argument("--preview-count", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--right-left-mapping",
        choices=("auto", "confirmed", "screen-sides"),
        default="auto",
    )
    p.add_argument(
        "--human-preview-accepted",
        action="store_true",
        help=(
            "Record ACCEPT_FOR_TRAINING only when a human has inspected previews "
            "and accepted bbox/A/B/C geometry."
        ),
    )
    return p.parse_args(argv)


def _count_files(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file())


def _du_sh(path: Path) -> str | None:
    if not path.exists():
        return None
    proc = subprocess.run(
        ["du", "-sh", str(path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return proc.stdout.split()[0]


def collect_counts(project_root: Path, quarantine_root: Path | None) -> ExportCounts:
    return ExportCounts(
        images=_count_files(project_root / "Images"),
        keypoint_files=_count_files(project_root / "keyPoint"),
        depth_files=_count_files(project_root / "Depth"),
        goal_files=_count_files(project_root / "Goal"),
        project_files=_count_files(project_root),
        quarantine_files=(
            _count_files(quarantine_root)
            if quarantine_root is not None and quarantine_root.exists()
            else None
        ),
        quarantine_size=(
            _du_sh(quarantine_root)
            if quarantine_root is not None and quarantine_root.exists()
            else None
        ),
    )


def _default_source_name() -> str:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M")
    return f"neuraldata1_manual_capture_{stamp}"


def _resolve_source_name(raw: str | None) -> str:
    return accept.slugify(raw or _default_source_name())


def _report_paths(acceptance_root: Path, source_name: str) -> tuple[Path, Path, Path]:
    work_root = (REPO_ROOT / acceptance_root / source_name).resolve()
    return (
        work_root,
        work_root / "capture_report.json",
        work_root / "capture_report.md",
    )


def _write_capture_report(
    *,
    report: dict[str, Any],
    json_path: Path,
    md_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    counts = report["counts"]
    lines = [
        "# NeuralData1 Capture Acceptance",
        "",
        f"- Status: **{report['status']}**",
        f"- Training decision: **{report['training_decision']}**",
        f"- Training allowed: **{report['training_allowed']}**",
        f"- Project root: `{report['project_root']}`",
        f"- Raw export root: `{report.get('raw_export_root')}`",
        f"- Acceptance report: `{report.get('acceptance_report')}`",
        "",
        "## Current Export Counts",
        "",
        f"- Images: `{counts['images']}`",
        f"- keyPoint files: `{counts['keypoint_files']}`",
        f"- Depth files: `{counts['depth_files']}`",
        f"- Goal files: `{counts['goal_files']}`",
        f"- Unreal project files: `{counts['project_files']}`",
        f"- Quarantine files: `{counts.get('quarantine_files')}`",
        f"- Quarantine size: `{counts.get('quarantine_size')}`",
        "",
        "## Next Actions",
        "",
    ]
    for action in report["next_actions"]:
        lines.append(f"- {action}")
    if report.get("preview_paths"):
        lines += ["", "## Preview Paths", ""]
        for value in report["preview_paths"]:
            lines.append(f"- `{value}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _counts_to_dict(counts: ExportCounts) -> dict[str, Any]:
    return {
        "images": counts.images,
        "keypoint_files": counts.keypoint_files,
        "depth_files": counts.depth_files,
        "goal_files": counts.goal_files,
        "project_files": counts.project_files,
        "quarantine_files": counts.quarantine_files,
        "quarantine_size": counts.quarantine_size,
    }


def _blocked_no_export_report(
    *,
    args: argparse.Namespace,
    source_name: str,
    counts: ExportCounts,
    raw_export_root: Path,
    acceptance_report: Path | None,
) -> dict[str, Any]:
    return {
        "status": "BLOCKED_NO_CAPTURE_EXPORT",
        "training_decision": "NOT_APPROVED_FOR_TRAINING_NO_EXPORT",
        "training_allowed": False,
        "project_root": str(args.project_root.expanduser().resolve()),
        "raw_export_root": str(raw_export_root),
        "acceptance_report": str(acceptance_report) if acceptance_report else None,
        "counts": _counts_to_dict(counts),
        "next_actions": [
            "Open NeuralData.uproject in Unreal.",
            "Open map 01, 02, 03, or standartWheelsRoom.",
            "Verify CameraCaptureWheels is on the scene and Floor points to Plane/floor.",
            "Press Play for a short smoke capture.",
            "Re-run this script after Images and keyPoint contain files.",
        ],
        "preview_paths": [],
    }


def _copy_export(project_root: Path, raw_root: Path, overwrite: bool) -> None:
    if raw_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"raw export root already exists: {raw_root}; pass --overwrite"
            )
        shutil.rmtree(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    for name in EXPORT_DIRS:
        src = project_root / name
        if src.exists():
            shutil.copytree(src, raw_root / name)


def _run_acceptance(args: argparse.Namespace, raw_root: Path, source_name: str) -> int:
    command = [
        sys.executable,
        "scripts/accept_unreal_export.py",
        "--source-root",
        str(raw_root),
        "--source-name",
        source_name,
        "--out-root",
        str(args.acceptance_out_root),
        "--preview-count",
        str(args.preview_count),
        "--seed",
        str(args.seed),
        "--val-ratio",
        str(args.val_ratio),
        "--right-left-mapping",
        args.right_left_mapping,
        "--overwrite",
        # NeuralData1 capture smoke exports currently do not provide plugin BBox.
        # Keep this wrapper on the explicit debug fallback path until BBox lands.
        "--allow-synthetic-bbox",
    ]
    print("\n==> accept_neuraldata1_capture")
    print(" ".join(command))
    proc = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return proc.returncode


def _training_decision(
    acceptance: dict[str, Any],
    human_preview_accepted: bool,
) -> tuple[str, bool]:
    if acceptance.get("technical_status") != "PASS":
        return "NOT_APPROVED_FOR_TRAINING_TECHNICAL_GATE_FAILED", False
    if not (acceptance.get("data_quality_gate") or {}).get("passed"):
        return "NOT_APPROVED_FOR_TRAINING_DATA_QUALITY_GATE_FAILED", False
    if not human_preview_accepted:
        return "NOT_APPROVED_FOR_TRAINING_UNTIL_HUMAN_PREVIEW_ACCEPTS_GEOMETRY", False
    return "ACCEPT_FOR_TRAINING", True


def _accepted_report(
    *,
    args: argparse.Namespace,
    counts: ExportCounts,
    raw_root: Path,
    acceptance: dict[str, Any],
    acceptance_report_path: Path,
) -> dict[str, Any]:
    decision, allowed = _training_decision(
        acceptance, bool(args.human_preview_accepted)
    )
    status = (
        "ACCEPT_FOR_TRAINING"
        if allowed
        else (
            "READY_FOR_HUMAN_PREVIEW"
            if acceptance.get("technical_status") == "PASS"
            else "BLOCKED_ACCEPTANCE_FAILED"
        )
    )
    artifacts = acceptance.get("artifacts") or {}
    return {
        "status": status,
        "training_decision": decision,
        "training_allowed": allowed,
        "project_root": str(args.project_root.expanduser().resolve()),
        "raw_export_root": str(raw_root),
        "acceptance_report": str(acceptance_report_path),
        "counts": _counts_to_dict(counts),
        "acceptance": {
            "technical_status": acceptance.get("technical_status"),
            "review_status": acceptance.get("review_status"),
            "training_status": acceptance.get("training_status"),
            "data_quality_gate": acceptance.get("data_quality_gate"),
            "raw_export": acceptance.get("raw_export"),
            "import": acceptance.get("import"),
            "conversion": acceptance.get("conversion"),
        },
        "next_actions": (
            [
                "Open the preview paths and verify bbox/A/B/C geometry manually.",
                "If previews are correct, rerun with --human-preview-accepted to record ACCEPT_FOR_TRAINING.",
                "If previews fail, fix wheels actors in Unreal and capture again.",
            ]
            if not allowed
            else [
                "Training is allowed for this accepted batch.",
                "Start training in a separate run using the accepted pose_dataset path.",
            ]
        ),
        "preview_paths": [
            artifacts.get("incoming_preview_dir"),
            artifacts.get("pose_preview_dir"),
            artifacts.get("status_preview_root"),
        ],
    }


def run(args: argparse.Namespace) -> int:
    project_root = args.project_root.expanduser().resolve()
    if not project_root.is_dir():
        print(f"ERROR: project root not found: {project_root}", file=sys.stderr)
        return 2

    source_name = _resolve_source_name(args.source_name)
    raw_root = (REPO_ROOT / args.raw_out_root / source_name).resolve()
    work_root, capture_json, capture_md = _report_paths(
        args.acceptance_out_root, source_name
    )
    acceptance_report_path = work_root / "acceptance_report.json"
    counts = collect_counts(
        project_root,
        args.quarantine_root.expanduser().resolve()
        if args.quarantine_root is not None
        else None,
    )

    if not counts.has_required_export:
        report = _blocked_no_export_report(
            args=args,
            source_name=source_name,
            counts=counts,
            raw_export_root=raw_root,
            acceptance_report=None,
        )
        _write_capture_report(report=report, json_path=capture_json, md_path=capture_md)
        print(
            "BLOCKED: NeuralData1 has no generated capture export yet "
            f"(Images={counts.images}, keyPoint files={counts.keypoint_files})."
        )
        print(f"Capture report: {capture_json}")
        return 2

    try:
        _copy_export(project_root, raw_root, overwrite=args.overwrite)
    except FileExistsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    acceptance_rc = _run_acceptance(args, raw_root, source_name)
    acceptance = {}
    if acceptance_report_path.is_file():
        acceptance = json.loads(acceptance_report_path.read_text(encoding="utf-8"))

    report = _accepted_report(
        args=args,
        counts=counts,
        raw_root=raw_root,
        acceptance=acceptance,
        acceptance_report_path=acceptance_report_path,
    )
    _write_capture_report(report=report, json_path=capture_json, md_path=capture_md)

    print(f"Capture status: {report['status']}")
    print(f"Training decision: {report['training_decision']}")
    print(f"Capture report: {capture_json}")
    if acceptance_rc != 0:
        return acceptance_rc
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
