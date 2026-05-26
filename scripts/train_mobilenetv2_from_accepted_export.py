"""Guarded MobileNetV2 training entry point for accepted Unreal exports.

This wrapper is intended for the Mac Studio handoff. It refuses to train unless
the selected Unreal acceptance folder has passed the technical gate, data-quality
gate, and an explicit human preview flag is provided at launch time.

It does not relax the ML/AR contract and does not inspect Unreal directly. The
input must be an ``accept_unreal_export.py`` work directory containing:

    acceptance_report.json
    incoming/metadata/import_report.json
    pose_dataset/

Use ``--dry-run`` to print the exact training command without starting training.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_mobilenetv2_skipless.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train MobileNetV2 only from an accepted Unreal export folder."
    )
    p.add_argument(
        "--acceptance-root",
        required=True,
        type=Path,
        help="Acceptance work dir with acceptance_report.json and pose_dataset/.",
    )
    p.add_argument("--name", default=None, help="Training run name.")
    p.add_argument("--project", type=Path, default=Path("runs/pose_mn2"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="mps")
    p.add_argument("--lr", default=None)
    p.add_argument("--weight-decay", default=None)
    p.add_argument("--init-from", type=Path, default=None)
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument(
        "--no-pretrained",
        action="store_false",
        dest="pretrained",
        help="Disable torchvision ImageNet initialization.",
    )
    p.add_argument(
        "--human-preview-accepted",
        action="store_true",
        help="Required: confirms bbox/A/B/C previews were manually reviewed.",
    )
    p.add_argument(
        "--accept-synthetic-bbox-after-review",
        action="store_true",
        help=(
            "Allow training when bbox_source is synthesized_by_adapter. Use only "
            "after human preview explicitly accepts every sampled bbox."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate gates and print the command without launching training.",
    )
    return p.parse_args(argv)


def _json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"ERROR: required file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ERROR: invalid JSON: {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise SystemExit(f"ERROR: JSON root must be an object: {path}")
    return loaded


def _slug(raw: str) -> str:
    out = []
    for ch in raw:
        out.append(ch if ch.isalnum() or ch in "._-" else "_")
    slug = "".join(out).strip("._-")
    return slug or "mobilenetv2_unreal_accepted"


def _default_name(acceptance: dict[str, Any], acceptance_root: Path) -> str:
    source_name = acceptance.get("source_name")
    if isinstance(source_name, str) and source_name.strip():
        return f"mn2_{_slug(source_name)}_e50"
    return f"mn2_{_slug(acceptance_root.name)}_e50"


def validate_training_gate(args: argparse.Namespace) -> dict[str, Any]:
    acceptance_root = args.acceptance_root.expanduser().resolve()
    acceptance = _json(acceptance_root / "acceptance_report.json")
    import_report = _json(acceptance_root / "incoming" / "metadata" / "import_report.json")
    dataset_root = acceptance_root / "pose_dataset"

    failures: list[str] = []
    if acceptance.get("technical_status") != "PASS":
        failures.append(
            f"technical_status is {acceptance.get('technical_status')!r}, expected 'PASS'"
        )
    data_gate = acceptance.get("data_quality_gate") or {}
    if not isinstance(data_gate, dict) or not data_gate.get("passed"):
        failures.append("data_quality_gate.passed is not true")
    conversion_gate = ((acceptance.get("conversion") or {}).get("quality_gate") or {})
    if not isinstance(conversion_gate, dict) or not conversion_gate.get("passed"):
        failures.append("conversion.quality_gate.passed is not true")
    if not dataset_root.is_dir():
        failures.append(f"pose_dataset directory is missing: {dataset_root}")
    if not args.human_preview_accepted:
        failures.append("--human-preview-accepted is required")

    bbox_source_counts = import_report.get("bbox_source_counts") or {}
    plugin_bbox = int(bbox_source_counts.get("plugin_provided") or 0)
    synthesized_bbox = int(bbox_source_counts.get("synthesized_by_adapter") or 0)
    if plugin_bbox <= 0 and synthesized_bbox > 0 and not args.accept_synthetic_bbox_after_review:
        failures.append(
            "bbox is synthesized_by_adapter; pass "
            "--accept-synthetic-bbox-after-review only after explicit human review"
        )
    if plugin_bbox <= 0 and synthesized_bbox <= 0:
        failures.append("no plugin-provided or synthesized bbox count found")

    return {
        "acceptance_root": str(acceptance_root),
        "dataset_root": str(dataset_root),
        "source_name": acceptance.get("source_name"),
        "technical_status": acceptance.get("technical_status"),
        "training_status": acceptance.get("training_status"),
        "data_quality_gate_passed": bool(data_gate.get("passed")),
        "conversion_quality_gate_passed": bool(conversion_gate.get("passed")),
        "bbox_source_counts": bbox_source_counts,
        "human_preview_accepted": bool(args.human_preview_accepted),
        "accept_synthetic_bbox_after_review": bool(
            args.accept_synthetic_bbox_after_review
        ),
        "failures": failures,
    }


def build_train_command(args: argparse.Namespace, gate: dict[str, Any]) -> list[str]:
    acceptance_root = args.acceptance_root.expanduser().resolve()
    name = args.name or _default_name({"source_name": gate.get("source_name")}, acceptance_root)
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--dataset-root",
        gate["dataset_root"],
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--device",
        args.device,
        "--imgsz",
        str(args.imgsz),
        "--num-workers",
        str(args.num_workers),
        "--project",
        str(args.project),
        "--name",
        name,
    ]
    if args.pretrained:
        cmd.append("--pretrained")
    if args.lr is not None:
        cmd += ["--lr", str(args.lr)]
    if args.weight_decay is not None:
        cmd += ["--weight-decay", str(args.weight_decay)]
    if args.init_from is not None:
        cmd += ["--init-from", str(args.init_from.expanduser())]
    return cmd


def run(args: argparse.Namespace) -> int:
    gate = validate_training_gate(args)
    if gate["failures"]:
        print("Training gate: FAIL")
        for failure in gate["failures"]:
            print(f"- {failure}")
        print(json.dumps(gate, indent=2))
        return 2

    cmd = build_train_command(args, gate)
    print("Training gate: PASS")
    print("Command:")
    print(" ".join(cmd))
    if args.dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
