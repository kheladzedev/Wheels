"""Audit all YOLO-pose dataset configs used by the wheel model work.

This goes beyond a single dataset format check:

- resolves each YAML config to a dataset root;
- validates image/label pairing and YOLO-pose label schema;
- counts wheels per split;
- checks train/val leakage by image stem and SHA1 hash;
- samples image readability through OpenCV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from check_yolo_pose_dataset import IMAGE_EXTS, SPLITS, validate_label_file


DEFAULT_CONFIG_GLOB = "configs/pose_dataset*.yaml"
DEFAULT_JSON_OUT = Path("outputs/production_audit/dataset_audit.json")
DEFAULT_MD_OUT = Path("docs/DATASET_AUDIT.md")


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _rel(path: Path) -> str:
    return str(path).replace("\\", "/")


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def list_images(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        return []
    return sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def count_label_lines(label_path: Path) -> int:
    try:
        return sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0


def image_readability_sample(images: list[Path], limit: int) -> dict[str, Any]:
    sample = images[:limit] if limit > 0 else images
    unreadable: list[str] = []
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return {"checked": 0, "unreadable": [], "cv2_available": False}
    for path in sample:
        if cv2.imread(str(path), cv2.IMREAD_UNCHANGED) is None:
            unreadable.append(_rel(path))
    return {
        "checked": len(sample),
        "unreadable": unreadable,
        "cv2_available": True,
    }


def audit_split(root: Path, split: str, image_sample_limit: int) -> dict[str, Any]:
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    images = list_images(images_dir)
    labels = sorted(labels_dir.glob("*.txt")) if labels_dir.is_dir() else []
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}

    label_errors: list[str] = []
    empty_labels = 0
    wheel_labels = 0
    for image in images:
        label_path = labels_dir / f"{image.stem}.txt"
        if not label_path.is_file():
            continue
        if label_path.stat().st_size == 0:
            empty_labels += 1
        wheel_labels += count_label_lines(label_path)
        label_errors.extend(validate_label_file(label_path))

    image_hashes: dict[str, str] = {}
    for image in images:
        try:
            image_hashes[_sha1(image)] = _rel(image)
        except OSError:
            label_errors.append(f"{image}: cannot hash image")

    return {
        "split": split,
        "images_dir": _rel(images_dir),
        "labels_dir": _rel(labels_dir),
        "images": len(images),
        "labels": len(labels),
        "wheel_labels": wheel_labels,
        "missing_label_stems": sorted(image_stems - label_stems),
        "orphan_label_stems": sorted(label_stems - image_stems),
        "empty_labels": empty_labels,
        "label_errors": label_errors[:200],
        "label_error_count": len(label_errors),
        "image_readability": image_readability_sample(images, image_sample_limit),
        "image_stems": sorted(image_stems),
        "image_hashes": image_hashes,
    }


def resolve_dataset_root(config_path: Path, config: dict[str, Any]) -> Path:
    raw = config.get("path")
    if raw is None:
        return Path("")
    root = Path(str(raw))
    if root.is_absolute():
        return root
    return (config_path.parent.parent / root).resolve()


def resolve_report_path(value: object, *, config_path: Path, dataset_root: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    for base in (Path.cwd(), config_path.parent.parent, dataset_root.parent):
        candidate = base / path
        if candidate.exists():
            return candidate.resolve()
    return (config_path.parent.parent / path).resolve()


def conversion_report_consistency(
    config_path: Path,
    root: Path,
    split_reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    path = root / "metadata" / "conversion_report.json"
    report = _read_json(path)
    failures: list[str] = []
    if not path.is_file():
        return {
            "present": False,
            "path": _rel(path),
            "failures": [],
        }

    source_root = resolve_report_path(
        report.get("source_root", ""),
        config_path=config_path,
        dataset_root=root,
    )
    source_images_dir = source_root / "images"
    source_images = list_images(source_images_dir)
    expected_source_images = int(report.get("source_images", -1) or -1)
    converted_images = int(
        report.get("converted_images", report.get("converted", -1)) or -1
    )
    expected_wheels = int(report.get("wheels", -1) or -1)
    dataset_images = sum(split_reports[split]["images"] for split in SPLITS)
    dataset_wheels = sum(split_reports[split]["wheel_labels"] for split in SPLITS)
    quality_gate = report.get("quality_gate", {})

    if not source_root.is_dir():
        failures.append(f"conversion_source_root_missing:{source_root}")
    elif expected_source_images != len(source_images):
        failures.append(
            f"conversion_source_image_count_mismatch:{expected_source_images}!={len(source_images)}"
        )
    if converted_images != dataset_images:
        failures.append(f"conversion_dataset_image_count_mismatch:{converted_images}!={dataset_images}")
    if expected_wheels >= 0 and expected_wheels != dataset_wheels:
        failures.append(f"conversion_wheel_count_mismatch:{expected_wheels}!={dataset_wheels}")
    if isinstance(quality_gate, dict) and quality_gate.get("passed") is not True:
        failures.append(f"conversion_quality_gate_not_passed:{quality_gate.get('reasons', [])}")

    return {
        "present": True,
        "path": _rel(path),
        "source_root": _rel(source_root),
        "source_images_reported": expected_source_images,
        "source_images_current": len(source_images),
        "converted_images_reported": converted_images,
        "dataset_images_current": dataset_images,
        "wheels_reported": expected_wheels,
        "dataset_wheels_current": dataset_wheels,
        "quality_gate_passed": quality_gate.get("passed") if isinstance(quality_gate, dict) else None,
        "failures": failures,
    }


def audit_config(config_path: Path, image_sample_limit: int) -> dict[str, Any]:
    config = _read_yaml(config_path)
    root = resolve_dataset_root(config_path, config)
    failures: list[str] = []
    warnings: list[str] = []

    if not config:
        failures.append("invalid_or_empty_yaml")
    if config.get("kpt_shape") != [3, 3]:
        failures.append(f"unexpected_kpt_shape:{config.get('kpt_shape')}")
    if config.get("flip_idx") != [1, 0, 2]:
        failures.append(f"unexpected_flip_idx:{config.get('flip_idx')}")
    names = config.get("names", {})
    if names not in ({0: "wheel"}, {"0": "wheel"}):
        warnings.append(f"unexpected_names:{names}")
    if not root.is_dir():
        failures.append(f"dataset_root_missing:{root}")

    split_reports = {
        split: audit_split(root, split, image_sample_limit)
        for split in SPLITS
    }
    for split, report in split_reports.items():
        if report["images"] == 0:
            failures.append(f"{split}_has_no_images")
        if report["labels"] == 0:
            failures.append(f"{split}_has_no_labels")
        if report["missing_label_stems"]:
            failures.append(f"{split}_missing_labels:{len(report['missing_label_stems'])}")
        if report["orphan_label_stems"]:
            failures.append(f"{split}_orphan_labels:{len(report['orphan_label_stems'])}")
        if report["label_error_count"]:
            failures.append(f"{split}_label_errors:{report['label_error_count']}")
        unreadable = report["image_readability"]["unreadable"]
        if unreadable:
            failures.append(f"{split}_unreadable_images:{len(unreadable)}")

    train = split_reports["train"]
    val = split_reports["val"]
    stem_overlap = sorted(set(train["image_stems"]) & set(val["image_stems"]))
    hash_overlap = sorted(set(train["image_hashes"]) & set(val["image_hashes"]))
    if stem_overlap:
        failures.append(f"train_val_stem_overlap:{len(stem_overlap)}")
    if hash_overlap:
        failures.append(f"train_val_hash_overlap:{len(hash_overlap)}")

    conversion_consistency = conversion_report_consistency(config_path, root, split_reports)
    for failure in conversion_consistency["failures"]:
        failures.append(f"conversion_report:{failure}")

    # Avoid duplicating large helper fields in the final JSON.
    for report in split_reports.values():
        report.pop("image_stems", None)
        report.pop("image_hashes", None)

    return {
        "config": _rel(config_path),
        "root": _rel(root),
        "ok": not failures,
        "failures": failures,
        "warnings": warnings,
        "splits": split_reports,
        "conversion_report": conversion_consistency,
        "leakage": {
            "stem_overlap_count": len(stem_overlap),
            "stem_overlap_sample": stem_overlap[:20],
            "hash_overlap_count": len(hash_overlap),
            "hash_overlap_sample": hash_overlap[:20],
        },
    }


def build_audit(configs: list[Path], image_sample_limit: int) -> dict[str, Any]:
    reports = [audit_config(path, image_sample_limit) for path in sorted(configs)]
    return {
        "ok": all(report["ok"] for report in reports),
        "counts": {
            "configs": len(reports),
            "ok": sum(1 for report in reports if report["ok"]),
            "failed": sum(1 for report in reports if not report["ok"]),
            "total_train_images": sum(report["splits"]["train"]["images"] for report in reports),
            "total_val_images": sum(report["splits"]["val"]["images"] for report in reports),
            "total_wheel_labels": sum(
                split["wheel_labels"]
                for report in reports
                for split in report["splits"].values()
            ),
        },
        "reports": reports,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Dataset Audit",
        "",
        "Generated from `configs/pose_dataset*.yaml`.",
        "",
        "## Summary",
        "",
        f"- Overall OK: {audit['ok']}",
        f"- Configs: {audit['counts']['configs']}",
        f"- Passed: {audit['counts']['ok']}",
        f"- Failed: {audit['counts']['failed']}",
        f"- Total train images across configs: {audit['counts']['total_train_images']}",
        f"- Total val images across configs: {audit['counts']['total_val_images']}",
        f"- Total wheel label lines across configs: {audit['counts']['total_wheel_labels']}",
        "",
        "## Configs",
        "",
        "| Config | Root | OK | Train img / wheels | Val img / wheels | Leakage stem/hash | Failures |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for report in audit["reports"]:
        train = report["splits"]["train"]
        val = report["splits"]["val"]
        failures = ", ".join(report["failures"]) if report["failures"] else "none"
        lines.append(
            "| "
            f"`{report['config']}` | "
            f"`{report['root']}` | "
            f"{report['ok']} | "
            f"{train['images']} / {train['wheel_labels']} | "
            f"{val['images']} / {val['wheel_labels']} | "
            f"{report['leakage']['stem_overlap_count']} / {report['leakage']['hash_overlap_count']} | "
            f"{failures} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-glob", default=DEFAULT_CONFIG_GLOB)
    parser.add_argument("--image-sample-limit", type=int, default=50)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configs = sorted(Path(".").glob(args.config_glob))
    if not configs:
        raise FileNotFoundError(f"no configs matched {args.config_glob!r}")
    audit = build_audit(configs, args.image_sample_limit)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(audit), encoding="utf-8")
    print(
        f"ok={audit['ok']} configs={audit['counts']['configs']} "
        f"failed={audit['counts']['failed']} wheels={audit['counts']['total_wheel_labels']}"
    )
    print(f"json={args.json_out}")
    print(f"markdown={args.md_out}")
    return 0 if audit["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
