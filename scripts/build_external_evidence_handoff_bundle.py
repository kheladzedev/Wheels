"""Build a deterministic Android/AR external-evidence handoff bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path("outputs/production_audit/external_evidence_handoff_bundle.zip")
DEFAULT_MANIFEST_OUT = Path("outputs/production_audit/external_evidence_handoff_bundle_manifest.json")
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

DEFAULT_BUNDLE_ARTIFACTS = [
    "outputs/production_audit/tflite_export/best_float32.tflite",
    "outputs/production_audit/coreml_export/best.mlmodel",
    "outputs/production_audit/coreml_certification.json",
    "outputs/production_audit/data_readiness_decision.json",
    "outputs/production_audit/android_litert_device_report.template.json",
    "outputs/production_audit/ar_device_holdout_provenance.template.json",
    "outputs/production_audit/ar_3d_replay.template.jsonl",
    "outputs/production_audit/external_evidence_return_template.zip",
    "outputs/production_audit/external_evidence_return_template_manifest.json",
    "android_litert_harness/README.md",
    "android_litert_harness/AndroidLiteRtDeviceValidationTest.kt",
    "ar_holdout_harness/README.md",
    "ar_holdout_harness/ArHoldoutAnnotationWriter.kt",
    "ar_replay_harness/README.md",
    "ar_replay_harness/ArReplayLogger.kt",
    "docs/ANDROID_LITERT_DEVICE_REPORT.md",
    "docs/COREML_CERTIFICATION.md",
    "docs/DATA_READINESS_DECISION.md",
    "ios_coreml_handoff/README.md",
    "ios_coreml_handoff/WheelsCoreMLSmoke.swift",
    "docs/AR_ML_CONTRACT.md",
    "docs/AR_MOCK_LOG_CONTRACT.md",
    "docs/AR_REPLAY_METRIC_PLAN.md",
    "docs/PRODUCTION_EVIDENCE_CHECKLIST.md",
    "docs/PRODUCTION_EVIDENCE_INTAKE.md",
    "src/validate_android_litert_report.py",
    "src/evaluate_ar_holdout.py",
    "src/validate_ar_replay.py",
    "src/eval_ar_replay_metric.py",
    "src/production_evidence_audit.py",
    "src/import_external_evidence_drop.py",
    "src/run_production_evidence_intake.py",
    "scripts/create_external_evidence_return_template.py",
    "scripts/build_ios_coreml_handoff.py",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_artifact(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    size_bytes = path.stat().st_size if exists else 0
    return {
        "path": str(path).replace("\\", "/"),
        "exists": exists,
        "size_bytes": size_bytes,
        "sha256": sha256_file(path) if exists and size_bytes > 0 else None,
    }


def build_manifest(paths: list[Path], bundle_path: Path) -> dict[str, Any]:
    artifacts = [inspect_artifact(path) for path in paths]
    failures = [
        f"missing:{artifact['path']}"
        for artifact in artifacts
        if not artifact["exists"] or artifact["size_bytes"] <= 0
    ]
    return {
        "schema_version": 1,
        "ok": not failures,
        "bundle": str(bundle_path).replace("\\", "/"),
        "artifact_count": len(artifacts),
        "failures": failures,
        "artifacts": artifacts,
    }


def write_zip(paths: list[Path], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(paths, key=lambda p: str(p).replace("\\", "/")):
            arcname = str(path).replace("\\", "/")
            info = zipfile.ZipInfo(arcname, date_time=ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = [Path(path) for path in (args.artifact or DEFAULT_BUNDLE_ARTIFACTS)]
    manifest = build_manifest(paths, args.out)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not manifest["ok"]:
        print(f"ok=False failures={manifest['failures']}")
        print(f"manifest={args.manifest_out}")
        return 1
    write_zip(paths, args.out)
    bundle_sha = sha256_file(args.out)
    manifest["bundle_sha256"] = bundle_sha
    manifest["bundle_size_bytes"] = args.out.stat().st_size
    manifest["zip_sha256"] = bundle_sha
    manifest["zip_size_bytes"] = args.out.stat().st_size
    args.manifest_out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"ok=True bundle={args.out} artifacts={manifest['artifact_count']} sha256={bundle_sha}")
    print(f"manifest={args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
