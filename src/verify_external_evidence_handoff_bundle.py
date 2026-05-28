"""Verify the Android/AR external-evidence handoff bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.build_external_evidence_handoff_bundle import DEFAULT_BUNDLE_ARTIFACTS
except ImportError:  # pragma: no cover - used when executed from unusual cwd
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.build_external_evidence_handoff_bundle import DEFAULT_BUNDLE_ARTIFACTS


DEFAULT_MANIFEST = Path("outputs/production_audit/external_evidence_handoff_bundle_manifest.json")
DEFAULT_REPORT_OUT = Path("outputs/production_audit/external_evidence_handoff_bundle_verification.json")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def integer_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def verify_bundle(
    manifest_path: Path, *, required_artifacts: Iterable[str] | None = None
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    bundle_path = Path(str(manifest.get("bundle", "")))
    failures: list[str] = []
    if not manifest_path.is_file():
        failures.append(f"missing_manifest:{manifest_path}")
    if not bundle_path.is_file():
        failures.append(f"missing_bundle:{bundle_path}")
        return {
            "schema_version": 1,
            "ok": False,
            "manifest": str(manifest_path),
            "bundle": str(bundle_path),
            "failures": failures,
        }

    actual_bundle_sha = sha256_file(bundle_path)
    expected_bundle_sha = manifest.get("bundle_sha256") or manifest.get("zip_sha256")
    if not expected_bundle_sha:
        failures.append("missing_bundle_sha256")
    elif actual_bundle_sha != expected_bundle_sha:
        failures.append("bundle_sha256_mismatch")
    expected_bundle_size = (
        manifest.get("bundle_size_bytes")
        if "bundle_size_bytes" in manifest
        else manifest.get("zip_size_bytes")
    )
    if not integer_count(expected_bundle_size) or expected_bundle_size <= 0:
        failures.append("missing_bundle_size_bytes")
    elif bundle_path.stat().st_size != expected_bundle_size:
        failures.append("bundle_size_bytes_mismatch")

    expected_artifacts = {
        str(artifact.get("path")): artifact
        for artifact in manifest.get("artifacts", [])
        if isinstance(artifact, dict) and artifact.get("path")
    }
    required = sorted(required_artifacts if required_artifacts is not None else DEFAULT_BUNDLE_ARTIFACTS)
    missing_required = sorted(set(required) - set(expected_artifacts))
    if missing_required:
        failures.extend(f"missing_required_artifact:{path}" for path in missing_required)

    manifest_artifact_count = manifest.get("artifact_count")
    if not integer_count(manifest_artifact_count):
        failures.append(f"invalid_manifest_artifact_count:{manifest_artifact_count}")
    elif manifest_artifact_count != len(expected_artifacts):
        failures.append("manifest_artifact_count_mismatch")

    for path, artifact in expected_artifacts.items():
        if artifact.get("exists") is not True:
            failures.append(f"manifest_artifact_not_existing:{path}")
        size_bytes = artifact.get("size_bytes")
        if not integer_count(size_bytes) or size_bytes <= 0:
            failures.append(f"manifest_artifact_empty:{path}")
        if not artifact.get("sha256"):
            failures.append(f"manifest_artifact_missing_sha256:{path}")

    current_artifact_results: list[dict[str, Any]] = []
    for path, artifact in expected_artifacts.items():
        current_path = Path(path)
        current_sha = sha256_file(current_path)
        current_size = current_path.stat().st_size if current_path.is_file() else None
        expected_sha = artifact.get("sha256")
        expected_size = artifact.get("size_bytes")
        ok = (
            current_sha == expected_sha
            and current_size == expected_size
            and isinstance(current_size, int)
            and current_size > 0
        )
        if current_sha is None:
            failures.append(f"current_artifact_missing:{path}")
        elif current_sha != expected_sha:
            failures.append(f"current_artifact_sha256_mismatch:{path}")
        if current_size != expected_size:
            failures.append(f"current_artifact_size_bytes_mismatch:{path}")
        current_artifact_results.append(
            {
                "path": path,
                "ok": ok,
                "sha256": current_sha,
                "expected_sha256": expected_sha,
                "size_bytes": current_size,
                "expected_size_bytes": expected_size,
            }
        )

    try:
        with zipfile.ZipFile(bundle_path) as zf:
            entries = sorted(info.filename for info in zf.infolist() if not info.is_dir())
            expected_entries = sorted(expected_artifacts)
            if entries != expected_entries:
                failures.append("zip_entries_mismatch")
            artifact_results: list[dict[str, Any]] = []
            for entry in entries:
                expected = expected_artifacts.get(entry, {})
                actual_sha = sha256_bytes(zf.read(entry))
                expected_sha = expected.get("sha256")
                ok = actual_sha == expected_sha
                if not ok:
                    failures.append(f"artifact_sha256_mismatch:{entry}")
                artifact_results.append(
                    {
                        "path": entry,
                        "ok": ok,
                        "sha256": actual_sha,
                        "expected_sha256": expected_sha,
                    }
                )
    except zipfile.BadZipFile:
        failures.append("bad_zip_file")
        artifact_results = []
        entries = []

    return {
        "schema_version": 1,
        "ok": not failures,
        "manifest": str(manifest_path),
        "bundle": str(bundle_path),
        "bundle_sha256": actual_bundle_sha,
        "expected_bundle_sha256": expected_bundle_sha,
        "bundle_size_bytes": bundle_path.stat().st_size,
        "expected_bundle_size_bytes": expected_bundle_size,
        "entry_count": len(entries),
        "expected_entry_count": len(expected_artifacts),
        "required_artifact_count": len(required),
        "missing_required_artifacts": missing_required,
        "failures": failures,
        "artifacts": artifact_results,
        "current_artifacts": current_artifact_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = verify_bundle(args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"ok={report['ok']} failures={report['failures']}")
    print(f"report={args.out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
