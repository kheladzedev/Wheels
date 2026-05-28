from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

from scripts.build_external_evidence_handoff_bundle import (
    DEFAULT_BUNDLE_ARTIFACTS,
    build_manifest,
    sha256_file,
    write_zip,
)
from src.verify_external_evidence_handoff_bundle import verify_bundle


def test_verify_external_evidence_bundle_passes_valid_manifest(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("payload", encoding="utf-8")
    bundle = tmp_path / "bundle.zip"
    manifest_path = tmp_path / "manifest.json"
    write_zip([artifact], bundle)
    manifest = build_manifest([artifact], bundle)
    manifest["bundle_sha256"] = sha256_file(bundle)
    manifest["bundle_size_bytes"] = bundle.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = verify_bundle(manifest_path, required_artifacts=[str(artifact)])

    assert report["ok"] is True
    assert report["failures"] == []
    assert report["bundle_sha256"] == report["expected_bundle_sha256"]
    assert report["bundle_size_bytes"] == report["expected_bundle_size_bytes"]
    assert report["entry_count"] == 1
    assert report["missing_required_artifacts"] == []


def test_verify_external_evidence_bundle_detects_bad_artifact_hash(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("payload", encoding="utf-8")
    bundle = tmp_path / "bundle.zip"
    manifest_path = tmp_path / "manifest.json"
    write_zip([artifact], bundle)
    manifest = build_manifest([artifact], bundle)
    manifest["bundle_sha256"] = sha256_file(bundle)
    manifest["bundle_size_bytes"] = bundle.stat().st_size
    manifest["artifacts"][0]["sha256"] = "bad"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = verify_bundle(manifest_path, required_artifacts=[])

    assert report["ok"] is False
    assert any(failure.startswith("artifact_sha256_mismatch") for failure in report["failures"])
    assert any(
        failure.startswith("current_artifact_sha256_mismatch")
        for failure in report["failures"]
    )


def test_verify_external_evidence_bundle_rejects_stale_current_artifact(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("payload", encoding="utf-8")
    bundle = tmp_path / "bundle.zip"
    manifest_path = tmp_path / "manifest.json"
    write_zip([artifact], bundle)
    manifest = build_manifest([artifact], bundle)
    manifest["bundle_sha256"] = sha256_file(bundle)
    manifest["bundle_size_bytes"] = bundle.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    artifact.write_text("changed-after-bundle-build", encoding="utf-8")

    report = verify_bundle(manifest_path, required_artifacts=[str(artifact)])

    assert report["ok"] is False
    assert any(
        failure.startswith("current_artifact_sha256_mismatch")
        for failure in report["failures"]
    )
    assert any(
        failure.startswith("current_artifact_size_bytes_mismatch")
        for failure in report["failures"]
    )
    assert report["current_artifacts"][0]["ok"] is False


def test_verify_external_evidence_bundle_requires_bundle_hash_and_size(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("payload", encoding="utf-8")
    bundle = tmp_path / "bundle.zip"
    manifest_path = tmp_path / "manifest.json"
    write_zip([artifact], bundle)
    manifest = build_manifest([artifact], bundle)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = verify_bundle(manifest_path, required_artifacts=[str(artifact)])

    assert report["ok"] is False
    assert "missing_bundle_sha256" in report["failures"]
    assert "missing_bundle_size_bytes" in report["failures"]


def test_verify_external_evidence_bundle_accepts_zip_hash_aliases(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("payload", encoding="utf-8")
    bundle = tmp_path / "bundle.zip"
    manifest_path = tmp_path / "manifest.json"
    write_zip([artifact], bundle)
    manifest = build_manifest([artifact], bundle)
    manifest["zip_sha256"] = sha256_file(bundle)
    manifest["zip_size_bytes"] = bundle.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = verify_bundle(manifest_path, required_artifacts=[str(artifact)])

    assert report["ok"] is True


def test_verify_external_evidence_bundle_detects_entry_mismatch(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("payload", encoding="utf-8")
    bundle = tmp_path / "bundle.zip"
    manifest_path = tmp_path / "manifest.json"
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("unexpected.txt", "payload")
    manifest = build_manifest([artifact], bundle)
    manifest["bundle_sha256"] = sha256_file(bundle)
    manifest["bundle_size_bytes"] = bundle.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = verify_bundle(manifest_path, required_artifacts=[])

    assert report["ok"] is False
    assert "zip_entries_mismatch" in report["failures"]


def test_verify_external_evidence_bundle_requires_canonical_artifacts(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("payload", encoding="utf-8")
    bundle = tmp_path / "bundle.zip"
    manifest_path = tmp_path / "manifest.json"
    write_zip([artifact], bundle)
    manifest = build_manifest([artifact], bundle)
    manifest["bundle_sha256"] = sha256_file(bundle)
    manifest["bundle_size_bytes"] = bundle.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = verify_bundle(manifest_path, required_artifacts=DEFAULT_BUNDLE_ARTIFACTS)

    assert report["ok"] is False
    assert report["missing_required_artifacts"]
    assert any(
        failure.startswith("missing_required_artifact:")
        for failure in report["failures"]
    )


def test_verify_external_evidence_bundle_rejects_incomplete_manifest_entry(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("payload", encoding="utf-8")
    bundle = tmp_path / "bundle.zip"
    manifest_path = tmp_path / "manifest.json"
    write_zip([artifact], bundle)
    manifest = build_manifest([artifact], bundle)
    manifest["bundle_sha256"] = sha256_file(bundle)
    manifest["bundle_size_bytes"] = bundle.stat().st_size
    manifest["artifact_count"] = 99
    manifest["artifacts"][0]["exists"] = False
    manifest["artifacts"][0]["size_bytes"] = 0
    manifest["artifacts"][0]["sha256"] = ""
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = verify_bundle(manifest_path, required_artifacts=[str(artifact)])

    assert report["ok"] is False
    assert "manifest_artifact_count_mismatch" in report["failures"]
    assert any(
        failure.startswith("manifest_artifact_not_existing:")
        for failure in report["failures"]
    )
    assert any(
        failure.startswith("manifest_artifact_empty:")
        for failure in report["failures"]
    )
    assert any(
        failure.startswith("manifest_artifact_missing_sha256:")
        for failure in report["failures"]
    )


def test_verify_external_evidence_bundle_rejects_boolean_counts_and_sizes(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("payload", encoding="utf-8")
    bundle = tmp_path / "bundle.zip"
    manifest_path = tmp_path / "manifest.json"
    write_zip([artifact], bundle)
    manifest = build_manifest([artifact], bundle)
    manifest["bundle_sha256"] = sha256_file(bundle)
    manifest["bundle_size_bytes"] = True
    manifest["artifact_count"] = True
    manifest["artifacts"][0]["size_bytes"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = verify_bundle(manifest_path, required_artifacts=[str(artifact)])

    assert report["ok"] is False
    assert "missing_bundle_size_bytes" in report["failures"]
    assert "invalid_manifest_artifact_count:True" in report["failures"]
    assert any(
        failure.startswith("manifest_artifact_empty:")
        for failure in report["failures"]
    )
    assert any(
        failure.startswith("current_artifact_size_bytes_mismatch:")
        for failure in report["failures"]
    )


def test_verify_external_evidence_bundle_cli_uses_canonical_required_artifacts(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("payload", encoding="utf-8")
    bundle = tmp_path / "bundle.zip"
    manifest_path = tmp_path / "manifest.json"
    report_path = tmp_path / "report.json"
    write_zip([artifact], bundle)
    manifest = build_manifest([artifact], bundle)
    manifest["bundle_sha256"] = sha256_file(bundle)
    manifest["bundle_size_bytes"] = bundle.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    script = Path(__file__).resolve().parents[1] / "src" / "verify_external_evidence_handoff_bundle.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest_path),
            "--out",
            str(report_path),
        ],
        check=False,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert completed.returncode == 1
    assert report["required_artifact_count"] == len(DEFAULT_BUNDLE_ARTIFACTS)
    assert report["missing_required_artifacts"]
