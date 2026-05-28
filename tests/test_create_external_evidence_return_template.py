from __future__ import annotations

import zipfile

from scripts.create_external_evidence_return_template import (
    build_template_files,
    build_manifest,
    write_template_zip,
)
from scripts.build_external_evidence_handoff_bundle import DEFAULT_BUNDLE_ARTIFACTS
from src.release_integrity import DEFAULT_REQUIRED_ARTIFACTS


def test_external_evidence_return_template_zip_is_deterministic(tmp_path):
    out_a = tmp_path / "a.zip"
    out_b = tmp_path / "b.zip"
    artifact = tmp_path / "model.tflite"
    artifact.write_bytes(b"model")

    artifacts_a = write_template_zip(out_a, expected_android_artifact=artifact)
    artifacts_b = write_template_zip(out_b, expected_android_artifact=artifact)

    assert out_a.read_bytes() == out_b.read_bytes()
    assert artifacts_a == artifacts_b
    with zipfile.ZipFile(out_a) as zf:
        assert zf.namelist() == sorted(build_template_files(artifact))
        assert zf.getinfo("README_RETURN_EVIDENCE.md").date_time == (1980, 1, 1, 0, 0, 0)


def test_external_evidence_return_template_contains_expected_placeholders(tmp_path):
    out = tmp_path / "template.zip"
    artifact = tmp_path / "model.tflite"
    artifact.write_bytes(b"model")
    write_template_zip(out, expected_android_artifact=artifact)

    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        expected_artifact = zf.read("EXPECTED_ANDROID_ARTIFACT.json").decode("utf-8")
        readme = zf.read("README_RETURN_EVIDENCE.md").decode("utf-8")

    assert "EXPECTED_ANDROID_ARTIFACT.json" in names
    assert "android_litert_device_report.json.PLACEHOLDER" in names
    assert "ar_device_holdout/images/PLACE_FRAMES_HERE.txt" in names
    assert "ar_device_holdout/annotations/PLACE_ANNOTATIONS_HERE.txt" in names
    assert "ar_device_holdout/metadata/provenance.json.PLACEHOLDER" in names
    assert "ar_3d_replay/ar_replay.jsonl.PLACEHOLDER" in names
    assert "model.tflite" in expected_artifact
    assert "model.tflite" in readme


def test_external_evidence_return_template_manifest_and_release_sets(tmp_path):
    out = tmp_path / "template.zip"
    artifact = tmp_path / "model.tflite"
    artifact.write_bytes(b"model")
    artifacts = write_template_zip(out, expected_android_artifact=artifact)
    manifest = build_manifest(out, artifacts, expected_android_artifact=artifact)

    assert manifest["ok"] is True
    assert manifest["artifact_count"] == len(build_template_files(artifact))
    assert manifest["zip_sha256"]
    assert manifest["expected_android_artifact"]["path"] == str(artifact)
    assert manifest["expected_android_artifact"]["sha256"]
    assert "src/import_external_evidence_drop.py" in manifest["next_command"]
    assert "outputs/production_audit/external_evidence_return_template.zip" in DEFAULT_BUNDLE_ARTIFACTS
    assert "outputs/production_audit/external_evidence_return_template_manifest.json" in DEFAULT_BUNDLE_ARTIFACTS
    assert "outputs/production_audit/external_evidence_return_template.zip" in DEFAULT_REQUIRED_ARTIFACTS
    assert "outputs/production_audit/external_evidence_return_template_manifest.json" in DEFAULT_REQUIRED_ARTIFACTS
