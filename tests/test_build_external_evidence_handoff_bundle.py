from __future__ import annotations

import json
import zipfile

from scripts.build_external_evidence_handoff_bundle import (
    DEFAULT_BUNDLE_ARTIFACTS,
    build_manifest,
    main,
    write_zip,
)


def test_external_evidence_bundle_manifest_fails_missing_artifact(tmp_path):
    existing = tmp_path / "a.txt"
    existing.write_text("a", encoding="utf-8")
    missing = tmp_path / "missing.txt"

    manifest = build_manifest([existing, missing], tmp_path / "bundle.zip")

    assert manifest["ok"] is False
    assert f"missing:{missing}" in manifest["failures"]


def test_external_evidence_bundle_zip_is_deterministic_and_contains_paths(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "nested" / "second.txt"
    second.parent.mkdir()
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    out_a = tmp_path / "a.zip"
    out_b = tmp_path / "b.zip"

    write_zip([second, first], out_a)
    write_zip([first, second], out_b)

    assert out_a.read_bytes() == out_b.read_bytes()
    with zipfile.ZipFile(out_a) as zf:
        assert zf.namelist() == [str(first), str(second)]
        assert zf.getinfo(str(first)).date_time == (1980, 1, 1, 0, 0, 0)


def test_external_evidence_bundle_cli_manifest_has_zip_hash_aliases(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("payload", encoding="utf-8")
    bundle = tmp_path / "bundle.zip"
    manifest_path = tmp_path / "manifest.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_external_evidence_handoff_bundle.py",
            "--artifact",
            str(artifact),
            "--out",
            str(bundle),
            "--manifest-out",
            str(manifest_path),
        ],
    )

    assert main() == 0

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["bundle_sha256"] == manifest["zip_sha256"]
    assert manifest["bundle_size_bytes"] == manifest["zip_size_bytes"]


def test_external_evidence_bundle_includes_ios_coreml_artifact():
    assert "outputs/production_audit/coreml_export/best.mlmodel" in DEFAULT_BUNDLE_ARTIFACTS
    assert "outputs/production_audit/coreml_certification.json" in DEFAULT_BUNDLE_ARTIFACTS
    assert "docs/COREML_CERTIFICATION.md" in DEFAULT_BUNDLE_ARTIFACTS
    assert "ios_coreml_handoff/README.md" in DEFAULT_BUNDLE_ARTIFACTS
    assert "ios_coreml_handoff/WheelsCoreMLSmoke.swift" in DEFAULT_BUNDLE_ARTIFACTS
    assert "scripts/build_ios_coreml_handoff.py" in DEFAULT_BUNDLE_ARTIFACTS


def test_external_evidence_bundle_includes_data_readiness_decision():
    assert "outputs/production_audit/data_readiness_decision.json" in DEFAULT_BUNDLE_ARTIFACTS
    assert "docs/DATA_READINESS_DECISION.md" in DEFAULT_BUNDLE_ARTIFACTS


def test_external_evidence_bundle_includes_ar_replay_metric():
    assert "src/validate_ar_replay.py" in DEFAULT_BUNDLE_ARTIFACTS
    assert "src/eval_ar_replay_metric.py" in DEFAULT_BUNDLE_ARTIFACTS
    assert "docs/AR_REPLAY_METRIC_PLAN.md" in DEFAULT_BUNDLE_ARTIFACTS
