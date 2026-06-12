from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

from scripts.create_web_floor_evidence_request_bundle import (
    DEFAULT_MANIFEST_OUT,
    DEFAULT_OUT,
    build_manifest,
    build_template_files,
    csv_header,
    main,
    write_template_zip,
)
from web_floor_annotation_import import REQUIRED_COLUMNS


def test_csv_header_contains_required_and_optional_columns() -> None:
    header = csv_header().strip().split(",")

    for column in REQUIRED_COLUMNS:
        assert column in header
    assert "provenance_capture_date" in header
    assert "fov_mode" in header


def test_web_floor_request_bundle_zip_is_deterministic(tmp_path: Path) -> None:
    files = build_template_files(include_repo_docs=False)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    first_artifacts = write_template_zip(first, files)
    second_artifacts = write_template_zip(second, files)

    assert first.read_bytes() == second.read_bytes()
    assert first_artifacts == second_artifacts


def test_web_floor_request_bundle_contains_placeholders_and_commands(tmp_path: Path) -> None:
    out = tmp_path / "bundle.zip"
    artifacts = write_template_zip(out, build_template_files(include_repo_docs=False))
    manifest = build_manifest(out, artifacts)

    assert manifest["ok"] is True
    assert manifest["zip_sha256"] == manifest["bundle_sha256"]
    assert manifest["zip_size_bytes"] == manifest["bundle_size_bytes"]
    with zipfile.ZipFile(out) as archive:
        names = set(archive.namelist())
        readme = archive.read("README_WEB_FLOOR_EVIDENCE.md").decode("utf-8")
        template = archive.read("annotations/web_floor_annotations.csv").decode("utf-8")

    assert "README_WEB_FLOOR_EVIDENCE.md" in names
    assert "annotations/web_floor_annotations.csv" in names
    assert "annotations/web_floor_annotations_example.csv" in names
    assert "images/PLACE_REAL_FRAMES_HERE.txt" in names
    assert "scripts/import_web_floor_annotations.py" in readme
    assert "scripts/audit_web_floor_real_data.py" in readme
    assert template == csv_header()


def test_web_floor_request_bundle_can_include_repo_docs(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    configs = tmp_path / "configs"
    docs.mkdir()
    configs.mkdir()
    (docs / "WEB_FLOOR_REAL_DATA_INTAKE.md").write_text("intake doc\n", encoding="utf-8")
    (configs / "pose_dataset_web_floor_real_template.yaml").write_text("path: data/web_floor_real_v1\n", encoding="utf-8")

    files = build_template_files(repo_root=tmp_path, include_repo_docs=True)

    assert files["docs/WEB_FLOOR_REAL_DATA_INTAKE.md"] == "intake doc\n"
    assert files["configs/pose_dataset_web_floor_real_template.yaml"] == "path: data/web_floor_real_v1\n"


def test_web_floor_request_bundle_cli_writes_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "bundle.zip"
    manifest_out = tmp_path / "manifest.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "create_web_floor_evidence_request_bundle.py",
            "--out",
            str(out),
            "--manifest-out",
            str(manifest_out),
            "--no-repo-docs",
        ],
    )

    assert main() == 0
    manifest = json.loads(manifest_out.read_text(encoding="utf-8"))

    assert out.is_file()
    assert manifest["template"] == str(out)
    assert manifest["artifact_count"] >= 4
    assert manifest["zip_sha256"] == manifest["bundle_sha256"]


def test_web_floor_request_bundle_defaults_live_under_outputs() -> None:
    assert DEFAULT_OUT == Path("outputs/web_floor_network/web_floor_real_data_request_bundle.zip")
    assert DEFAULT_MANIFEST_OUT == Path("outputs/web_floor_network/web_floor_real_data_request_bundle_manifest.json")
