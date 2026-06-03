from __future__ import annotations

import zipfile

from scripts.build_ios_coreml_handoff import (
    ZIP_TIMESTAMP,
    build_manifest,
    package_files,
    write_zip,
)


def test_ios_handoff_manifest_requires_model_and_compiled_bundle(tmp_path):
    root = tmp_path / "handoff"
    root.mkdir()
    (root / "README.md").write_text("readme", encoding="utf-8")
    (root / "WheelsCoreMLSmoke.swift").write_text("swift", encoding="utf-8")
    (root / "COREML_CERTIFICATION.md").write_text("cert", encoding="utf-8")
    (root / "AR_ML_CONTRACT.md").write_text("contract", encoding="utf-8")

    manifest = build_manifest(root, tmp_path / "handoff.zip")

    assert manifest["ok"] is False
    assert "missing:best.mlmodel" in manifest["failures"]
    assert "missing:best.mlmodelc/model.espresso.net" in manifest["failures"]


def test_ios_handoff_zip_is_deterministic(tmp_path):
    root = tmp_path / "handoff"
    compiled = root / "best.mlmodelc"
    compiled.mkdir(parents=True)
    (root / "README.md").write_text("readme", encoding="utf-8")
    (compiled / "model.espresso.net").write_text("net", encoding="utf-8")
    (compiled / "model.espresso.weights").write_text("weights", encoding="utf-8")
    out_a = tmp_path / "a.zip"
    out_b = tmp_path / "b.zip"

    write_zip(root, out_a)
    write_zip(root, out_b)

    assert out_a.read_bytes() == out_b.read_bytes()
    with zipfile.ZipFile(out_a) as zf:
        assert zf.namelist() == [
            "README.md",
            "best.mlmodelc/model.espresso.net",
            "best.mlmodelc/model.espresso.weights",
        ]
        assert zf.getinfo("README.md").date_time == ZIP_TIMESTAMP


def test_ios_handoff_package_files_are_sorted(tmp_path):
    root = tmp_path / "handoff"
    nested = root / "z"
    nested.mkdir(parents=True)
    (nested / "b.txt").write_text("b", encoding="utf-8")
    (root / "a.txt").write_text("a", encoding="utf-8")

    names = [path.relative_to(root).as_posix() for path in package_files(root)]

    assert names == ["a.txt", "z/b.txt"]
