from __future__ import annotations

from src.coreml_certification import build_certification


def test_coreml_certification_fails_missing_artifact(tmp_path):
    report = build_certification(
        coreml_artifact=tmp_path / "missing.mlmodel",
        pytorch_artifact=tmp_path / "best.pt",
    )

    assert report["certified"] is False
    assert "missing_coreml_artifact" in report["failures"]


def test_coreml_certification_accepts_valid_coreml_summary(tmp_path, monkeypatch):
    coreml_artifact = tmp_path / "best.mlmodel"
    pytorch_artifact = tmp_path / "best.pt"
    coreml_artifact.write_bytes(b"coreml")
    pytorch_artifact.write_bytes(b"pytorch")

    monkeypatch.setattr(
        "src.coreml_certification.load_coreml_spec_summary",
        lambda _path: {
            "specification_version": 4,
            "inputs": [{"name": "image", "kind": "image", "width": 640, "height": 640}],
            "outputs": [{"name": "var_1347", "kind": "multiArrayType", "shape": []}],
        },
    )

    report = build_certification(
        coreml_artifact=coreml_artifact,
        pytorch_artifact=pytorch_artifact,
    )

    assert report["certified"] is True
    assert report["format"] == "coreml_mlmodel_neuralnetwork"
    assert report["artifact"]["sha256"]
    assert report["pytorch_reference"]["sha256"]
    assert report["failures"] == []
