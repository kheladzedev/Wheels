from __future__ import annotations

import zipfile
from pathlib import Path

from scripts.build_mobile_onnx_handoff import (
    build_manifest,
    handoff_paths,
    render_markdown,
    write_zip,
)


def _signature() -> dict:
    return {
        "ok": True,
        "inputs": [{"name": "images", "shape": [1, 3, 384, 384], "elem_type": 1}],
        "outputs": [{"name": "output0", "shape": [1, 14, 3024], "elem_type": 1}],
        "opsets": [{"domain": "ai.onnx", "version": 20}],
        "node_count": 359,
    }


def test_mobile_onnx_manifest_passes_model_signature_and_smoke(tmp_path: Path):
    model = tmp_path / "best_mobile_384.onnx"
    smoke = tmp_path / "smoke.json"
    quality = tmp_path / "quality.json"
    model.write_bytes(b"onnx")
    smoke.write_text("{}", encoding="utf-8")
    quality.write_text("{}", encoding="utf-8")

    manifest = build_manifest(
        model_path=model,
        source_onnx=tmp_path / "source.onnx",
        signature=_signature(),
        smoke_report={"ok": True, "latency_ms": {"avg": 1.0}},
        smoke_report_path=smoke,
        quality_report=quality,
    )

    assert manifest["ok"] is True
    assert manifest["failures"] == []
    assert manifest["policy"]["target_runtime"] == "ONNX Runtime Mobile"
    assert manifest["artifacts"][0]["format"] == "onnx"
    assert manifest["artifacts"][0]["sha256"]


def test_mobile_onnx_manifest_fails_bad_signature(tmp_path: Path):
    model = tmp_path / "best_mobile_384.onnx"
    model.write_bytes(b"onnx")

    manifest = build_manifest(
        model_path=model,
        source_onnx=tmp_path / "source.onnx",
        signature={"ok": False, "error": "bad model"},
        smoke_report={"ok": True},
        smoke_report_path=None,
        quality_report=None,
    )

    assert manifest["ok"] is False
    assert manifest["failures"] == ["onnx_signature_failed"]


def test_handoff_paths_include_model_smoke_quality_and_docs(tmp_path: Path):
    model = tmp_path / "best_mobile_384.onnx"
    smoke = tmp_path / "smoke.json"
    quality = tmp_path / "quality.json"
    manifest_path = tmp_path / "manifest.json"
    markdown_path = tmp_path / "handoff.md"
    for path in (model, smoke, quality):
        path.write_bytes(b"x")
    manifest = build_manifest(
        model_path=model,
        source_onnx=tmp_path / "source.onnx",
        signature=_signature(),
        smoke_report={"ok": True},
        smoke_report_path=smoke,
        quality_report=quality,
    )

    paths = handoff_paths(manifest, manifest_path, markdown_path)

    assert paths == [model, smoke, quality, manifest_path, markdown_path]


def test_mobile_onnx_zip_is_deterministic(tmp_path: Path):
    first = tmp_path / "b.txt"
    second = tmp_path / "a.txt"
    first.write_text("b", encoding="utf-8")
    second.write_text("a", encoding="utf-8")
    out_a = tmp_path / "a.zip"
    out_b = tmp_path / "b.zip"

    write_zip([first, second], out_a)
    write_zip([second, first], out_b)

    assert out_a.read_bytes() == out_b.read_bytes()
    with zipfile.ZipFile(out_a) as zf:
        assert zf.namelist() == [str(second), str(first)]
        assert zf.getinfo(str(second)).date_time == (1980, 1, 1, 0, 0, 0)


def test_render_markdown_states_not_production_promotion(tmp_path: Path):
    model = tmp_path / "best_mobile_384.onnx"
    model.write_bytes(b"onnx")
    manifest = build_manifest(
        model_path=model,
        source_onnx=tmp_path / "source.onnx",
        signature=_signature(),
        smoke_report={"ok": True, "latency_ms": {"avg": 1.0}},
        smoke_report_path=None,
        quality_report=None,
    )

    markdown = render_markdown(manifest)

    assert "# Mobile ONNX Handoff" in markdown
    assert "ONNX Runtime Mobile" in markdown
    assert "not a production promotion" in markdown
