from __future__ import annotations

import zipfile

from scripts.build_mobile_6mb_handoff import (
    ArtifactSpec,
    DEFAULT_ARTIFACTS,
    MAX_MODEL_SIZE_MB,
    build_manifest,
    platform_handoff_paths,
    write_zip,
)


def test_mobile_6mb_manifest_passes_artifacts_under_limit(tmp_path):
    model = tmp_path / "model.tflite"
    report = tmp_path / "report.json"
    model.write_bytes(b"a" * 128)
    report.write_text("{}", encoding="utf-8")

    manifest = build_manifest(
        [
            ArtifactSpec(
                path=model,
                platform="android",
                role="model",
                max_size_mb=MAX_MODEL_SIZE_MB,
            ),
            ArtifactSpec(path=report, platform="android", role="validation"),
        ]
    )

    assert manifest["ok"] is True
    assert manifest["failures"] == []
    assert manifest["artifacts"][0]["within_size_limit"] is True


def test_mobile_6mb_manifest_fails_model_over_limit(tmp_path):
    model = tmp_path / "too_large.mlmodel"
    model.write_bytes(b"a" * (6 * 1024 * 1024 + 1))

    manifest = build_manifest(
        [
            ArtifactSpec(
                path=model,
                platform="ios",
                role="model",
                max_size_mb=MAX_MODEL_SIZE_MB,
            )
        ]
    )

    assert manifest["ok"] is False
    assert manifest["failures"] == ["over_size:too_large.mlmodel"]
    assert manifest["artifacts"][0]["within_size_limit"] is False


def test_mobile_6mb_zip_is_deterministic(tmp_path):
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


def test_platform_handoff_paths_filters_by_platform(tmp_path):
    android = tmp_path / "android.tflite"
    ios = tmp_path / "ios.mlmodel"
    shared = tmp_path / "quality.json"
    manifest_path = tmp_path / "manifest.json"
    markdown_path = tmp_path / "handoff.md"
    android.write_bytes(b"a")
    ios.write_bytes(b"i")
    shared.write_text("{}", encoding="utf-8")
    manifest = build_manifest(
        [
            ArtifactSpec(path=android, platform="android", role="model"),
            ArtifactSpec(path=ios, platform="ios", role="model"),
            ArtifactSpec(path=shared, platform="shared", role="quality_reference"),
        ]
    )

    paths = platform_handoff_paths(manifest, "android", manifest_path, markdown_path)

    assert paths == [android, shared, manifest_path, markdown_path]


def test_default_artifacts_include_android_and_ios_models():
    paths = {artifact.path.as_posix() for artifact in DEFAULT_ARTIFACTS}

    assert "outputs/production_audit/mobile_6mb/tflite_nano_fp16_384/best_float16.tflite" in paths
    assert "outputs/production_audit/mobile_6mb/coreml_nano_int8_384/best_int8.mlmodel" in paths
    assert "outputs/production_audit/mobile_6mb/coreml_nano_linear4_384/best_linear4.mlmodel" in paths
    assert "outputs/production_audit/mobile_6mb/nano_source_eval_self_plus_ue_conf025.json" in paths
