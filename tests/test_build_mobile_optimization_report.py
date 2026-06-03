from __future__ import annotations

import json
import zipfile

from scripts.build_mobile_optimization_report import (
    BaselineSpec,
    CandidateSpec,
    DEFAULT_CANDIDATES,
    build_report,
    handoff_paths,
    materialize_candidates,
    render_markdown,
    write_zip,
)


def test_mobile_optimization_report_computes_compression_ratio(tmp_path):
    baseline = tmp_path / "best_float32.tflite"
    candidate = tmp_path / "best_float16.tflite"
    baseline.write_bytes(b"a" * 100)
    candidate.write_bytes(b"b" * 25)

    report = build_report(
        baselines=[
            BaselineSpec(
                id="tflite_float32_640",
                platform="android",
                path=baseline,
                input_shape=[1, 640, 640, 3],
                output_shape=[1, 14, 8400],
            )
        ],
        candidates=[
            CandidateSpec(
                id="tflite_fp16_640",
                baseline_id="tflite_float32_640",
                platform="android",
                precision="fp16",
                source_path=candidate,
                target_path=tmp_path / "out" / "best_float16.tflite",
                input_shape=[1, 640, 640, 3],
                output_shape=[1, 14, 8400],
                required=True,
            )
        ],
    )

    assert report["ok"] is True
    assert report["candidates"][0]["status"] == "ready"
    assert report["candidates"][0]["size_bytes"] == 25
    assert report["candidates"][0]["compression_ratio_vs_baseline"] == 4.0
    assert report["candidates"][0]["path"] == str(tmp_path / "out" / "best_float16.tflite")
    assert report["candidates"][0]["failures"] == []


def test_mobile_optimization_report_marks_missing_required_candidate(tmp_path):
    baseline = tmp_path / "best_float32.tflite"
    baseline.write_bytes(b"a" * 100)

    report = build_report(
        baselines=[
            BaselineSpec(
                id="tflite_float32_640",
                platform="android",
                path=baseline,
                input_shape=[1, 640, 640, 3],
                output_shape=[1, 14, 8400],
            )
        ],
        candidates=[
            CandidateSpec(
                id="tflite_fp16_640",
                baseline_id="tflite_float32_640",
                platform="android",
                precision="fp16",
                source_path=tmp_path / "missing.tflite",
                target_path=tmp_path / "out" / "missing.tflite",
                input_shape=[1, 640, 640, 3],
                output_shape=[1, 14, 8400],
                required=True,
            )
        ],
    )

    assert report["ok"] is False
    assert report["failures"] == ["missing_candidate:tflite_fp16_640"]
    assert report["candidates"][0]["status"] == "missing"


def test_materialize_candidates_copies_ready_artifacts_and_writes_reports(tmp_path):
    baseline = tmp_path / "best_float32.tflite"
    source = tmp_path / "source" / "best_float16.tflite"
    target = tmp_path / "out" / "best_float16.tflite"
    manifest = tmp_path / "mobile_optimization_report.json"
    markdown = tmp_path / "MOBILE_OPTIMIZATION_REPORT.md"
    zip_out = tmp_path / "mobile_optimization_handoff.zip"
    source.parent.mkdir()
    baseline.write_bytes(b"a" * 100)
    source.write_bytes(b"b" * 25)

    report = materialize_candidates(
        baselines=[
            BaselineSpec(
                id="tflite_float32_640",
                platform="android",
                path=baseline,
                input_shape=[1, 640, 640, 3],
                output_shape=[1, 14, 8400],
            )
        ],
        candidates=[
            CandidateSpec(
                id="tflite_fp16_640",
                baseline_id="tflite_float32_640",
                platform="android",
                precision="fp16",
                source_path=source,
                target_path=target,
                input_shape=[1, 640, 640, 3],
                output_shape=[1, 14, 8400],
                required=True,
            )
        ],
        manifest_out=manifest,
        markdown_out=markdown,
        zip_out=zip_out,
    )

    assert report["ok"] is True
    assert target.read_bytes() == source.read_bytes()
    assert json.loads(manifest.read_text(encoding="utf-8"))["ok"] is True
    assert "tflite_fp16_640" in markdown.read_text(encoding="utf-8")
    assert zip_out.is_file()


def test_handoff_paths_include_ready_candidate_and_validation_report(tmp_path):
    baseline = tmp_path / "best_float32.tflite"
    candidate = tmp_path / "best_float16.tflite"
    validation = tmp_path / "litert_smoke.json"
    manifest = tmp_path / "mobile_optimization_report.json"
    markdown = tmp_path / "MOBILE_OPTIMIZATION_REPORT.md"
    baseline.write_bytes(b"a" * 100)
    candidate.write_bytes(b"b" * 25)
    validation.write_text("{}", encoding="utf-8")
    report = build_report(
        baselines=[
            BaselineSpec(
                id="tflite_float32_640",
                platform="android",
                path=baseline,
                input_shape=[1, 640, 640, 3],
                output_shape=[1, 14, 8400],
            )
        ],
        candidates=[
            CandidateSpec(
                id="tflite_fp16_640",
                baseline_id="tflite_float32_640",
                platform="android",
                precision="fp16",
                source_path=candidate,
                target_path=candidate,
                input_shape=[1, 640, 640, 3],
                output_shape=[1, 14, 8400],
                validation_report=validation,
            )
        ],
    )

    paths = handoff_paths(report, manifest, markdown)

    assert paths == [candidate, validation, manifest, markdown]


def test_mobile_optimization_zip_is_deterministic(tmp_path):
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


def test_default_candidates_include_android_dynamic_range_quant():
    ids = {candidate.id for candidate in DEFAULT_CANDIDATES}

    assert "tflite_dynamic_range_int8_640" in ids


def test_render_markdown_keeps_baseline_and_candidates_visible(tmp_path):
    baseline = tmp_path / "best_float32.tflite"
    candidate = tmp_path / "best_float16.tflite"
    baseline.write_bytes(b"a" * 100)
    candidate.write_bytes(b"b" * 25)
    report = build_report(
        baselines=[
            BaselineSpec(
                id="tflite_float32_640",
                platform="android",
                path=baseline,
                input_shape=[1, 640, 640, 3],
                output_shape=[1, 14, 8400],
            )
        ],
        candidates=[
            CandidateSpec(
                id="tflite_fp16_640",
                baseline_id="tflite_float32_640",
                platform="android",
                precision="fp16",
                source_path=candidate,
                target_path=tmp_path / "out" / "best_float16.tflite",
                input_shape=[1, 640, 640, 3],
                output_shape=[1, 14, 8400],
            )
        ],
    )

    markdown = render_markdown(report)

    assert "# Mobile Optimization Report" in markdown
    assert "tflite_float32_640" in markdown
    assert "tflite_fp16_640" in markdown
    assert "4.00x" in markdown
