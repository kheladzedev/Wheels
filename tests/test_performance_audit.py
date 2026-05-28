from __future__ import annotations

import json

from src.performance_audit import load_litert_smoke, percentile, render_markdown, summarize_ms


def test_percentile_interpolates():
    values = [10.0, 20.0, 30.0, 40.0]

    assert percentile(values, 0) == 10.0
    assert percentile(values, 50) == 25.0
    assert percentile(values, 95) == 38.5
    assert percentile(values, 100) == 40.0


def test_summarize_ms_handles_empty_and_values():
    assert summarize_ms([])["runs"] == 0

    summary = summarize_ms([3.0, 1.0, 2.0])

    assert summary["runs"] == 3
    assert summary["mean"] == 2.0
    assert summary["p50"] == 2.0
    assert summary["min"] == 1.0
    assert summary["max"] == 3.0


def test_load_litert_smoke_maps_latency(tmp_path):
    smoke = tmp_path / "litert.json"
    smoke.write_text(
        json.dumps(
            {
                "ok": True,
                "runtime": "ai_edge_litert",
                "model": "best.tflite",
                "latency_ms": {"runs": 10, "mean": 12.5, "p50": 12.0, "p95": 14.0},
            }
        ),
        encoding="utf-8",
    )

    report = load_litert_smoke(smoke)

    assert report["ok"] is True
    assert report["latency_ms"]["mean"] == 12.5
    assert report["source_report"] == str(smoke)


def test_render_markdown_lists_runtime_table():
    markdown = render_markdown(
        {
            "ok": True,
            "scope": "desktop",
            "sample_count": 2,
            "images_dir": "images/val",
            "benchmarks": {
                "pytorch_cpu": {
                    "ok": True,
                    "device": "cpu",
                    "latency_ms": {"runs": 4, "mean": 10.0, "p50": 9.0, "p95": 12.0},
                    "detections": {"mean_per_image": 2.5},
                }
            },
            "notes": ["local only"],
        }
    )

    assert "Performance Audit" in markdown
    assert "pytorch_cpu" in markdown
    assert "10.000" in markdown
