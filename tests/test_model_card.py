from __future__ import annotations

from src.model_card import build_model_card, eval_line, metric


def test_metric_reads_nested_value_with_default():
    report = {"a": {"b": 1.5}}

    assert metric(report, "a", "b") == 1.5
    assert metric(report, "a", "missing") == "n/a"


def test_eval_line_formats_core_metrics():
    report = {
        "metrics_bbox": {"mAP50": 0.9, "mAP50_95": 0.8},
        "oks": {"mean": 0.85},
        "rates": {"false_negative_rate": 0.1, "false_positive_rate": 0.2},
        "counts": {"gt_wheels": 10, "pred_wheels_above_conf": 11, "matched": 9},
    }

    line = eval_line("eval", report)

    assert line.startswith("| eval |")
    assert "0.9" in line
    assert "10/11/9" in line


def test_model_card_avoids_embedding_release_manifest_counts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    audit_root = tmp_path / "outputs" / "production_audit"
    audit_root.mkdir(parents=True)
    (audit_root / "release_integrity.json").write_text(
        '{"ok": true, "artifact_count": 72, "total_size_mb": 125.0}',
        encoding="utf-8",
    )

    text = build_model_card()

    assert "Release integrity OK: True" in text
    assert "Release artifacts:" not in text
    assert "Release size:" not in text
