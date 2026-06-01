from __future__ import annotations

import json
from pathlib import Path

from src.operating_point_audit import build_audit, render_markdown


def _write_report(
    path: Path,
    *,
    conf: float,
    map50: float,
    oks: float,
    fn: float,
    fp: float,
) -> None:
    path.write_text(
        json.dumps(
            {
                "thresholds": {"conf": conf},
                "metrics_bbox": {"mAP50": map50},
                "oks": {"mean": oks},
                "rates": {
                    "false_negative_rate": fn,
                    "false_positive_rate": fp,
                },
                "counts": {
                    "gt_wheels": 64,
                    "pred_wheels_above_conf": 68,
                    "matched": 58,
                    "false_positives": 10,
                    "false_negatives": 6,
                },
            }
        ),
        encoding="utf-8",
    )


def test_operating_point_selects_lowest_threshold_that_passes(tmp_path: Path) -> None:
    low = tmp_path / "threshold_conf075_real_val.json"
    high = tmp_path / "threshold_conf080_real_val.json"
    _write_report(low, conf=0.75, map50=0.90, oks=0.88, fn=0.09, fp=0.18)
    _write_report(high, conf=0.80, map50=0.90, oks=0.88, fn=0.09, fp=0.147)

    audit = build_audit([low, high])

    assert audit["ok"] is True
    assert audit["selected"]["path"] == str(high)
    assert audit["selected"]["conf"] == 0.80
    assert audit["selected"]["false_positive_rate"] == 0.147
    assert audit["failures"] == []


def test_operating_point_fails_when_no_threshold_passes(tmp_path: Path) -> None:
    report = tmp_path / "threshold_conf080_real_val.json"
    _write_report(report, conf=0.80, map50=0.90, oks=0.88, fn=0.12, fp=0.147)

    audit = build_audit([report])

    assert audit["ok"] is False
    assert audit["selected"] is None
    assert audit["failures"] == ["no_threshold_candidate_meets_quality_gates"]


def test_operating_point_markdown_names_selected_report(tmp_path: Path) -> None:
    report = tmp_path / "threshold_conf080_real_val.json"
    _write_report(report, conf=0.80, map50=0.90, oks=0.88, fn=0.09, fp=0.147)

    markdown = render_markdown(build_audit([report]))

    assert "Operating Point Audit" in markdown
    assert str(report) in markdown
    assert "0.800" in markdown
