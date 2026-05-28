from __future__ import annotations

import json

from src.runtime_contract_audit import build_report, validate_payload


def _payload(frame_id: str = "frame_1", wheels: int = 1) -> dict:
    return {
        "frame_id": frame_id,
        "wheels": [
            {
                "bbox_xyxy": [1.0, 2.0, 10.0, 20.0],
                "confidence": 0.9,
                "points": {
                    "a": [2.0, 18.0],
                    "b": [8.0, 18.0],
                    "c_disc_bottom": [5.0, 16.0],
                },
            }
            for _ in range(wheels)
        ],
    }


def test_validate_payload_accepts_confirmed_schema():
    assert validate_payload(_payload(), source="sample") == []


def test_validate_payload_rejects_forbidden_3d_fields():
    payload = _payload()
    payload["wheels"][0]["world_position"] = [1, 2, 3]

    errors = validate_payload(payload, source="sample")

    assert any("forbidden key" in error for error in errors)
    assert any("keys" in error for error in errors)


def test_build_report_cross_checks_batch_summary(tmp_path):
    single = tmp_path / "single.json"
    batch = tmp_path / "batch.jsonl"
    summary = tmp_path / "summary.json"
    single.write_text(json.dumps(_payload(wheels=2)), encoding="utf-8")
    batch.write_text(json.dumps(_payload("f1", wheels=1)) + "\n" + json.dumps(_payload("f2", wheels=3)) + "\n", encoding="utf-8")
    summary.write_text(json.dumps({"frames_inferred": 2, "wheels_detected_total": 4}), encoding="utf-8")

    report = build_report(single, batch, summary)

    assert report["ok"] is True
    assert report["counts"]["single_wheels"] == 2
    assert report["counts"]["batch_frames"] == 2
    assert report["counts"]["batch_wheels"] == 4


def test_build_report_detects_summary_mismatch(tmp_path):
    single = tmp_path / "single.json"
    batch = tmp_path / "batch.jsonl"
    summary = tmp_path / "summary.json"
    single.write_text(json.dumps(_payload()), encoding="utf-8")
    batch.write_text(json.dumps(_payload("f1", wheels=1)) + "\n", encoding="utf-8")
    summary.write_text(json.dumps({"frames_inferred": 99, "wheels_detected_total": 99}), encoding="utf-8")

    report = build_report(single, batch, summary)

    assert report["ok"] is False
    assert any("frames_inferred" in failure for failure in report["failures"])
    assert any("wheels_detected_total" in failure for failure in report["failures"])
