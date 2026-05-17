from __future__ import annotations

import json
from pathlib import Path

import pytest

from run_service_scenario_suite import audit_confirmed_payload, load_manifest


def test_load_manifest_requires_scenarios(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"scenarios": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty"):
        load_manifest(manifest)


def test_audit_confirmed_payload_accepts_floor_ray_geometry(tmp_path: Path):
    payload = {
        "frame_id": "frame_001",
        "wheels": [
            {
                "bbox_xyxy": [10, 20, 110, 120],
                "confidence": 0.91,
                "points": {
                    "a": [20, 105],
                    "b": [100, 106],
                    "c_disc_bottom": [60, 86],
                },
            }
        ],
    }
    path = tmp_path / "pred.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    audit = audit_confirmed_payload(path, image_hw=(200, 200))

    assert audit["schema_ok"] is True
    assert audit["geometry_ok"] is True
    assert audit["wheel_count"] == 1
    assert audit["issues"] == []


def test_audit_confirmed_payload_flags_bad_geometry(tmp_path: Path):
    payload = {
        "frame_id": "frame_001",
        "wheels": [
            {
                "bbox_xyxy": [10, 20, 110, 120],
                "confidence": 0.91,
                "points": {
                    "a": [95, 60],
                    "b": [45, 62],
                    "c_disc_bottom": [60, 112],
                },
            }
        ],
    }
    path = tmp_path / "pred.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    audit = audit_confirmed_payload(path, image_hw=(200, 200))

    assert audit["schema_ok"] is True
    assert audit["geometry_ok"] is False
    assert any("A is not left of B" in issue for issue in audit["issues"])
    assert any("C is not above" in issue for issue in audit["issues"])
