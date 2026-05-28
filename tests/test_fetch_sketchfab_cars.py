"""Offline tests for the Sketchfab GLB candidate filters."""

from __future__ import annotations

import time
from pathlib import Path

import fetch_sketchfab_cars as sfc


def _model(name: str, **extra: object) -> dict:
    return {"uid": "u1", "name": name, **extra}


def test_vehicle_filter_keeps_standalone_wheel_assets():
    assert sfc._is_vehicle_like(_model("FREE Detailed Tire - public domain (CC0)"))
    assert sfc._is_vehicle_like(_model("Car wheel low poly"))


def test_car_body_filter_accepts_whole_vehicle_names():
    names = [
        "Toyota Corolla - PS1 Low Poly",
        "White Van - PS1 Low Poly",
        "Low poly Tokyo Taxi",
        "Ford Crown Victoria NYC Taxi Cab",
        "Civil Car Pack - New York Taxi License 3D",
        "Volkswagen Golf V - PS1 Low Poly",
    ]

    for name in names:
        assert sfc._is_car_body_like(_model(name)), name


def test_car_body_filter_rejects_non_body_vehicle_adjacent_assets():
    names = [
        "FREE Detailed Tire - public domain (CC0)",
        "Car wheel low poly",
        "Taxi Drone",
        "Police Car Lightbar",
        "Police Siren",
        "Rim set for sports car",
        "Bus Stop - low",
        "bus stop in ps1 style",
        "Race Car Driver - Low Poly",
        "Audi Nameplate",
        "Hand Truck Cart",
    ]

    for name in names:
        assert not sfc._is_car_body_like(_model(name)), name


def test_name_filter_switches_between_vehicle_and_body_modes():
    tire = _model("Car tire realistic")

    assert sfc._matches_name_filter(tire, car_body_only=False)
    assert not sfc._matches_name_filter(tire, car_body_only=True)


def test_complexity_limits_use_manifest_count_variants():
    assert sfc._passes_complexity_limits(
        _model("Toyota car", faceCount=240_000, vertex_count=120_000),
        max_face_count=250_000,
        max_vertex_count=150_000,
    )
    assert not sfc._passes_complexity_limits(
        _model("Toyota car", face_count=260_000, vertexCount=120_000),
        max_face_count=250_000,
        max_vertex_count=150_000,
    )
    assert not sfc._passes_complexity_limits(
        _model("Toyota car", face_count=240_000, vertexCount=160_000),
        max_face_count=250_000,
        max_vertex_count=150_000,
    )


def test_complexity_sort_puts_known_small_models_first():
    models = [
        _model("unknown", uid="c"),
        _model("large", uid="b", faceCount=1000, vertexCount=1000),
        _model("small", uid="a", face_count=10, vertex_count=20),
    ]

    assert [m["name"] for m in sorted(models, key=sfc._complexity_score)] == [
        "small",
        "large",
        "unknown",
    ]


def test_recent_download_failures_can_be_skipped_from_manifest_queue(tmp_path: Path):
    fresh = tmp_path / "fresh.json"
    failed = tmp_path / "failed.json"
    old_failed = tmp_path / "old_failed.json"
    now = int(time.time())

    fresh.write_text('{"uid": "fresh", "name": "Toyota car"}')
    failed.write_text(
        '{"uid": "failed", "name": "Nissan car", '
        '"_download_failure": {"reason": "download", "time": %d}}' % now
    )
    old_failed.write_text(
        '{"uid": "old_failed", "name": "Ford car", '
        '"_download_failure": {"reason": "download", "time": %d}}'
        % (now - 72 * 3600)
    )

    candidates = sfc._load_manifest_candidates(
        tmp_path,
        vehicle_only=True,
        car_body_only=True,
        skip_failed_within_hours=24,
    )

    assert [c["uid"] for c in candidates] == ["fresh", "old_failed"]


def test_download_failure_writer_preserves_source_metadata(tmp_path: Path):
    path = tmp_path / "u1.json"
    model = _model("Toyota Corolla", uid="u1", faceCount=123, vertexCount=456)

    sfc._write_download_failure(path, model, "download")
    manifest = sfc._read_manifest(path)

    assert manifest["uid"] == "u1"
    assert manifest["name"] == "Toyota Corolla"
    assert manifest["face_count"] == 123
    assert manifest["vertex_count"] == 456
    assert manifest["_download_failure"]["reason"] == "download"


def test_main_returns_temporary_block_exit_code(monkeypatch, tmp_path: Path):
    manifest = tmp_path / "blocked.json"
    manifest.write_text('{"uid": "blocked", "name": "Toyota car"}')

    monkeypatch.setattr(sfc, "_resolve_token", lambda token: "token")

    def raise_block(*args, **kwargs):
        raise sfc.SketchfabTemporaryBlock("rate limited")

    monkeypatch.setattr(sfc, "_download_model", raise_block)

    rc = sfc.main(
        [
            "--output-dir",
            str(tmp_path),
            "--from-existing-manifests",
            "--car-body-only",
            "--candidate-limit",
            "1",
            "--download-retries",
            "0",
        ]
    )

    assert rc == sfc.EXIT_TEMPORARY_BLOCK


def test_main_returns_consecutive_failures_exit_code(monkeypatch, tmp_path: Path):
    for uid in ("a", "b"):
        (tmp_path / f"{uid}.json").write_text(
            '{"uid": "%s", "name": "Toyota car"}' % uid
        )

    monkeypatch.setattr(sfc, "_resolve_token", lambda token: "token")

    def fail_download(token, model, output_dir, max_bytes, **kwargs):
        return {
            "status": "failed",
            "uid": model["uid"],
            "name": model["name"],
            "reason": "download",
        }

    monkeypatch.setattr(sfc, "_download_model", fail_download)

    rc = sfc.main(
        [
            "--output-dir",
            str(tmp_path),
            "--from-existing-manifests",
            "--car-body-only",
            "--candidate-limit",
            "2",
            "--max-consecutive-failures",
            "2",
            "--download-delay",
            "0",
        ]
    )

    assert rc == sfc.EXIT_CONSECUTIVE_FAILURES
