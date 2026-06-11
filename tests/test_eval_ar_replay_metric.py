from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import subprocess
import sys

from src.eval_ar_replay_metric import MetricConfig, build_metric_report, main


def _obs(
    i: int,
    *,
    wheel_index: int = 0,
    inlier: bool = True,
    residual: float = 0.004,
    c_dx: float = 0.0,
    normal_x: float = 0.998,
    normal_z: float = 0.063,
) -> dict:
    norm = math.sqrt(normal_x * normal_x + normal_z * normal_z)
    return {
        "schema_version": 1,
        "source_type": "android_ar_device_replay",
        "capture_device": "Pixel test",
        "capture_app_version": "1.2.3",
        "capture_date_utc": "2026-05-27",
        "session_id": "s1",
        "frame_id": f"frame_{i:04d}",
        "capture_index": i,
        "wheel_index": wheel_index,
        "camera_transform": None,
        "camera_pose_ref": f"pose_{i:04d}",
        "screen_points": {
            "a": [100.0, 200.0],
            "b": [150.0, 200.0],
            "c_disc_bottom": [125.0, 170.0],
        },
        "floor_raycast_hits": {
            "a": [1.0 + i * 0.001, 0.0, 2.0],
            "b": [1.5 + i * 0.001, 0.0, 2.0],
        },
        "inlier": inlier,
        "residual": residual,
        "recovered_plane": {
            "normal": [normal_x / norm, 0.0, normal_z / norm],
            "point": [1.25, 0.0, 2.0],
            "support": 18,
        },
        "c_plane_hit": [1.25 + c_dx, 0.4, 2.0],
        "c_height_value": 0.4,
        "final_disc_bottom_position": [1.25, 0.4, 2.0],
    }


def test_ar_replay_metric_reports_per_wheel_3d_stability(tmp_path):
    source = tmp_path / "ar_replay.jsonl"
    source.write_text("\n".join(json.dumps(_obs(i, c_dx=i * 0.01)) for i in range(5)) + "\n")

    report = build_metric_report(
        [_obs(i, c_dx=i * 0.01) for i in range(5)],
        MetricConfig(min_observations_per_wheel=4),
        source=source,
    )

    assert report["ok"] is True
    assert report["counts"]["sessions"] == 1
    assert report["counts"]["wheels"] == 1
    assert report["aggregate"]["failure_rate"] == 0.0
    wheel = report["sessions"]["s1"]["wheels"]["wheel_index:0"]
    assert wheel["counts"]["observations"] == 5
    assert wheel["metrics"]["inlier_ratio"] == 1.0
    assert wheel["metrics"]["median_residual"] == 0.004
    assert wheel["metrics"]["p95_residual"] == 0.004
    assert wheel["metrics"]["plane_normal_stability_deg"] == 0.0
    assert wheel["metrics"]["plane_verticality_deg"] == 90.0
    assert wheel["metrics"]["c_plane_hit_std"] > 0.0


def test_ar_replay_metric_fails_when_required_metric_fields_are_missing(tmp_path):
    observations = [_obs(i) for i in range(4)]
    observations[0].pop("c_plane_hit")

    report = build_metric_report(
        observations,
        MetricConfig(min_observations_per_wheel=4),
        source=tmp_path / "missing_metric_fields.jsonl",
    )

    assert report["ok"] is False
    assert "missing_metric_fields" in report["failures"]
    wheel = report["sessions"]["s1"]["wheels"]["wheel_index:0"]
    assert any("c_plane_hit" in item for item in wheel["metric_field_errors"])


def test_ar_replay_metric_fails_empty_log(tmp_path):
    report = build_metric_report(
        [],
        MetricConfig(min_observations_per_wheel=4),
        source=tmp_path / "empty.jsonl",
    )

    assert report["ok"] is False
    assert "no_wheels" in report["failures"]
    assert report["aggregate"]["failure_rate"] == 1.0


def test_ar_replay_metric_cli_writes_json_and_per_frame_csv(tmp_path):
    source = tmp_path / "ar_replay.jsonl"
    out = tmp_path / "metric.json"
    csv_out = tmp_path / "per_frame.csv"
    source.write_text("\n".join(json.dumps(_obs(i, c_dx=i * 0.01)) for i in range(4)) + "\n")

    rc = main(
        [
            "--jsonl",
            str(source),
            "--out",
            str(out),
            "--per-frame-csv",
            str(csv_out),
            "--min-observations-per-wheel",
            "4",
        ]
    )

    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["ok"] is True
    rows = list(csv.DictReader(csv_out.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 4
    assert rows[0]["session_id"] == "s1"
    assert rows[0]["wheel_id"] == "wheel_index:0"
    assert rows[0]["inlier"] == "true"


def test_ar_replay_metric_file_cli_runs_without_pythonpath(tmp_path):
    source = tmp_path / "ar_replay.jsonl"
    out = tmp_path / "metric.json"
    csv_out = tmp_path / "per_frame.csv"
    source.write_text("\n".join(json.dumps(_obs(i, c_dx=i * 0.01)) for i in range(4)) + "\n")

    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            "src/eval_ar_replay_metric.py",
            "--jsonl",
            str(source),
            "--out",
            str(out),
            "--per-frame-csv",
            str(csv_out),
            "--min-observations-per-wheel",
            "4",
        ],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out.is_file()
    assert csv_out.is_file()
