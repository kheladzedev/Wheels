from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.run_production_evidence_intake import (
    EvidenceDropImportResult,
    FinalizationResult,
    IntakeResult,
    POST_FINALIZATION_REFRESH_COMMANDS,
    POST_FINALIZATION_RELEASE_REFRESH_COMMANDS,
    POST_FINALIZATION_REPORT_REFRESH_COMMANDS,
    RefreshResult,
    build_preflight_status,
    build_status,
    build_steps,
    fail_import_for_destination_mismatch,
    finalization_canonical_path_failures,
    run_evidence_drop_import,
    write_status,
)

EXPECTED_ANDROID_ARTIFACT_BYTES = b"test tflite"
EXPECTED_ANDROID_ARTIFACT_SHA = hashlib.sha256(EXPECTED_ANDROID_ARTIFACT_BYTES).hexdigest()


def _args(tmp_path):
    expected_artifact = tmp_path / "expected.tflite"
    expected_artifact.write_bytes(EXPECTED_ANDROID_ARTIFACT_BYTES)
    return argparse.Namespace(
        android_litert_source=tmp_path / "android_litert_device_report.json",
        android_litert_eval=tmp_path / "android_eval.json",
        ar_holdout_source=tmp_path / "ar_device_holdout",
        ar_holdout_eval=tmp_path / "ar_holdout_eval.json",
        ar_holdout_pipeline=tmp_path / "ar_holdout_pipeline.json",
        ar_replay_jsonl=tmp_path / "ar_3d_replay" / "ar_replay.jsonl",
        ar_replay_eval=tmp_path / "ar_replay_eval.json",
        evidence_drop=None,
        evidence_drop_dest_root=tmp_path,
        evidence_drop_report_out=tmp_path / "external_evidence_drop_import.json",
        evidence_drop_overwrite=False,
        expected_android_artifact=expected_artifact,
    )


def _canonical_args():
    return argparse.Namespace(
        android_litert_source=Path("data/incoming/android_litert_device_report.json"),
        android_litert_eval=Path("outputs/production_audit/android_litert_device_eval.json"),
        ar_holdout_source=Path("data/incoming/ar_device_holdout"),
        ar_holdout_eval=Path("outputs/production_audit/ar_device_holdout_eval.json"),
        ar_holdout_pipeline=Path("outputs/production_audit/ar_device_holdout_pipeline.json"),
        ar_replay_jsonl=Path("data/incoming/ar_3d_replay/ar_replay.jsonl"),
        ar_replay_eval=Path("outputs/production_audit/ar_3d_replay_eval.json"),
        evidence_drop=None,
        evidence_drop_dest_root=Path("data/incoming"),
        evidence_drop_report_out=Path("outputs/production_audit/external_evidence_drop_import.json"),
        evidence_drop_overwrite=False,
        expected_android_artifact=Path("outputs/production_audit/tflite_export/best_float32.tflite"),
    )


def _annotation(frame_id: str, image: str | None = None) -> str:
    image_name = image or f"{frame_id}.jpg"
    return json.dumps(
        {
            "schema_version": 1,
            "frame_id": frame_id,
            "image": image_name,
            "wheels": [
                {
                    "bbox_xyxy": [10, 10, 30, 30],
                    "points": {"a": [12, 28], "b": [28, 28], "c_disc_bottom": [20, 24]},
                },
                {
                    "bbox_xyxy": [40, 10, 60, 30],
                    "points": {"a": [42, 28], "b": [58, 28], "c_disc_bottom": [50, 24]},
                },
            ],
        }
    )


def _write_drop(root):
    (root / "ar_device_holdout" / "images").mkdir(parents=True)
    (root / "ar_device_holdout" / "annotations").mkdir()
    (root / "ar_device_holdout" / "metadata").mkdir()
    (root / "ar_3d_replay").mkdir()
    (root / "android_litert_device_report.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_type": "android_litert_device_validation",
                "test_session_id": "android-litert-001",
                "test_app_version": "1.2.3",
                "test_date_utc": "2026-05-27",
                "device": {
                    "model": "Pixel test",
                    "manufacturer": "Google",
                    "android_version": "15",
                    "soc": "Tensor test",
                    "is_emulator": False,
                },
                "runtime": "LiteRT",
                "artifact": {"sha256": EXPECTED_ANDROID_ARTIFACT_SHA, "format": "tflite_float32"},
                "input": {"shape": [1, 640, 640, 3], "dtype": "float32", "profile": "zero_float32_smoke"},
                "latency_ms": {"runs": 30, "mean": 40.0, "p95": 70.0},
                "memory_mb": {"peak": 128.0},
                "output": {"shape": [1, 14, 8400], "finite": True, "min": 0.0, "max": 1.0, "mean": 0.5},
            }
        ),
        encoding="utf-8",
    )
    for i in range(50):
        frame_id = f"frame_{i:04d}"
        (root / "ar_device_holdout" / "images" / f"{frame_id}.jpg").write_bytes(b"jpg")
        (root / "ar_device_holdout" / "annotations" / f"{frame_id}.json").write_text(
            _annotation(frame_id),
            encoding="utf-8",
        )
    (root / "ar_device_holdout" / "metadata" / "provenance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_type": "android_ar_device_human_labelled",
                "label_type": "human_reviewed",
                "capture_device": "Pixel test",
                "capture_app_version": "1.2.3",
                "capture_date_utc": "2026-05-27",
                "annotator": "annotator_a",
                "reviewer": "reviewer_b",
                "review_status": "accepted",
            }
        ),
        encoding="utf-8",
    )
    (root / "ar_3d_replay" / "ar_replay.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_type": "android_ar_device_replay",
                    "capture_device": "Pixel test",
                    "capture_app_version": "1.2.3",
                    "capture_date_utc": "2026-05-27",
                    "session_id": "s1",
                    "frame_id": f"frame_{i:04d}",
                    "capture_index": i,
                    "camera_transform": None,
                    "camera_pose_ref": f"pose_{i:04d}",
                    "screen_points": {
                        "a": [100.0, 200.0],
                        "b": [150.0, 200.0],
                        "c_disc_bottom": [125.0, 170.0],
                    },
                    "floor_raycast_hits": {"a": [1.0, 0.0, 2.0], "b": [1.5, 0.0, 2.0]},
                    "inlier": True,
                    "residual": 0.004,
                    "recovered_plane": {
                        "normal": [0.998, 0.0, 0.062],
                        "point": [1.25, 0.0, 2.0],
                        "support": 18,
                    },
                    "c_plane_hit": [1.25, 0.4, 2.0],
                    "c_height_value": 0.4,
                    "final_disc_bottom_position": [1.25, 0.0, 2.0] if i == 0 else None,
                }
            )
            for i in range(30)
        )
        + "\n",
        encoding="utf-8",
    )


def _write_android(args) -> None:
    args.android_litert_source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_type": "android_litert_device_validation",
                "test_session_id": "android-litert-001",
                "test_app_version": "1.2.3",
                "test_date_utc": "2026-05-27",
                "device": {
                    "model": "Pixel test",
                    "manufacturer": "Google",
                    "android_version": "15",
                    "soc": "Tensor test",
                    "is_emulator": False,
                },
                "runtime": "LiteRT",
                "artifact": {"sha256": EXPECTED_ANDROID_ARTIFACT_SHA, "format": "tflite_float32"},
                "input": {"shape": [1, 640, 640, 3], "dtype": "float32", "profile": "zero_float32_smoke"},
                "latency_ms": {"runs": 30, "mean": 40.0, "p95": 70.0},
                "memory_mb": {"peak": 128.0},
                "output": {"shape": [1, 14, 8400], "finite": True, "min": 0.0, "max": 1.0, "mean": 0.5},
            }
        ),
        encoding="utf-8",
    )


def _valid_replay_line() -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "source_type": "android_ar_device_replay",
            "capture_device": "Pixel test",
            "capture_app_version": "1.2.3",
            "capture_date_utc": "2026-05-27",
            "session_id": "s1",
            "frame_id": "frame_0001",
            "capture_index": 0,
            "camera_transform": None,
            "camera_pose_ref": "pose_0001",
            "screen_points": {
                "a": [100.0, 200.0],
                "b": [150.0, 200.0],
                "c_disc_bottom": [125.0, 170.0],
            },
            "floor_raycast_hits": {"a": [1.0, 0.0, 2.0], "b": [1.5, 0.0, 2.0]},
            "inlier": True,
            "residual": 0.004,
            "recovered_plane": {
                "normal": [0.998, 0.0, 0.062],
                "point": [1.25, 0.0, 2.0],
                "support": 18,
            },
            "c_plane_hit": [1.25, 0.4, 2.0],
            "c_height_value": 0.4,
            "final_disc_bottom_position": [1.25, 0.0, 2.0],
        }
    )


def _write_replay(args, text: str | None = None) -> None:
    args.ar_replay_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.ar_replay_jsonl.write_text(text if text is not None else _valid_replay_line() + "\n", encoding="utf-8")


def _valid_provenance() -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "source_type": "android_ar_device_human_labelled",
            "label_type": "human_reviewed",
            "capture_device": "Pixel test",
            "capture_app_version": "1.2.3",
            "capture_date_utc": "2026-05-27",
            "annotator": "annotator_a",
            "reviewer": "reviewer_b",
            "review_status": "accepted",
        }
    )


def _write_valid_provenance(args) -> None:
    (args.ar_holdout_source / "metadata").mkdir(parents=True, exist_ok=True)
    (args.ar_holdout_source / "metadata" / "provenance.json").write_text(
        _valid_provenance(),
        encoding="utf-8",
    )


def _write_min_holdout(args) -> None:
    (args.ar_holdout_source / "images").mkdir(parents=True)
    (args.ar_holdout_source / "annotations").mkdir()
    (args.ar_holdout_source / "metadata").mkdir()
    (args.ar_holdout_source / "images" / "frame.jpg").write_bytes(b"jpg")
    (args.ar_holdout_source / "annotations" / "frame.json").write_text(
        _annotation("frame"),
        encoding="utf-8",
    )
    _write_valid_provenance(args)


def test_evidence_intake_builds_required_validation_order(tmp_path):
    names = [step.name for step in build_steps(_args(tmp_path))]

    assert names[:3] == [
        "android_litert_device_validation",
        "human_labelled_ar_device_holdout",
        "ar_3d_replay_validation",
    ]
    assert names.index("production_evidence_audit") < names.index("production_gate")
    assert names.index("production_gate") < names.index("senior_ml_audit")
    assert names.index("senior_ml_audit") < names.index("requirements_traceability")
    assert names.index("executive_report_ru") < names.index("objective_completion_audit")
    assert names.index("objective_completion_audit") < names.index("release_integrity")


def test_evidence_intake_propagates_custom_evidence_paths_to_audits_and_gates(tmp_path):
    args = _args(tmp_path)
    steps = {step.name: step for step in build_steps(args)}

    evidence_cmd = steps["production_evidence_audit"].cmd
    android_cmd = steps["android_litert_device_validation"].cmd
    holdout_cmd = steps["human_labelled_ar_device_holdout"].cmd
    replay_cmd = steps["ar_3d_replay_validation"].cmd
    assert android_cmd[android_cmd.index("--expected-artifact") + 1] == str(args.expected_android_artifact)
    assert android_cmd[android_cmd.index("--min-runs") + 1] == "20"
    assert android_cmd[android_cmd.index("--max-mean-latency-ms") + 1] == "120.0"
    assert android_cmd[android_cmd.index("--max-p95-latency-ms") + 1] == "180.0"
    assert android_cmd[android_cmd.index("--max-peak-memory-mb") + 1] == "512.0"
    assert holdout_cmd[holdout_cmd.index("--status-out") + 1] == str(args.ar_holdout_pipeline)
    assert holdout_cmd[holdout_cmd.index("--min-map50") + 1] == "0.85"
    assert holdout_cmd[holdout_cmd.index("--min-oks") + 1] == "0.8"
    assert holdout_cmd[holdout_cmd.index("--max-fn") + 1] == "0.1"
    assert holdout_cmd[holdout_cmd.index("--min-images") + 1] == "50"
    assert holdout_cmd[holdout_cmd.index("--min-gt-wheels") + 1] == "80"
    assert replay_cmd[replay_cmd.index("--min-observations") + 1] == "30"
    assert replay_cmd[replay_cmd.index("--min-sessions") + 1] == "1"
    assert replay_cmd[replay_cmd.index("--min-floor-hit-rate") + 1] == "0.9"
    assert replay_cmd[replay_cmd.index("--min-inlier-rate") + 1] == "0.7"
    assert replay_cmd[replay_cmd.index("--max-median-residual") + 1] == "0.02"
    assert replay_cmd[replay_cmd.index("--max-p95-residual") + 1] == "0.05"
    assert replay_cmd[replay_cmd.index("--min-final-positions") + 1] == "1"
    assert evidence_cmd[evidence_cmd.index("--android-litert-source") + 1] == str(args.android_litert_source)
    assert evidence_cmd[evidence_cmd.index("--android-litert-eval") + 1] == str(args.android_litert_eval)
    assert evidence_cmd[evidence_cmd.index("--ar-holdout-source") + 1] == str(args.ar_holdout_source)
    assert evidence_cmd[evidence_cmd.index("--ar-holdout-eval") + 1] == str(args.ar_holdout_eval)
    assert evidence_cmd[evidence_cmd.index("--ar-holdout-pipeline") + 1] == str(args.ar_holdout_pipeline)
    assert evidence_cmd[evidence_cmd.index("--ar-replay-jsonl") + 1] == str(args.ar_replay_jsonl)
    assert evidence_cmd[evidence_cmd.index("--ar-replay-eval") + 1] == str(args.ar_replay_eval)
    assert evidence_cmd[evidence_cmd.index("--external-evidence-import-report") + 1] == str(
        args.evidence_drop_report_out
    )
    assert evidence_cmd[evidence_cmd.index("--expected-android-artifact") + 1] == str(
        args.expected_android_artifact
    )

    senior_cmd = steps["senior_ml_audit"].cmd
    assert senior_cmd[senior_cmd.index("--android-litert-eval") + 1] == str(args.android_litert_eval)
    assert senior_cmd[senior_cmd.index("--ar-holdout-eval") + 1] == str(args.ar_holdout_eval)
    assert senior_cmd[senior_cmd.index("--ar-replay-eval") + 1] == str(args.ar_replay_eval)

    for name in ("integration_gate", "production_gate"):
        gate_cmd = steps[name].cmd
        assert gate_cmd[gate_cmd.index("--android-litert-eval") + 1] == str(args.android_litert_eval)
        assert gate_cmd[gate_cmd.index("--ar-holdout-eval") + 1] == str(args.ar_holdout_eval)
        assert gate_cmd[gate_cmd.index("--ar-3d-eval") + 1] == str(args.ar_replay_eval)


def test_evidence_intake_finalize_accepts_only_canonical_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert finalization_canonical_path_failures(_canonical_args()) == []

    failures = finalization_canonical_path_failures(_args(tmp_path))

    assert any(failure.startswith("non_canonical_android_litert_source:") for failure in failures)
    assert any(failure.startswith("non_canonical_evidence_drop_report_out:") for failure in failures)


def test_evidence_intake_post_finalization_refresh_order():
    names = [command[1] for command in POST_FINALIZATION_REFRESH_COMMANDS]

    assert names == [
        "scripts/write_production_audit_report.py",
        "scripts/write_handoff_report.py",
        "src/release_integrity.py",
        "src/report_consistency_audit.py",
    ]
    assert [command[1] for command in POST_FINALIZATION_REPORT_REFRESH_COMMANDS] == [
        "scripts/write_production_audit_report.py",
        "scripts/write_handoff_report.py",
    ]
    assert [command[1] for command in POST_FINALIZATION_RELEASE_REFRESH_COMMANDS] == [
        "src/release_integrity.py",
        "src/report_consistency_audit.py",
    ]
    assert names.index("src/release_integrity.py") > names.index("scripts/write_handoff_report.py")
    assert names.index("src/report_consistency_audit.py") > names.index("src/release_integrity.py")


def test_evidence_intake_write_status_creates_parent_and_trailing_newline(tmp_path):
    path = tmp_path / "nested" / "status.json"

    write_status(path, {"ok": True})

    assert path.read_text(encoding="utf-8").endswith("\n")
    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}


def test_evidence_intake_status_requires_evidence_and_production_gate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    audit_root = tmp_path / "outputs" / "production_audit"
    audit_root.mkdir(parents=True)
    (audit_root / "production_evidence_audit.json").write_text(
        '{"production_evidence_ready": false, "blockers": ["ar_3d_replay_validation"]}',
        encoding="utf-8",
    )
    (audit_root / "integration_gate.json").write_text('{"ok": true}', encoding="utf-8")
    (audit_root / "production_gate.json").write_text('{"ok": false}', encoding="utf-8")
    (audit_root / "objective_completion_audit.json").write_text(
        '{"objective_complete": false, "failed_requirements": ["production_gate_passed"]}',
        encoding="utf-8",
    )

    status = build_status(
        [
            IntakeResult(
                name="ar_3d_replay_validation",
                returncode=None,
                ok=False,
                skipped=True,
                missing_input=True,
                input_path="missing.jsonl",
                cmd=["python", "src/validate_ar_replay.py"],
            )
        ]
    )

    assert status["ok"] is False
    assert status["production_ready"] is False
    assert status["objective_complete"] is False
    assert status["finalization_required"] is True
    assert status["finalization_command"] == [
        "./.venv/bin/python",
        "src/production_audit_suite.py",
        "--with-pytest",
    ]
    assert status["objective_failed_requirements"] == ["production_gate_passed"]
    assert status["production_blockers"] == ["ar_3d_replay_validation"]
    assert status["production_required_failures"] == ["ar_3d_replay_validation"]


def test_evidence_intake_status_includes_failed_drop_import(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    audit_root = tmp_path / "outputs" / "production_audit"
    audit_root.mkdir(parents=True)
    (audit_root / "production_evidence_audit.json").write_text(
        '{"production_evidence_ready": false, "blockers": []}',
        encoding="utf-8",
    )
    status = build_status(
        [],
        evidence_drop_import=EvidenceDropImportResult(
            source="drop.zip",
            dest_root=str(tmp_path),
            report_out="external_evidence_drop_import.json",
            ok=False,
            dry_run=False,
            file_count=0,
            failures=["bad_zip_file:drop.zip"],
        ),
    )

    assert status["ok"] is False
    assert status["production_required_failures"] == ["external_evidence_drop_import"]
    assert status["evidence_drop_import"]["failures"] == ["bad_zip_file:drop.zip"]


def test_evidence_intake_status_passes_when_evidence_and_gate_pass(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    audit_root = tmp_path / "outputs" / "production_audit"
    audit_root.mkdir(parents=True)
    (audit_root / "production_evidence_audit.json").write_text(
        '{"production_evidence_ready": true, "blockers": []}',
        encoding="utf-8",
    )
    (audit_root / "integration_gate.json").write_text('{"ok": true}', encoding="utf-8")
    (audit_root / "production_gate.json").write_text('{"ok": true}', encoding="utf-8")
    (audit_root / "objective_completion_audit.json").write_text(
        '{"objective_complete": true, "failed_requirements": []}',
        encoding="utf-8",
    )

    status = build_status([])

    assert status["ok"] is True
    assert status["production_ready"] is True
    assert status["objective_complete"] is True
    assert status["finalization_required"] is True
    assert status["finalization"] is None


def test_evidence_intake_status_clears_finalization_after_green_suite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    audit_root = tmp_path / "outputs" / "production_audit"
    audit_root.mkdir(parents=True)
    (audit_root / "production_evidence_audit.json").write_text(
        '{"production_evidence_ready": true, "blockers": []}',
        encoding="utf-8",
    )
    (audit_root / "integration_gate.json").write_text('{"ok": true}', encoding="utf-8")
    (audit_root / "production_gate.json").write_text('{"ok": true}', encoding="utf-8")
    (audit_root / "objective_completion_audit.json").write_text(
        '{"objective_complete": true, "failed_requirements": []}',
        encoding="utf-8",
    )

    status = build_status(
        [],
        finalization=FinalizationResult(
            command=["./.venv/bin/python", "src/production_audit_suite.py", "--with-pytest"],
            returncode=0,
            ok=True,
            skipped=False,
        ),
    )

    assert status["ok"] is True
    assert status["finalization_required"] is False
    assert status["finalization"]["ok"] is True
    assert status["production_required_failures"] == []
    assert status["post_finalization_refresh"] == []


def test_evidence_intake_status_records_post_finalization_refresh_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    audit_root = tmp_path / "outputs" / "production_audit"
    audit_root.mkdir(parents=True)
    (audit_root / "production_evidence_audit.json").write_text(
        '{"production_evidence_ready": true, "blockers": []}',
        encoding="utf-8",
    )
    (audit_root / "integration_gate.json").write_text('{"ok": true}', encoding="utf-8")
    (audit_root / "production_gate.json").write_text('{"ok": true}', encoding="utf-8")
    (audit_root / "objective_completion_audit.json").write_text(
        '{"objective_complete": true, "failed_requirements": []}',
        encoding="utf-8",
    )

    status = build_status(
        [],
        finalization=FinalizationResult(
            command=["./.venv/bin/python", "src/production_audit_suite.py", "--with-pytest"],
            returncode=0,
            ok=True,
            skipped=False,
        ),
        post_finalization_refresh=[
            RefreshResult(
                name="release_integrity",
                command=["python", "src/release_integrity.py"],
                returncode=1,
                ok=False,
            )
        ],
    )

    assert status["ok"] is False
    assert "post_finalization_refresh" in status["production_required_failures"]
    assert status["post_finalization_refresh"][0]["name"] == "release_integrity"


def test_evidence_intake_status_records_successful_release_refresh_after_finalization(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    audit_root = tmp_path / "outputs" / "production_audit"
    audit_root.mkdir(parents=True)
    (audit_root / "production_evidence_audit.json").write_text(
        '{"production_evidence_ready": true, "blockers": []}',
        encoding="utf-8",
    )
    (audit_root / "integration_gate.json").write_text('{"ok": true}', encoding="utf-8")
    (audit_root / "production_gate.json").write_text('{"ok": true}', encoding="utf-8")
    (audit_root / "objective_completion_audit.json").write_text(
        '{"objective_complete": true, "failed_requirements": []}',
        encoding="utf-8",
    )

    status = build_status(
        [],
        finalization=FinalizationResult(
            command=["./.venv/bin/python", "src/production_audit_suite.py", "--with-pytest"],
            returncode=0,
            ok=True,
            skipped=False,
        ),
        post_finalization_refresh=[
            RefreshResult(
                name="write_production_audit_report",
                command=["python", "scripts/write_production_audit_report.py"],
                returncode=0,
                ok=True,
            ),
            RefreshResult(
                name="write_handoff_report",
                command=["python", "scripts/write_handoff_report.py"],
                returncode=0,
                ok=True,
            ),
            RefreshResult(
                name="release_integrity",
                command=["python", "src/release_integrity.py"],
                returncode=0,
                ok=True,
            ),
            RefreshResult(
                name="report_consistency_audit",
                command=["python", "src/report_consistency_audit.py"],
                returncode=0,
                ok=True,
            ),
        ],
    )

    assert status["ok"] is True
    assert [item["name"] for item in status["post_finalization_refresh"]] == [
        "write_production_audit_report",
        "write_handoff_report",
        "release_integrity",
        "report_consistency_audit",
    ]


def test_evidence_intake_status_fails_when_final_suite_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    audit_root = tmp_path / "outputs" / "production_audit"
    audit_root.mkdir(parents=True)
    (audit_root / "production_evidence_audit.json").write_text(
        '{"production_evidence_ready": true, "blockers": []}',
        encoding="utf-8",
    )
    (audit_root / "integration_gate.json").write_text('{"ok": true}', encoding="utf-8")
    (audit_root / "production_gate.json").write_text('{"ok": true}', encoding="utf-8")
    (audit_root / "objective_completion_audit.json").write_text(
        '{"objective_complete": true, "failed_requirements": []}',
        encoding="utf-8",
    )

    status = build_status(
        [],
        finalization=FinalizationResult(
            command=["./.venv/bin/python", "src/production_audit_suite.py", "--with-pytest"],
            returncode=1,
            ok=False,
            skipped=False,
        ),
    )

    assert status["ok"] is False
    assert status["finalization_required"] is True
    assert "finalization" in status["production_required_failures"]


def test_evidence_intake_status_stays_failed_when_final_suite_exits_zero_but_gate_regresses(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    audit_root = tmp_path / "outputs" / "production_audit"
    audit_root.mkdir(parents=True)
    (audit_root / "production_evidence_audit.json").write_text(
        '{"production_evidence_ready": true, "blockers": []}',
        encoding="utf-8",
    )
    (audit_root / "integration_gate.json").write_text('{"ok": true}', encoding="utf-8")
    (audit_root / "production_gate.json").write_text(
        '{"ok": false, "failed": ["human_ar_holdout_eval"]}',
        encoding="utf-8",
    )
    (audit_root / "objective_completion_audit.json").write_text(
        '{"objective_complete": false, "failed_requirements": ["production_gate_passed"]}',
        encoding="utf-8",
    )

    status = build_status(
        [],
        finalization=FinalizationResult(
            command=["./.venv/bin/python", "src/production_audit_suite.py", "--with-pytest"],
            returncode=0,
            ok=True,
            skipped=False,
        ),
    )

    assert status["ok"] is False
    assert status["finalization_required"] is False
    assert status["production_ready"] is False
    assert status["objective_complete"] is False
    assert status["production_required_failures"] == []


def test_evidence_intake_status_requires_objective_completion_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    audit_root = tmp_path / "outputs" / "production_audit"
    audit_root.mkdir(parents=True)
    (audit_root / "production_evidence_audit.json").write_text(
        '{"production_evidence_ready": true, "blockers": []}',
        encoding="utf-8",
    )
    (audit_root / "integration_gate.json").write_text('{"ok": true}', encoding="utf-8")
    (audit_root / "production_gate.json").write_text('{"ok": true}', encoding="utf-8")

    status = build_status([])

    assert status["ok"] is False
    assert status["production_ready"] is True
    assert status["objective_complete"] is False
    assert status["finalization_required"] is True


def test_evidence_intake_preflight_reports_missing_required_inputs(tmp_path):
    args = _args(tmp_path)
    steps = build_steps(args)

    status = build_preflight_status(steps)

    assert status["dry_run"] is True
    assert status["ok"] is False
    assert status["finalization_required"] is True
    assert str(args.android_litert_source) in status["missing_inputs"]
    assert str(args.ar_holdout_source / "metadata" / "provenance.json") in status["missing_inputs"]
    assert status["invalid_inputs"] == []
    assert len(status["required_inputs"]) == 5


def test_evidence_intake_preflight_accepts_valid_drop_without_existing_inputs(tmp_path):
    args = _args(tmp_path)
    drop = tmp_path / "drop"
    _write_drop(drop)
    args.evidence_drop = drop

    import_result = run_evidence_drop_import(args, dry_run=True)
    status = build_preflight_status(build_steps(args), evidence_drop_import=import_result)

    assert import_result is not None
    assert import_result.ok is True
    assert import_result.dry_run is True
    assert status["ok"] is True
    assert status["missing_inputs"] == []
    assert status["evidence_drop_import"]["file_count"] == 103
    assert not (args.evidence_drop_dest_root / "android_litert_device_report.json").exists()


def test_evidence_intake_rejects_non_integer_import_file_count(tmp_path, monkeypatch):
    args = _args(tmp_path)
    drop = tmp_path / "drop"
    drop.mkdir()
    args.evidence_drop = drop

    def fake_build_import_report(*_args, **_kwargs):
        return {
            "ok": True,
            "file_count": True,
            "failures": [],
        }

    monkeypatch.setattr(
        "src.run_production_evidence_intake.build_import_report",
        fake_build_import_report,
    )

    import_result = run_evidence_drop_import(args, dry_run=True)

    assert import_result is not None
    assert import_result.ok is False
    assert import_result.file_count == 0
    assert "invalid_import_file_count:True" in import_result.failures


def test_evidence_intake_preflight_valid_drop_supersedes_invalid_existing_inputs(tmp_path):
    args = _args(tmp_path)
    args.evidence_drop_dest_root = tmp_path
    args.evidence_drop_overwrite = True
    drop = tmp_path / "drop"
    _write_drop(drop)
    args.evidence_drop = drop
    args.android_litert_source.write_text("", encoding="utf-8")
    (args.ar_holdout_source / "images").mkdir(parents=True)
    (args.ar_holdout_source / "annotations").mkdir()
    (args.ar_holdout_source / "metadata").mkdir()
    (args.ar_holdout_source / "images" / "PLACE_FRAMES_HERE.txt").write_text("replace", encoding="utf-8")
    (args.ar_holdout_source / "annotations" / "bad.txt").write_text("{}", encoding="utf-8")
    (args.ar_holdout_source / "metadata" / "provenance.json").write_text("", encoding="utf-8")
    args.ar_replay_jsonl.parent.mkdir(parents=True, exist_ok=True)
    _write_replay(args, "")

    import_result = run_evidence_drop_import(args, dry_run=True)
    status = build_preflight_status(build_steps(args), evidence_drop_import=import_result)

    assert import_result is not None
    assert import_result.ok is True
    assert status["ok"] is True
    assert status["missing_inputs"] == []
    assert status["invalid_inputs"] == []


def test_evidence_intake_preflight_rejects_drop_destination_mismatch(tmp_path):
    args = _args(tmp_path)
    args.evidence_drop_dest_root = tmp_path / "incoming"
    drop = tmp_path / "drop"
    _write_drop(drop)
    args.evidence_drop = drop

    import_result = run_evidence_drop_import(args, dry_run=True)
    status = build_preflight_status(build_steps(args), evidence_drop_import=import_result)

    assert import_result is not None
    assert import_result.ok is True
    assert status["ok"] is False
    assert {
        "path": str(args.android_litert_source),
        "reason": f"evidence_drop_destination_mismatch:{args.evidence_drop_dest_root}",
        "step": "android_litert_device_validation",
    } in status["invalid_inputs"]


def test_evidence_intake_preflight_accepts_absolute_dest_matching_relative_sources(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(
        android_litert_source=Path("data/incoming/android_litert_device_report.json"),
        android_litert_eval=tmp_path / "android_eval.json",
        ar_holdout_source=Path("data/incoming/ar_device_holdout"),
        ar_holdout_eval=tmp_path / "ar_holdout_eval.json",
        ar_holdout_pipeline=tmp_path / "ar_holdout_pipeline.json",
        ar_replay_jsonl=Path("data/incoming/ar_3d_replay/ar_replay.jsonl"),
        ar_replay_eval=tmp_path / "ar_replay_eval.json",
        evidence_drop=tmp_path / "drop",
        evidence_drop_dest_root=(tmp_path / "data" / "incoming").resolve(),
        evidence_drop_report_out=tmp_path / "external_evidence_drop_import.json",
        evidence_drop_overwrite=False,
        expected_android_artifact=tmp_path / "expected.tflite",
    )
    args.expected_android_artifact.write_bytes(EXPECTED_ANDROID_ARTIFACT_BYTES)
    _write_drop(args.evidence_drop)

    import_result = run_evidence_drop_import(args, dry_run=True)
    status = build_preflight_status(build_steps(args), evidence_drop_import=import_result)

    assert import_result is not None
    assert import_result.ok is True
    assert status["ok"] is True
    assert status["invalid_inputs"] == []


def test_evidence_intake_marks_import_failed_on_destination_mismatch(tmp_path):
    args = _args(tmp_path)
    args.evidence_drop_dest_root = tmp_path / "incoming"
    drop = tmp_path / "drop"
    _write_drop(drop)
    args.evidence_drop = drop

    import_result = fail_import_for_destination_mismatch(
        build_steps(args),
        run_evidence_drop_import(args, dry_run=False),
    )

    assert import_result is not None
    assert import_result.ok is False
    assert any(
        failure.startswith(f"evidence_drop_destination_mismatch:{args.evidence_drop_dest_root}")
        for failure in import_result.failures
    )


def test_evidence_intake_import_copies_valid_drop_for_validation(tmp_path):
    args = _args(tmp_path)
    drop = tmp_path / "drop"
    _write_drop(drop)
    args.evidence_drop = drop

    import_result = run_evidence_drop_import(args, dry_run=False)

    assert import_result is not None
    assert import_result.ok is True
    assert (args.evidence_drop_dest_root / "android_litert_device_report.json").is_file()
    assert (args.evidence_drop_report_out).is_file()
    report = json.loads(args.evidence_drop_report_out.read_text(encoding="utf-8"))
    assert report["copied_artifacts"]


def test_evidence_intake_preflight_rejects_invalid_drop(tmp_path):
    args = _args(tmp_path)
    drop = tmp_path / "drop"
    drop.mkdir()
    args.evidence_drop = drop

    import_result = run_evidence_drop_import(args, dry_run=True)
    status = build_preflight_status(build_steps(args), evidence_drop_import=import_result)

    assert import_result is not None
    assert import_result.ok is False
    assert status["ok"] is False
    assert "missing_required:android_litert_device_report" in import_result.failures


def test_evidence_intake_preflight_passes_when_required_inputs_exist(tmp_path):
    args = _args(tmp_path)
    _write_android(args)
    (args.ar_holdout_source / "images").mkdir(parents=True)
    (args.ar_holdout_source / "annotations").mkdir()
    (args.ar_holdout_source / "images" / "frame.jpg").write_bytes(b"jpg")
    (args.ar_holdout_source / "annotations" / "frame.json").write_text(
        _annotation("frame"),
        encoding="utf-8",
    )
    (args.ar_holdout_source / "metadata").mkdir()
    _write_valid_provenance(args)
    _write_replay(args)

    status = build_preflight_status(build_steps(args))

    assert status["ok"] is True
    assert status["missing_inputs"] == []
    assert status["invalid_inputs"] == []


def test_evidence_intake_preflight_rejects_empty_inputs(tmp_path):
    args = _args(tmp_path)
    args.android_litert_source.write_text("", encoding="utf-8")
    (args.ar_holdout_source / "images").mkdir(parents=True)
    (args.ar_holdout_source / "annotations").mkdir()
    (args.ar_holdout_source / "metadata").mkdir()
    _write_valid_provenance(args)
    _write_replay(args, "")

    status = build_preflight_status(build_steps(args))

    assert status["ok"] is False
    assert {"path": str(args.android_litert_source), "reason": "empty_file", "step": "android_litert_device_validation"} in status["invalid_inputs"]
    assert {"path": str(args.ar_holdout_source / "images"), "reason": "empty_directory", "step": "human_labelled_ar_device_holdout"} in status["invalid_inputs"]
    assert {"path": str(args.ar_holdout_source / "annotations"), "reason": "empty_directory", "step": "human_labelled_ar_device_holdout"} in status["invalid_inputs"]
    assert {"path": str(args.ar_replay_jsonl), "reason": "empty_file", "step": "ar_3d_replay_validation"} in status["invalid_inputs"]


def test_evidence_intake_preflight_rejects_malformed_android_report(tmp_path):
    args = _args(tmp_path)
    args.android_litert_source.write_text("{", encoding="utf-8")
    _write_min_holdout(args)
    _write_replay(args)

    status = build_preflight_status(build_steps(args))

    assert status["ok"] is False
    assert {
        "path": str(args.android_litert_source),
        "reason": "android_report_invalid_json_object",
        "step": "android_litert_device_validation",
    } in status["invalid_inputs"]


def test_evidence_intake_preflight_rejects_android_schema_and_source_type(tmp_path):
    args = _args(tmp_path)
    args.android_litert_source.write_text(
        '{"schema_version":2,"source_type":"desktop_smoke"}',
        encoding="utf-8",
    )
    _write_min_holdout(args)
    _write_replay(args)

    status = build_preflight_status(build_steps(args))

    assert status["ok"] is False
    assert {
        "path": str(args.android_litert_source),
        "reason": "android_report_unsupported_schema_version:2",
        "step": "android_litert_device_validation",
    } in status["invalid_inputs"]
    assert {
        "path": str(args.android_litert_source),
        "reason": "android_report_invalid_source_type:desktop_smoke",
        "step": "android_litert_device_validation",
    } in status["invalid_inputs"]


def test_evidence_intake_preflight_rejects_incomplete_android_report_contract(tmp_path):
    args = _args(tmp_path)
    payload = {
        "schema_version": 1,
        "source_type": "android_litert_device_validation",
        "test_session_id": "FILL_ME",
        "test_app_version": "FILL_ME",
        "test_date_utc": "2026-99-99",
        "device": {
            "model": "Pixel test",
            "manufacturer": "FILL_ME",
            "android_version": "15",
            "soc": "Tensor test",
            "is_emulator": True,
        },
        "runtime": "desktop",
        "artifact": {"sha256": "FILL_ME", "format": "onnx"},
        "input": {"shape": [1, 3, 640, 640], "dtype": "uint8", "profile": "unknown"},
        "latency_ms": {"runs": 1, "mean": 250.0, "p95": -1.0},
        "memory_mb": {"peak": 900.0},
        "output": {"shape": [1, 14, 8400], "finite": True, "min": 0.0, "max": 0.0, "mean": 0.0},
    }
    args.android_litert_source.write_text(json.dumps(payload), encoding="utf-8")
    _write_min_holdout(args)
    _write_replay(args)

    status = build_preflight_status(build_steps(args))

    assert status["ok"] is False
    expected_reasons = {
        "android_report_missing_test_session_id",
        "android_report_missing_test_app_version",
        "android_report_invalid_test_date_utc",
        "android_report_missing_device_manufacturer",
        "android_report_device_must_be_physical:True",
        "android_report_unsupported_runtime:desktop",
        "android_report_missing_artifact_sha256",
        "android_report_unexpected_artifact_format:onnx",
        "android_report_unexpected_input_shape:[1, 3, 640, 640]",
        "android_report_unexpected_input_dtype:uint8",
        "android_report_unexpected_input_profile:unknown",
        "android_report_too_few_runs:1<20",
        "android_report_mean_latency_high:250.000>120.000",
        "android_report_invalid_p95_latency:-1.000",
        "android_report_peak_memory_high:900.000>512.000",
        "android_report_degenerate_output_range",
    }
    actual_reasons = {
        item["reason"]
        for item in status["invalid_inputs"]
        if item["path"] == str(args.android_litert_source)
    }
    assert expected_reasons <= actual_reasons


def test_evidence_intake_preflight_rejects_android_non_finite_latency_and_memory(tmp_path):
    args = _args(tmp_path)
    _write_android(args)
    payload = json.loads(args.android_litert_source.read_text(encoding="utf-8"))
    payload["latency_ms"] = {"runs": float("nan"), "mean": float("nan"), "p95": float("inf")}
    payload["memory_mb"] = {"peak": float("nan")}
    args.android_litert_source.write_text(json.dumps(payload), encoding="utf-8")
    _write_min_holdout(args)
    _write_replay(args)

    status = build_preflight_status(build_steps(args))

    assert status["ok"] is False
    expected_reasons = {
        "android_report_missing_latency_runs",
        "android_report_missing_latency_mean",
        "android_report_missing_latency_p95",
        "android_report_missing_peak_memory",
    }
    actual_reasons = {
        item["reason"]
        for item in status["invalid_inputs"]
        if item["path"] == str(args.android_litert_source)
    }
    assert expected_reasons <= actual_reasons


def test_evidence_intake_preflight_rejects_android_non_integer_shapes(tmp_path):
    args = _args(tmp_path)
    _write_android(args)
    payload = json.loads(args.android_litert_source.read_text(encoding="utf-8"))
    payload["input"]["shape"] = [True, 640, 640, 3]
    payload["output"]["shape"] = [1.0, 14, 8400]
    args.android_litert_source.write_text(json.dumps(payload), encoding="utf-8")
    _write_min_holdout(args)
    _write_replay(args)

    status = build_preflight_status(build_steps(args))

    assert status["ok"] is False
    assert {
        "path": str(args.android_litert_source),
        "reason": "android_report_unexpected_input_shape:[True, 640, 640, 3]",
        "step": "android_litert_device_validation",
    } in status["invalid_inputs"]
    assert {
        "path": str(args.android_litert_source),
        "reason": "android_report_unexpected_output_shape:[1.0, 14, 8400]",
        "step": "android_litert_device_validation",
    } in status["invalid_inputs"]


def test_evidence_intake_preflight_rejects_android_fractional_latency_runs(tmp_path):
    args = _args(tmp_path)
    _write_android(args)
    payload = json.loads(args.android_litert_source.read_text(encoding="utf-8"))
    payload["latency_ms"]["runs"] = 30.5
    args.android_litert_source.write_text(json.dumps(payload), encoding="utf-8")
    _write_min_holdout(args)
    _write_replay(args)

    status = build_preflight_status(build_steps(args))

    assert status["ok"] is False
    assert {
        "path": str(args.android_litert_source),
        "reason": "android_report_invalid_latency_runs",
        "step": "android_litert_device_validation",
    } in status["invalid_inputs"]


def test_evidence_intake_preflight_rejects_android_boolean_latency_runs(tmp_path):
    args = _args(tmp_path)
    _write_android(args)
    payload = json.loads(args.android_litert_source.read_text(encoding="utf-8"))
    payload["latency_ms"]["runs"] = True
    args.android_litert_source.write_text(json.dumps(payload), encoding="utf-8")
    _write_min_holdout(args)
    _write_replay(args)

    status = build_preflight_status(build_steps(args))

    assert status["ok"] is False
    assert {
        "path": str(args.android_litert_source),
        "reason": "android_report_invalid_latency_runs",
        "step": "android_litert_device_validation",
    } in status["invalid_inputs"]


def test_evidence_intake_preflight_rejects_wrong_android_artifact_hash(tmp_path):
    args = _args(tmp_path)
    _write_android(args)
    payload = json.loads(args.android_litert_source.read_text(encoding="utf-8"))
    payload["artifact"]["sha256"] = "wrong-sha"
    args.android_litert_source.write_text(json.dumps(payload), encoding="utf-8")
    _write_min_holdout(args)
    _write_replay(args)

    status = build_preflight_status(build_steps(args))

    assert status["ok"] is False
    assert {
        "path": str(args.android_litert_source),
        "reason": "android_report_artifact_sha256_mismatch",
        "step": "android_litert_device_validation",
    } in status["invalid_inputs"]


def test_evidence_intake_preflight_rejects_missing_expected_android_artifact(tmp_path):
    args = _args(tmp_path)
    args.expected_android_artifact.unlink()
    _write_android(args)
    _write_min_holdout(args)
    _write_replay(args)

    status = build_preflight_status(build_steps(args))

    assert status["ok"] is False
    assert {
        "path": str(args.android_litert_source),
        "reason": f"android_report_missing_expected_artifact:{args.expected_android_artifact}",
        "step": "android_litert_device_validation",
    } in status["invalid_inputs"]


def test_evidence_intake_preflight_rejects_malformed_ar_replay_jsonl(tmp_path):
    args = _args(tmp_path)
    _write_android(args)
    _write_min_holdout(args)
    _write_replay(
        args,
        'not-json\n[]\n{"schema_version":2,"source_type":"desktop_replay"}\n',
    )

    status = build_preflight_status(build_steps(args))

    assert status["ok"] is False
    assert {
        "path": str(args.ar_replay_jsonl),
        "reason": "ar_replay_invalid_json_line:1",
        "step": "ar_3d_replay_validation",
    } in status["invalid_inputs"]
    assert {
        "path": str(args.ar_replay_jsonl),
        "reason": "ar_replay_line_not_object:2",
        "step": "ar_3d_replay_validation",
    } in status["invalid_inputs"]
    assert {
        "path": str(args.ar_replay_jsonl),
        "reason": "ar_replay_line_unsupported_schema_version:3:2",
        "step": "ar_3d_replay_validation",
    } in status["invalid_inputs"]
    assert {
        "path": str(args.ar_replay_jsonl),
        "reason": "ar_replay_line_invalid_source_type:3:desktop_replay",
        "step": "ar_3d_replay_validation",
    } in status["invalid_inputs"]


def test_evidence_intake_preflight_rejects_boolean_schema_versions(tmp_path):
    args = _args(tmp_path)
    _write_android(args)
    android = json.loads(args.android_litert_source.read_text(encoding="utf-8"))
    android["schema_version"] = True
    args.android_litert_source.write_text(json.dumps(android), encoding="utf-8")
    _write_min_holdout(args)
    payload = json.loads(_valid_replay_line())
    payload["schema_version"] = True
    _write_replay(args, json.dumps(payload) + "\n")

    status = build_preflight_status(build_steps(args))

    assert status["ok"] is False
    assert {
        "path": str(args.android_litert_source),
        "reason": "android_report_unsupported_schema_version:True",
        "step": "android_litert_device_validation",
    } in status["invalid_inputs"]
    assert {
        "path": str(args.ar_replay_jsonl),
        "reason": "ar_replay_line_unsupported_schema_version:1:True",
        "step": "ar_3d_replay_validation",
    } in status["invalid_inputs"]


def test_evidence_intake_preflight_rejects_incomplete_ar_replay_contract(tmp_path):
    args = _args(tmp_path)
    _write_android(args)
    _write_min_holdout(args)
    payload = json.loads(_valid_replay_line())
    payload["capture_index"] = -1
    payload["camera_pose_ref"] = None
    payload["screen_points"]["a"] = [100.0]
    payload["floor_raycast_hits"].pop("b")
    payload["residual"] = -0.1
    payload["recovered_plane"]["normal"] = [2.0, 0.0, 0.0]
    payload["c_height_value"] = -0.1
    _write_replay(args, json.dumps(payload) + "\n")

    status = build_preflight_status(build_steps(args))

    assert status["ok"] is False
    expected_reasons = {
        "ar_replay_line_negative_capture_index:1:-1",
        "ar_replay_line_missing_camera_pose_evidence:1",
        "ar_replay_line_invalid_screen_point_a:1",
        "ar_replay_line_invalid_floor_hit_b:1",
        "ar_replay_line_negative_residual:1",
        "ar_replay_line_invalid_recovered_plane:1",
        "ar_replay_line_negative_c_height_value:1",
    }
    actual_reasons = {
        item["reason"]
        for item in status["invalid_inputs"]
        if item["path"] == str(args.ar_replay_jsonl)
    }
    assert expected_reasons <= actual_reasons


def test_evidence_intake_preflight_rejects_decreasing_ar_replay_capture_index(tmp_path):
    args = _args(tmp_path)
    _write_android(args)
    _write_min_holdout(args)
    observations = [json.loads(_valid_replay_line()) for _ in range(3)]
    for index, observation in enumerate(observations):
        observation["frame_id"] = f"frame_{index:04d}"
        observation["camera_pose_ref"] = f"pose_{index:04d}"
        observation["capture_index"] = index
    observations[2]["capture_index"] = 0
    _write_replay(args, "\n".join(json.dumps(observation) for observation in observations) + "\n")

    status = build_preflight_status(build_steps(args))

    assert status["ok"] is False
    assert {
        "path": str(args.ar_replay_jsonl),
        "reason": "ar_replay_line_decreasing_capture_index:3:0<1:previous_line=2",
        "step": "ar_3d_replay_validation",
    } in status["invalid_inputs"]


def test_evidence_intake_preflight_rejects_repeated_replay_frame_without_wheel_identity(tmp_path):
    args = _args(tmp_path)
    _write_android(args)
    _write_min_holdout(args)
    observations = [json.loads(_valid_replay_line()) for _ in range(3)]
    for index, observation in enumerate(observations):
        observation["frame_id"] = f"frame_{index:04d}"
        observation["camera_pose_ref"] = f"pose_{index:04d}"
        observation["capture_index"] = index
    observations[2]["capture_index"] = 1
    observations[2]["frame_id"] = "frame_0001"
    observations[2]["camera_pose_ref"] = "pose_0001"
    _write_replay(args, "\n".join(json.dumps(observation) for observation in observations) + "\n")

    status = build_preflight_status(build_steps(args))

    assert status["ok"] is False
    assert {
        "path": str(args.ar_replay_jsonl),
        "reason": "ar_replay_repeated_frame_missing_wheel_identity:s1:frame_0001:1:lines=2,3",
        "step": "ar_3d_replay_validation",
    } in status["invalid_inputs"]


def test_evidence_intake_preflight_accepts_repeated_replay_frame_with_unique_wheel_identity(tmp_path):
    args = _args(tmp_path)
    _write_android(args)
    _write_min_holdout(args)
    observations = [json.loads(_valid_replay_line()) for _ in range(3)]
    for index, observation in enumerate(observations):
        observation["frame_id"] = f"frame_{index:04d}"
        observation["camera_pose_ref"] = f"pose_{index:04d}"
        observation["capture_index"] = index
    observations[1]["wheel_index"] = 0
    observations[2]["capture_index"] = 1
    observations[2]["frame_id"] = "frame_0001"
    observations[2]["camera_pose_ref"] = "pose_0001"
    observations[2]["wheel_index"] = 1
    _write_replay(args, "\n".join(json.dumps(observation) for observation in observations) + "\n")

    status = build_preflight_status(build_steps(args))

    reasons = {
        item["reason"]
        for item in status["invalid_inputs"]
        if item["path"] == str(args.ar_replay_jsonl)
    }
    assert "ar_replay_repeated_frame_missing_wheel_identity:s1:frame_0001:1:lines=2,3" not in reasons
    assert "ar_replay_line_decreasing_capture_index:3:0<1:previous_line=2" not in reasons


def test_evidence_intake_preflight_rejects_invalid_holdout_provenance(tmp_path):
    args = _args(tmp_path)
    _write_android(args)
    _write_min_holdout(args)
    provenance = json.loads((args.ar_holdout_source / "metadata" / "provenance.json").read_text(encoding="utf-8"))
    provenance["schema_version"] = 2
    provenance["source_type"] = "desktop_holdout"
    provenance["label_type"] = "human_labelled"
    provenance["review_status"] = "approved"
    provenance["capture_device"] = "FILL_ME"
    provenance["capture_app_version"] = "FILL_ME"
    provenance["capture_date_utc"] = "2026-99-99"
    provenance["annotator"] = "same_person"
    provenance["reviewer"] = "same_person"
    (args.ar_holdout_source / "metadata" / "provenance.json").write_text(
        json.dumps(provenance),
        encoding="utf-8",
    )
    _write_replay(args)

    status = build_preflight_status(build_steps(args))

    assert status["ok"] is False
    expected_reasons = {
        "ar_holdout_unsupported_schema_version:2",
        "ar_holdout_invalid_source_type:desktop_holdout",
        "ar_holdout_invalid_label_type:human_labelled",
        "ar_holdout_invalid_review_status:approved",
        "ar_holdout_missing_capture_device",
        "ar_holdout_missing_capture_app_version",
        "ar_holdout_invalid_capture_date_utc",
        "ar_holdout_annotator_reviewer_not_independent",
    }
    actual_reasons = {
        item["reason"]
        for item in status["invalid_inputs"]
        if item["path"] == str(args.ar_holdout_source)
    }
    assert expected_reasons <= actual_reasons


def test_evidence_intake_preflight_rejects_placeholder_files(tmp_path):
    args = _args(tmp_path)
    _write_android(args)
    (args.ar_holdout_source / "images").mkdir(parents=True)
    (args.ar_holdout_source / "annotations").mkdir()
    (args.ar_holdout_source / "metadata").mkdir()
    (args.ar_holdout_source / "images" / "PLACE_FRAMES_HERE.txt").write_text("replace", encoding="utf-8")
    (args.ar_holdout_source / "images" / "frame.jpg").write_bytes(b"jpg")
    (args.ar_holdout_source / "annotations" / "frame.json").write_text(
        _annotation("frame"),
        encoding="utf-8",
    )
    _write_valid_provenance(args)
    _write_replay(args)

    status = build_preflight_status(build_steps(args))

    assert status["ok"] is False
    assert status["invalid_inputs"] == [
        {
            "path": str(args.ar_holdout_source / "images"),
            "reason": "placeholder_files:PLACE_FRAMES_HERE.txt",
            "step": "human_labelled_ar_device_holdout",
        }
    ]


def test_evidence_intake_preflight_rejects_holdout_stem_mismatch(tmp_path):
    args = _args(tmp_path)
    _write_android(args)
    (args.ar_holdout_source / "images").mkdir(parents=True)
    (args.ar_holdout_source / "annotations").mkdir()
    (args.ar_holdout_source / "metadata").mkdir()
    (args.ar_holdout_source / "images" / "frame_0001.jpg").write_bytes(b"jpg")
    (args.ar_holdout_source / "annotations" / "frame_9999.json").write_text(
        _annotation("frame_9999"),
        encoding="utf-8",
    )
    _write_valid_provenance(args)
    _write_replay(args)

    status = build_preflight_status(build_steps(args))

    assert status["ok"] is False
    assert {
        "path": str(args.ar_holdout_source),
        "reason": "ar_holdout_missing_annotations:frame_0001",
        "step": "human_labelled_ar_device_holdout",
    } in status["invalid_inputs"]
    assert {
        "path": str(args.ar_holdout_source),
        "reason": "ar_holdout_missing_images:frame_9999",
        "step": "human_labelled_ar_device_holdout",
    } in status["invalid_inputs"]


def test_evidence_intake_preflight_rejects_bad_holdout_extensions(tmp_path):
    args = _args(tmp_path)
    _write_android(args)
    (args.ar_holdout_source / "images").mkdir(parents=True)
    (args.ar_holdout_source / "annotations").mkdir()
    (args.ar_holdout_source / "metadata").mkdir()
    (args.ar_holdout_source / "images" / "frame_0001.txt").write_text("not an image", encoding="utf-8")
    (args.ar_holdout_source / "annotations" / "frame_0001.txt").write_text("{}", encoding="utf-8")
    _write_valid_provenance(args)
    _write_replay(args)

    status = build_preflight_status(build_steps(args))

    assert status["ok"] is False
    assert {
        "path": str(args.ar_holdout_source),
        "reason": "ar_holdout_bad_image_extensions:frame_0001.txt",
        "step": "human_labelled_ar_device_holdout",
    } in status["invalid_inputs"]
    assert {
        "path": str(args.ar_holdout_source),
        "reason": "ar_holdout_bad_annotation_extensions:frame_0001.txt",
        "step": "human_labelled_ar_device_holdout",
    } in status["invalid_inputs"]


def test_evidence_intake_preflight_rejects_bad_holdout_annotation_contract(tmp_path):
    args = _args(tmp_path)
    _write_android(args)
    (args.ar_holdout_source / "images").mkdir(parents=True)
    (args.ar_holdout_source / "annotations").mkdir()
    (args.ar_holdout_source / "metadata").mkdir()
    (args.ar_holdout_source / "images" / "frame_0001.jpg").write_bytes(b"jpg")
    (args.ar_holdout_source / "annotations" / "frame_0001.json").write_text(
        '{"schema_version":1,"frame_id":"wrong","image":"nested/frame_9999.jpg","wheels":{}}',
        encoding="utf-8",
    )
    _write_valid_provenance(args)
    _write_replay(args)

    status = build_preflight_status(build_steps(args))

    assert status["ok"] is False
    assert {
        "path": str(args.ar_holdout_source),
        "reason": "ar_holdout_annotation_frame_id_mismatch:frame_0001.json:wrong",
        "step": "human_labelled_ar_device_holdout",
    } in status["invalid_inputs"]
    assert {
        "path": str(args.ar_holdout_source),
        "reason": "ar_holdout_annotation_image_not_filename:frame_0001.json:nested/frame_9999.jpg",
        "step": "human_labelled_ar_device_holdout",
    } in status["invalid_inputs"]
    assert {
        "path": str(args.ar_holdout_source),
        "reason": "ar_holdout_annotation_wheels_not_array:frame_0001.json",
        "step": "human_labelled_ar_device_holdout",
    } in status["invalid_inputs"]


def test_evidence_intake_preflight_rejects_holdout_annotation_missing_schema_version(tmp_path):
    args = _args(tmp_path)
    _write_android(args)
    (args.ar_holdout_source / "images").mkdir(parents=True)
    (args.ar_holdout_source / "annotations").mkdir()
    (args.ar_holdout_source / "metadata").mkdir()
    (args.ar_holdout_source / "images" / "frame_0001.jpg").write_bytes(b"jpg")
    (args.ar_holdout_source / "annotations" / "frame_0001.json").write_text(
        '{"frame_id":"frame_0001","image":"frame_0001.jpg","wheels":[]}',
        encoding="utf-8",
    )
    _write_valid_provenance(args)
    _write_replay(args)

    status = build_preflight_status(build_steps(args))

    assert status["ok"] is False
    assert {
        "path": str(args.ar_holdout_source),
        "reason": "ar_holdout_annotation_unsupported_schema_version:frame_0001.json:missing",
        "step": "human_labelled_ar_device_holdout",
    } in status["invalid_inputs"]


def test_evidence_intake_preflight_rejects_invalid_holdout_wheel_schema(tmp_path):
    args = _args(tmp_path)
    _write_android(args)
    (args.ar_holdout_source / "images").mkdir(parents=True)
    (args.ar_holdout_source / "annotations").mkdir()
    (args.ar_holdout_source / "metadata").mkdir()
    (args.ar_holdout_source / "images" / "frame_0001.jpg").write_bytes(b"jpg")
    (args.ar_holdout_source / "annotations" / "frame_0001.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frame_id": "frame_0001",
                "image": "frame_0001.jpg",
                "wheels": [
                    {
                        "bbox_xyxy": [30, 10, 10, 30],
                        "points": {"a": [12, 28], "b": [28, 28]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_valid_provenance(args)
    _write_replay(args)

    status = build_preflight_status(build_steps(args))

    assert status["ok"] is False
    assert {
        "path": str(args.ar_holdout_source),
        "reason": "ar_holdout_annotation_wheel_nonpositive_bbox:frame_0001.json:wheel[0]",
        "step": "human_labelled_ar_device_holdout",
    } in status["invalid_inputs"]
    assert {
        "path": str(args.ar_holdout_source),
        "reason": "ar_holdout_annotation_wheel_invalid_point_c_disc_bottom:frame_0001.json:wheel[0]",
        "step": "human_labelled_ar_device_holdout",
    } in status["invalid_inputs"]


def test_evidence_intake_preflight_rejects_holdout_points_outside_bbox(tmp_path):
    args = _args(tmp_path)
    _write_android(args)
    (args.ar_holdout_source / "images").mkdir(parents=True)
    (args.ar_holdout_source / "annotations").mkdir()
    (args.ar_holdout_source / "metadata").mkdir()
    (args.ar_holdout_source / "images" / "frame_0001.jpg").write_bytes(b"jpg")
    (args.ar_holdout_source / "annotations" / "frame_0001.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frame_id": "frame_0001",
                "image": "frame_0001.jpg",
                "wheels": [
                    {
                        "bbox_xyxy": [10, 10, 30, 30],
                        "points": {"a": [9, 28], "b": [28, 28], "c_disc_bottom": [20, 31]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_valid_provenance(args)
    _write_replay(args)

    status = build_preflight_status(build_steps(args))

    assert status["ok"] is False
    assert {
        "path": str(args.ar_holdout_source),
        "reason": "ar_holdout_annotation_wheel_point_a_outside_bbox:frame_0001.json:wheel[0]",
        "step": "human_labelled_ar_device_holdout",
    } in status["invalid_inputs"]
    assert {
        "path": str(args.ar_holdout_source),
        "reason": "ar_holdout_annotation_wheel_point_c_disc_bottom_outside_bbox:frame_0001.json:wheel[0]",
        "step": "human_labelled_ar_device_holdout",
    } in status["invalid_inputs"]
