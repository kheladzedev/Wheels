from __future__ import annotations

from src.data_readiness_decision import build_decision, render_markdown


def _dataset_audit() -> dict:
    return {
        "ok": False,
        "counts": {
            "configs": 22,
            "ok": 2,
            "failed": 20,
            "total_train_images": 5705,
            "total_val_images": 1287,
            "total_wheel_labels": 12148,
        },
        "gate": {
            "ok": True,
            "scope": "configured_subset",
            "configs": [
                "configs/pose_dataset_real_v1_self_strict.yaml",
                "configs/pose_dataset_real_v1_self_plus_ue_synthetic_strict.yaml",
            ],
            "counts": {
                "configs": 2,
                "ok": 2,
                "failed": 0,
                "total_train_images": 423,
                "total_val_images": 106,
                "total_wheel_labels": 602,
            },
        },
        "reports": [
            {
                "config": "configs/pose_dataset_real_v1_self_strict.yaml",
                "ok": True,
                "leakage": {"stem_overlap_count": 0, "hash_overlap_count": 0},
            },
            {
                "config": "configs/pose_dataset_real_v1_self_plus_ue_synthetic_strict.yaml",
                "ok": True,
                "leakage": {"stem_overlap_count": 0, "hash_overlap_count": 0},
            },
        ],
    }


def _production_evidence() -> dict:
    return {
        "ok": True,
        "production_evidence_ready": False,
        "blockers": [
            "android_litert_device_validation",
            "human_labelled_ar_device_holdout",
            "ar_3d_replay_validation",
        ],
    }


def test_data_readiness_marks_handoff_ready_but_not_production() -> None:
    report = build_decision(
        dataset_audit=_dataset_audit(),
        model_inventory={"counts": {"train_runs": 13, "artifacts": 35}},
        operating_point={"selected": {"conf": 0.8, "false_positive_rate": 0.147}},
        production_evidence=_production_evidence(),
        audit_suite={"ok": True, "integration_ready": True, "production_ready": False},
    )

    assert report["verdict"]["test_handoff_ready"] is True
    assert report["verdict"]["production_data_ready"] is False
    assert report["verdict"]["production_training_data_sufficient"] is False
    assert report["risk_flags"]["dirty_legacy_dataset_configs"]["failed_configs"] == 20
    assert "human_labelled_ar_device_holdout" in report["production_blockers"]


def test_data_readiness_flags_small_strict_subset_and_no_leakage() -> None:
    report = build_decision(
        dataset_audit=_dataset_audit(),
        model_inventory={"counts": {}},
        operating_point={"selected": {}},
        production_evidence=_production_evidence(),
        audit_suite={"ok": True, "integration_ready": True, "production_ready": False},
    )

    assert report["current_data"]["strict_gate"]["train_images"] == 423
    assert report["current_data"]["strict_gate"]["val_images"] == 106
    assert report["current_data"]["strict_gate"]["wheel_labels"] == 602
    assert report["current_data"]["strict_gate"]["leakage_ok"] is True
    assert report["risk_flags"]["strict_subset_size"]["status"] == "too_small_for_production"


def test_data_readiness_can_run_before_suite_status_exists() -> None:
    report = build_decision(
        dataset_audit=_dataset_audit(),
        model_inventory={"counts": {}},
        operating_point={"selected": {}},
        production_evidence=_production_evidence(),
        audit_suite={},
    )

    assert report["verdict"]["test_handoff_ready"] is True
    assert report["verdict"]["production_data_ready"] is False


def test_data_readiness_markdown_has_senior_decision() -> None:
    report = build_decision(
        dataset_audit=_dataset_audit(),
        model_inventory={"counts": {"tflite_artifacts": 1, "coreml_artifacts": 1}},
        operating_point={"selected": {"conf": 0.8}},
        production_evidence=_production_evidence(),
        audit_suite={"ok": True, "integration_ready": True, "production_ready": False},
    )

    text = render_markdown(report)

    assert "Для тестовой передачи: можно отдавать" in text
    assert "Для production: данных недостаточно" in text
    assert "не дообучать вслепую" in text
    assert "2000" in text
