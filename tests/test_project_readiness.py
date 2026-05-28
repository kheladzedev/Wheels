"""Unit tests for the handoff readiness summary."""

from __future__ import annotations

import project_readiness as pr


def test_model_pool_check_reports_missing_models(tmp_path):
    root = tmp_path / "sketchfab"
    root.mkdir()
    for i in range(2):
        (root / f"{i}.glb").write_bytes(b"glb")
    (root / "ov_objaverse.glb").write_bytes(b"glb")
    (root / "rejected").mkdir()
    (root / "rejected" / "bad.glb").write_bytes(b"bad")

    check = pr.model_pool_check(root, target_total=4)

    assert check.ok is False
    assert check.name == "car_body_model_pool"
    assert "3/4" in check.detail
    assert "sketchfab=2" in check.detail
    assert "objaverse_fallback=1" in check.detail
    assert "missing=1" in check.detail
    assert "rejected=1" in check.detail


def test_yolo_dataset_check_requires_matching_labels(tmp_path):
    root = tmp_path / "dataset"
    for sub in ("images/train", "labels/train", "images/val", "labels/val"):
        (root / sub).mkdir(parents=True)
    (root / "images/train" / "a.jpg").write_bytes(b"img")
    (root / "labels/train" / "a.txt").write_text("0 0 0 1 1")
    (root / "images/val" / "b.jpg").write_bytes(b"img")

    check = pr.yolo_dataset_check("ds", root)

    assert check.ok is False
    assert "train=1/1" in check.detail
    assert "val=1/0" in check.detail


def test_incoming_min_check_counts_wheels(tmp_path):
    root = tmp_path / "incoming"
    for sub in ("images", "annotations"):
        (root / sub).mkdir(parents=True)
    (root / "images" / "a.png").write_bytes(b"img")
    (root / "annotations" / "a.json").write_text(
        '{"wheels": [{"id": 1}, {"id": 2}]}',
        encoding="utf-8",
    )

    check = pr.incoming_min_check("incoming", root, min_images=1, min_wheels=2)

    assert check.ok is True
    assert "wheels=2/2" in check.detail


def test_incoming_diagnostic_check_never_blocks(tmp_path):
    root = tmp_path / "incoming"
    for sub in ("images", "annotations"):
        (root / sub).mkdir(parents=True)
    (root / "images" / "a.png").write_bytes(b"img")
    (root / "annotations" / "a.json").write_text(
        '{"wheels": [{"id": 1}]}',
        encoding="utf-8",
    )

    check = pr.incoming_diagnostic_check("diagnostic", root)

    assert check.ok is True
    assert "images=1" in check.detail
    assert "annotations=1" in check.detail
    assert "wheels=1" in check.detail


def test_eval_comparison_check_reports_not_promoted(tmp_path):
    champion = tmp_path / "champion.json"
    candidate = tmp_path / "candidate.json"
    champion.write_text(
        '{"oks": {"mean": 0.9}, "rates": {"false_negative_rate": 0.2, '
        '"false_positive_rate": 0.1}, "metrics_bbox": {"mAP50": 0.7}}',
        encoding="utf-8",
    )
    candidate.write_text(
        '{"oks": {"mean": 0.8}, "rates": {"false_negative_rate": 0.3, '
        '"false_positive_rate": 0.2}, "metrics_bbox": {"mAP50": 0.6}}',
        encoding="utf-8",
    )

    check = pr.eval_comparison_check("candidate", champion, candidate)

    assert check.ok is True
    assert "not_promoted" in check.detail
    assert "oks=0.800" in check.detail


def test_export_drift_diagnostic_check_never_blocks_present_report(tmp_path):
    report = tmp_path / "drift.json"
    report.write_text(
        '{"ok": false, "samples_matched": 14, "samples_checked": 20, '
        '"max_bbox_drift_px": 8.5, "max_kp_drift_px": 13.4, '
        '"max_conf_drift": 0.23}',
        encoding="utf-8",
    )

    check = pr.export_drift_diagnostic_check("drift", report)

    assert check.ok is True
    assert "not_certified" in check.detail
    assert "14/20" in check.detail


def test_certification_diagnostic_check_reports_not_certified(tmp_path):
    report = tmp_path / "tflite.json"
    report.write_text(
        '{"certified": false, "artifact": {"path": "model.tflite"}, '
        '"aggregate_eval": {"bbox_map50": 0.69, "oks_mean": 0.88, '
        '"false_negative_rate": 0.28, "false_positive_rate": 0.27}}',
        encoding="utf-8",
    )

    check = pr.certification_diagnostic_check("tflite", report)

    assert check.ok is True
    assert "not_certified" in check.detail
    assert "model.tflite" in check.detail
    assert "bbox_mAP50=0.690" in check.detail


def test_certification_diagnostic_check_reports_backend_certification(tmp_path):
    report = tmp_path / "export_certification.json"
    report.write_text(
        '{"certified": true, "scope": "desktop_export_backend_certification_not_android_device", '
        '"backends": {"onnx": {"certified": true}, "tflite": {"certified": true}}}',
        encoding="utf-8",
    )

    check = pr.certification_diagnostic_check("export_certification", report)

    assert check.ok is True
    assert "certified" in check.detail
    assert "onnx=True" in check.detail
    assert "tflite=True" in check.detail


def test_render_images_check_requires_min_pngs(tmp_path):
    root = tmp_path / "renders"
    root.mkdir()
    (root / "a.png").write_bytes(b"img")

    check = pr.render_images_check("renders", root, min_images=2)

    assert check.ok is False
    assert "png=1/2" in check.detail


def test_image_content_check_reports_missing_cv2(monkeypatch, tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    (root / "a.png").write_bytes(b"not-an-image")
    monkeypatch.setitem(__import__("sys").modules, "cv2", None)

    check = pr.image_content_check("content", root, min_nonblack=1)

    assert check.ok is False
    assert "cv2" in check.detail


def test_file_check_ok(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"x")

    check = pr.file_check("artifact", path)

    assert check.ok is True
    assert check.name == "artifact"


def test_collect_checks_includes_handoff_orchestration(monkeypatch):
    class Args:
        model_pool_root = None
        model_target = 300
        mcp_host = "127.0.0.1"
        mcp_port = 55557
        mcp_timeout = 0.01

    monkeypatch.setattr(pr, "model_pool_check", lambda root, target: pr.Check("car_body_model_pool", True, "ok"))
    monkeypatch.setattr(pr, "mcp_check", lambda host, port, timeout: pr.Check("unreal_mcp", True, "ok"))
    monkeypatch.setattr(pr, "file_check", lambda name, path: pr.Check(name, True, str(path)))
    monkeypatch.setattr(pr, "dataset_check", lambda name, root: pr.Check(name, True, str(root)))
    monkeypatch.setattr(pr, "yolo_dataset_check", lambda name, root: pr.Check(name, True, str(root)))
    monkeypatch.setattr(
        pr,
        "incoming_min_check",
        lambda name, root, min_images, min_wheels: pr.Check(name, True, str(root)),
    )
    monkeypatch.setattr(
        pr,
        "incoming_diagnostic_check",
        lambda name, root: pr.Check(name, True, str(root)),
    )
    monkeypatch.setattr(
        pr,
        "render_images_check",
        lambda name, root, min_images: pr.Check(name, True, str(root)),
    )
    monkeypatch.setattr(
        pr,
        "image_content_check",
        lambda name, root, min_nonblack: pr.Check(name, True, str(root)),
    )
    monkeypatch.setattr(
        pr,
        "eval_comparison_check",
        lambda name, champion_path, candidate_path: pr.Check(name, True, str(candidate_path)),
    )
    monkeypatch.setattr(
        pr,
        "export_drift_diagnostic_check",
        lambda name, path: pr.Check(name, True, str(path)),
    )
    monkeypatch.setattr(
        pr,
        "certification_diagnostic_check",
        lambda name, path: pr.Check(name, True, str(path)),
    )

    names = [check.name for check in pr.collect_checks(Args())]

    assert "mcp_wait_wrapper" in names
    assert "finish_orchestrator" in names
    assert "production_audit_suite_runner" in names
    assert "ue_sketchfab_geometry_incoming" in names
    assert "ue_sketchfab_geometry_rgb_content" in names
    assert "ue_sketchfab_geometry_yolo" in names
    assert "ue_sketchfab_geometry_clean_incoming" in names
    assert "ue_sketchfab_geometry_clean_rgb_content" in names
    assert "ue_sketchfab_geometry_clean_yolo" in names
    assert "real_self_ue_plus_sketchfab_clean_yolo" in names
    assert "real_self_ue_plus_sketchfab_clean_eval_diagnostic" in names
    assert "production_readiness_audit" in names
    assert "model_package_manifest" in names
    assert "model_inventory_json" in names
    assert "model_inventory_report" in names
    assert "model_selection_audit_json" in names
    assert "model_selection_audit_report" in names
    assert "spec_compliance_audit_json" in names
    assert "spec_compliance_audit_report" in names
    assert "dataset_audit_json" in names
    assert "dataset_audit_report" in names
    assert "release_integrity_json" in names
    assert "release_package_report" in names
    assert "performance_audit_json" in names
    assert "performance_audit_report" in names
    assert "senior_ml_audit_json" in names
    assert "senior_ml_audit_report" in names
    assert "objective_completion_audit_json" in names
    assert "objective_completion_audit_report" in names
    assert "objective_completion_audit_runner" in names
    assert "report_consistency_audit_json" in names
    assert "report_consistency_audit_report" in names
    assert "report_consistency_audit_runner" in names
    assert "runtime_contract_audit" in names
    assert "integration_gate_report" in names
    assert "production_gate_report" in names
    assert "champion_onnx_drift_diagnostic" in names
    assert "export_parity_audit_json" in names
    assert "export_parity_audit_report" in names
    assert "production_evidence_intake_doc" in names
    assert "production_evidence_intake_runner" in names
    assert "production_evidence_intake_preflight_status" in names
    assert "external_evidence_drop_importer" in names
    assert "external_evidence_drop_import_runner" in names
    assert "external_evidence_return_template" in names
    assert "external_evidence_return_template_manifest" in names
    assert "external_evidence_return_template_writer" in names
    assert "android_litert_harness_doc" in names
    assert "android_litert_harness_test" in names
    assert "external_evidence_handoff_bundle_doc" in names
    assert "external_evidence_handoff_bundle" in names
    assert "external_evidence_handoff_bundle_manifest" in names
    assert "external_evidence_handoff_bundle_verification" in names
    assert "external_evidence_handoff_bundle_builder" in names
    assert "external_evidence_handoff_bundle_verifier" in names
    assert "champion_tflite_certification_diagnostic" in names
    assert "ue_sketchfab_pseudo_yield_diagnostic" in names
    assert "ar_replay_validator" in names
    assert "ar_holdout_provenance_template" in names
    assert "ar_holdout_harness_doc" in names
    assert "ar_holdout_harness_writer" in names
    assert "ar_holdout_provenance_template_writer" in names
    assert "ar_replay_log_template" in names
    assert "ar_replay_harness_doc" in names
    assert "ar_replay_harness_logger" in names
    assert "ar_replay_template_writer" in names
    assert "ar_holdout_evaluator" in names
    assert "litert_runtime_smoke" in names
    assert "litert_runtime_checker" in names
    assert "android_litert_harness_readme" in names
    assert "android_litert_harness_kotlin_test" in names
    assert "ar_holdout_harness_readme" in names
    assert "ar_holdout_harness_kotlin_writer" in names
    assert "ar_replay_harness_readme" in names
    assert "ar_replay_harness_kotlin_logger" in names
