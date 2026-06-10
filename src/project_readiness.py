"""Summarize ML/synthetic-data readiness for the current handoff.

This is intentionally filesystem-based and conservative: it reports what
is present, what is missing, and which external services are unavailable
without trying to mutate the workspace.
"""

from __future__ import annotations

import argparse
import json
import socket
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_MODEL_TARGET = 300
DEFAULT_MODEL_POOL_ROOT = Path("data/sketchfab_cars")
DEFAULT_MCP_HOST = "127.0.0.1"
DEFAULT_MCP_PORT = 55557


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _count_files(root: Path, pattern: str) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for _ in root.glob(pattern))


def _tcp_open(host: str, port: int, timeout_s: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def model_pool_check(root: Path, target_total: int) -> Check:
    glbs = _count_files(root, "*.glb")
    objaverse = _count_files(root, "ov_*.glb")
    sketchfab = max(0, glbs - objaverse)
    rejected = _count_files(root / "rejected", "*.glb")
    missing = max(0, target_total - glbs)
    return Check(
        name="car_body_model_pool",
        ok=glbs >= target_total,
        detail=(
            f"{glbs}/{target_total} clean GLBs, sketchfab={sketchfab}, "
            f"objaverse_fallback={objaverse}, missing={missing}, rejected={rejected}"
        ),
    )


def sketchfab_check(root: Path, target_total: int) -> Check:
    """Backward-compatible wrapper for older tests/callers."""
    return model_pool_check(root, target_total)


def mcp_check(host: str, port: int, timeout_s: float) -> Check:
    ok = _tcp_open(host, port, timeout_s)
    return Check(
        name="unreal_mcp",
        ok=ok,
        detail=f"{host}:{port} {'reachable' if ok else 'not reachable'}",
    )


def file_check(name: str, path: Path) -> Check:
    return Check(name=name, ok=path.is_file(), detail=str(path))


def dataset_check(name: str, root: Path) -> Check:
    images = _count_files(root / "images", "*")
    annotations = _count_files(root / "annotations", "*.json")
    ok = images > 0 and annotations > 0 and images == annotations
    return Check(
        name=name,
        ok=ok,
        detail=f"{root} images={images} annotations={annotations}",
    )


def incoming_min_check(name: str, root: Path, *, min_images: int, min_wheels: int) -> Check:
    images = _count_files(root / "images", "*")
    annotations = _count_files(root / "annotations", "*.json")
    wheels = 0
    if (root / "annotations").is_dir():
        for path in (root / "annotations").glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            value = payload.get("wheels", [])
            if isinstance(value, list):
                wheels += len(value)
    ok = images >= min_images and annotations >= min_images and wheels >= min_wheels
    return Check(
        name=name,
        ok=ok,
        detail=(
            f"{root} images={images}/{min_images} annotations={annotations} "
            f"wheels={wheels}/{min_wheels}"
        ),
    )


def incoming_diagnostic_check(name: str, root: Path) -> Check:
    images = _count_files(root / "images", "*")
    annotations = _count_files(root / "annotations", "*.json")
    wheels = 0
    if (root / "annotations").is_dir():
        for path in (root / "annotations").glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            value = payload.get("wheels", [])
            if isinstance(value, list):
                wheels += len(value)
    return Check(
        name=name,
        ok=True,
        detail=f"{root} images={images} annotations={annotations} wheels={wheels}",
    )


def eval_comparison_check(name: str, champion_path: Path, candidate_path: Path) -> Check:
    if not champion_path.is_file() or not candidate_path.is_file():
        return Check(
            name=name,
            ok=False,
            detail=f"champion={champion_path.is_file()} candidate={candidate_path.is_file()}",
        )
    try:
        champion = json.loads(champion_path.read_text(encoding="utf-8"))
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return Check(name=name, ok=False, detail=f"invalid eval JSON: {exc}")

    champion_oks = float(champion.get("oks", {}).get("mean", 0.0))
    candidate_oks = float(candidate.get("oks", {}).get("mean", 0.0))
    champion_fn = float(champion.get("rates", {}).get("false_negative_rate", 1.0))
    candidate_fn = float(candidate.get("rates", {}).get("false_negative_rate", 1.0))
    champion_fp = float(champion.get("rates", {}).get("false_positive_rate", 1.0))
    candidate_fp = float(candidate.get("rates", {}).get("false_positive_rate", 1.0))
    champion_map50 = float(champion.get("metrics_bbox", {}).get("mAP50", 0.0))
    candidate_map50 = float(candidate.get("metrics_bbox", {}).get("mAP50", 0.0))
    promoted = (
        candidate_oks >= champion_oks
        and candidate_fn <= champion_fn
        and candidate_fp <= champion_fp
        and candidate_map50 >= champion_map50
    )
    return Check(
        name=name,
        ok=True,
        detail=(
            f"{'promoted' if promoted else 'not_promoted'} "
            f"candidate oks={candidate_oks:.3f} fn={candidate_fn:.3f} "
            f"fp={candidate_fp:.3f} bbox_mAP50={candidate_map50:.3f}; "
            f"champion oks={champion_oks:.3f} fn={champion_fn:.3f} "
            f"fp={champion_fp:.3f} bbox_mAP50={champion_map50:.3f}"
        ),
    )


def export_drift_diagnostic_check(name: str, path: Path) -> Check:
    if not path.is_file():
        return Check(name=name, ok=False, detail=str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return Check(name=name, ok=False, detail=f"invalid drift JSON: {exc}")
    status = "certified" if payload.get("ok") else "not_certified"
    return Check(
        name=name,
        ok=True,
        detail=(
            f"{status} samples={payload.get('samples_matched', 'n/a')}/"
            f"{payload.get('samples_checked', 'n/a')} "
            f"max_bbox={float(payload.get('max_bbox_drift_px', 0.0)):.3f}px "
            f"max_kp={float(payload.get('max_kp_drift_px', 0.0)):.3f}px "
            f"max_conf={float(payload.get('max_conf_drift', 0.0)):.3f}"
        ),
    )


def certification_diagnostic_check(name: str, path: Path) -> Check:
    if not path.is_file():
        return Check(name=name, ok=False, detail=str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return Check(name=name, ok=False, detail=f"invalid certification JSON: {exc}")
    status = "certified" if payload.get("certified") else "not_certified"
    artifact = payload.get("artifact", {}).get("path", "n/a")
    aggregate = payload.get("aggregate_eval", {})
    if not aggregate and isinstance(payload.get("backends"), dict):
        backend_bits = ", ".join(
            f"{backend_name}={bool(backend.get('certified'))}"
            for backend_name, backend in payload["backends"].items()
            if isinstance(backend, dict)
        )
        return Check(
            name=name,
            ok=True,
            detail=f"{status} scope={payload.get('scope', 'n/a')} backends={backend_bits}",
        )
    return Check(
        name=name,
        ok=True,
        detail=(
            f"{status} artifact={artifact} "
            f"bbox_mAP50={float(aggregate.get('bbox_map50', 0.0)):.3f} "
            f"oks={float(aggregate.get('oks_mean', 0.0)):.3f} "
            f"fn={float(aggregate.get('false_negative_rate', 0.0)):.3f} "
            f"fp={float(aggregate.get('false_positive_rate', 0.0)):.3f}"
        ),
    )


def render_images_check(name: str, root: Path, *, min_images: int) -> Check:
    images = _count_files(root, "*.png")
    return Check(
        name=name,
        ok=images >= min_images,
        detail=f"{root} png={images}/{min_images}",
    )


def image_content_check(name: str, root: Path, *, min_nonblack: int, sample_limit: int = 50) -> Check:
    images = sorted(root.glob("*.png"))[:sample_limit] if root.is_dir() else []
    nonblack = 0
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return Check(name=name, ok=False, detail="opencv/cv2 unavailable")
    for path in images:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        rgb = image[:, :, :3] if getattr(image, "ndim", 0) == 3 else image
        if float(rgb.max()) > 10.0:
            nonblack += 1
    return Check(
        name=name,
        ok=nonblack >= min_nonblack,
        detail=f"{root} nonblack_png={nonblack}/{min_nonblack} sampled={len(images)}",
    )


def yolo_dataset_check(name: str, root: Path) -> Check:
    train_images = _count_files(root / "images/train", "*")
    val_images = _count_files(root / "images/val", "*")
    train_labels = _count_files(root / "labels/train", "*.txt")
    val_labels = _count_files(root / "labels/val", "*.txt")
    ok = (
        train_images > 0
        and val_images > 0
        and train_images == train_labels
        and val_images == val_labels
    )
    return Check(
        name=name,
        ok=ok,
        detail=(
            f"{root} train={train_images}/{train_labels} "
            f"val={val_images}/{val_labels}"
        ),
    )


def collect_checks(args: argparse.Namespace) -> list[Check]:
    model_pool_root = getattr(args, "model_pool_root", None)
    if model_pool_root is None:
        model_pool_root = getattr(args, "sketchfab_root", DEFAULT_MODEL_POOL_ROOT)
    return [
        model_pool_check(model_pool_root, args.model_target),
        mcp_check(args.mcp_host, args.mcp_port, args.mcp_timeout),
        file_check(
            "champion_pt",
            Path("runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt"),
        ),
        file_check(
            "champion_onnx",
            Path("runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.onnx"),
        ),
        file_check(
            "champion_eval",
            Path("outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s.json"),
        ),
        dataset_check(
            "ue_synthetic_pseudo_incoming",
            Path("data/incoming/ue_synthetic_pseudo_from_champion"),
        ),
        yolo_dataset_check(
            "ue_synthetic_pseudo_yolo",
            Path("data/wheel_pose_dataset_ue_synthetic_pseudo_from_champion"),
        ),
        incoming_min_check(
            "ue_neuraldata_keypoint_full_incoming",
            Path("data/incoming/ue_neuraldata_keypoint_full"),
            min_images=50,
            min_wheels=80,
        ),
        yolo_dataset_check(
            "ue_neuraldata_keypoint_full_yolo",
            Path("data/wheel_pose_dataset_ue_neuraldata_keypoint_full"),
        ),
        render_images_check(
            "ue_sketchfab_render_pool",
            Path("outputs/ue_sketchfab_renders/images"),
            min_images=1200,
        ),
        incoming_min_check(
            "ue_sketchfab_geometry_incoming",
            Path("data/incoming/ue_sketchfab_geometry"),
            min_images=150,
            min_wheels=500,
        ),
        image_content_check(
            "ue_sketchfab_geometry_rgb_content",
            Path("data/incoming/ue_sketchfab_geometry/images"),
            min_nonblack=25,
        ),
        yolo_dataset_check(
            "ue_sketchfab_geometry_yolo",
            Path("data/wheel_pose_dataset_ue_sketchfab_geometry"),
        ),
        incoming_min_check(
            "ue_sketchfab_geometry_clean_incoming",
            Path("data/incoming/ue_sketchfab_geometry_clean"),
            min_images=120,
            min_wheels=500,
        ),
        image_content_check(
            "ue_sketchfab_geometry_clean_rgb_content",
            Path("data/incoming/ue_sketchfab_geometry_clean/images"),
            min_nonblack=25,
        ),
        yolo_dataset_check(
            "ue_sketchfab_geometry_clean_yolo",
            Path("data/wheel_pose_dataset_ue_sketchfab_geometry_clean"),
        ),
        yolo_dataset_check(
            "real_self_ue_plus_sketchfab_clean_yolo",
            Path("data/wheel_pose_dataset_real_self_ue_plus_sketchfab_clean"),
        ),
        file_check(
            "real_self_ue_plus_sketchfab_clean_config",
            Path("configs/pose_dataset_real_self_ue_plus_sketchfab_clean.yaml"),
        ),
        file_check(
            "real_self_ue_plus_sketchfab_clean_checkpoint",
            Path("runs/pose/wheel_real_self_ue_plus_sketchfab_clean_ft20_v2/weights/best.pt"),
        ),
        eval_comparison_check(
            "real_self_ue_plus_sketchfab_clean_eval_diagnostic",
            Path("outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s.json"),
            Path("outputs/eval/wheel_real_self_ue_plus_sketchfab_clean_ft20_v2_on_real.json"),
        ),
        file_check(
            "production_readiness_audit",
            Path("docs/PRODUCTION_READINESS_AUDIT.md"),
        ),
        file_check(
            "model_package_manifest",
            Path("outputs/production_audit/model_package_manifest.json"),
        ),
        file_check(
            "model_inventory_json",
            Path("outputs/production_audit/model_inventory.json"),
        ),
        file_check(
            "model_inventory_report",
            Path("docs/MODEL_INVENTORY.md"),
        ),
        file_check(
            "model_selection_audit_json",
            Path("outputs/production_audit/model_selection_audit.json"),
        ),
        file_check(
            "model_selection_audit_report",
            Path("docs/MODEL_SELECTION_AUDIT.md"),
        ),
        file_check(
            "spec_compliance_audit_json",
            Path("outputs/production_audit/spec_compliance_audit.json"),
        ),
        file_check(
            "spec_compliance_audit_report",
            Path("docs/SPEC_COMPLIANCE_AUDIT.md"),
        ),
        file_check(
            "model_card",
            Path("docs/MODEL_CARD.md"),
        ),
        file_check(
            "dataset_audit_json",
            Path("outputs/production_audit/dataset_audit.json"),
        ),
        file_check(
            "dataset_audit_report",
            Path("docs/DATASET_AUDIT.md"),
        ),
        file_check(
            "release_integrity_json",
            Path("outputs/production_audit/release_integrity.json"),
        ),
        file_check(
            "release_package_report",
            Path("docs/RELEASE_PACKAGE.md"),
        ),
        file_check(
            "performance_audit_json",
            Path("outputs/production_audit/performance_audit.json"),
        ),
        file_check(
            "performance_audit_report",
            Path("docs/PERFORMANCE_AUDIT.md"),
        ),
        file_check(
            "senior_ml_audit_json",
            Path("outputs/production_audit/senior_ml_audit.json"),
        ),
        file_check(
            "senior_ml_audit_report",
            Path("docs/SENIOR_ML_AUDIT.md"),
        ),
        file_check(
            "runtime_contract_audit",
            Path("outputs/production_audit/runtime_contract_audit.json"),
        ),
        file_check(
            "integration_gate_report",
            Path("outputs/production_audit/integration_gate.json"),
        ),
        file_check(
            "production_gate_report",
            Path("outputs/production_audit/production_gate.json"),
        ),
        export_drift_diagnostic_check(
            "champion_onnx_drift_diagnostic",
            Path("outputs/production_audit/onnx_drift_20.json"),
        ),
        file_check(
            "export_parity_audit_json",
            Path("outputs/production_audit/export_parity_audit.json"),
        ),
        file_check(
            "export_parity_audit_report",
            Path("docs/EXPORT_PARITY_AUDIT.md"),
        ),
        certification_diagnostic_check(
            "export_certification_diagnostic",
            Path("outputs/production_audit/export_certification.json"),
        ),
        file_check(
            "export_certification_report",
            Path("docs/EXPORT_CERTIFICATION.md"),
        ),
        file_check(
            "android_litert_device_report_doc",
            Path("docs/ANDROID_LITERT_DEVICE_REPORT.md"),
        ),
        file_check(
            "production_evidence_checklist",
            Path("docs/PRODUCTION_EVIDENCE_CHECKLIST.md"),
        ),
        file_check(
            "production_evidence_intake_doc",
            Path("docs/PRODUCTION_EVIDENCE_INTAKE.md"),
        ),
        file_check(
            "external_evidence_handoff_bundle_doc",
            Path("docs/EXTERNAL_EVIDENCE_HANDOFF_BUNDLE.md"),
        ),
        file_check(
            "production_evidence_audit_json",
            Path("outputs/production_audit/production_evidence_audit.json"),
        ),
        file_check(
            "production_evidence_intake_preflight_status",
            Path("outputs/production_audit/production_evidence_intake_preflight_status.json"),
        ),
        file_check(
            "external_evidence_drop_importer",
            Path("src/import_external_evidence_drop.py"),
        ),
        file_check(
            "external_evidence_return_template",
            Path("outputs/production_audit/external_evidence_return_template.zip"),
        ),
        file_check(
            "external_evidence_return_template_manifest",
            Path("outputs/production_audit/external_evidence_return_template_manifest.json"),
        ),
        file_check(
            "external_evidence_handoff_bundle",
            Path("outputs/production_audit/external_evidence_handoff_bundle.zip"),
        ),
        file_check(
            "external_evidence_handoff_bundle_manifest",
            Path("outputs/production_audit/external_evidence_handoff_bundle_manifest.json"),
        ),
        file_check(
            "external_evidence_handoff_bundle_verification",
            Path("outputs/production_audit/external_evidence_handoff_bundle_verification.json"),
        ),
        file_check(
            "production_evidence_audit_report",
            Path("docs/PRODUCTION_EVIDENCE_AUDIT.md"),
        ),
        file_check(
            "requirements_traceability_json",
            Path("outputs/production_audit/requirements_traceability.json"),
        ),
        file_check(
            "requirements_traceability_report",
            Path("docs/REQUIREMENTS_TRACEABILITY.md"),
        ),
        file_check(
            "objective_completion_audit_json",
            Path("outputs/production_audit/objective_completion_audit.json"),
        ),
        file_check(
            "objective_completion_audit_report",
            Path("docs/OBJECTIVE_COMPLETION_AUDIT.md"),
        ),
        file_check(
            "report_consistency_audit_json",
            Path("outputs/production_audit/report_consistency_audit.json"),
        ),
        file_check(
            "report_consistency_audit_report",
            Path("docs/REPORT_CONSISTENCY_AUDIT.md"),
        ),
        file_check(
            "executive_report_ru",
            Path("docs/EXECUTIVE_REPORT_RU.md"),
        ),
        file_check(
            "android_litert_report_template",
            Path("outputs/production_audit/android_litert_device_report.template.json"),
        ),
        file_check(
            "android_litert_harness_doc",
            Path("android_litert_harness/README.md"),
        ),
        file_check(
            "android_litert_harness_test",
            Path("android_litert_harness/AndroidLiteRtDeviceValidationTest.kt"),
        ),
        file_check(
            "ar_holdout_provenance_template",
            Path("outputs/production_audit/ar_device_holdout_provenance.template.json"),
        ),
        file_check(
            "ar_holdout_harness_doc",
            Path("ar_holdout_harness/README.md"),
        ),
        file_check(
            "ar_holdout_harness_writer",
            Path("ar_holdout_harness/ArHoldoutAnnotationWriter.kt"),
        ),
        file_check(
            "ar_replay_log_template",
            Path("outputs/production_audit/ar_3d_replay.template.jsonl"),
        ),
        file_check(
            "ar_replay_harness_doc",
            Path("ar_replay_harness/README.md"),
        ),
        file_check(
            "ar_replay_harness_logger",
            Path("ar_replay_harness/ArReplayLogger.kt"),
        ),
        certification_diagnostic_check(
            "champion_tflite_certification_diagnostic",
            Path("outputs/production_audit/tflite_certification.json"),
        ),
        incoming_diagnostic_check(
            "ue_sketchfab_pseudo_yield_diagnostic",
            Path("data/incoming/ue_sketchfab_pseudo_conf005"),
        ),
        file_check(
            "ue_import_script",
            Path("scripts/ue/import_sketchfab_glbs.py"),
        ),
        file_check(
            "ue_render_script",
            Path("scripts/ue/render_sketchfab_cars.py"),
        ),
        file_check(
            "ue_geometry_label_script",
            Path("scripts/ue/render_sketchfab_geometry_labels.py"),
        ),
        file_check(
            "ue_geometry_label_status",
            Path("outputs/ue_tasks/render_sketchfab_geometry_labels_status.json"),
        ),
        file_check(
            "ue_sketchfab_geometry_config",
            Path("configs/pose_dataset_ue_sketchfab_geometry.yaml"),
        ),
        file_check(
            "ue_geometry_filter_script",
            Path("src/filter_geometry_incoming.py"),
        ),
        file_check(
            "ue_wheel_asset_filter",
            Path("src/ue_wheel_asset_filter.py"),
        ),
        file_check(
            "ar_replay_validator",
            Path("src/validate_ar_replay.py"),
        ),
        file_check(
            "ar_replay_metric_scorer",
            Path("src/eval_ar_replay_metric.py"),
        ),
        file_check(
            "ar_holdout_evaluator",
            Path("src/evaluate_ar_holdout.py"),
        ),
        file_check(
            "litert_runtime_smoke",
            Path("outputs/production_audit/litert_runtime_smoke.json"),
        ),
        file_check(
            "litert_runtime_checker",
            Path("src/check_litert_runtime.py"),
        ),
        file_check(
            "tflite_certification_builder",
            Path("src/tflite_certification.py"),
        ),
        file_check(
            "android_litert_validator",
            Path("src/validate_android_litert_report.py"),
        ),
        file_check(
            "android_litert_template_writer",
            Path("scripts/create_android_litert_report_template.py"),
        ),
        file_check(
            "android_litert_harness_readme",
            Path("android_litert_harness/README.md"),
        ),
        file_check(
            "android_litert_harness_kotlin_test",
            Path("android_litert_harness/AndroidLiteRtDeviceValidationTest.kt"),
        ),
        file_check(
            "ar_holdout_provenance_template_writer",
            Path("scripts/create_ar_holdout_provenance_template.py"),
        ),
        file_check(
            "ar_holdout_harness_readme",
            Path("ar_holdout_harness/README.md"),
        ),
        file_check(
            "ar_holdout_harness_kotlin_writer",
            Path("ar_holdout_harness/ArHoldoutAnnotationWriter.kt"),
        ),
        file_check(
            "ar_replay_template_writer",
            Path("scripts/create_ar_replay_log_template.py"),
        ),
        file_check(
            "external_evidence_return_template_writer",
            Path("scripts/create_external_evidence_return_template.py"),
        ),
        file_check(
            "ar_replay_harness_readme",
            Path("ar_replay_harness/README.md"),
        ),
        file_check(
            "ar_replay_harness_kotlin_logger",
            Path("ar_replay_harness/ArReplayLogger.kt"),
        ),
        file_check(
            "external_evidence_handoff_bundle_builder",
            Path("scripts/build_external_evidence_handoff_bundle.py"),
        ),
        file_check(
            "external_evidence_handoff_bundle_verifier",
            Path("src/verify_external_evidence_handoff_bundle.py"),
        ),
        file_check(
            "production_evidence_audit_runner",
            Path("src/production_evidence_audit.py"),
        ),
        file_check(
            "production_evidence_intake_runner",
            Path("src/run_production_evidence_intake.py"),
        ),
        file_check(
            "external_evidence_drop_import_runner",
            Path("src/import_external_evidence_drop.py"),
        ),
        file_check(
            "requirements_traceability_runner",
            Path("src/requirements_traceability.py"),
        ),
        file_check(
            "executive_report_ru_runner",
            Path("src/executive_report_ru.py"),
        ),
        file_check(
            "objective_completion_audit_runner",
            Path("src/objective_completion_audit.py"),
        ),
        file_check(
            "report_consistency_audit_runner",
            Path("src/report_consistency_audit.py"),
        ),
        file_check(
            "ue_sketchfab_geometry_clean_config",
            Path("configs/pose_dataset_ue_sketchfab_geometry_clean.yaml"),
        ),
        file_check(
            "ue_pseudo_wrapper",
            Path("scripts/prepare_ue_sketchfab_pseudo_data.sh"),
        ),
        file_check(
            "sketchfab_autofetch_wrapper",
            Path("scripts/fetch_sketchfab_until_target.sh"),
        ),
        file_check(
            "mcp_wait_wrapper",
            Path("scripts/wait_for_unreal_mcp.sh"),
        ),
        file_check(
            "finish_orchestrator",
            Path("scripts/finish_project_today.sh"),
        ),
        file_check(
            "production_audit_suite_runner",
            Path("src/production_audit_suite.py"),
        ),
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-target", type=int, default=DEFAULT_MODEL_TARGET)
    parser.add_argument("--model-pool-root", type=Path, default=DEFAULT_MODEL_POOL_ROOT)
    parser.add_argument(
        "--sketchfab-root",
        type=Path,
        dest="model_pool_root",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--mcp-host", default=DEFAULT_MCP_HOST)
    parser.add_argument("--mcp-port", type=int, default=DEFAULT_MCP_PORT)
    parser.add_argument("--mcp-timeout", type=float, default=0.5)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checks = collect_checks(args)
    payload = {
        "checks": [asdict(check) for check in checks],
        "ok": all(check.ok for check in checks),
        "failed": [check.name for check in checks if not check.ok],
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        for check in checks:
            status = "OK" if check.ok else "MISSING"
            print(f"{status:7} {check.name}: {check.detail}")
        print(f"Overall: {'OK' if payload['ok'] else 'NOT READY'}")
        if payload["failed"]:
            print("Failed: " + ", ".join(payload["failed"]))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
