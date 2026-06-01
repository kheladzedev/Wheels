"""Summarize whether the current data is enough for handoff vs production.

This report exists because a green integration pipeline is not the same
thing as enough real-world AR evidence for production. It reads existing
audits and turns them into a senior ML decision: what can be handed off
today, what must be collected next, and why blindly adding more data is
not the right immediate move.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_DATASET_AUDIT = Path("outputs/production_audit/dataset_audit.json")
DEFAULT_MODEL_INVENTORY = Path("outputs/production_audit/model_inventory.json")
DEFAULT_OPERATING_POINT = Path("outputs/production_audit/operating_point_audit.json")
DEFAULT_PRODUCTION_EVIDENCE = Path("outputs/production_audit/production_evidence_audit.json")
DEFAULT_AUDIT_SUITE = Path("outputs/production_audit/audit_suite_status.json")
DEFAULT_JSON_OUT = Path("outputs/production_audit/data_readiness_decision.json")
DEFAULT_MD_OUT = Path("docs/DATA_READINESS_DECISION.md")

PRODUCTION_RECOMMENDED_REAL_FRAMES = 2000
PRODUCTION_RECOMMENDED_HARD_NEGATIVES = 300
RECOMMENDED_AR_HOLDOUT_IMAGES = 300
RECOMMENDED_AR_HOLDOUT_WHEELS = 500
GATE_MIN_AR_HOLDOUT_IMAGES = 50
GATE_MIN_AR_HOLDOUT_WHEELS = 80


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _gate_reports(dataset_audit: dict[str, Any]) -> list[dict[str, Any]]:
    gate = dataset_audit.get("gate", {})
    configs = set(gate.get("configs", [])) if isinstance(gate, dict) else set()
    reports = dataset_audit.get("reports", [])
    if not isinstance(reports, list):
        return []
    return [
        report
        for report in reports
        if isinstance(report, dict) and report.get("config") in configs
    ]


def _strict_gate_leakage_ok(dataset_audit: dict[str, Any]) -> bool:
    reports = _gate_reports(dataset_audit)
    if not reports:
        return False
    for report in reports:
        leakage = report.get("leakage", {})
        if not isinstance(leakage, dict):
            return False
        if _int(leakage.get("stem_overlap_count")) != 0:
            return False
        if _int(leakage.get("hash_overlap_count")) != 0:
            return False
    return True


def _strict_gate_has_no_label_errors(dataset_audit: dict[str, Any]) -> bool:
    reports = _gate_reports(dataset_audit)
    if not reports:
        return False
    for report in reports:
        if report.get("ok") is not True:
            return False
        failures = report.get("failures", [])
        if failures:
            return False
    return True


def _selected_metric(operating_point: dict[str, Any], key: str) -> float | None:
    selected = operating_point.get("selected", {})
    if not isinstance(selected, dict):
        return None
    return _float(selected.get(key))


def build_decision(
    *,
    dataset_audit: dict[str, Any],
    model_inventory: dict[str, Any],
    operating_point: dict[str, Any],
    production_evidence: dict[str, Any],
    audit_suite: dict[str, Any],
) -> dict[str, Any]:
    gate = dataset_audit.get("gate", {})
    gate_counts = gate.get("counts", {}) if isinstance(gate, dict) else {}
    all_counts = dataset_audit.get("counts", {})
    inventory_counts = model_inventory.get("counts", {})

    strict_train_images = _int(gate_counts.get("total_train_images"))
    strict_val_images = _int(gate_counts.get("total_val_images"))
    strict_wheel_labels = _int(gate_counts.get("total_wheel_labels"))
    failed_configs = _int(all_counts.get("failed"))
    total_configs = _int(all_counts.get("configs"))
    production_blockers = production_evidence.get("blockers", [])
    if not isinstance(production_blockers, list):
        production_blockers = []

    gate_ok = isinstance(gate, dict) and gate.get("ok") is True
    strict_leakage_ok = _strict_gate_leakage_ok(dataset_audit)
    strict_label_ok = _strict_gate_has_no_label_errors(dataset_audit)
    # This report is generated inside the audit suite before the final
    # suite status JSON is rewritten, so absent suite status must not make
    # a fresh handoff report look red.
    suite_status_known = bool(audit_suite)
    suite_ok = audit_suite.get("ok") is True if suite_status_known else True
    integration_ready = (
        audit_suite.get("integration_ready") is True if suite_status_known else True
    )
    production_ready = audit_suite.get("production_ready") is True
    production_evidence_ready = production_evidence.get("production_evidence_ready") is True
    test_handoff_ready = bool(suite_ok and integration_ready and gate_ok and strict_label_ok)

    production_training_data_sufficient = bool(
        production_ready
        and production_evidence_ready
        and strict_train_images >= PRODUCTION_RECOMMENDED_REAL_FRAMES
    )

    current_data = {
        "strict_gate": {
            "ok": gate_ok,
            "configs": gate.get("configs", []) if isinstance(gate, dict) else [],
            "train_images": strict_train_images,
            "val_images": strict_val_images,
            "wheel_labels": strict_wheel_labels,
            "label_schema_ok": strict_label_ok,
            "leakage_ok": strict_leakage_ok,
        },
        "all_configured_datasets": {
            "configs": total_configs,
            "ok": _int(all_counts.get("ok")),
            "failed": failed_configs,
            "train_images": _int(all_counts.get("total_train_images")),
            "val_images": _int(all_counts.get("total_val_images")),
            "wheel_labels": _int(all_counts.get("total_wheel_labels")),
        },
        "model_inventory": inventory_counts if isinstance(inventory_counts, dict) else {},
        "operating_point": {
            "conf": _selected_metric(operating_point, "conf"),
            "bbox_mAP50": _selected_metric(operating_point, "bbox_mAP50"),
            "oks_mean": _selected_metric(operating_point, "oks_mean"),
            "false_negative_rate": _selected_metric(
                operating_point, "false_negative_rate"
            ),
            "false_positive_rate": _selected_metric(
                operating_point, "false_positive_rate"
            ),
        },
    }

    strict_size_status = (
        "enough_for_integration"
        if strict_train_images >= 300 and strict_val_images >= 50
        else "thin_even_for_integration"
    )
    if strict_train_images < PRODUCTION_RECOMMENDED_REAL_FRAMES:
        strict_size_status = "too_small_for_production"

    return {
        "schema_version": 1,
        "verdict": {
            "test_handoff_ready": test_handoff_ready,
            "production_data_ready": bool(production_ready and production_evidence_ready),
            "production_training_data_sufficient": production_training_data_sufficient,
            "senior_confidence": (
                "medium_for_integration_handoff_low_for_production"
                if test_handoff_ready and not production_training_data_sufficient
                else "low_until_audits_pass"
            ),
            "decision_ru": (
                "Для тестовой передачи данных и артефактов достаточно; "
                "для production данных и device/AR evidence недостаточно."
            ),
        },
        "current_data": current_data,
        "production_blockers": production_blockers,
        "risk_flags": {
            "dirty_legacy_dataset_configs": {
                "status": "present_but_excluded_from_gate"
                if failed_configs
                else "none_detected",
                "failed_configs": failed_configs,
                "total_configs": total_configs,
            },
            "strict_subset_size": {
                "status": strict_size_status,
                "train_images": strict_train_images,
                "val_images": strict_val_images,
                "wheel_labels": strict_wheel_labels,
                "recommended_real_frames_for_production": PRODUCTION_RECOMMENDED_REAL_FRAMES,
            },
            "ground_truth_quality": {
                "status": "insufficient_for_production",
                "reason": (
                    "Current strict validation data is clean structurally, but not a "
                    "human-labelled AR-device holdout."
                ),
            },
            "external_evidence": {
                "status": "missing" if production_blockers else "ready",
                "blockers": production_blockers,
            },
            "blind_retrain": {
                "status": "not_recommended",
                "reason": (
                    "More uncurated data can make labels noisier. Collect AR-device "
                    "holdout and hard negatives first, then retrain behind promotion gates."
                ),
            },
        },
        "recommended_data_plan": [
            {
                "priority": "P0",
                "owner": "android",
                "action": "Run exact TFLite artifact on physical Android LiteRT harness.",
                "why": "Closes serving/runtime skew for Android before any retrain.",
            },
            {
                "priority": "P0",
                "owner": "ar_data_collection",
                "action": (
                    f"Collect at least gate minimum {GATE_MIN_AR_HOLDOUT_IMAGES} "
                    f"AR frames / {GATE_MIN_AR_HOLDOUT_WHEELS} wheels, recommended "
                    f"{RECOMMENDED_AR_HOLDOUT_IMAGES}+ frames / "
                    f"{RECOMMENDED_AR_HOLDOUT_WHEELS}+ wheels for confidence."
                ),
                "why": "The current val set is local/static, not a real AR-device holdout.",
            },
            {
                "priority": "P0",
                "owner": "ar_runtime",
                "action": "Collect AR replay JSONL with floor hits, RANSAC and residuals.",
                "why": "2D keypoints are not enough; production requires 3D floor-hit behavior.",
            },
            {
                "priority": "P1",
                "owner": "ml",
                "action": (
                    f"Mine {PRODUCTION_RECOMMENDED_HARD_NEGATIVES}-1000 hard-negative "
                    "frames from false positives and add them with empty labels."
                ),
                "why": "This directly targets FP risk instead of just increasing data volume.",
            },
            {
                "priority": "P2",
                "owner": "ml_ar",
                "action": (
                    f"Build a production training pool of {PRODUCTION_RECOMMENDED_REAL_FRAMES}+ "
                    "real labelled AR/app frames with WheelBBox/keypoints and "
                    "scene/device/session groups."
                ),
                "why": "Needed for production retrain without leakage and domain skew.",
            },
        ],
        "do_now": [
            "Отдать текущий bundle в тест.",
            "Не дообучать вслепую на старых грязных датасетах.",
            "Сначала собрать device/AR evidence, потом решать retrain.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    verdict = report["verdict"]
    strict = report["current_data"]["strict_gate"]
    all_data = report["current_data"]["all_configured_datasets"]
    op = report["current_data"]["operating_point"]
    flags = report["risk_flags"]

    lines = [
        "# Data Readiness Decision",
        "",
        "## Verdict",
        "",
        f"- Для тестовой передачи: {'можно отдавать' if verdict['test_handoff_ready'] else 'нельзя отдавать'}.",
        f"- Для production: {'данных достаточно' if verdict['production_training_data_sufficient'] else 'данных недостаточно'}.",
        f"- Confidence: `{verdict['senior_confidence']}`.",
        "",
        "## Current Data",
        "",
        (
            f"- Strict gate subset: train={strict['train_images']}, "
            f"val={strict['val_images']}, wheels={strict['wheel_labels']}, "
            f"labels_ok={strict['label_schema_ok']}, leakage_ok={strict['leakage_ok']}."
        ),
        (
            f"- All configured datasets: configs={all_data['configs']}, "
            f"failed={all_data['failed']}, train_images={all_data['train_images']}, "
            f"val_images={all_data['val_images']}, wheels={all_data['wheel_labels']}."
        ),
        (
            f"- Operating point: conf={op['conf']}, bbox_mAP50={op['bbox_mAP50']}, "
            f"OKS={op['oks_mean']}, FN={op['false_negative_rate']}, "
            f"FP={op['false_positive_rate']}."
        ),
        "",
        "## Risk Flags",
        "",
        (
            f"- Legacy/experimental configs: {flags['dirty_legacy_dataset_configs']['failed_configs']} "
            f"failed out of {flags['dirty_legacy_dataset_configs']['total_configs']}; "
            "they are excluded from the production gate."
        ),
        (
            f"- Strict subset size: {flags['strict_subset_size']['status']} "
            f"(recommended {PRODUCTION_RECOMMENDED_REAL_FRAMES}+ real frames for production retrain)."
        ),
        f"- Ground truth quality: {flags['ground_truth_quality']['reason']}",
        (
            "- External evidence: "
            + ", ".join(flags["external_evidence"]["blockers"])
            if flags["external_evidence"]["blockers"]
            else "- External evidence: ready."
        ),
        "- Retrain decision: не дообучать вслепую; first collect AR holdout, replay and hard negatives.",
        "",
        "## Data Plan",
        "",
    ]
    for item in report["recommended_data_plan"]:
        lines.append(
            f"- {item['priority']} `{item['owner']}`: {item['action']} Reason: {item['why']}"
        )
    lines.extend(
        [
            "",
            "## Do Now",
            "",
            "- Отдать текущий bundle в тест.",
            "- Собирать возврат от Android/iOS/AR по шаблонам.",
            "- После внешнего evidence делать targeted retrain, not generic data download.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-audit", type=Path, default=DEFAULT_DATASET_AUDIT)
    parser.add_argument("--model-inventory", type=Path, default=DEFAULT_MODEL_INVENTORY)
    parser.add_argument("--operating-point", type=Path, default=DEFAULT_OPERATING_POINT)
    parser.add_argument(
        "--production-evidence", type=Path, default=DEFAULT_PRODUCTION_EVIDENCE
    )
    parser.add_argument("--audit-suite", type=Path, default=DEFAULT_AUDIT_SUITE)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_decision(
        dataset_audit=read_json(args.dataset_audit),
        model_inventory=read_json(args.model_inventory),
        operating_point=read_json(args.operating_point),
        production_evidence=read_json(args.production_evidence),
        audit_suite=read_json(args.audit_suite),
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(report), encoding="utf-8")
    print(
        "test_handoff_ready="
        f"{report['verdict']['test_handoff_ready']} "
        "production_training_data_sufficient="
        f"{report['verdict']['production_training_data_sufficient']}"
    )
    print(f"json={args.json_out}")
    print(f"markdown={args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
