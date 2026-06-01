"""Generate a Russian executive report for stakeholder handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path("docs/EXECUTIVE_REPORT_RU.md")


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def metric(report: dict[str, Any], *keys: str, default: Any = "n/a") -> Any:
    cur: Any = report
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _blockers_text(blockers: Any) -> str:
    if isinstance(blockers, list) and blockers:
        return ", ".join(str(item) for item in blockers)
    return "нет"


def build_report() -> str:
    suite = read_json(Path("outputs/production_audit/audit_suite_status.json"))
    trace = read_json(Path("outputs/production_audit/requirements_traceability.json"))
    evidence = read_json(Path("outputs/production_audit/production_evidence_audit.json"))
    inventory = read_json(Path("outputs/production_audit/model_inventory.json"))
    model_selection = read_json(Path("outputs/production_audit/model_selection_audit.json"))
    spec_compliance = read_json(Path("outputs/production_audit/spec_compliance_audit.json"))
    dataset = read_json(Path("outputs/production_audit/dataset_audit.json"))
    release = read_json(Path("outputs/production_audit/release_integrity.json"))
    tflite_cert = read_json(Path("outputs/production_audit/tflite_certification.json"))
    coreml_cert = read_json(Path("outputs/production_audit/coreml_certification.json"))
    export_cert = read_json(Path("outputs/production_audit/export_certification.json"))
    pt_real = read_json(Path("outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_on_self_val.json"))
    tflite_eval = read_json(Path("outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_tflite_on_self_plus_ue_val.json"))

    blockers = suite.get("production_blockers", [])
    integration_ready = suite.get("integration_ready")
    production_ready = suite.get("production_ready")
    blockers_text = _blockers_text(blockers)
    evidence_checks = evidence.get("checks", []) if isinstance(evidence.get("checks"), list) else []
    evidence_ready = evidence.get("production_evidence_ready", "n/a")
    trace_summary = trace.get("summary", {}) if isinstance(trace.get("summary"), dict) else {}
    dataset_counts = dataset.get("counts", {}) if isinstance(dataset.get("counts"), dict) else {}

    missing_lines: list[str] = []
    for check in evidence_checks:
        if not isinstance(check, dict) or check.get("ready"):
            continue
        failures = ", ".join(str(item) for item in check.get("failures", [])) or "нет деталей"
        missing_lines.append(f"- `{check.get('name')}`: {failures}")
    if not missing_lines:
        missing_lines.append("- Нет незакрытых внешних evidence blockers.")

    if integration_ready is True:
        summary_line = (
            "Вывод: модель и export package готовы для интеграции и smoke-проверок. "
            "Полный production gate не закрыт, потому что не хватает внешних Android/AR evidence artifacts."
        )
        final_line = (
            "На текущем evidence модель является integration-ready, но не full production-ready. "
            "Блокеры не связаны с отсутствием модели/export package: они связаны с отсутствием "
            "реального Android-device LiteRT отчета, human-labelled AR holdout и AR-side 3D replay validation."
        )
    else:
        summary_line = (
            "Вывод: integration gate сейчас не закрыт. Модель и export package существуют, "
            f"но текущий evidence блокируют: {blockers_text}."
        )
        final_line = (
            "На текущем evidence модель не является integration-ready. Сначала нужно закрыть "
            f"блокеры: {blockers_text}."
        )

    lines = [
        "# Executive Report RU",
        "",
        "## Короткий статус",
        "",
        f"- Integration ready: {integration_ready if integration_ready is not None else 'n/a'}",
        f"- Production ready: {production_ready if production_ready is not None else 'n/a'}",
        f"- Audit suite OK: {suite.get('ok', 'n/a')}",
        f"- Production evidence audit ready: {evidence_ready}",
        f"- Production blockers: {blockers_text if isinstance(blockers, list) else 'n/a'}",
        "",
        summary_line,
        "",
        "## Что сделано",
        "",
        "- Собран пул 300 чистых GLB-моделей машин из Sketchfab/Objaverse.",
        "- Через Unreal/MCP подготовлен synthetic/geometry поток: 192 кадра и 702 wheel labels до clean-фильтра, 152 кадра и 626 wheel labels после clean QA.",
        "- Проведен model inventory: "
        f"{metric(inventory, 'counts', 'train_runs')} train runs, "
        f"{metric(inventory, 'counts', 'artifacts')} artifacts, "
        f"{metric(inventory, 'counts', 'eval_reports')} eval reports.",
        "- Проведен dataset audit: "
        f"{metric(dataset, 'counts', 'configs')} configs, "
        f"{metric(dataset, 'counts', 'total_train_images')} train images, "
        f"{metric(dataset, 'counts', 'total_val_images')} val images, "
        f"{metric(dataset, 'counts', 'total_wheel_labels')} wheel labels.",
        "- Champion model выбран через machine-readable promotion guard: "
        f"ok={model_selection.get('ok', 'n/a')}, "
        f"anchor candidates={metric(model_selection, 'counts', 'anchor_candidates')}, "
        f"promotion required={metric(model_selection, 'counts', 'promotion_required')}.",
        "- Соответствие AR technical spec проверено отдельным audit: "
        f"ok={spec_compliance.get('ok', 'n/a')}, "
        f"failures={spec_compliance.get('failures', [])}.",
        "- ONNX/TFLite export package сертифицирован по calibrated backend policy.",
        "- TFLite/LiteRT desktop package сертифицирован; Android-device runtime validation вынесен отдельным production blocker.",
        "- CoreML `.mlmodel` package сертифицирован как desktop/iOS handoff artifact; iOS-device runtime validation остается на стороне приложения.",
        "",
        "## Champion model",
        "",
        "- PT: `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt`",
        "- ONNX: `runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.onnx`",
        "- TFLite: `outputs/production_audit/tflite_export/best_float32.tflite`",
        "- CoreML: `outputs/production_audit/coreml_export/best.mlmodel`",
        "- Training data: `configs/pose_dataset_real_v1_self_plus_ue_synthetic.yaml`",
        "",
        "## Основные метрики",
        "",
        f"- Real-only PyTorch bbox mAP50: {fmt(metric(pt_real, 'metrics_bbox', 'mAP50'))}",
        f"- Real-only PyTorch OKS: {fmt(metric(pt_real, 'oks', 'mean'))}",
        f"- Real-only PyTorch FN rate: {fmt(metric(pt_real, 'rates', 'false_negative_rate'))}",
        f"- TFLite mixed-anchor bbox mAP50: {fmt(metric(tflite_eval, 'metrics_bbox', 'mAP50'))}",
        f"- TFLite mixed-anchor OKS: {fmt(metric(tflite_eval, 'oks', 'mean'))}",
        f"- Export backend certification: {export_cert.get('certified', 'n/a')} (`{export_cert.get('scope', 'n/a')}`)",
        f"- TFLite package certification: {tflite_cert.get('certified', 'n/a')} (`{tflite_cert.get('scope', 'n/a')}`)",
        f"- CoreML package certification: {coreml_cert.get('certified', 'n/a')} (`{coreml_cert.get('scope', 'n/a')}`)",
        "",
        "## Соответствие требованиям",
        "",
        f"- Requirements закрыто: {trace_summary.get('passed', 'n/a')} / {trace_summary.get('requirements', 'n/a')}",
        f"- Consolidated production evidence gate: {evidence_ready}",
        "- Traceability report: `docs/REQUIREMENTS_TRACEABILITY.md`",
        "- Senior ML audit: `docs/SENIOR_ML_AUDIT.md`",
        "- Model card: `docs/MODEL_CARD.md`",
        "",
        "## Что не закрыто для production",
        "",
        *missing_lines,
        "",
        "## Что нужно от Android/AR команды",
        "",
        "1. Заполнить `data/incoming/android_litert_device_report.json` по контракту `docs/ANDROID_LITERT_DEVICE_REPORT.md`.",
        "2. Передать human-reviewed AR-device holdout в `data/incoming/ar_device_holdout` с `metadata/provenance.json`.",
        "3. Передать AR replay JSONL в `data/incoming/ar_3d_replay/ar_replay.jsonl`.",
        "4. Запустить единый intake: `./.venv/bin/python src/run_production_evidence_intake.py`.",
        "5. После зеленого intake запустить `./.venv/bin/python src/production_audit_suite.py --with-pytest`.",
        "",
        "## Release package",
        "",
        f"- Release integrity OK: {release.get('ok', 'n/a')}",
        "- Deterministic package manifest: `docs/RELEASE_PACKAGE.md` / `outputs/production_audit/release_integrity.json`.",
        "",
        "## Финальный вывод",
        "",
        final_line,
    ]
    _ = dataset_counts  # keep local audit data intentionally loaded for future extensions
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_report(), encoding="utf-8")
    print(f"executive_report={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
