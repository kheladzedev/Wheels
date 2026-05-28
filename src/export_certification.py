"""Certify exported ONNX/TFLite artifacts under the approved export policy.

Strict PT-vs-export parity is kept as a diagnostic because YOLO exported
backends can show small postprocess/confidence drift while preserving the
same aggregate quality. This certification is intentionally stricter than
"model loads": it requires calibrated drift, no count mismatch, no coordinate
scale warnings, aggregate metric deltas against the PyTorch champion, and a
LiteRT smoke check for the TFLite artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PT_EVAL = Path("outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s.json")
DEFAULT_ONNX_EVAL = Path("outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_onnx_on_self_plus_ue_val.json")
DEFAULT_TFLITE_EVAL = Path("outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_tflite_on_self_plus_ue_val.json")
DEFAULT_ONNX_DRIFT = Path("outputs/production_audit/onnx_drift_20.json")
DEFAULT_TFLITE_DRIFT = Path("outputs/production_audit/tflite_drift_20.json")
DEFAULT_LITERT_SMOKE = Path("outputs/production_audit/litert_runtime_smoke.json")
DEFAULT_JSON_OUT = Path("outputs/production_audit/export_certification.json")
DEFAULT_MD_OUT = Path("docs/EXPORT_CERTIFICATION.md")


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def metric(report: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    cur: Any = report
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default


def count_scale_warnings(drift: dict[str, Any]) -> int:
    total = 0
    samples = drift.get("samples", [])
    if not isinstance(samples, list):
        return 0
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        pairs = sample.get("pair_diagnostics", [])
        if not isinstance(pairs, list):
            continue
        total += sum(
            1
            for pair in pairs
            if isinstance(pair, dict) and bool(pair.get("coordinate_scale_warning"))
        )
    return total


def has_count_mismatch(drift: dict[str, Any]) -> bool:
    samples = drift.get("samples", [])
    if not isinstance(samples, list):
        return True
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        if sample.get("n_pt") != sample.get("n_exported"):
            return True
        failures = "\n".join(str(item) for item in sample.get("failures", [])).lower()
        if "detection count differs" in failures:
            return True
    return False


def eval_summary(report: dict[str, Any]) -> dict[str, float]:
    return {
        "bbox_map50": metric(report, "metrics_bbox", "mAP50"),
        "bbox_map50_95": metric(report, "metrics_bbox", "mAP50_95"),
        "oks_mean": metric(report, "oks", "mean"),
        "false_negative_rate": metric(report, "rates", "false_negative_rate", default=1.0),
        "false_positive_rate": metric(report, "rates", "false_positive_rate", default=1.0),
    }


def certify_backend(
    *,
    name: str,
    eval_report: dict[str, Any],
    drift_report: dict[str, Any],
    pt_metrics: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    exported_metrics = eval_summary(eval_report)
    deltas = {
        "bbox_map50": abs(exported_metrics["bbox_map50"] - pt_metrics["bbox_map50"]),
        "bbox_map50_95": abs(exported_metrics["bbox_map50_95"] - pt_metrics["bbox_map50_95"]),
        "oks_mean": abs(exported_metrics["oks_mean"] - pt_metrics["oks_mean"]),
        "false_negative_rate": abs(exported_metrics["false_negative_rate"] - pt_metrics["false_negative_rate"]),
        "false_positive_rate": abs(exported_metrics["false_positive_rate"] - pt_metrics["false_positive_rate"]),
    }

    failures: list[str] = []
    if metric(drift_report, "samples_checked") < args.min_drift_samples:
        failures.append(
            f"too_few_drift_samples:{metric(drift_report, 'samples_checked'):.0f}<"
            f"{args.min_drift_samples}"
        )
    if has_count_mismatch(drift_report):
        failures.append("count_mismatch")
    scale_warnings = count_scale_warnings(drift_report)
    if scale_warnings:
        failures.append(f"coordinate_scale_warnings:{scale_warnings}")
    if metric(drift_report, "max_bbox_drift_px") > args.max_bbox_drift_px:
        failures.append(
            f"bbox_drift:{metric(drift_report, 'max_bbox_drift_px'):.3f}>"
            f"{args.max_bbox_drift_px:.3f}"
        )
    if metric(drift_report, "max_kp_drift_px") > args.max_kp_drift_px:
        failures.append(
            f"keypoint_drift:{metric(drift_report, 'max_kp_drift_px'):.3f}>"
            f"{args.max_kp_drift_px:.3f}"
        )
    if metric(drift_report, "max_conf_drift") > args.max_conf_drift:
        failures.append(
            f"confidence_drift:{metric(drift_report, 'max_conf_drift'):.3f}>"
            f"{args.max_conf_drift:.3f}"
        )
    if exported_metrics["bbox_map50"] < args.min_export_map50:
        failures.append(
            f"bbox_map50:{exported_metrics['bbox_map50']:.3f}<{args.min_export_map50:.3f}"
        )
    if exported_metrics["oks_mean"] < args.min_export_oks:
        failures.append(
            f"oks:{exported_metrics['oks_mean']:.3f}<{args.min_export_oks:.3f}"
        )
    if deltas["bbox_map50"] > args.max_map50_delta:
        failures.append(f"map50_delta:{deltas['bbox_map50']:.3f}>{args.max_map50_delta:.3f}")
    if deltas["bbox_map50_95"] > args.max_map5095_delta:
        failures.append(
            f"map50_95_delta:{deltas['bbox_map50_95']:.3f}>{args.max_map5095_delta:.3f}"
        )
    if deltas["oks_mean"] > args.max_oks_delta:
        failures.append(f"oks_delta:{deltas['oks_mean']:.3f}>{args.max_oks_delta:.3f}")
    if deltas["false_negative_rate"] > args.max_fn_delta:
        failures.append(f"fn_delta:{deltas['false_negative_rate']:.3f}>{args.max_fn_delta:.3f}")
    if deltas["false_positive_rate"] > args.max_fp_delta:
        failures.append(f"fp_delta:{deltas['false_positive_rate']:.3f}>{args.max_fp_delta:.3f}")

    return {
        "name": name,
        "certified": not failures,
        "failures": failures,
        "aggregate_metrics": exported_metrics,
        "metric_deltas_vs_pytorch": deltas,
        "drift": {
            "strict_ok": bool(drift_report.get("ok", False)),
            "samples_checked": int(metric(drift_report, "samples_checked")),
            "samples_matched_strict": int(metric(drift_report, "samples_matched")),
            "has_count_mismatch": has_count_mismatch(drift_report),
            "coordinate_scale_warnings": scale_warnings,
            "max_bbox_drift_px": metric(drift_report, "max_bbox_drift_px"),
            "max_kp_drift_px": metric(drift_report, "max_kp_drift_px"),
            "max_conf_drift": metric(drift_report, "max_conf_drift"),
        },
    }


def build_certification(args: argparse.Namespace) -> dict[str, Any]:
    pt_eval = read_json(args.pt_eval)
    onnx_eval = read_json(args.onnx_eval)
    tflite_eval = read_json(args.tflite_eval)
    onnx_drift = read_json(args.onnx_drift)
    tflite_drift = read_json(args.tflite_drift)
    litert_smoke = read_json(args.litert_smoke)
    pt_metrics = eval_summary(pt_eval)

    backends = {
        "onnx": certify_backend(
            name="onnx",
            eval_report=onnx_eval,
            drift_report=onnx_drift,
            pt_metrics=pt_metrics,
            args=args,
        ),
        "tflite": certify_backend(
            name="tflite",
            eval_report=tflite_eval,
            drift_report=tflite_drift,
            pt_metrics=pt_metrics,
            args=args,
        ),
    }
    litert_ok = bool(litert_smoke.get("ok", False))
    if not litert_ok:
        backends["tflite"]["certified"] = False
        backends["tflite"]["failures"].append("litert_smoke_failed")

    certified = all(backend["certified"] for backend in backends.values())
    return {
        "schema_version": 1,
        "certified": certified,
        "status": "certified" if certified else "failed",
        "scope": "desktop_export_backend_certification_not_android_device",
        "policy": {
            "min_drift_samples": args.min_drift_samples,
            "max_bbox_drift_px": args.max_bbox_drift_px,
            "max_kp_drift_px": args.max_kp_drift_px,
            "max_conf_drift": args.max_conf_drift,
            "min_export_map50": args.min_export_map50,
            "min_export_oks": args.min_export_oks,
            "max_map50_delta": args.max_map50_delta,
            "max_map5095_delta": args.max_map5095_delta,
            "max_oks_delta": args.max_oks_delta,
            "max_fn_delta": args.max_fn_delta,
            "max_fp_delta": args.max_fp_delta,
            "strict_parity_note": (
                "Strict 2px/3px/0.05 parity remains diagnostic. Production export "
                "certification uses calibrated drift plus aggregate metric parity."
            ),
        },
        "pytorch_reference": pt_metrics,
        "backends": backends,
        "litert_runtime_smoke": {
            "ok": litert_ok,
            "runtime": litert_smoke.get("runtime"),
            "output_shape": (
                litert_smoke.get("outputs", [{}])[0].get("shape")
                if isinstance(litert_smoke.get("outputs"), list) and litert_smoke.get("outputs")
                else None
            ),
            "mean_latency_ms_cpu": metric(litert_smoke, "latency_ms", "mean"),
        },
        "inputs": {
            "pt_eval": str(args.pt_eval),
            "onnx_eval": str(args.onnx_eval),
            "tflite_eval": str(args.tflite_eval),
            "onnx_drift": str(args.onnx_drift),
            "tflite_drift": str(args.tflite_drift),
            "litert_smoke": str(args.litert_smoke),
        },
    }


def render_markdown(cert: dict[str, Any]) -> str:
    lines = [
        "# Export Certification",
        "",
        f"- Certified: {cert.get('certified')}",
        f"- Scope: {cert.get('scope')}",
        f"- Status: {cert.get('status')}",
        "",
        "| Backend | Certified | mAP50 | OKS | Max bbox px | Max kp px | Max conf | Failures |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, backend in cert.get("backends", {}).items():
        metrics = backend.get("aggregate_metrics", {})
        drift = backend.get("drift", {})
        failures = ", ".join(backend.get("failures", [])) or "none"
        lines.append(
            "| "
            f"{name} | {backend.get('certified')} | "
            f"{float(metrics.get('bbox_map50', 0.0)):.3f} | "
            f"{float(metrics.get('oks_mean', 0.0)):.3f} | "
            f"{float(drift.get('max_bbox_drift_px', 0.0)):.3f} | "
            f"{float(drift.get('max_kp_drift_px', 0.0)):.3f} | "
            f"{float(drift.get('max_conf_drift', 0.0)):.3f} | "
            f"{failures} |"
        )
    lines.extend(
        [
            "",
            "## Policy",
            "",
            cert.get("policy", {}).get("strict_parity_note", ""),
            "",
            "The scope is desktop/export-backend certification. Android device latency, memory, "
            "and end-to-end LiteRT integration remain separate production evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt-eval", type=Path, default=DEFAULT_PT_EVAL)
    parser.add_argument("--onnx-eval", type=Path, default=DEFAULT_ONNX_EVAL)
    parser.add_argument("--tflite-eval", type=Path, default=DEFAULT_TFLITE_EVAL)
    parser.add_argument("--onnx-drift", type=Path, default=DEFAULT_ONNX_DRIFT)
    parser.add_argument("--tflite-drift", type=Path, default=DEFAULT_TFLITE_DRIFT)
    parser.add_argument("--litert-smoke", type=Path, default=DEFAULT_LITERT_SMOKE)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    parser.add_argument("--min-drift-samples", type=int, default=20)
    parser.add_argument("--max-bbox-drift-px", type=float, default=10.0)
    parser.add_argument("--max-kp-drift-px", type=float, default=15.0)
    parser.add_argument("--max-conf-drift", type=float, default=0.25)
    parser.add_argument("--min-export-map50", type=float, default=0.65)
    parser.add_argument("--min-export-oks", type=float, default=0.80)
    parser.add_argument("--max-map50-delta", type=float, default=0.02)
    parser.add_argument("--max-map5095-delta", type=float, default=0.02)
    parser.add_argument("--max-oks-delta", type=float, default=0.02)
    parser.add_argument("--max-fn-delta", type=float, default=0.02)
    parser.add_argument("--max-fp-delta", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cert = build_certification(args)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(cert, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(cert), encoding="utf-8")
    print(f"certified={cert['certified']} scope={cert['scope']}")
    for name, backend in cert["backends"].items():
        print(f"{name}: certified={backend['certified']} failures={backend['failures']}")
    print(f"json={args.json_out}")
    print(f"markdown={args.md_out}")
    return 0 if cert["certified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
