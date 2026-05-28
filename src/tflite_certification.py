"""Build the desktop TFLite/LiteRT package certification report.

This report certifies the exported `.tflite` artifact for package
handoff: the artifact exists, the calibrated export certification passes
for the TFLite backend, aggregate eval is present, and the desktop
ai_edge_litert smoke test executes with finite output.

It is intentionally not an Android-device certification. Target-device
evidence is validated separately by `src/validate_android_litert_report.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_TFLITE = Path("outputs/production_audit/tflite_export/best_float32.tflite")
DEFAULT_EXPORT_CERT = Path("outputs/production_audit/export_certification.json")
DEFAULT_TFLITE_EVAL = Path("outputs/eval/wheel_real_v1_self_plus_ue_synthetic_s_tflite_on_self_plus_ue_val.json")
DEFAULT_TFLITE_DRIFT = Path("outputs/production_audit/tflite_drift_20.json")
DEFAULT_LITERT_SMOKE = Path("outputs/production_audit/litert_runtime_smoke.json")
DEFAULT_JSON_OUT = Path("outputs/production_audit/tflite_certification.json")


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def output_shape(smoke: dict[str, Any]) -> list[int] | None:
    outputs = smoke.get("outputs", [])
    if not isinstance(outputs, list) or not outputs:
        return None
    first = outputs[0]
    if not isinstance(first, dict) or not isinstance(first.get("shape"), list):
        return None
    return [int(v) for v in first["shape"]]


def build_certification(args: argparse.Namespace) -> dict[str, Any]:
    export_cert = read_json(args.export_certification)
    tflite_eval = read_json(args.tflite_eval)
    drift = read_json(args.tflite_drift)
    smoke = read_json(args.litert_smoke)
    tflite_backend = export_cert.get("backends", {}).get("tflite", {})

    failures: list[str] = []
    if not args.artifact.is_file():
        failures.append(f"missing_artifact:{args.artifact}")
    if not bool(tflite_backend.get("certified", False)):
        failures.append("export_backend_tflite_not_certified")
    if not bool(smoke.get("ok", False)):
        failures.append("litert_smoke_failed")
    if output_shape(smoke) != [1, 14, 8400]:
        failures.append(f"unexpected_output_shape:{output_shape(smoke)}")
    if metric(tflite_eval, "metrics_bbox", "mAP50") < args.min_map50:
        failures.append(
            f"bbox_map50:{metric(tflite_eval, 'metrics_bbox', 'mAP50'):.3f}<"
            f"{args.min_map50:.3f}"
        )
    if metric(tflite_eval, "oks", "mean") < args.min_oks:
        failures.append(f"oks:{metric(tflite_eval, 'oks', 'mean'):.3f}<{args.min_oks:.3f}")

    certified = not failures
    return {
        "schema_version": 2,
        "certified": certified,
        "status": "certified" if certified else "failed",
        "scope": "desktop_tflite_litert_package_not_android_device",
        "failures": failures,
        "reason": (
            "TFLite package passes calibrated export certification and desktop "
            "ai_edge_litert smoke. Android device validation is tracked separately."
            if certified
            else "TFLite package certification failed; see failures."
        ),
        "artifact": {
            "format": "tflite_float32",
            "path": str(args.artifact),
            "size_mb": round(args.artifact.stat().st_size / (1024 * 1024), 3)
            if args.artifact.is_file()
            else 0.0,
            "sha256": sha256_file(args.artifact),
            "source_model": "runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt",
            "model_task": "pose",
        },
        "export_certification": {
            "report": str(args.export_certification),
            "certified": bool(export_cert.get("certified", False)),
            "tflite_backend_certified": bool(tflite_backend.get("certified", False)),
            "tflite_backend_failures": tflite_backend.get("failures", []),
        },
        "multi_frame_drift": {
            "report": str(args.tflite_drift),
            "strict_ok": bool(drift.get("ok", False)),
            "samples_checked": drift.get("samples_checked"),
            "samples_matched_strict": drift.get("samples_matched"),
            "max_bbox_drift_px": drift.get("max_bbox_drift_px"),
            "max_keypoint_drift_px": drift.get("max_kp_drift_px"),
            "max_conf_drift": drift.get("max_conf_drift"),
            "note": "Strict parity is diagnostic; calibrated export certification is authoritative.",
        },
        "litert_runtime_smoke": {
            "report": str(args.litert_smoke),
            "ok": bool(smoke.get("ok", False)),
            "runtime": smoke.get("runtime"),
            "output_shape": output_shape(smoke),
            "mean_latency_ms_cpu": metric(smoke, "latency_ms", "mean"),
            "p95_latency_ms_cpu": metric(smoke, "latency_ms", "p95"),
        },
        "aggregate_eval": {
            "report": str(args.tflite_eval),
            "bbox_map50": metric(tflite_eval, "metrics_bbox", "mAP50"),
            "bbox_map50_95": metric(tflite_eval, "metrics_bbox", "mAP50_95"),
            "oks_mean": metric(tflite_eval, "oks", "mean"),
            "false_negative_rate": metric(tflite_eval, "rates", "false_negative_rate", default=1.0),
            "false_positive_rate": metric(tflite_eval, "rates", "false_positive_rate", default=1.0),
            "images": int(metric(tflite_eval, "counts", "images")),
            "gt_wheels": int(metric(tflite_eval, "counts", "gt_wheels")),
            "pred_wheels_above_conf": int(metric(tflite_eval, "counts", "pred_wheels_above_conf")),
            "matched": int(metric(tflite_eval, "counts", "matched")),
        },
        "next_steps": [
            "Run the exact artifact on the target Android LiteRT integration.",
            "Validate latency, memory, output shape, and finite outputs with src/validate_android_litert_report.py.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_TFLITE)
    parser.add_argument("--export-certification", type=Path, default=DEFAULT_EXPORT_CERT)
    parser.add_argument("--tflite-eval", type=Path, default=DEFAULT_TFLITE_EVAL)
    parser.add_argument("--tflite-drift", type=Path, default=DEFAULT_TFLITE_DRIFT)
    parser.add_argument("--litert-smoke", type=Path, default=DEFAULT_LITERT_SMOKE)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--min-map50", type=float, default=0.65)
    parser.add_argument("--min-oks", type=float, default=0.80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_certification(args)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"certified={report['certified']} scope={report['scope']} "
        f"failures={report['failures']}"
    )
    print(f"json={args.json_out}")
    return 0 if report["certified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
