"""Benchmark the current wheel-pose release artifacts on local runtimes.

The numbers produced here are desktop-local diagnostics, not Android
production certification. The goal is to keep a reproducible latency
record for the exact PT/ONNX/TFLite artifacts in the handoff package.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any


DEFAULT_PT_MODEL = Path("runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt")
DEFAULT_ONNX_MODEL = Path("runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.onnx")
DEFAULT_IMAGES_DIR = Path("data/wheel_pose_dataset_real_v1_self_plus_ue_synthetic/images/val")
DEFAULT_LITERT_SMOKE = Path("outputs/production_audit/litert_runtime_smoke.json")
DEFAULT_JSON_OUT = Path("outputs/production_audit/performance_audit.json")
DEFAULT_MD_OUT = Path("docs/PERFORMANCE_AUDIT.md")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if q <= 0:
        return float(min(values))
    if q >= 100:
        return float(max(values))
    ordered = sorted(float(v) for v in values)
    pos = (len(ordered) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def summarize_ms(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"runs": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    return {
        "runs": len(values),
        "mean": float(statistics.fmean(values)),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def iter_images(root: Path, limit: int) -> list[Path]:
    if not root.is_dir():
        return []
    images = [path for path in sorted(root.iterdir()) if path.suffix.lower() in IMAGE_SUFFIXES]
    return images[: max(0, limit)] if limit else images


def _count_detections(result: Any) -> int:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return 0
    try:
        return int(len(boxes))
    except TypeError:
        return 0


def benchmark_yolo(
    *,
    name: str,
    model_path: Path,
    images: list[Path],
    device: str,
    warmup: int,
    repeats: int,
    conf: float,
    iou: float,
    max_det: int,
    task: str,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "name": name,
        "ok": False,
        "model": str(model_path),
        "runtime": "ultralytics",
        "device": device,
        "task": task,
        "images": len(images),
        "warmup": warmup,
        "repeats": repeats,
        "latency_ms": summarize_ms([]),
        "detections": {"total": 0, "mean_per_image": None},
    }
    if not model_path.is_file():
        report["error"] = f"missing model: {model_path}"
        return report
    if not images:
        report["error"] = "no benchmark images"
        return report

    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]  # noqa: PLC0415

        model_kwargs = {} if task == "auto" else {"task": task}
        model = YOLO(str(model_path), **model_kwargs)
        for _ in range(max(0, warmup)):
            model.predict(
                source=str(images[0]),
                conf=conf,
                iou=iou,
                max_det=max_det,
                device=device,
                verbose=False,
            )

        latencies_ms: list[float] = []
        detections = 0
        measured_images = 0
        for _ in range(max(1, repeats)):
            for image in images:
                start = time.perf_counter()
                results = model.predict(
                    source=str(image),
                    conf=conf,
                    iou=iou,
                    max_det=max_det,
                    device=device,
                    verbose=False,
                )
                latencies_ms.append((time.perf_counter() - start) * 1000.0)
                if results:
                    detections += _count_detections(results[0])
                measured_images += 1

        report["ok"] = bool(latencies_ms)
        report["latency_ms"] = summarize_ms(latencies_ms)
        report["detections"] = {
            "total": detections,
            "mean_per_image": float(detections / measured_images) if measured_images else None,
        }
    except Exception as exc:  # pragma: no cover - exercised only when a backend breaks locally.
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def load_litert_smoke(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "name": "litert_cpu_smoke",
            "ok": False,
            "runtime": "ai_edge_litert",
            "source_report": str(path),
            "error": f"missing report: {path}",
            "latency_ms": summarize_ms([]),
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "name": "litert_cpu_smoke",
            "ok": False,
            "runtime": "ai_edge_litert",
            "source_report": str(path),
            "error": f"invalid JSON: {exc}",
            "latency_ms": summarize_ms([]),
        }

    return {
        "name": "litert_cpu_smoke",
        "ok": bool(payload.get("ok")),
        "runtime": payload.get("runtime", "ai_edge_litert"),
        "source_report": str(path),
        "model": payload.get("model"),
        "image": payload.get("image"),
        "input": payload.get("input"),
        "outputs": payload.get("outputs", []),
        "latency_ms": payload.get("latency_ms", summarize_ms([])),
    }


def build_report(
    *,
    pt_model: Path,
    onnx_model: Path,
    images_dir: Path,
    litert_smoke: Path,
    limit: int,
    warmup: int,
    repeats: int,
    device: str,
    conf: float,
    iou: float,
    max_det: int,
    task: str,
) -> dict[str, Any]:
    images = iter_images(images_dir, limit)
    benchmarks = {
        "pytorch_cpu": benchmark_yolo(
            name="pytorch_cpu",
            model_path=pt_model,
            images=images,
            device=device,
            warmup=warmup,
            repeats=repeats,
            conf=conf,
            iou=iou,
            max_det=max_det,
            task=task,
        ),
        "onnx_cpu": benchmark_yolo(
            name="onnx_cpu",
            model_path=onnx_model,
            images=images,
            device=device,
            warmup=warmup,
            repeats=repeats,
            conf=conf,
            iou=iou,
            max_det=max_det,
            task=task,
        ),
        "litert_cpu_smoke": load_litert_smoke(litert_smoke),
    }
    failures = [name for name, item in benchmarks.items() if not item.get("ok")]
    return {
        "ok": not failures,
        "schema_version": 1,
        "scope": "desktop_local_runtime_diagnostic_not_android_certification",
        "images_dir": str(images_dir),
        "sample_count": len(images),
        "sample_images": [str(path) for path in images],
        "settings": {
            "device": device,
            "warmup": warmup,
            "repeats": repeats,
            "conf": conf,
            "iou": iou,
            "max_det": max_det,
            "task": task,
        },
        "benchmarks": benchmarks,
        "failures": failures,
        "notes": [
            "Ultralytics PT/ONNX measurements include preprocess, inference, and postprocess wall time.",
            "LiteRT value is imported from the raw ai_edge_litert smoke report.",
            "Android production latency must still be measured inside the target app/runtime.",
        ],
    }


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Performance Audit",
        "",
        "Desktop-local inference latency diagnostic for the current wheel-pose release artifacts.",
        "",
        f"- OK: {report.get('ok')}",
        f"- Scope: `{report.get('scope', 'n/a')}`",
        f"- Sample count: {report.get('sample_count', 'n/a')}",
        f"- Images dir: `{report.get('images_dir', 'n/a')}`",
        "",
        "| Runtime | OK | Device | Runs | Mean ms | P50 ms | P95 ms | Detections/image |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for name, bench in report.get("benchmarks", {}).items():
        latency = bench.get("latency_ms", {}) if isinstance(bench, dict) else {}
        detections = bench.get("detections", {}) if isinstance(bench, dict) else {}
        lines.append(
            "| "
            f"{name} | "
            f"{bench.get('ok', False)} | "
            f"{bench.get('device', bench.get('runtime', 'n/a'))} | "
            f"{_fmt(latency.get('runs'))} | "
            f"{_fmt(latency.get('mean'))} | "
            f"{_fmt(latency.get('p50'))} | "
            f"{_fmt(latency.get('p95'))} | "
            f"{_fmt(detections.get('mean_per_image'))} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
        ]
    )
    for note in report.get("notes", []):
        lines.append(f"- {note}")
    if report.get("failures"):
        lines.extend(["", f"Failures: `{', '.join(report['failures'])}`"])
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt-model", type=Path, default=DEFAULT_PT_MODEL)
    parser.add_argument("--onnx-model", type=Path, default=DEFAULT_ONNX_MODEL)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--litert-smoke", type=Path, default=DEFAULT_LITERT_SMOKE)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--task", choices=("auto", "pose"), default="pose")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        pt_model=args.pt_model,
        onnx_model=args.onnx_model,
        images_dir=args.images_dir,
        litert_smoke=args.litert_smoke,
        limit=args.limit,
        warmup=args.warmup,
        repeats=args.repeats,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        task=args.task,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(report), encoding="utf-8")
    print(f"ok={report['ok']} samples={report['sample_count']} failures={report['failures']}")
    for name, bench in report["benchmarks"].items():
        latency = bench.get("latency_ms", {})
        print(
            f"{name}: ok={bench.get('ok')} "
            f"mean_ms={_fmt(latency.get('mean'))} p95_ms={_fmt(latency.get('p95'))}"
        )
    print(f"json={args.json_out}")
    print(f"markdown={args.md_out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
