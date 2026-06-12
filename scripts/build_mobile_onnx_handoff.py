"""Build the Android ONNX Runtime Mobile handoff bundle.

This bundle is separate from the <=6 MB TFLite/CoreML handoff. It packages
the 384x384 nano ONNX candidate for Android/ONNX Runtime Mobile integration
smoke tests, records the graph interface, and writes a deterministic zip for
handoff. It is an integration artifact, not a production promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("outputs/production_audit/mobile_onnx")
DEFAULT_SOURCE_ONNX = Path("outputs/production_audit/mobile_6mb/export_tmp_nano_384/best.onnx")
DEFAULT_TARGET_ONNX = DEFAULT_ROOT / "best_mobile_384.onnx"
DEFAULT_MANIFEST = DEFAULT_ROOT / "mobile_onnx_handoff_manifest.json"
DEFAULT_MARKDOWN = Path("docs/MOBILE_ONNX_HANDOFF.md")
DEFAULT_ZIP = DEFAULT_ROOT / "mobile_onnx_handoff.zip"
DEFAULT_SMOKE_REPORT = DEFAULT_ROOT / "onnxruntime_mobile_cpu_smoke.json"
DEFAULT_QUALITY_REPORT = Path("outputs/production_audit/mobile_6mb/nano_source_eval_self_plus_ue_conf025.json")
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _path(path: Path) -> str:
    return str(path).replace("\\", "/")


def inspect_file(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    size_bytes = path.stat().st_size if exists else 0
    return {
        "path": _path(path),
        "exists": exists,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 3),
        "sha256": sha256_file(path) if exists and size_bytes > 0 else None,
    }


def _shape_from_value(value: Any) -> list[int | str]:
    dims = value.type.tensor_type.shape.dim
    shape: list[int | str] = []
    for dim in dims:
        if dim.dim_value:
            shape.append(int(dim.dim_value))
        elif dim.dim_param:
            shape.append(str(dim.dim_param))
        else:
            shape.append("?")
    return shape


def inspect_onnx_graph(model_path: Path) -> dict[str, Any]:
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        return {"ok": False, "error": f"onnx import failed: {exc}"}

    try:
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)
    except Exception as exc:  # pragma: no cover - depends on corrupt external files
        return {"ok": False, "error": f"onnx check failed: {exc}"}

    return {
        "ok": True,
        "ir_version": int(model.ir_version),
        "opsets": [
            {"domain": item.domain or "ai.onnx", "version": int(item.version)}
            for item in model.opset_import
        ],
        "node_count": len(model.graph.node),
        "inputs": [
            {
                "name": item.name,
                "shape": _shape_from_value(item),
                "elem_type": int(item.type.tensor_type.elem_type),
            }
            for item in model.graph.input
        ],
        "outputs": [
            {
                "name": item.name,
                "shape": _shape_from_value(item),
                "elem_type": int(item.type.tensor_type.elem_type),
            }
            for item in model.graph.output
        ],
    }


def _first_static_input_shape(signature: dict[str, Any]) -> list[int]:
    inputs = signature.get("inputs") or []
    if inputs:
        shape = inputs[0].get("shape") or []
        if shape and all(isinstance(v, int) and v > 0 for v in shape):
            return list(shape)
    return [1, 3, 384, 384]


def run_onnxruntime_smoke(
    model_path: Path,
    signature: dict[str, Any],
    report_out: Path,
    *,
    runs: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": 1,
        "ok": False,
        "model": _path(model_path),
        "provider": "CPUExecutionProvider",
        "runs": runs,
    }
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        report["error"] = f"onnxruntime smoke import failed: {exc}"
        _write_json(report_out, report)
        return report

    try:
        input_shape = _first_static_input_shape(signature)
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        sample = np.zeros(input_shape, dtype=np.float32)
        timings_ms: list[float] = []
        output_shapes: dict[str, list[int]] = {}
        for _ in range(runs):
            t0 = time.perf_counter()
            outputs = session.run(None, {input_name: sample})
            timings_ms.append((time.perf_counter() - t0) * 1000.0)
        for meta, value in zip(session.get_outputs(), outputs):
            output_shapes[meta.name] = [int(v) for v in value.shape]
        report.update(
            {
                "ok": True,
                "input_name": input_name,
                "input_shape": input_shape,
                "output_shapes": output_shapes,
                "latency_ms": {
                    "min": round(min(timings_ms), 3),
                    "avg": round(sum(timings_ms) / len(timings_ms), 3),
                    "max": round(max(timings_ms), 3),
                },
            }
        )
    except Exception as exc:  # pragma: no cover - depends on external model/runtime
        report["error"] = f"onnxruntime smoke failed: {exc}"
    report["duration_seconds"] = round(time.perf_counter() - started, 3)
    _write_json(report_out, report)
    return report


def build_manifest(
    *,
    model_path: Path,
    source_onnx: Path,
    signature: dict[str, Any],
    smoke_report: dict[str, Any] | None,
    smoke_report_path: Path | None,
    quality_report: Path | None,
) -> dict[str, Any]:
    model = inspect_file(model_path)
    quality = inspect_file(quality_report) if quality_report else None
    smoke_file = inspect_file(smoke_report_path) if smoke_report_path else None
    failures: list[str] = []
    if not model["exists"] or model["size_bytes"] <= 0:
        failures.append(f"missing_model:{model_path}")
    if not signature.get("ok"):
        failures.append("onnx_signature_failed")
    if smoke_report is not None and not smoke_report.get("ok"):
        failures.append("onnxruntime_smoke_failed")

    artifacts = [
        {
            "platform": "android",
            "role": "model",
            "format": "onnx",
            "precision": "float32",
            "runtime": "onnxruntime-mobile",
            "source_path": _path(source_onnx),
            **model,
        }
    ]
    if smoke_file:
        artifacts.append({"platform": "android", "role": "validation", **smoke_file})
    if quality:
        artifacts.append({"platform": "shared", "role": "quality_reference", **quality})

    return {
        "schema_version": 1,
        "ok": not failures,
        "scope": "mobile_onnx_runtime_handoff_candidate_not_production_promotion",
        "model_status": "integration_smoke_not_production",
        "policy": {
            "target_runtime": "ONNX Runtime Mobile",
            "target_platform": "Android",
            "input_shape": [1, 3, 384, 384],
            "output_shape": [1, 14, 3024],
            "size_note": "ONNX candidate is not part of the <=6 MB TFLite/CoreML constraint.",
            "promotion_note": (
                "Promote only after real AR holdout validation and target-device "
                "latency/memory evidence."
            ),
        },
        "failures": failures,
        "onnx_signature": signature,
        "onnxruntime_smoke": smoke_report,
        "artifacts": artifacts,
    }


def render_markdown(manifest: dict[str, Any]) -> str:
    policy = manifest.get("policy", {})
    model = next(
        (row for row in manifest.get("artifacts", []) if row.get("role") == "model"),
        {},
    )
    smoke = manifest.get("onnxruntime_smoke") or {}
    signature = manifest.get("onnx_signature") or {}
    inputs_json = json.dumps(signature.get("inputs"), ensure_ascii=False)
    outputs_json = json.dumps(signature.get("outputs"), ensure_ascii=False)
    opsets_json = json.dumps(signature.get("opsets"), ensure_ascii=False)
    latency_json = json.dumps(smoke.get("latency_ms"), ensure_ascii=False)
    smoke_outputs_json = json.dumps(smoke.get("output_shapes"), ensure_ascii=False)
    lines = [
        "# Mobile ONNX Handoff",
        "",
        f"- OK: {manifest.get('ok')}",
        f"- Scope: {manifest.get('scope')}",
        f"- Runtime: {policy.get('target_runtime')} on {policy.get('target_platform')}",
        f"- Model: `{model.get('path')}`",
        f"- Size: {model.get('size_mb')} MB",
        f"- SHA256: `{model.get('sha256')}`",
        f"- Status: **{manifest.get('model_status')}**",
        f"- Failures: {', '.join(manifest.get('failures', [])) if manifest.get('failures') else 'none'}",
        "",
        "## Interface",
        "",
        f"- Input shape: `{policy.get('input_shape')}`",
        f"- Output shape: `{policy.get('output_shape')}`",
        f"- ONNX inputs: `{inputs_json}`",
        f"- ONNX outputs: `{outputs_json}`",
        f"- Opsets: `{opsets_json}`",
        "",
        "## Runtime Smoke",
        "",
        f"- OK: {smoke.get('ok')}",
        f"- Provider: {smoke.get('provider')}",
        f"- Runs: {smoke.get('runs')}",
        f"- Latency ms: `{latency_json}`",
        f"- Output shapes: `{smoke_outputs_json}`",
        "",
        "## Package",
        "",
        f"- Zip: `{manifest.get('zip')}`",
        f"- Zip SHA256: `{manifest.get('zip_sha256')}`",
        "",
        "## Rebuild",
        "",
        "```bash",
        "./.venv/bin/python scripts/build_mobile_onnx_handoff.py",
        "```",
        "",
        "## Notes",
        "",
        "- This is the ONNX Runtime Mobile integration candidate, not a production promotion.",
        "- The existing <=6 MB mobile package remains TFLite/CoreML-only.",
        "- Do not claim production readiness until Android-device latency/memory and real AR holdout evidence are attached.",
        "",
    ]
    return "\n".join(lines)


def handoff_paths(
    manifest: dict[str, Any],
    manifest_out: Path,
    markdown_out: Path,
) -> list[Path]:
    paths = [Path(row["path"]) for row in manifest.get("artifacts", []) if row.get("exists")]
    paths.extend([manifest_out, markdown_out])
    return list(dict.fromkeys(paths))


def write_zip(paths: list[Path], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(paths, key=_path):
            info = zipfile.ZipInfo(_path(path), date_time=ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def materialize_handoff(
    *,
    source_onnx: Path,
    target_onnx: Path,
    manifest_out: Path,
    markdown_out: Path,
    zip_out: Path,
    smoke_report_out: Path,
    quality_report: Path | None,
    runs: int,
    run_smoke: bool,
) -> dict[str, Any]:
    if not source_onnx.is_file():
        raise FileNotFoundError(f"source ONNX not found: {source_onnx}")
    target_onnx.parent.mkdir(parents=True, exist_ok=True)
    if source_onnx.resolve() != target_onnx.resolve():
        shutil.copy2(source_onnx, target_onnx)

    signature = inspect_onnx_graph(target_onnx)
    smoke = (
        run_onnxruntime_smoke(target_onnx, signature, smoke_report_out, runs=runs)
        if run_smoke
        else None
    )
    manifest = build_manifest(
        model_path=target_onnx,
        source_onnx=source_onnx,
        signature=signature,
        smoke_report=smoke,
        smoke_report_path=smoke_report_out if run_smoke else None,
        quality_report=quality_report if quality_report and quality_report.exists() else None,
    )
    _write_json(manifest_out, manifest)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text(render_markdown(manifest), encoding="utf-8")
    if manifest["ok"]:
        write_zip(handoff_paths(manifest, manifest_out, markdown_out), zip_out)
        manifest["zip"] = _path(zip_out)
        manifest["zip_size_bytes"] = zip_out.stat().st_size
        manifest["zip_sha256"] = sha256_file(zip_out)
        _write_json(manifest_out, manifest)
        markdown_out.write_text(render_markdown(manifest), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-onnx", type=Path, default=DEFAULT_SOURCE_ONNX)
    parser.add_argument("--target-onnx", type=Path, default=DEFAULT_TARGET_ONNX)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--zip-out", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--smoke-report-out", type=Path, default=DEFAULT_SMOKE_REPORT)
    parser.add_argument("--quality-report", type=Path, default=DEFAULT_QUALITY_REPORT)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--no-smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = materialize_handoff(
        source_onnx=args.source_onnx,
        target_onnx=args.target_onnx,
        manifest_out=args.manifest_out,
        markdown_out=args.markdown_out,
        zip_out=args.zip_out,
        smoke_report_out=args.smoke_report_out,
        quality_report=args.quality_report,
        runs=args.runs,
        run_smoke=not args.no_smoke,
    )
    print(f"ok={manifest['ok']} failures={manifest['failures']}")
    print(f"model={args.target_onnx}")
    print(f"manifest={args.manifest_out}")
    print(f"markdown={args.markdown_out}")
    if manifest.get("zip"):
        print(f"zip={args.zip_out}")
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
