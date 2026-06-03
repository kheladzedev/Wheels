"""Build a mobile optimization report for compressed model candidates.

The production handoff keeps the exact float32 artifacts as the baseline.
This script tracks lighter mobile candidates separately so iOS/Android can
compare file size, input shape, output shape, and validation status without
confusing an optimized experiment with the already-certified baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUT_DIR = Path("outputs/production_audit/mobile_optimization")
DEFAULT_MANIFEST = DEFAULT_OUT_DIR / "mobile_optimization_report.json"
DEFAULT_ZIP = DEFAULT_OUT_DIR / "mobile_optimization_handoff.zip"
DEFAULT_MARKDOWN = Path("docs/MOBILE_OPTIMIZATION_REPORT.md")
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class BaselineSpec:
    id: str
    platform: str
    path: Path
    input_shape: list[int]
    output_shape: list[int]


@dataclass(frozen=True)
class CandidateSpec:
    id: str
    baseline_id: str
    platform: str
    precision: str
    source_path: Path
    target_path: Path
    input_shape: list[int]
    output_shape: list[int]
    required: bool = False
    validation_report: Path | None = None
    note: str = ""


DEFAULT_BASELINES = [
    BaselineSpec(
        id="tflite_float32_640",
        platform="android",
        path=Path("outputs/production_audit/tflite_export/best_float32.tflite"),
        input_shape=[1, 640, 640, 3],
        output_shape=[1, 14, 8400],
    ),
    BaselineSpec(
        id="coreml_float32_640",
        platform="ios",
        path=Path("outputs/production_audit/coreml_export/best.mlmodel"),
        input_shape=[1, 640, 640, 3],
        output_shape=[1, 14, 8400],
    ),
]


DEFAULT_CANDIDATES = [
    CandidateSpec(
        id="tflite_fp16_640",
        baseline_id="tflite_float32_640",
        platform="android",
        precision="fp16",
        source_path=Path(
            "runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/"
            "best_saved_model/best_float16.tflite"
        ),
        target_path=DEFAULT_OUT_DIR / "tflite_fp16_640/best_float16.tflite",
        input_shape=[1, 640, 640, 3],
        output_shape=[1, 14, 8400],
        required=True,
        validation_report=DEFAULT_OUT_DIR / "litert_smoke_tflite_fp16_640.json",
        note="Existing Ultralytics FP16 TFLite export; first safe mobile candidate.",
    ),
    CandidateSpec(
        id="tflite_fp16_416",
        baseline_id="tflite_float32_640",
        platform="android",
        precision="fp16",
        source_path=DEFAULT_OUT_DIR / "tflite_fp16_416/best_float16.tflite",
        target_path=DEFAULT_OUT_DIR / "tflite_fp16_416/best_float16.tflite",
        input_shape=[1, 416, 416, 3],
        output_shape=[1, 14, 3549],
        validation_report=DEFAULT_OUT_DIR / "litert_smoke_tflite_fp16_416.json",
        note="Lower-resolution candidate; requires quality/latency comparison.",
    ),
    CandidateSpec(
        id="tflite_dynamic_range_int8_640",
        baseline_id="tflite_float32_640",
        platform="android",
        precision="dynamic_range_int8_weights",
        source_path=DEFAULT_OUT_DIR
        / "tflite_dynamic_range_int8_640/best_dynamic_range_quant.tflite",
        target_path=DEFAULT_OUT_DIR
        / "tflite_dynamic_range_int8_640/best_dynamic_range_quant.tflite",
        input_shape=[1, 640, 640, 3],
        output_shape=[1, 14, 8400],
        validation_report=DEFAULT_OUT_DIR
        / "litert_smoke_tflite_dynamic_range_int8_640.json",
        note=(
            "TFLite dynamic-range quantized candidate. Full INT8 export needs "
            "more calibration images; current exporter found 58 and recommends >300."
        ),
    ),
    CandidateSpec(
        id="tflite_fp16_384",
        baseline_id="tflite_float32_640",
        platform="android",
        precision="fp16",
        source_path=DEFAULT_OUT_DIR / "tflite_fp16_384/best_float16.tflite",
        target_path=DEFAULT_OUT_DIR / "tflite_fp16_384/best_float16.tflite",
        input_shape=[1, 384, 384, 3],
        output_shape=[1, 14, 3024],
        validation_report=DEFAULT_OUT_DIR / "litert_smoke_tflite_fp16_384.json",
        note="Lower-resolution candidate; requires quality/latency comparison.",
    ),
    CandidateSpec(
        id="coreml_linear_int8_640",
        baseline_id="coreml_float32_640",
        platform="ios",
        precision="int8_weights",
        source_path=DEFAULT_OUT_DIR / "coreml_linear_int8_640/best_int8.mlmodel",
        target_path=DEFAULT_OUT_DIR / "coreml_linear_int8_640/best_int8.mlmodel",
        input_shape=[1, 640, 640, 3],
        output_shape=[1, 14, 8400],
        validation_report=DEFAULT_OUT_DIR / "coreml_linear_int8_640/coreml_certification.json",
        note="CoreML linear weight quantization candidate.",
    ),
    CandidateSpec(
        id="coreml_linear_4bit_640",
        baseline_id="coreml_float32_640",
        platform="ios",
        precision="linear_4bit_weights",
        source_path=DEFAULT_OUT_DIR / "coreml_linear_4bit_640/best_linear4.mlmodel",
        target_path=DEFAULT_OUT_DIR / "coreml_linear_4bit_640/best_linear4.mlmodel",
        input_shape=[1, 640, 640, 3],
        output_shape=[1, 14, 8400],
        validation_report=DEFAULT_OUT_DIR / "coreml_linear_4bit_640/coreml_certification.json",
        note="CoreML 4-bit linear weight quantization candidate; aggressive quality check required.",
    ),
]


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


def inspect_baseline(spec: BaselineSpec) -> dict[str, Any]:
    artifact = inspect_file(spec.path)
    return {
        "id": spec.id,
        "platform": spec.platform,
        "input_shape": spec.input_shape,
        "output_shape": spec.output_shape,
        **artifact,
    }


def inspect_candidate(spec: CandidateSpec, baselines: dict[str, dict[str, Any]]) -> dict[str, Any]:
    artifact = inspect_file(spec.source_path)
    baseline = baselines.get(spec.baseline_id, {})
    baseline_size = int(baseline.get("size_bytes") or 0)
    failures: list[str] = []
    status = "ready" if artifact["exists"] and artifact["size_bytes"] > 0 else "missing"

    if spec.baseline_id not in baselines:
        failures.append(f"unknown_baseline:{spec.baseline_id}")
    if status == "missing" and spec.required:
        failures.append(f"missing_candidate:{spec.id}")

    candidate_size = int(artifact["size_bytes"] or 0)
    ratio = round(baseline_size / candidate_size, 2) if baseline_size > 0 and candidate_size > 0 else None
    validation = inspect_file(spec.validation_report) if spec.validation_report else None

    return {
        "id": spec.id,
        "platform": spec.platform,
        "baseline_id": spec.baseline_id,
        "precision": spec.precision,
        "status": status,
        **artifact,
        "source_path": _path(spec.source_path),
        "path": _path(spec.target_path),
        "input_shape": spec.input_shape,
        "output_shape": spec.output_shape,
        "compression_ratio_vs_baseline": ratio,
        "required": spec.required,
        "validation_report": validation,
        "note": spec.note,
        "failures": failures,
    }


def build_report(
    *,
    baselines: list[BaselineSpec] | None = None,
    candidates: list[CandidateSpec] | None = None,
) -> dict[str, Any]:
    baseline_specs = baselines if baselines is not None else DEFAULT_BASELINES
    candidate_specs = candidates if candidates is not None else DEFAULT_CANDIDATES
    baseline_rows = [inspect_baseline(spec) for spec in baseline_specs]
    baseline_by_id = {row["id"]: row for row in baseline_rows}

    failures: list[str] = []
    for row in baseline_rows:
        if not row["exists"] or row["size_bytes"] <= 0:
            failures.append(f"missing_baseline:{row['id']}")

    candidate_rows = [inspect_candidate(spec, baseline_by_id) for spec in candidate_specs]
    for row in candidate_rows:
        failures.extend(row["failures"])

    ready_candidates = [row for row in candidate_rows if row["status"] == "ready"]
    return {
        "schema_version": 1,
        "ok": not failures,
        "scope": "mobile_optimization_candidates_not_production_promotion",
        "baseline_note": (
            "Float32 artifacts remain the certified integration baseline. "
            "Optimized candidates need app-device latency and quality validation before promotion."
        ),
        "failures": failures,
        "ready_candidate_count": len(ready_candidates),
        "baselines": baseline_rows,
        "candidates": candidate_rows,
    }


def materialize_candidates(
    *,
    baselines: list[BaselineSpec] | None = None,
    candidates: list[CandidateSpec] | None = None,
    manifest_out: Path = DEFAULT_MANIFEST,
    markdown_out: Path = DEFAULT_MARKDOWN,
    zip_out: Path = DEFAULT_ZIP,
    copy_artifacts: bool = True,
) -> dict[str, Any]:
    candidate_specs = candidates if candidates is not None else DEFAULT_CANDIDATES
    if copy_artifacts:
        for spec in candidate_specs:
            if not spec.source_path.is_file():
                continue
            spec.target_path.parent.mkdir(parents=True, exist_ok=True)
            if spec.source_path.resolve() != spec.target_path.resolve():
                shutil.copy2(spec.source_path, spec.target_path)

    report = build_report(baselines=baselines, candidates=candidate_specs)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text(render_markdown(report), encoding="utf-8")
    write_zip(handoff_paths(report, manifest_out, markdown_out), zip_out)
    report["zip"] = _path(zip_out)
    report["zip_size_bytes"] = zip_out.stat().st_size
    report["zip_sha256"] = sha256_file(zip_out)
    manifest_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def handoff_paths(report: dict[str, Any], manifest_out: Path, markdown_out: Path) -> list[Path]:
    paths: list[Path] = []
    for row in report.get("candidates", []):
        if row.get("status") != "ready":
            continue
        candidate_path = Path(str(row.get("path", "")))
        if candidate_path.is_file():
            paths.append(candidate_path)
        validation = row.get("validation_report")
        if isinstance(validation, dict) and validation.get("exists") and validation.get("path"):
            paths.append(Path(str(validation["path"])))
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


def _shape(shape: list[int]) -> str:
    return "[" + ", ".join(str(v) for v in shape) + "]"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Mobile Optimization Report",
        "",
        f"- OK: {report.get('ok')}",
        f"- Scope: {report.get('scope')}",
        f"- Ready candidates: {report.get('ready_candidate_count')}",
        f"- Failures: {', '.join(report.get('failures', [])) if report.get('failures') else 'none'}",
        "",
        "## Baselines",
        "",
        "| ID | Platform | Size MB | Input | Output | Path |",
        "|---|---|---:|---|---|---|",
    ]
    for row in report.get("baselines", []):
        lines.append(
            "| "
            f"{row.get('id')} | {row.get('platform')} | {float(row.get('size_mb', 0.0)):.3f} | "
            f"`{_shape(row.get('input_shape', []))}` | `{_shape(row.get('output_shape', []))}` | "
            f"`{row.get('path')}` |"
        )

    lines.extend(
        [
            "",
            "## Candidates",
            "",
            "| ID | Platform | Precision | Status | Size MB | Ratio | Input | Output | Path |",
            "|---|---|---|---|---:|---:|---|---|---|",
        ]
    )
    for row in report.get("candidates", []):
        ratio = row.get("compression_ratio_vs_baseline")
        ratio_text = f"{float(ratio):.2f}x" if ratio is not None else ""
        lines.append(
            "| "
            f"{row.get('id')} | {row.get('platform')} | {row.get('precision')} | "
            f"{row.get('status')} | {float(row.get('size_mb', 0.0)):.3f} | {ratio_text} | "
            f"`{_shape(row.get('input_shape', []))}` | `{_shape(row.get('output_shape', []))}` | "
            f"`{row.get('path')}` |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Use the float32 artifacts as the current integration baseline.",
            "Use optimized candidates only after device latency and quality validation.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--zip-out", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--no-copy", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = materialize_candidates(
        manifest_out=args.manifest_out,
        markdown_out=args.markdown_out,
        zip_out=args.zip_out,
        copy_artifacts=not args.no_copy,
    )
    print(
        f"ok={report['ok']} ready_candidates={report['ready_candidate_count']} "
        f"failures={report['failures']}"
    )
    print(f"manifest={args.manifest_out}")
    print(f"markdown={args.markdown_out}")
    print(f"zip={args.zip_out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
