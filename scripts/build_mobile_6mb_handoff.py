"""Build the <=6 MB mobile handoff bundle.

This bundle is intentionally separate from the float32 integration baseline.
It contains mobile-small candidates that fit Igor's current constraints:
model file <=6 MB and input resolution <=390x390, implemented as 384x384
because YOLO expects stride-32 image sizes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


MAX_MODEL_SIZE_MB = 6.0
DEFAULT_ROOT = Path("outputs/production_audit/mobile_6mb")
DEFAULT_MANIFEST = DEFAULT_ROOT / "mobile_6mb_handoff_manifest.json"
DEFAULT_ZIP = DEFAULT_ROOT / "mobile_6mb_handoff.zip"
DEFAULT_ANDROID_ZIP = DEFAULT_ROOT / "android_6mb_handoff.zip"
DEFAULT_IOS_ZIP = DEFAULT_ROOT / "ios_6mb_handoff.zip"
DEFAULT_MARKDOWN = Path("docs/MOBILE_6MB_HANDOFF.md")
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class ArtifactSpec:
    path: Path
    platform: str
    role: str
    max_size_mb: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


DEFAULT_ARTIFACTS = [
    ArtifactSpec(
        path=DEFAULT_ROOT / "tflite_nano_fp16_384/best_float16.tflite",
        platform="android",
        role="model",
        max_size_mb=MAX_MODEL_SIZE_MB,
        metadata={
            "format": "tflite",
            "precision": "fp16",
            "input_shape": [1, 384, 384, 3],
            "input_dtype": "float32",
            "output_shape": [1, 14, 3024],
            "output_dtype": "float32",
            "source_checkpoint": "runs/pose/runs/pose/wheel_real_v1_soft_n_aug/weights/best.pt",
        },
    ),
    ArtifactSpec(
        path=DEFAULT_ROOT / "litert_smoke_tflite_nano_fp16_384.json",
        platform="android",
        role="validation",
    ),
    ArtifactSpec(
        path=DEFAULT_ROOT / "nano_source_eval_self_plus_ue_conf025.json",
        platform="shared",
        role="quality_reference",
        metadata={
            "model": "runs/pose/runs/pose/wheel_real_v1_soft_n_aug/weights/best.pt",
            "data": "configs/pose_dataset_real_v1_self_plus_ue_synthetic.yaml",
            "scope": "source_checkpoint_eval_not_exact_export_runtime_parity",
            "conf": 0.25,
            "images": 58,
            "gt_wheels": 84,
            "matched": 55,
            "mAP50": 0.671,
            "mAP50_95": 0.570,
            "oks_mean": 0.866,
            "false_negative_rate": 0.345,
            "false_positive_rate": 0.154,
        },
    ),
    ArtifactSpec(
        path=DEFAULT_ROOT / "coreml_nano_int8_384/best_int8.mlmodel",
        platform="ios",
        role="model",
        max_size_mb=MAX_MODEL_SIZE_MB,
        metadata={
            "format": "coreml_mlmodel",
            "precision": "int8_linear_weights",
            "input_name": "image",
            "input_size": [384, 384],
            "output_name": "var_1344",
            "logical_output_shape": [1, 14, 3024],
            "source_checkpoint": "runs/pose/runs/pose/wheel_real_v1_soft_n_aug/weights/best.pt",
        },
    ),
    ArtifactSpec(
        path=DEFAULT_ROOT / "coreml_nano_int8_384/coreml_certification.json",
        platform="ios",
        role="validation",
    ),
    ArtifactSpec(
        path=DEFAULT_ROOT / "coreml_nano_linear4_384/best_linear4.mlmodel",
        platform="ios",
        role="model",
        max_size_mb=MAX_MODEL_SIZE_MB,
        metadata={
            "format": "coreml_mlmodel",
            "precision": "linear_4bit_weights",
            "input_name": "image",
            "input_size": [384, 384],
            "output_name": "var_1344",
            "logical_output_shape": [1, 14, 3024],
            "source_checkpoint": "runs/pose/runs/pose/wheel_real_v1_soft_n_aug/weights/best.pt",
        },
    ),
    ArtifactSpec(
        path=DEFAULT_ROOT / "coreml_nano_linear4_384/coreml_certification.json",
        platform="ios",
        role="validation",
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


def inspect_artifact(spec: ArtifactSpec) -> dict[str, Any]:
    exists = spec.path.is_file()
    size_bytes = spec.path.stat().st_size if exists else 0
    max_size_bytes = (
        int(spec.max_size_mb * 1024 * 1024) if spec.max_size_mb is not None else None
    )
    within_limit = (
        None if max_size_bytes is None else bool(exists and size_bytes <= max_size_bytes)
    )
    return {
        "path": _path(spec.path),
        "name": spec.path.name,
        "platform": spec.platform,
        "role": spec.role,
        "exists": exists,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 3),
        "max_size_mb": spec.max_size_mb,
        "within_size_limit": within_limit,
        "sha256": sha256_file(spec.path) if exists and size_bytes > 0 else None,
        "metadata": spec.metadata,
    }


def build_manifest(artifacts: list[ArtifactSpec] | None = None) -> dict[str, Any]:
    specs = artifacts if artifacts is not None else DEFAULT_ARTIFACTS
    rows = [inspect_artifact(spec) for spec in specs]
    failures: list[str] = []
    for row in rows:
        if not row["exists"] or row["size_bytes"] <= 0:
            failures.append(f"missing:{row['path']}")
        if row["within_size_limit"] is False:
            failures.append(f"over_size:{row['name']}")
    return {
        "schema_version": 1,
        "ok": not failures,
        "scope": "mobile_6mb_handoff_candidates_not_production_promotion",
        "policy": {
            "max_model_size_mb": MAX_MODEL_SIZE_MB,
            "max_input_size": [390, 390],
            "actual_input_size": [384, 384],
            "note": (
                "These are mobile-small candidates. They satisfy file-size and "
                "input-size constraints but still need quality and target-device validation."
            ),
        },
        "failures": failures,
        "artifacts": rows,
    }


def render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Mobile 6MB Handoff",
        "",
        f"- OK: {manifest.get('ok')}",
        f"- Scope: {manifest.get('scope')}",
        f"- Max model size MB: {manifest.get('policy', {}).get('max_model_size_mb')}",
        f"- Actual input size: `{manifest.get('policy', {}).get('actual_input_size')}`",
        f"- Failures: {', '.join(manifest.get('failures', [])) if manifest.get('failures') else 'none'}",
        "",
        "## Artifacts",
        "",
        "| Platform | Role | File | Size MB | Limit MB | Path |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in manifest.get("artifacts", []):
        limit = row.get("max_size_mb")
        limit_text = "" if limit is None else f"{float(limit):.1f}"
        lines.append(
            "| "
            f"{row.get('platform')} | {row.get('role')} | {row.get('name')} | "
            f"{float(row.get('size_mb', 0.0)):.3f} | {limit_text} | "
            f"`{row.get('path')}` |"
        )
    lines.extend(
        [
            "",
            "## Model Interfaces",
            "",
            "- Android TFLite: float32 input `[1, 384, 384, 3]`, float32 output `[1, 14, 3024]`.",
            "- iOS CoreML: image input `image` 384x384, output `var_1344`, logical output `[1, 14, 3024]`.",
        ]
    )
    quality_rows = [
        row
        for row in manifest.get("artifacts", [])
        if row.get("role") == "quality_reference" and row.get("exists")
    ]
    if quality_rows:
        quality = quality_rows[0].get("metadata", {})
        lines.extend(
            [
                "",
                "## Quality Reference",
                "",
                f"- Scope: {quality.get('scope')}",
                f"- Model: `{quality.get('model')}`",
                f"- Data: `{quality.get('data')}`",
                f"- Images / GT wheels / matched: {quality.get('images')} / {quality.get('gt_wheels')} / {quality.get('matched')}",
                f"- mAP50 / mAP50-95: {quality.get('mAP50')} / {quality.get('mAP50_95')}",
                f"- OKS mean: {quality.get('oks_mean')}",
                f"- FN rate / FP rate: {quality.get('false_negative_rate')} / {quality.get('false_positive_rate')}",
            ]
        )
    platform_zips = manifest.get("platform_zips", {})
    if platform_zips:
        lines.extend(
            [
                "",
                "## Platform Zips",
                "",
                "| Platform | Size MB | Path |",
                "|---|---:|---|",
            ]
        )
        for platform in ("android", "ios"):
            item = platform_zips.get(platform, {})
            if not item:
                continue
            size_mb = float(item.get("size_bytes", 0)) / (1024 * 1024)
            lines.append(f"| {platform} | {size_mb:.3f} | `{item.get('path')}` |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Android artifact uses the nano checkpoint, FP16 TFLite, and 384x384 input.",
            "- iOS includes int8 and 4-bit CoreML candidates; int8 is the safer first test candidate.",
            "- These candidates fit the mobile size constraint but are not production-promoted.",
            "- Promote only after app-device latency plus quality validation.",
            "",
        ]
    )
    return "\n".join(lines)


def handoff_paths(manifest: dict[str, Any], manifest_out: Path, markdown_out: Path) -> list[Path]:
    paths = [Path(row["path"]) for row in manifest.get("artifacts", []) if row.get("exists")]
    paths.extend([manifest_out, markdown_out])
    return list(dict.fromkeys(paths))


def platform_handoff_paths(
    manifest: dict[str, Any],
    platform: str,
    manifest_out: Path,
    markdown_out: Path,
) -> list[Path]:
    paths = [
        Path(row["path"])
        for row in manifest.get("artifacts", [])
        if row.get("exists") and row.get("platform") in {platform, "shared"}
    ]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--zip-out", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--android-zip-out", type=Path, default=DEFAULT_ANDROID_ZIP)
    parser.add_argument("--ios-zip-out", type=Path, default=DEFAULT_IOS_ZIP)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest()
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.markdown_out.write_text(render_markdown(manifest), encoding="utf-8")
    if manifest["ok"]:
        write_zip(handoff_paths(manifest, args.manifest_out, args.markdown_out), args.zip_out)
        write_zip(
            platform_handoff_paths(manifest, "android", args.manifest_out, args.markdown_out),
            args.android_zip_out,
        )
        write_zip(
            platform_handoff_paths(manifest, "ios", args.manifest_out, args.markdown_out),
            args.ios_zip_out,
        )
        manifest["zip"] = _path(args.zip_out)
        manifest["zip_size_bytes"] = args.zip_out.stat().st_size
        manifest["zip_sha256"] = sha256_file(args.zip_out)
        manifest["platform_zips"] = {
            "android": {
                "path": _path(args.android_zip_out),
                "size_bytes": args.android_zip_out.stat().st_size,
                "sha256": sha256_file(args.android_zip_out),
            },
            "ios": {
                "path": _path(args.ios_zip_out),
                "size_bytes": args.ios_zip_out.stat().st_size,
                "sha256": sha256_file(args.ios_zip_out),
            },
        }
        args.manifest_out.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    args.markdown_out.write_text(render_markdown(manifest), encoding="utf-8")
    print(f"ok={manifest['ok']} failures={manifest['failures']}")
    print(f"manifest={args.manifest_out}")
    print(f"markdown={args.markdown_out}")
    if manifest.get("zip"):
        print(f"zip={args.zip_out}")
        print(f"android_zip={args.android_zip_out}")
        print(f"ios_zip={args.ios_zip_out}")
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
