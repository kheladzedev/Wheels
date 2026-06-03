"""Build a deterministic iOS CoreML handoff zip."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any


DEFAULT_MODEL = Path("outputs/production_audit/coreml_export/best.mlmodel")
DEFAULT_PACKAGE_ROOT = Path("outputs/production_audit/ios_coreml_handoff")
DEFAULT_ZIP = Path("outputs/production_audit/ios_coreml_handoff.zip")
DEFAULT_MANIFEST = Path("outputs/production_audit/ios_coreml_handoff_manifest.json")
SOURCE_FILES = [
    Path("ios_coreml_handoff/README.md"),
    Path("ios_coreml_handoff/WheelsCoreMLSmoke.swift"),
    Path("docs/COREML_CERTIFICATION.md"),
    Path("docs/AR_ML_CONTRACT.md"),
]
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_source_files(package_root: Path, model_path: Path) -> None:
    package_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, package_root / "best.mlmodel")
    for source in SOURCE_FILES:
        target = package_root / source.name
        shutil.copy2(source, target)


def compile_coreml_model(model_path: Path, package_root: Path) -> Path:
    compiled = package_root / "best.mlmodelc"
    if compiled.exists():
        shutil.rmtree(compiled)
    package_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(model_path), str(package_root)],
        check=True,
    )
    if not compiled.is_dir():
        raise FileNotFoundError(compiled)
    return compiled


def package_files(package_root: Path) -> list[Path]:
    return sorted(path for path in package_root.rglob("*") if path.is_file())


def inspect_file(package_root: Path, path: Path) -> dict[str, Any]:
    relative = path.relative_to(package_root).as_posix()
    return {
        "path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_manifest(package_root: Path, zip_path: Path) -> dict[str, Any]:
    files = package_files(package_root)
    names = {path.relative_to(package_root).as_posix() for path in files}
    required = {
        "README.md",
        "WheelsCoreMLSmoke.swift",
        "COREML_CERTIFICATION.md",
        "AR_ML_CONTRACT.md",
        "best.mlmodel",
        "best.mlmodelc/model.espresso.net",
        "best.mlmodelc/model.espresso.weights",
    }
    failures = [f"missing:{name}" for name in sorted(required - names)]
    return {
        "schema_version": 1,
        "ok": not failures,
        "handoff": str(package_root).replace("\\", "/"),
        "zip": str(zip_path).replace("\\", "/"),
        "model": {
            "input_name": "image",
            "input_shape": [1, 640, 640, 3],
            "extra_inputs": [],
            "output_name": "var_1347",
            "output_shape": [1, 14, 8400],
            "confidence_threshold": 0.80,
            "nms_iou": 0.45,
        },
        "failures": failures,
        "file_count": len(files),
        "files": [inspect_file(package_root, path) for path in files],
    }


def write_zip(package_root: Path, zip_path: Path) -> None:
    files = package_files(package_root)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            arcname = path.relative_to(package_root).as_posix()
            info = zipfile.ZipInfo(arcname, date_time=ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument("--zip-out", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--skip-compile", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True)
    if not args.skip_compile:
        compile_coreml_model(args.model, args.out_dir)
    copy_source_files(args.out_dir, args.model)
    manifest = build_manifest(args.out_dir, args.zip_out)
    if not manifest["ok"]:
        args.manifest_out.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"ok=False failures={manifest['failures']}")
        print(f"manifest={args.manifest_out}")
        return 1
    write_zip(args.out_dir, args.zip_out)
    manifest["zip_sha256"] = sha256_file(args.zip_out)
    manifest["zip_size_bytes"] = args.zip_out.stat().st_size
    args.manifest_out.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"ok=True zip={args.zip_out} files={manifest['file_count']} "
        f"sha256={manifest['zip_sha256']}"
    )
    print(f"manifest={args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

