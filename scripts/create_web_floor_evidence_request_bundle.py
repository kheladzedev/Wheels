#!/usr/bin/env python3
"""Create a return-template zip for real web-floor training evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from web_floor_annotation_import import REQUIRED_COLUMNS


DEFAULT_OUT = Path("outputs/web_floor_network/web_floor_real_data_request_bundle.zip")
DEFAULT_MANIFEST_OUT = Path("outputs/web_floor_network/web_floor_real_data_request_bundle_manifest.json")
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
OPTIONAL_COLUMNS = ("provenance_capture_date", "fov_mode")
REPO_DOCS = (
    "docs/WEB_FLOOR_REAL_DATA_INTAKE.md",
    "docs/WEB_FLOOR_NETWORK_CONTRACT.md",
    "docs/WEB_FLOOR_NETWORK_STATUS.md",
    "docs/WEB_FLOOR_NETWORK_HANDOFF.md",
    "configs/pose_dataset_web_floor_real_template.yaml",
)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def csv_header() -> str:
    return ",".join([*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS]) + "\n"


def csv_example() -> str:
    columns = [*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS]
    row = {
        "frame_id": "phone-floor-0001",
        "split": "train",
        "image": "frame_0001.jpg",
        "provenance_source": "phone_capture",
        "provenance_device": "iPhone 15 Pro",
        "provenance_annotator": "igor",
        "pitch": "-0.04",
        "roll": "0.01",
        "distance": "1.2",
        "distance_mode": "metric_anchor",
        "bbox_x1": "120",
        "bbox_y1": "340",
        "bbox_x2": "260",
        "bbox_y2": "520",
        "confidence": "1.0",
        "a_x": "142",
        "a_y": "505",
        "b_x": "238",
        "b_y": "506",
        "c_disc_bottom_x": "190",
        "c_disc_bottom_y": "452",
        "provenance_capture_date": "2026-06-12",
        "fov_mode": "provided",
    }
    return ",".join(columns) + "\n" + ",".join(row[column] for column in columns) + "\n"


def render_readme() -> str:
    return """# Web Floor Real Data Request

Return this bundle with real web/phone images plus wheel/floor annotations.

Required payload:

- images/frame_*.jpg or images/frame_*.png
- annotations/web_floor_annotations.csv

CSV rule: one row is one labelled wheel. Repeat frame-level fields for multiple
wheels in the same frame.

Minimum gate:

- at least 50 frames
- at least 80 labelled wheels
- train and holdout split values
- non-empty provenance fields for every frame
- floor pitch, roll, distance, and distance_mode
- distance_mode must not be unknown
- at least 0.5 distance span
- at least 0.05 rad pitch or roll span

ML-side import:

```bash
./.venv/bin/python scripts/import_web_floor_annotations.py \\
  --csv data/incoming/web_floor_real_v1/annotations/web_floor_annotations.csv \\
  --image-root data/incoming/web_floor_real_v1/images \\
  --dataset-root data/web_floor_real_v1 \\
  --config-out configs/pose_dataset_web_floor_real_v1.yaml \\
  --overwrite
```

ML-side gate:

```bash
./.venv/bin/python scripts/audit_web_floor_real_data.py \\
  --config configs/pose_dataset_web_floor_real_v1.yaml \\
  --output-json outputs/web_floor_network/real_data_gate.json \\
  --fail-on-not-ready
```

The web runtime target stays direct/lightweight: one RGB frame, one ONNX
forward, direct `[pitch, roll, distance]`; no runtime depth, segmentation,
RANSAC, multi-frame state, or heavy backend geometry postprocess.
"""


def build_template_files(
    *,
    repo_root: Path = Path("."),
    include_repo_docs: bool = True,
) -> dict[str, str]:
    files = {
        "README_WEB_FLOOR_EVIDENCE.md": render_readme(),
        "annotations/web_floor_annotations.csv": csv_header(),
        "annotations/web_floor_annotations_example.csv": csv_example(),
        "images/PLACE_REAL_FRAMES_HERE.txt": (
            "Place real web/phone frames here. Filenames must match the CSV image column.\n"
        ),
    }
    if include_repo_docs:
        for rel in REPO_DOCS:
            path = repo_root / rel
            if path.is_file():
                files[rel] = path.read_text(encoding="utf-8")
    return files


def write_template_zip(out: Path, files: dict[str, str]) -> list[dict[str, Any]]:
    out.parent.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            data = files[name].encode("utf-8")
            info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
            artifacts.append(
                {
                    "path": name,
                    "size_bytes": len(data),
                    "sha256": sha256_bytes(data),
                }
            )
    return artifacts


def build_manifest(out: Path, artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "schema": "web_floor_real_data_request_bundle_v1",
        "ok": True,
        "template": str(out).replace("\\", "/"),
        "artifact_count": len(artifacts),
        "zip_sha256": sha256_file(out),
        "zip_size_bytes": out.stat().st_size,
        "bundle_sha256": sha256_file(out),
        "bundle_size_bytes": out.stat().st_size,
        "artifacts": artifacts,
        "next_command": "./.venv/bin/python scripts/import_web_floor_annotations.py --csv data/incoming/web_floor_real_v1/annotations/web_floor_annotations.csv --image-root data/incoming/web_floor_real_v1/images --dataset-root data/web_floor_real_v1 --config-out configs/pose_dataset_web_floor_real_v1.yaml --overwrite",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST_OUT)
    parser.add_argument("--no-repo-docs", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    files = build_template_files(include_repo_docs=not args.no_repo_docs)
    artifacts = write_template_zip(args.out, files)
    manifest = build_manifest(args.out, artifacts)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"ok=True template={args.out} artifacts={manifest['artifact_count']} sha256={manifest['zip_sha256']}")
    print(f"manifest={args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
