"""Create the zip shape Android/AR should return with real evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


DEFAULT_OUT = Path("outputs/production_audit/external_evidence_return_template.zip")
DEFAULT_MANIFEST_OUT = Path("outputs/production_audit/external_evidence_return_template_manifest.json")
DEFAULT_EXPECTED_ANDROID_ARTIFACT = Path("outputs/production_audit/tflite_export/best_float32.tflite")
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

PLACEHOLDER_TEMPLATE_FILES = {
    "android_litert_device_report.json.PLACEHOLDER": (
        "Replace this file with android_litert_device_report.json produced by "
        "android_litert_harness/AndroidLiteRtDeviceValidationTest.kt.\n"
    ),
    "ar_device_holdout/images/PLACE_FRAMES_HERE.txt": (
        "Place real AR-device holdout images here. Image stems must match annotations.\n"
    ),
    "ar_device_holdout/annotations/PLACE_ANNOTATIONS_HERE.txt": (
        "Place human-reviewed annotation JSON files here. Stems must match images.\n"
    ),
    "ar_device_holdout/metadata/provenance.json.PLACEHOLDER": (
        "Replace this file with metadata/provenance.json produced by "
        "ar_holdout_harness/ArHoldoutAnnotationWriter.kt.\n"
    ),
    "ar_3d_replay/ar_replay.jsonl.PLACEHOLDER": (
        "Replace this file with ar_replay.jsonl produced by ar_replay_harness/ArReplayLogger.kt.\n"
    ),
}


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_artifact_payload(artifact: Path) -> dict[str, Any]:
    return {
        "path": str(artifact).replace("\\", "/"),
        "sha256": sha256_file(artifact),
        "format": "tflite_float32",
    }


def render_readme(expected_artifact: dict[str, Any]) -> str:
    artifact_sha = expected_artifact.get("sha256") or "MISSING_ARTIFACT_SHA"
    artifact_path = expected_artifact.get("path") or "outputs/production_audit/tflite_export/best_float32.tflite"
    return f"""# External Evidence Return Template

Replace the placeholder files in this zip with real Android/AR evidence,
then send the zip back to ML.

Required files:

- android_litert_device_report.json
- ar_device_holdout/images/<frame_id>.jpg
- ar_device_holdout/annotations/<frame_id>.json
- ar_device_holdout/metadata/provenance.json
- ar_3d_replay/ar_replay.jsonl

Minimum production contents:

- at least 50 AR holdout images with matching annotation JSON files
- at least 80 total annotated wheel instances
- each AR holdout annotation JSON must include `schema_version=1`
- provenance.json with human-reviewed AR-device source metadata and
  `schema_version=1`
- at least 30 AR replay observations
- every replay observation must include camera_transform or camera_pose_ref
- Android report must include source_type, test_session_id, SoC,
  TFLite artifact format/hash for the current ML-provided
  `{artifact_path}`, output stats, latency, and memory

Expected Android TFLite artifact:

- path: `{artifact_path}`
- sha256: `{artifact_sha}`
- format: `tflite_float32`

The same values are also stored in `EXPECTED_ANDROID_ARTIFACT.json`.
Android LiteRT evidence must report this exact SHA-256 or the ML-side
importer will reject the drop before copying it into `data/incoming`.

Before sending, the ML side can verify the shape with:

```bash
./.venv/bin/python src/import_external_evidence_drop.py path/to/evidence_drop.zip --dry-run
```
"""


def build_template_files(expected_artifact_path: Path = DEFAULT_EXPECTED_ANDROID_ARTIFACT) -> dict[str, str]:
    expected_artifact = expected_artifact_payload(expected_artifact_path)
    return {
        "EXPECTED_ANDROID_ARTIFACT.json": json.dumps(
            {
                "schema_version": 1,
                "expected_android_artifact": expected_artifact,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        "README_RETURN_EVIDENCE.md": render_readme(expected_artifact),
        **PLACEHOLDER_TEMPLATE_FILES,
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_template_zip(
    out: Path,
    files: dict[str, str] | None = None,
    *,
    expected_android_artifact: Path = DEFAULT_EXPECTED_ANDROID_ARTIFACT,
) -> list[dict[str, Any]]:
    out.parent.mkdir(parents=True, exist_ok=True)
    files = files if files is not None else build_template_files(expected_android_artifact)
    artifacts: list[dict[str, Any]] = []
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(files):
            data = files[name].encode("utf-8")
            info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, data)
            artifacts.append(
                {
                    "path": name,
                    "size_bytes": len(data),
                    "sha256": sha256_bytes(data),
                }
            )
    return artifacts


def build_manifest(
    out: Path,
    artifacts: list[dict[str, Any]],
    *,
    expected_android_artifact: Path = DEFAULT_EXPECTED_ANDROID_ARTIFACT,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ok": True,
        "template": str(out).replace("\\", "/"),
        "artifact_count": len(artifacts),
        "zip_sha256": sha256_file(out),
        "zip_size_bytes": out.stat().st_size,
        "expected_android_artifact": expected_artifact_payload(expected_android_artifact),
        "artifacts": artifacts,
        "next_command": "./.venv/bin/python src/import_external_evidence_drop.py path/to/evidence_drop.zip --dry-run",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST_OUT)
    parser.add_argument("--expected-android-artifact", type=Path, default=DEFAULT_EXPECTED_ANDROID_ARTIFACT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts = write_template_zip(args.out, expected_android_artifact=args.expected_android_artifact)
    manifest = build_manifest(args.out, artifacts, expected_android_artifact=args.expected_android_artifact)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"ok=True template={args.out} artifacts={manifest['artifact_count']} sha256={manifest['zip_sha256']}")
    print(f"manifest={args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
