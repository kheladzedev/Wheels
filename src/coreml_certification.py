"""Certify the iOS CoreML handoff artifact.

The current Python 3.14/coremltools environment cannot write ML Program
`.mlpackage` artifacts because the native BlobWriter extension is not
available. The iOS handoff therefore uses the legacy CoreML neuralnetwork
`.mlmodel` export (`format=mlmodel`), which still loads as a CoreML spec and
is suitable for app-side test integration.

This is package certification, not physical-device certification. iOS runtime
latency/memory must still be measured inside the app or an XCTest harness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_COREML = Path("outputs/production_audit/coreml_export/best.mlmodel")
DEFAULT_PT = Path("runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt")
DEFAULT_JSON_OUT = Path("outputs/production_audit/coreml_certification.json")
DEFAULT_MD_OUT = Path("docs/COREML_CERTIFICATION.md")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_file(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    size_bytes = path.stat().st_size if exists else 0
    return {
        "path": str(path).replace("\\", "/"),
        "exists": exists,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 3),
        "sha256": sha256_file(path) if exists and size_bytes > 0 else None,
    }


def _feature_kind(feature: Any) -> str | None:
    feature_type = getattr(feature, "type", None)
    if feature_type is None:
        return None
    try:
        return feature_type.WhichOneof("Type")
    except ValueError:
        return None


def load_coreml_spec_summary(path: Path) -> dict[str, Any]:
    import coremltools as ct  # noqa: PLC0415

    spec = ct.utils.load_spec(str(path))
    inputs: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []

    for feature in spec.description.input:
        item: dict[str, Any] = {"name": feature.name, "kind": _feature_kind(feature)}
        if item["kind"] == "imageType":
            image = feature.type.imageType
            item.update({"kind": "image", "width": int(image.width), "height": int(image.height)})
        elif item["kind"] == "multiArrayType":
            item.update({"shape": [int(v) for v in feature.type.multiArrayType.shape]})
        inputs.append(item)

    for feature in spec.description.output:
        item = {"name": feature.name, "kind": _feature_kind(feature)}
        if item["kind"] == "multiArrayType":
            item.update({"shape": [int(v) for v in feature.type.multiArrayType.shape]})
        outputs.append(item)

    return {
        "specification_version": int(spec.specificationVersion),
        "inputs": inputs,
        "outputs": outputs,
    }


def build_certification(
    *,
    coreml_artifact: Path = DEFAULT_COREML,
    pytorch_artifact: Path = DEFAULT_PT,
    expected_input_size: int = 640,
) -> dict[str, Any]:
    artifact = inspect_file(coreml_artifact)
    pytorch = inspect_file(pytorch_artifact)
    failures: list[str] = []
    spec_summary: dict[str, Any] = {}

    if not artifact["exists"] or artifact["size_bytes"] <= 0:
        failures.append("missing_coreml_artifact")
    if not pytorch["exists"] or pytorch["size_bytes"] <= 0:
        failures.append("missing_pytorch_reference")

    if artifact["exists"] and artifact["size_bytes"] > 0:
        try:
            spec_summary = load_coreml_spec_summary(coreml_artifact)
        except Exception as exc:  # pragma: no cover - exact coremltools errors vary by env
            failures.append(f"coreml_spec_load_failed:{type(exc).__name__}:{exc}")

    inputs = spec_summary.get("inputs", [])
    outputs = spec_summary.get("outputs", [])
    image_inputs = [item for item in inputs if item.get("kind") == "image"]
    if spec_summary:
        if len(image_inputs) != 1:
            failures.append(f"expected_one_image_input:{len(image_inputs)}")
        else:
            image = image_inputs[0]
            if image.get("width") != expected_input_size or image.get("height") != expected_input_size:
                failures.append(
                    f"input_size:{image.get('width')}x{image.get('height')}!={expected_input_size}"
                )
        if len(outputs) != 1:
            failures.append(f"expected_one_output:{len(outputs)}")
        elif outputs[0].get("kind") != "multiArrayType":
            failures.append(f"output_kind:{outputs[0].get('kind')}!=multiArrayType")

    certified = not failures
    return {
        "schema_version": 1,
        "certified": certified,
        "status": "certified" if certified else "failed",
        "scope": "desktop_coreml_package_not_ios_device",
        "format": "coreml_mlmodel_neuralnetwork",
        "artifact": artifact,
        "pytorch_reference": pytorch,
        "spec": spec_summary,
        "policy": {
            "expected_input_size": expected_input_size,
            "runtime_note": (
                "This certifies the CoreML package/spec on desktop. iOS app/device "
                "latency, memory, and end-to-end output validation remain external evidence."
            ),
            "mlpackage_note": (
                "ML Program .mlpackage export is blocked in the current Python 3.14 "
                "environment because coremltools cannot load BlobWriter. The .mlmodel "
                "neuralnetwork export is the iOS test handoff artifact."
            ),
        },
        "failures": failures,
    }


def render_markdown(report: dict[str, Any]) -> str:
    artifact = report.get("artifact", {})
    spec = report.get("spec", {})
    lines = [
        "# CoreML Certification",
        "",
        f"- Certified: {report.get('certified')}",
        f"- Scope: {report.get('scope')}",
        f"- Format: {report.get('format')}",
        f"- Artifact: `{artifact.get('path')}`",
        f"- Size MB: {artifact.get('size_mb')}",
        f"- SHA256: `{artifact.get('sha256')}`",
        f"- Failures: {', '.join(report.get('failures', [])) if report.get('failures') else 'none'}",
        "",
        "## CoreML Spec",
        "",
        f"- Specification version: {spec.get('specification_version')}",
        f"- Inputs: `{spec.get('inputs')}`",
        f"- Outputs: `{spec.get('outputs')}`",
        "",
        "This is desktop package certification, not iOS-device runtime certification.",
        "The iOS team should load the exact `.mlmodel` in the app/XCTest and report latency, memory, and output sanity.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coreml-artifact", type=Path, default=DEFAULT_COREML)
    parser.add_argument("--pytorch-artifact", type=Path, default=DEFAULT_PT)
    parser.add_argument("--expected-input-size", type=int, default=640)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_certification(
        coreml_artifact=args.coreml_artifact,
        pytorch_artifact=args.pytorch_artifact,
        expected_input_size=args.expected_input_size,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(report), encoding="utf-8")
    print(
        f"certified={report['certified']} scope={report['scope']} "
        f"artifact={report['artifact']['path']}"
    )
    print(f"json={args.json_out}")
    print(f"markdown={args.md_out}")
    return 0 if report["certified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
