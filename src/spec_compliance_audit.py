"""Machine-readable audit for AR spec / ML contract compliance.

The human-readable mapping lives in docs/SPEC_COMPLIANCE.md. This audit
turns the load-bearing pieces into executable evidence for production:
confirmed JSON shape, keypoint semantics, responsibility split, and
contract-test coverage.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_JSON_OUT = Path("outputs/production_audit/spec_compliance_audit.json")
DEFAULT_MD_OUT = Path("docs/SPEC_COMPLIANCE_AUDIT.md")
EXPECTED_KEYPOINT_NAMES = ("rim_left", "rim_right", "disc_bottom")
EXPECTED_CONFIRMED_POINTS = {"a", "b", "c_disc_bottom"}
EXPECTED_TOP_LEVEL = {"frame_id", "wheels"}
EXPECTED_WHEEL_KEYS = {"bbox_xyxy", "confidence", "points"}
FORBIDDEN_CONFIRMED_KEY_SUBSTRINGS = (
    "track_id",
    "track",
    "world",
    "plane",
    "ransac",
    "raycast",
    "intrinsic",
    "extrinsic",
    "imu",
    "depth",
    "z_world",
    "z_axis",
    "3d",
    "visibility",
    "keypoints_confidence",
    "point_confidence",
    "kp_confidence",
    "timestamp",
)


@dataclass
class ComplianceCheck:
    name: str
    ok: bool
    evidence: str
    detail: str


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _check_file_contains(name: str, path: Path, needles: list[str]) -> ComplianceCheck:
    content = _read_text(path)
    missing = [needle for needle in needles if needle not in content]
    return ComplianceCheck(
        name=name,
        ok=path.is_file() and not missing,
        evidence=str(path).replace("\\", "/"),
        detail=(
            "present and contains required spec anchors"
            if path.is_file() and not missing
            else f"exists={path.is_file()}, missing={missing}"
        ),
    )


def _collect_keys(payload: object, into: set[str]) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            into.add(str(key))
            _collect_keys(value, into)
    elif isinstance(payload, list):
        for item in payload:
            _collect_keys(item, into)


def _contract_payload() -> dict[str, Any]:
    from postprocess_wheels import build_ar_payload, to_confirmed_schema

    detections = [
        {
            "class_name": "wheel",
            # Valid floor-ray geometry: A/B sit on the lower band (rel_y>=0.80)
            # with >=0.50w horizontal separation, C is in the lower half and
            # above the A/B line (see postprocess_wheels.confirmed_geometry_issues).
            "bbox": [10, 20, 60, 80],
            "confidence": 0.93,
            "keypoints": [
                {"xy": [18, 76], "visibility": 2, "confidence": 0.91},
                {"xy": [52, 76], "visibility": 2, "confidence": 0.90},
                {"xy": [35, 70], "visibility": 2, "confidence": 0.88},
            ],
        },
        {
            "class_name": "wheel",
            "bbox": [110, 120, 160, 180],
            "confidence": 0.80,
            "keypoints": [
                {"xy": [115, 130], "visibility": 2, "confidence": 0.91},
                {"xy": [155, 175], "visibility": 2, "confidence": 0.90},
                {"xy": [135, 179], "visibility": 0, "confidence": 0.10},
            ],
        },
    ]
    legacy = build_ar_payload(
        detections, conf_threshold=0.25, frame_id="spec-audit-frame"
    )
    return to_confirmed_schema(legacy)


def build_audit() -> dict[str, Any]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from postprocess_wheels import INTERNAL_TO_CONFIRMED_KP, KEYPOINT_NAMES, N_KEYPOINTS

    confirmed = _contract_payload()
    wheels = confirmed.get("wheels", [])
    first_wheel = wheels[0] if wheels and isinstance(wheels[0], dict) else {}
    points = (
        first_wheel.get("points", {})
        if isinstance(first_wheel.get("points"), dict)
        else {}
    )
    all_keys: set[str] = set()
    _collect_keys(confirmed, all_keys)
    forbidden_hits = sorted(
        f"{key}:{needle}"
        for key in all_keys
        for needle in FORBIDDEN_CONFIRMED_KEY_SUBSTRINGS
        if needle in key.lower()
    )

    checks = [
        _check_file_contains(
            "spec_compliance_document",
            Path("docs/SPEC_COMPLIANCE.md"),
            [
                "ML deliverable per the spec",
                "No 3D world positions",
                "tests/test_confirmed_ar_schema_shape.py",
            ],
        ),
        _check_file_contains(
            "ar_ml_contract_document",
            Path("docs/AR_ML_CONTRACT.md"),
            [
                "ML returns **2D screen-space pixels only.**",
                "No 3D",
                "No `track_id` from ML",
                '"bbox_xyxy"',
                '"c_disc_bottom"',
            ],
        ),
        ComplianceCheck(
            "canonical_keypoint_names",
            tuple(KEYPOINT_NAMES) == EXPECTED_KEYPOINT_NAMES and int(N_KEYPOINTS) == 3,
            "src/postprocess_wheels.py",
            f"KEYPOINT_NAMES={tuple(KEYPOINT_NAMES)}, N_KEYPOINTS={N_KEYPOINTS}",
        ),
        ComplianceCheck(
            "confirmed_keypoint_mapping",
            dict(INTERNAL_TO_CONFIRMED_KP)
            == {
                "rim_left": "a",
                "rim_right": "b",
                "disc_bottom": "c_disc_bottom",
            },
            "src/postprocess_wheels.py",
            f"INTERNAL_TO_CONFIRMED_KP={dict(INTERNAL_TO_CONFIRMED_KP)}",
        ),
        ComplianceCheck(
            "confirmed_top_level_schema",
            set(confirmed.keys()) == EXPECTED_TOP_LEVEL,
            "src/postprocess_wheels.py::to_confirmed_schema",
            f"keys={sorted(confirmed.keys())}",
        ),
        ComplianceCheck(
            "confirmed_wheel_schema",
            set(first_wheel.keys()) == EXPECTED_WHEEL_KEYS,
            "src/postprocess_wheels.py::to_confirmed_schema",
            f"keys={sorted(first_wheel.keys())}",
        ),
        ComplianceCheck(
            "confirmed_points_schema",
            set(points.keys()) == EXPECTED_CONFIRMED_POINTS,
            "src/postprocess_wheels.py::to_confirmed_schema",
            f"points={sorted(points.keys())}",
        ),
        ComplianceCheck(
            "occluded_wheels_are_dropped",
            isinstance(wheels, list) and len(wheels) == 1,
            "src/postprocess_wheels.py::to_confirmed_schema",
            f"emitted_wheels={len(wheels) if isinstance(wheels, list) else 'n/a'}",
        ),
        ComplianceCheck(
            "confirmed_schema_has_no_forbidden_ml_fields",
            not forbidden_hits,
            "src/postprocess_wheels.py::to_confirmed_schema",
            f"forbidden_hits={forbidden_hits}",
        ),
        ComplianceCheck(
            "contract_tests_present",
            Path("tests/test_ar_contract.py").is_file()
            and Path("tests/test_confirmed_ar_schema_shape.py").is_file(),
            "tests/test_ar_contract.py; tests/test_confirmed_ar_schema_shape.py",
            "shape guards present",
        ),
        ComplianceCheck(
            "inference_wrappers_present",
            Path("src/infer_image.py").is_file()
            and Path("src/infer_batch.py").is_file(),
            "src/infer_image.py; src/infer_batch.py",
            "single-frame and batch AR payload entrypoints present",
        ),
    ]
    failures = [check.name for check in checks if not check.ok]
    return {
        "schema_version": 1,
        "ok": not failures,
        "failures": failures,
        "checks": [asdict(check) for check in checks],
        "confirmed_schema": confirmed,
        "policy": {
            "ml_scope": "per-frame, stateless 2D wheel detection with three keypoints",
            "ar_scope": "raycast, RANSAC, plane recovery, K-frame accumulation, tracking",
            "schema_change_requires_ar_signoff": True,
        },
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Spec Compliance Audit",
        "",
        "Executable audit for AR spec / ML contract compliance.",
        "",
        f"- OK: {audit.get('ok')}",
        f"- Failures: {', '.join(audit.get('failures', [])) if audit.get('failures') else 'none'}",
        f"- ML scope: {audit.get('policy', {}).get('ml_scope', 'n/a')}",
        f"- AR scope: {audit.get('policy', {}).get('ar_scope', 'n/a')}",
        "",
        "| Check | OK | Evidence | Detail |",
        "|---|---:|---|---|",
    ]
    for check in audit.get("checks", []):
        lines.append(
            "| "
            f"{check.get('name')} | "
            f"{check.get('ok')} | "
            f"`{check.get('evidence')}` | "
            f"{check.get('detail')} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = build_audit()
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(audit), encoding="utf-8")
    print(f"ok={audit['ok']} failures={audit['failures']}")
    print(f"json={args.json_out}")
    print(f"markdown={args.md_out}")
    return 0 if audit["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
