"""Geometry audit of real-image inference JSONs against the 2026-05-13
floor-ray contract semantics.

For each wheel in the confirmed-schema JSONs (bbox_xyxy + points.{a, b,
c_disc_bottom}), compute:
  - bbox width / height
  - rel_y_{a,b,c} = (point_y - y1) / (y2 - y1)
  - ab_sep_ratio = |b_x - a_x| / bbox_width
  - c_ab_order_ok: c_y strictly less than min(a_y, b_y)   (C above A/B)

Audit criteria (heuristics — not training labels):
  - A/B should sit in the LOWER region of the bbox (rel_y >= 0.80).
  - A/B should be horizontally separated (ab_sep_ratio >= 0.50).
  - C should be ABOVE both A and B in image coords (smaller y).

Outputs:
  outputs/real_infer_geometry_audit.json — per-frame, per-wheel result
  outputs/real_infer_geometry_audit.md   — human summary
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SINGLE_DIR = REPO / "outputs" / "test_infer_single"
BATCH_DIR = REPO / "outputs" / "test_batch_ar"
OUT_JSON = REPO / "outputs" / "real_infer_geometry_audit.json"
OUT_MD = REPO / "outputs" / "real_infer_geometry_audit.md"

REL_Y_AB_MIN = 0.80
AB_SEP_RATIO_MIN = 0.50

REQUIRED_WHEEL_KEYS = {"bbox_xyxy", "confidence", "points"}
REQUIRED_POINT_KEYS = {"a", "b", "c_disc_bottom"}
FORBIDDEN_TOP_LEVEL_KEYS = {"track_id"}


def _detect_schema(payload: dict[str, Any]) -> str:
    """Return one of: 'confirmed', 'legacy', 'target_draft', 'unknown'."""
    wheels = payload.get("wheels")
    if not isinstance(wheels, list) or not wheels:
        return "unknown"
    w = wheels[0]
    if "bbox_xyxy" in w and "points" in w:
        return "confirmed"
    if "wheel_bbox" in w and "keypoints" in w:
        return "legacy"
    if "bbox_xywh" in w and "keypoints" in w:
        return "target_draft"
    return "unknown"


def _check_confirmed_schema(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if "frame_id" not in payload or not isinstance(payload["frame_id"], str):
        issues.append("missing/invalid frame_id")
    if "wheels" not in payload or not isinstance(payload["wheels"], list):
        issues.append("missing/invalid wheels[]")
        return False, issues

    for forb in FORBIDDEN_TOP_LEVEL_KEYS:
        if forb in payload:
            issues.append(f"forbidden top-level key present: {forb}")

    for i, w in enumerate(payload["wheels"]):
        missing = REQUIRED_WHEEL_KEYS - set(w.keys())
        if missing:
            issues.append(f"wheel[{i}] missing keys: {sorted(missing)}")
        if "points" in w:
            kp_missing = REQUIRED_POINT_KEYS - set(w["points"].keys())
            if kp_missing:
                issues.append(f"wheel[{i}].points missing: {sorted(kp_missing)}")
            kp_extra = set(w["points"].keys()) - REQUIRED_POINT_KEYS
            if kp_extra:
                issues.append(f"wheel[{i}].points has extra keys: {sorted(kp_extra)}")
    return not issues, issues


def _audit_wheel(w: dict[str, Any]) -> dict[str, Any]:
    x1, y1, x2, y2 = w["bbox_xyxy"]
    bw = float(x2 - x1)
    bh = float(y2 - y1)
    pts = w["points"]
    ax, ay = pts["a"]
    bx, by = pts["b"]
    cx, cy = pts["c_disc_bottom"]

    rel_y_a = (ay - y1) / bh if bh > 0 else 0.0
    rel_y_b = (by - y1) / bh if bh > 0 else 0.0
    rel_y_c = (cy - y1) / bh if bh > 0 else 0.0
    ab_sep_ratio = abs(bx - ax) / bw if bw > 0 else 0.0
    c_ab_order_ok = cy < min(ay, by)

    checks = {
        "a_in_lower_band": rel_y_a >= REL_Y_AB_MIN,
        "b_in_lower_band": rel_y_b >= REL_Y_AB_MIN,
        "ab_horizontally_separated": ab_sep_ratio >= AB_SEP_RATIO_MIN,
        "c_above_ab_in_image": c_ab_order_ok,
    }
    n_pass = sum(checks.values())
    if n_pass == len(checks):
        verdict = "PASS"
    elif n_pass >= 2:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    reasons: list[str] = []
    if not checks["a_in_lower_band"]:
        reasons.append(
            f"A at rel_y={rel_y_a:.3f} (need >= {REL_Y_AB_MIN}). "
            f"Floor-ray semantics expect A near the wheel's ground line."
        )
    if not checks["b_in_lower_band"]:
        reasons.append(f"B at rel_y={rel_y_b:.3f} (need >= {REL_Y_AB_MIN}).")
    if not checks["ab_horizontally_separated"]:
        reasons.append(
            f"|B_x - A_x| / bbox_w = {ab_sep_ratio:.3f} (need >= {AB_SEP_RATIO_MIN}). "
            f"A and B should span the wheel's footprint width."
        )
    if not checks["c_above_ab_in_image"]:
        reasons.append(
            f"C_y={cy:.1f} >= min(A_y, B_y)={min(ay, by):.1f}. "
            f"C should be ABOVE A/B in image coords (smaller y)."
        )

    return {
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "w": bw, "h": bh},
        "points": {"a": [ax, ay], "b": [bx, by], "c_disc_bottom": [cx, cy]},
        "rel_y": {"a": rel_y_a, "b": rel_y_b, "c": rel_y_c},
        "ab_sep_ratio": ab_sep_ratio,
        "c_ab_order_ok": c_ab_order_ok,
        "checks": checks,
        "verdict": verdict,
        "reasons": reasons,
    }


def _audit_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = _detect_schema(payload)
    schema_ok = False
    schema_issues: list[str] = []
    wheels_audit: list[dict[str, Any]] = []

    if schema == "confirmed":
        schema_ok, schema_issues = _check_confirmed_schema(payload)
        wheels_audit = [_audit_wheel(w) for w in payload.get("wheels", [])]
    else:
        schema_issues.append(
            f"schema is {schema!r}, not 'confirmed' — geometry audit skipped "
            f"because A/B/C are not present in this payload shape"
        )

    return {
        "path": str(path.relative_to(REPO)),
        "frame_id": payload.get("frame_id"),
        "schema": schema,
        "schema_ok": schema_ok,
        "schema_issues": schema_issues,
        "n_wheels": len(wheels_audit),
        "wheels": wheels_audit,
    }


def _gather_files() -> list[Path]:
    files: list[Path] = []
    if SINGLE_DIR.is_dir():
        for p in sorted(SINGLE_DIR.glob("*.json")):
            files.append(p)
    if BATCH_DIR.is_dir():
        for p in sorted(BATCH_DIR.glob("*.json")):
            if p.stem == "batch_summary":
                continue
            files.append(p)
    return files


def main() -> None:
    files = _gather_files()
    by_file: list[dict[str, Any]] = []
    n_wheels = 0
    n_pass = n_warn = n_fail = 0
    n_files_confirmed = 0
    n_files_legacy = 0
    n_files_other = 0
    n_files_schema_ok = 0

    for p in files:
        a = _audit_file(p)
        by_file.append(a)
        if a["schema"] == "confirmed":
            n_files_confirmed += 1
            if a["schema_ok"]:
                n_files_schema_ok += 1
        elif a["schema"] == "legacy":
            n_files_legacy += 1
        else:
            n_files_other += 1
        for w in a["wheels"]:
            n_wheels += 1
            if w["verdict"] == "PASS":
                n_pass += 1
            elif w["verdict"] == "WARN":
                n_warn += 1
            else:
                n_fail += 1

    report = {
        "audit_version": 1,
        "criteria": {
            "rel_y_ab_min": REL_Y_AB_MIN,
            "ab_sep_ratio_min": AB_SEP_RATIO_MIN,
            "c_must_be_above_ab_in_image": True,
        },
        "files_audited": len(files),
        "files_by_schema": {
            "confirmed": n_files_confirmed,
            "legacy": n_files_legacy,
            "other_or_unknown": n_files_other,
        },
        "files_schema_ok": n_files_schema_ok,
        "wheels_audited": n_wheels,
        "wheels_pass": n_pass,
        "wheels_warn": n_warn,
        "wheels_fail": n_fail,
        "by_file": by_file,
    }
    OUT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"files={len(files)} (confirmed={n_files_confirmed}, "
        f"legacy={n_files_legacy}, other={n_files_other}); "
        f"wheels={n_wheels} pass={n_pass} warn={n_warn} fail={n_fail}"
    )
    print(f"wrote: {OUT_JSON.relative_to(REPO)}")


if __name__ == "__main__":
    main()
