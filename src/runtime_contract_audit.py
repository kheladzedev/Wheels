"""Audit actual inference outputs against the confirmed AR JSON contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ALLOWED_TOP_LEVEL = {"frame_id", "wheels"}
ALLOWED_WHEEL_KEYS = {"bbox_xyxy", "confidence", "points"}
ALLOWED_POINT_KEYS = {"a", "b", "c_disc_bottom"}
FORBIDDEN_KEY_SUBSTRINGS = (
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

DEFAULT_SINGLE_JSON = Path(
    "outputs/production_audit/smoke_single/"
    "real_v1_self__real_010_car_1930_Duesenberg_J_Murphy_Disappearing_Top_Torpedo_Convertibl.json"
)
DEFAULT_BATCH_JSONL = Path("outputs/production_audit/smoke_batch/val.jsonl")
DEFAULT_BATCH_SUMMARY = Path("outputs/production_audit/smoke_batch/batch_summary.json")
DEFAULT_OUT = Path("outputs/production_audit/runtime_contract_audit.json")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"_load_error": str(exc)}
    return payload if isinstance(payload, dict) else {"_load_error": "top-level is not object"}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [{"_line_no": 0, "_load_error": str(exc)}]
    for i, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            payloads.append({"_line_no": i, "_load_error": exc.msg})
            continue
        if isinstance(payload, dict):
            payload["_line_no"] = i
            payloads.append(payload)
        else:
            payloads.append({"_line_no": i, "_load_error": "line is not object"})
    return payloads


def _collect_keys(payload: Any, out: set[str]) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            out.add(str(key))
            _collect_keys(value, out)
    elif isinstance(payload, list):
        for item in payload:
            _collect_keys(item, out)


def validate_payload(payload: dict[str, Any], *, source: str) -> list[str]:
    errors: list[str] = []
    if "_load_error" in payload:
        return [f"{source}: {payload['_load_error']}"]
    if "_line_no" in payload:
        payload = {k: v for k, v in payload.items() if k != "_line_no"}
    if set(payload.keys()) != ALLOWED_TOP_LEVEL:
        errors.append(f"{source}: top-level keys {sorted(payload.keys())}")
    frame_id = payload.get("frame_id")
    if not isinstance(frame_id, str) or not frame_id:
        errors.append(f"{source}: frame_id must be non-empty string")
    wheels = payload.get("wheels")
    if not isinstance(wheels, list):
        errors.append(f"{source}: wheels must be list")
        wheels = []
    for idx, wheel in enumerate(wheels):
        if not isinstance(wheel, dict):
            errors.append(f"{source}: wheels[{idx}] must be object")
            continue
        if set(wheel.keys()) != ALLOWED_WHEEL_KEYS:
            errors.append(f"{source}: wheels[{idx}] keys {sorted(wheel.keys())}")
        bbox = wheel.get("bbox_xyxy")
        if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)):
            errors.append(f"{source}: wheels[{idx}].bbox_xyxy invalid")
        else:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            if not (x1 < x2 and y1 < y2):
                errors.append(f"{source}: wheels[{idx}].bbox_xyxy not ordered")
        confidence = wheel.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
            errors.append(f"{source}: wheels[{idx}].confidence invalid")
        points = wheel.get("points")
        if not isinstance(points, dict) or set(points.keys()) != ALLOWED_POINT_KEYS:
            errors.append(f"{source}: wheels[{idx}].points keys invalid")
            continue
        for name in ALLOWED_POINT_KEYS:
            xy = points.get(name)
            if not (isinstance(xy, list) and len(xy) == 2 and all(isinstance(v, (int, float)) for v in xy)):
                errors.append(f"{source}: wheels[{idx}].points.{name} invalid")

    keys: set[str] = set()
    _collect_keys(payload, keys)
    for key in keys:
        lower = key.lower()
        for forbidden in FORBIDDEN_KEY_SUBSTRINGS:
            if forbidden in lower:
                errors.append(f"{source}: forbidden key {key!r} contains {forbidden!r}")
    return errors


def build_report(single_json: Path, batch_jsonl: Path, batch_summary: Path) -> dict[str, Any]:
    failures: list[str] = []
    single_payload = _load_json(single_json)
    batch_payloads = _load_jsonl(batch_jsonl)
    summary = _load_json(batch_summary)

    failures.extend(validate_payload(single_payload, source=str(single_json)))
    for payload in batch_payloads:
        line = payload.get("_line_no", "?")
        failures.extend(validate_payload(payload, source=f"{batch_jsonl}:{line}"))

    single_wheels = len(single_payload.get("wheels", [])) if isinstance(single_payload.get("wheels"), list) else 0
    batch_wheels = sum(
        len(payload.get("wheels", []))
        for payload in batch_payloads
        if isinstance(payload.get("wheels"), list)
    )
    if "_load_error" in summary:
        failures.append(f"{batch_summary}: {summary['_load_error']}")
    else:
        if summary.get("frames_inferred") != len(batch_payloads):
            failures.append(
                f"{batch_summary}: frames_inferred={summary.get('frames_inferred')} "
                f"jsonl_lines={len(batch_payloads)}"
            )
        if summary.get("wheels_detected_total") != batch_wheels:
            failures.append(
                f"{batch_summary}: wheels_detected_total={summary.get('wheels_detected_total')} "
                f"jsonl_wheels={batch_wheels}"
            )

    return {
        "ok": not failures,
        "single_json": str(single_json),
        "batch_jsonl": str(batch_jsonl),
        "batch_summary": str(batch_summary),
        "counts": {
            "single_wheels": single_wheels,
            "batch_frames": len(batch_payloads),
            "batch_wheels": batch_wheels,
        },
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-json", type=Path, default=DEFAULT_SINGLE_JSON)
    parser.add_argument("--batch-jsonl", type=Path, default=DEFAULT_BATCH_JSONL)
    parser.add_argument("--batch-summary", type=Path, default=DEFAULT_BATCH_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args.single_json, args.batch_jsonl, args.batch_summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"ok={report['ok']} single_wheels={report['counts']['single_wheels']} "
        f"batch_frames={report['counts']['batch_frames']} batch_wheels={report['counts']['batch_wheels']}"
    )
    print(f"report={args.out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
