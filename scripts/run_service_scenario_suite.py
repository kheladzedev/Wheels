"""Run the wheel-pose inference service over a real-photo scenario suite.

This script intentionally calls ``src/infer_image.py`` as a subprocess instead
of importing postprocess helpers directly. The goal is to verify the same
service path an AR/client integration would exercise: image in, confirmed AR
JSON + overlay image out.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2

REQUIRED_POINT_KEYS = ("a", "b", "c_disc_bottom")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run src/infer_image.py against a real-photo scenario suite."
    )
    p.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSON manifest with a top-level 'scenarios' list.",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=Path("runs/pose/wheel_v4_real/weights/best.pt"),
        help="YOLO-pose weights passed to src/infer_image.py.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/service_scenario_suite"),
        help="Directory for per-scenario service outputs and summary files.",
    )
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--max-det", type=int, default=20)
    p.add_argument("--device", default="cpu")
    p.add_argument("--viz-mode", choices=("raw", "final", "both"), default="final")
    p.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used to run src/infer_image.py.",
    )
    return p.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    scenarios = data.get("scenarios") if isinstance(data, dict) else data
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Manifest must contain a non-empty 'scenarios' list")
    out: list[dict[str, Any]] = []
    for idx, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise ValueError(f"Scenario #{idx} is not an object")
        scenario_id = str(scenario.get("id") or "").strip()
        image = str(scenario.get("image") or "").strip()
        if not scenario_id:
            raise ValueError(f"Scenario #{idx} is missing 'id'")
        if not image:
            raise ValueError(f"Scenario {scenario_id!r} is missing 'image'")
        out.append(scenario)
    return out


def image_size_hw(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    h, w = img.shape[:2]
    return int(h), int(w)


def audit_wheel_geometry(wheel: dict[str, Any], image_hw: tuple[int, int]) -> list[str]:
    issues: list[str] = []
    h, w_img = image_hw

    bbox = wheel.get("bbox_xyxy")
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return ["missing bbox_xyxy[4]"]
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return ["bbox_xyxy contains non-numeric values"]

    if not (0 <= x1 < x2 <= w_img and 0 <= y1 < y2 <= h):
        issues.append("bbox outside image or degenerate")

    points = wheel.get("points")
    if not isinstance(points, dict):
        return issues + ["missing points object"]
    for key in REQUIRED_POINT_KEYS:
        value = points.get(key)
        if not (isinstance(value, list) and len(value) == 2):
            issues.append(f"missing points.{key}[2]")
            continue
        try:
            px, py = (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            issues.append(f"points.{key} contains non-numeric values")
            continue
        if not (0 <= px <= w_img and 0 <= py <= h):
            issues.append(f"points.{key} outside image")
        if not (x1 <= px <= x2 and y1 <= py <= y2):
            issues.append(f"points.{key} outside bbox")

    if not issues:
        a = points["a"]
        b = points["b"]
        c = points["c_disc_bottom"]
        width = max(x2 - x1, 1.0)
        height = max(y2 - y1, 1.0)
        a_rel_y = (float(a[1]) - y1) / height
        b_rel_y = (float(b[1]) - y1) / height
        c_rel_y = (float(c[1]) - y1) / height
        ab_sep = (float(b[0]) - float(a[0])) / width
        if float(a[0]) >= float(b[0]):
            issues.append("A is not left of B")
        if ab_sep < 0.50:
            issues.append("A/B horizontal separation < 50% bbox width")
        if min(a_rel_y, b_rel_y) < 0.80:
            issues.append("A/B are not in the lower 20% of the bbox")
        if c_rel_y <= 0.50:
            issues.append("C is not in the lower half of the bbox")
        if float(c[1]) >= min(float(a[1]), float(b[1])):
            issues.append("C is not above the A/B floor-ray line")

    return issues


def audit_confirmed_payload(path: Path, image_hw: tuple[int, int]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    issues: list[str] = []
    if "frame_id" not in payload:
        issues.append("missing frame_id")
    wheels = payload.get("wheels")
    if not isinstance(wheels, list):
        return {"schema_ok": False, "wheel_count": 0, "geometry_ok": False, "issues": issues + ["missing wheels list"]}

    bad_geometry = 0
    for idx, wheel in enumerate(wheels):
        if not isinstance(wheel, dict):
            issues.append(f"wheels[{idx}] is not an object")
            continue
        if "confidence" not in wheel:
            issues.append(f"wheels[{idx}] missing confidence")
        wheel_issues = audit_wheel_geometry(wheel, image_hw)
        if wheel_issues:
            bad_geometry += 1
            issues.extend(f"wheels[{idx}]: {issue}" for issue in wheel_issues)

    return {
        "schema_ok": not any("missing" in issue for issue in issues),
        "wheel_count": len(wheels),
        "geometry_ok": bad_geometry == 0,
        "issues": issues,
    }


def run_infer_image(
    *,
    python: Path,
    image: Path,
    model: Path,
    out_dir: Path,
    scenario_id: str,
    conf: float,
    iou: float,
    max_det: int,
    device: str,
    viz_mode: str,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        str(python),
        "src/infer_image.py",
        "--image",
        str(image),
        "--model",
        str(model),
        "--out-dir",
        str(out_dir),
        "--frame-id",
        scenario_id,
        "--conf",
        str(conf),
        "--iou",
        str(iou),
        "--max-det",
        str(max_det),
        "--viz-mode",
        viz_mode,
    ]
    if device:
        cmd.extend(["--device", device])
    return subprocess.run(cmd, text=True, capture_output=True, check=False)


def write_markdown_report(out_path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Service Scenario Suite",
        "",
        "| Scenario | Image | Wheels | Geometry | Output |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in rows:
        rel_overlay = row.get("overlay_path") or ""
        geometry = "ok" if row["audit"]["geometry_ok"] else "needs_review"
        lines.append(
            "| {scenario} | {image} | {wheels} | {geometry} | {overlay} |".format(
                scenario=row["id"],
                image=Path(row["image"]).name,
                wheels=row["audit"]["wheel_count"],
                geometry=geometry,
                overlay=rel_overlay,
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    scenarios = load_manifest(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failed = 0
    for scenario in scenarios:
        scenario_id = str(scenario["id"])
        image = Path(str(scenario["image"]))
        scenario_out = args.out_dir / scenario_id
        scenario_out.mkdir(parents=True, exist_ok=True)

        proc = run_infer_image(
            python=args.python,
            image=image,
            model=args.model,
            out_dir=scenario_out,
            scenario_id=scenario_id,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            viz_mode=args.viz_mode,
        )
        json_path = scenario_out / f"{image.stem}.json"
        overlay_path = scenario_out / f"{image.stem}_final_pred.jpg"
        row: dict[str, Any] = {
            **scenario,
            "service_command": " ".join(proc.args),
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "json_path": str(json_path),
            "overlay_path": str(overlay_path) if overlay_path.exists() else None,
        }

        if proc.returncode != 0 or not json_path.is_file():
            failed += 1
            row["audit"] = {
                "schema_ok": False,
                "wheel_count": 0,
                "geometry_ok": False,
                "issues": ["service command failed or JSON was not written"],
            }
        else:
            row["audit"] = audit_confirmed_payload(json_path, image_size_hw(image))
        rows.append(row)

        status = "ok" if row["audit"]["geometry_ok"] else "needs_review"
        print(
            f"{scenario_id}: rc={proc.returncode} wheels={row['audit']['wheel_count']} "
            f"geometry={status}"
        )

    summary = {
        "manifest": str(args.manifest),
        "model": str(args.model),
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "device": args.device,
        "scenario_count": len(rows),
        "service_failures": failed,
        "total_wheels": sum(int(r["audit"]["wheel_count"]) for r in rows),
        "geometry_ok_scenarios": sum(1 for r in rows if r["audit"]["geometry_ok"]),
        "rows": rows,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown_report(args.out_dir / "summary.md", rows)
    print(f"Summary: {args.out_dir / 'summary.json'}")
    print(f"Report:  {args.out_dir / 'summary.md'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
