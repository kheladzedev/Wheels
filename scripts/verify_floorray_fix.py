"""Independent verification of the floor-ray geometry fix.

Reads the raw JSON annotations in data/incoming/android_plugin/ and
the YOLO-pose label .txt files under data/wheel_pose_dataset/, decodes
both back to pixels, and asserts the floor-ray invariants on every
wheel. Fails loud the moment any wheel violates any rule. Writes
outputs/floorray_fix_verification.json and ..._verification.md.

Run from repo root after a fresh
  python src/create_sample_keypoint_incoming.py --count 10 --overwrite
  python src/convert_keypoint_incoming_to_yolo_pose.py ...
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2

REPO = Path(__file__).resolve().parents[1]
INCOMING_DIR = REPO / "data" / "incoming" / "android_plugin"
YOLO_DIR = REPO / "data" / "wheel_pose_dataset"
OUT_JSON = REPO / "outputs" / "floorray_fix_verification.json"
OUT_MD = REPO / "outputs" / "floorray_fix_verification.md"

REL_Y_AB_MIN = 0.85
AB_SEP_RATIO_MIN = 0.50


def _wheel_geom(
    bbox: list[float], a: list[float], b: list[float], c: list[float]
) -> dict:
    x1, y1, x2, y2 = bbox
    bw = float(x2 - x1)
    bh = float(y2 - y1)
    ax, ay = a
    bx, by = b
    cx, cy = c
    rel_y_a = (ay - y1) / bh
    rel_y_b = (by - y1) / bh
    rel_y_c = (cy - y1) / bh
    rel_x_c = (cx - x1) / bw
    ab_sep = abs(bx - ax) / bw
    inside = (
        (x1 <= ax <= x2 and y1 <= ay <= y2)
        and (x1 <= bx <= x2 and y1 <= by <= y2)
        and (x1 <= cx <= x2 and y1 <= cy <= y2)
    )
    checks = {
        "rel_y_a_ge_0_85": rel_y_a >= REL_Y_AB_MIN,
        "rel_y_b_ge_0_85": rel_y_b >= REL_Y_AB_MIN,
        "c_above_a": cy < ay,
        "c_above_b": cy < by,
        "ab_sep_ratio_ge_0_50": ab_sep >= AB_SEP_RATIO_MIN,
        "points_inside_bbox": inside,
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"
    return {
        "bbox": {"w": bw, "h": bh, "x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "rel_y": {"a": rel_y_a, "b": rel_y_b, "c": rel_y_c},
        "rel_x_c": rel_x_c,
        "ab_sep_ratio": ab_sep,
        "checks": checks,
        "verdict": verdict,
    }


def _audit_incoming() -> dict:
    rows: list[dict] = []
    anno_dir = INCOMING_DIR / "annotations"
    img_dir = INCOMING_DIR / "images"
    files = sorted(anno_dir.glob("*.json"))
    n_pass = n_fail = 0
    for ann_path in files:
        payload = json.loads(ann_path.read_text(encoding="utf-8"))
        for i, w in enumerate(payload["wheels"]):
            geom = _wheel_geom(
                w["bbox_xyxy"],
                w["points"]["a"],
                w["points"]["b"],
                w["points"]["c_disc_bottom"],
            )
            row = {"file": ann_path.name, "wheel_idx": i, **geom}
            rows.append(row)
            if geom["verdict"] == "PASS":
                n_pass += 1
            else:
                n_fail += 1
    return {
        "source": "incoming_json",
        "annotations_dir": str(anno_dir.relative_to(REPO)),
        "images_dir": str(img_dir.relative_to(REPO)),
        "n_files": len(files),
        "n_wheels": n_pass + n_fail,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "wheels": rows,
    }


def _decode_yolo_label(line: str, image_w: int, image_h: int) -> dict | None:
    parts = line.split()
    if len(parts) != 1 + 4 + 3 * 3:
        return None
    _, cx, cy, w, h = (float(x) for x in parts[:5])
    x1 = (cx - 0.5 * w) * image_w
    y1 = (cy - 0.5 * h) * image_h
    x2 = (cx + 0.5 * w) * image_w
    y2 = (cy + 0.5 * h) * image_h

    def _kp(i: int) -> list[float]:
        kx = float(parts[5 + i * 3]) * image_w
        ky = float(parts[5 + i * 3 + 1]) * image_h
        return [kx, ky]

    return {
        "bbox": [x1, y1, x2, y2],
        "a": _kp(0),
        "b": _kp(1),
        "c": _kp(2),
    }


def _audit_yolo() -> dict:
    rows: list[dict] = []
    n_pass = n_fail = 0
    splits = ("train", "val")
    n_files = 0
    for split in splits:
        img_dir = YOLO_DIR / "images" / split
        lbl_dir = YOLO_DIR / "labels" / split
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            n_files += 1
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                decoded = _decode_yolo_label(line, w, h)
                if decoded is None:
                    continue
                geom = _wheel_geom(
                    decoded["bbox"], decoded["a"], decoded["b"], decoded["c"]
                )
                row = {
                    "file": str(img_path.relative_to(REPO)),
                    "split": split,
                    **geom,
                }
                rows.append(row)
                if geom["verdict"] == "PASS":
                    n_pass += 1
                else:
                    n_fail += 1
    return {
        "source": "yolo_pose_decoded",
        "dataset_root": str(YOLO_DIR.relative_to(REPO)),
        "n_files": n_files,
        "n_wheels": n_pass + n_fail,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "wheels": rows,
    }


def _markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# Floor-Ray Fix — Independent Verification")
    lines.append("")
    lines.append(f"- Date: {report['date']}")
    lines.append(f"- Overall verdict: **{report['verdict']}**")
    lines.append("")
    lines.append("## Thresholds")
    lines.append(f"- `rel_y_{{a,b}} >= {REL_Y_AB_MIN}`")
    lines.append(f"- `ab_sep_ratio >= {AB_SEP_RATIO_MIN}`")
    lines.append("- `c_y < min(a_y, b_y)`")
    lines.append("- all three points strictly inside bbox")
    lines.append("")
    for section_key, header in (
        ("incoming", "Incoming JSON (raw generator output)"),
        ("yolo", "Converted YOLO-pose labels (decoded back to pixels)"),
    ):
        sec = report[section_key]
        lines.append(f"## {header}")
        lines.append(
            f"- Files: {sec['n_files']}, wheels: {sec['n_wheels']}, "
            f"PASS: {sec['n_pass']}, FAIL: {sec['n_fail']}"
        )
        lines.append("")
        # First 10 wheel rows for inspection.
        lines.append(
            "| file | rel_y_a | rel_y_b | rel_y_c | ab_sep | c<a&c<b | inside | verdict |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in sec["wheels"][:10]:
            checks = r["checks"]
            file_label = r["file"]
            if "wheel_idx" in r:
                file_label = f"{file_label}#{r['wheel_idx']}"
            lines.append(
                f"| `{file_label}` "
                f"| {r['rel_y']['a']:.3f} "
                f"| {r['rel_y']['b']:.3f} "
                f"| {r['rel_y']['c']:.3f} "
                f"| {r['ab_sep_ratio']:.3f} "
                f"| {checks['c_above_a'] and checks['c_above_b']} "
                f"| {checks['points_inside_bbox']} "
                f"| **{r['verdict']}** |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    import time

    incoming = _audit_incoming()
    yolo = _audit_yolo()
    overall_fail = incoming["n_fail"] + yolo["n_fail"]
    report = {
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "thresholds": {
            "rel_y_ab_min": REL_Y_AB_MIN,
            "ab_sep_ratio_min": AB_SEP_RATIO_MIN,
        },
        "incoming": incoming,
        "yolo": yolo,
        "verdict": "PASS" if overall_fail == 0 else "FAIL",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    OUT_MD.write_text(_markdown(report), encoding="utf-8")

    print(
        f"incoming: files={incoming['n_files']} wheels={incoming['n_wheels']} "
        f"pass={incoming['n_pass']} fail={incoming['n_fail']}"
    )
    print(
        f"yolo:     files={yolo['n_files']} wheels={yolo['n_wheels']} "
        f"pass={yolo['n_pass']} fail={yolo['n_fail']}"
    )
    print(f"verdict:  {report['verdict']}")
    print(f"wrote:    {OUT_JSON.relative_to(REPO)}")
    print(f"          {OUT_MD.relative_to(REPO)}")
    return 0 if overall_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
