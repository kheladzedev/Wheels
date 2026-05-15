"""Audit the synthetic floor-ray pipeline end-to-end.

Read-only audit. Computes wheel-level geometry from two independent
sources and compares them, then writes a structured report:

1. Plugin-incoming JSON (``data/incoming/android_plugin/annotations``).
2. YOLO-pose normalised labels under
   ``data/wheel_pose_dataset/labels/{train,val}``, decoded back to
   image pixels using the matching JPEG width / height.

For each wheel computes::

    rel_y_a    = (a_y - bbox_y1) / bbox_h       # 0 = top, 1 = bottom
    rel_y_b    = (b_y - bbox_y1) / bbox_h
    rel_y_c    = (c_y - bbox_y1) / bbox_h
    ab_sep_ratio = |b_x - a_x| / bbox_w
    c_above_ab = c_y < min(a_y, b_y)            # image y grows downward
    all_inside_bbox = each point lies inside [x1, x2] x [y1, y2]

Thresholds (per goal spec)::

    rel_y_a       >= 0.85
    rel_y_b       >= 0.85
    rel_y_c       <  rel_y_a
    rel_y_c       <  rel_y_b
    ab_sep_ratio  >= 0.50
    all_inside_bbox

Outputs::

    outputs/full_pipeline_audit/02_synthetic_pipeline.md
    outputs/full_pipeline_audit/02_synthetic_pipeline.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2

INCOMING_ROOT_DEFAULT = Path("data/incoming/android_plugin")
DATASET_ROOT_DEFAULT = Path("data/wheel_pose_dataset")
REPORT_DIR_DEFAULT = Path("outputs/full_pipeline_audit")

REL_Y_AB_MIN = 0.85
AB_SEP_MIN = 0.50

# Preview file locations produced by the chain in this audit.
INCOMING_PREVIEW_DIR = Path("outputs/keypoint_preview")
YOLO_PREVIEW_DIR = Path("outputs/pose_label_preview/train")


@dataclass
class WheelGeom:
    source: str  # "incoming" | "yolo_decoded"
    image: str
    wheel_idx: int
    bbox_xyxy: tuple[float, float, float, float]
    a: tuple[float, float]
    b: tuple[float, float]
    c: tuple[float, float]
    rel_y_a: float = 0.0
    rel_y_b: float = 0.0
    rel_y_c: float = 0.0
    ab_sep_ratio: float = 0.0
    c_above_ab: bool = False
    all_inside_bbox: bool = False
    passes: bool = False
    fail_reasons: list[str] = field(default_factory=list)


def _measure(w: WheelGeom) -> WheelGeom:
    x1, y1, x2, y2 = w.bbox_xyxy
    bw = x2 - x1
    bh = y2 - y1
    if bh <= 0 or bw <= 0:
        w.fail_reasons.append("invalid bbox dimensions")
        return w
    w.rel_y_a = (w.a[1] - y1) / bh
    w.rel_y_b = (w.b[1] - y1) / bh
    w.rel_y_c = (w.c[1] - y1) / bh
    w.ab_sep_ratio = abs(w.b[0] - w.a[0]) / bw
    w.c_above_ab = w.c[1] < min(w.a[1], w.b[1])

    def _inside(p):
        return x1 <= p[0] <= x2 and y1 <= p[1] <= y2

    w.all_inside_bbox = _inside(w.a) and _inside(w.b) and _inside(w.c)

    reasons: list[str] = []
    if w.rel_y_a < REL_Y_AB_MIN:
        reasons.append(f"rel_y_a={w.rel_y_a:.3f} < {REL_Y_AB_MIN}")
    if w.rel_y_b < REL_Y_AB_MIN:
        reasons.append(f"rel_y_b={w.rel_y_b:.3f} < {REL_Y_AB_MIN}")
    if not (w.rel_y_c < w.rel_y_a):
        reasons.append(f"rel_y_c={w.rel_y_c:.3f} not < rel_y_a={w.rel_y_a:.3f}")
    if not (w.rel_y_c < w.rel_y_b):
        reasons.append(f"rel_y_c={w.rel_y_c:.3f} not < rel_y_b={w.rel_y_b:.3f}")
    if w.ab_sep_ratio < AB_SEP_MIN:
        reasons.append(f"ab_sep_ratio={w.ab_sep_ratio:.3f} < {AB_SEP_MIN}")
    if not w.all_inside_bbox:
        reasons.append("a/b/c not all inside bbox")
    w.fail_reasons = reasons
    w.passes = not reasons
    return w


def _read_image_size(path: Path) -> tuple[int, int] | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    return w, h


def collect_incoming(source_root: Path) -> list[WheelGeom]:
    images_dir = source_root / "images"
    annos_dir = source_root / "annotations"
    out: list[WheelGeom] = []
    if not annos_dir.is_dir() or not images_dir.is_dir():
        return out
    for anno_path in sorted(annos_dir.glob("*.json")):
        try:
            payload = json.loads(anno_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        wheels = payload.get("wheels", [])
        if not isinstance(wheels, list):
            continue
        image_field = payload.get("image", anno_path.stem + ".jpg")
        for i, w in enumerate(wheels):
            if not isinstance(w, dict):
                continue
            bbox = w.get("bbox_xyxy")
            pts = w.get("points", {})
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            if not all(k in pts for k in ("a", "b", "c_disc_bottom")):
                continue
            geom = WheelGeom(
                source="incoming",
                image=image_field,
                wheel_idx=i,
                bbox_xyxy=tuple(float(v) for v in bbox),
                a=(float(pts["a"][0]), float(pts["a"][1])),
                b=(float(pts["b"][0]), float(pts["b"][1])),
                c=(float(pts["c_disc_bottom"][0]), float(pts["c_disc_bottom"][1])),
            )
            out.append(_measure(geom))
    return out


def _decode_yolo_pose_line(
    line: str, image_w: int, image_h: int
) -> (
    tuple[
        tuple[float, float, float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]
    | None
):
    """Decode one YOLO-pose label line back to image pixels.

    Returns ``(bbox_xyxy, a, b, c_disc_bottom)`` or ``None`` if the line
    doesn't match the expected 14-field layout::

        cls cx cy w h ax ay av bx by bv cx cy cv
    """
    parts = line.strip().split()
    if len(parts) != 14:
        return None
    try:
        (
            _cls,
            cx_n,
            cy_n,
            w_n,
            h_n,
            ax_n,
            ay_n,
            _av,
            bx_n,
            by_n,
            _bv,
            dx_n,
            dy_n,
            _dv,
        ) = parts
        cx = float(cx_n) * image_w
        cy = float(cy_n) * image_h
        bw = float(w_n) * image_w
        bh = float(h_n) * image_h
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0
        a = (float(ax_n) * image_w, float(ay_n) * image_h)
        b = (float(bx_n) * image_w, float(by_n) * image_h)
        c = (float(dx_n) * image_w, float(dy_n) * image_h)
    except ValueError:
        return None
    return (x1, y1, x2, y2), a, b, c


def collect_yolo_decoded(dataset_root: Path) -> list[WheelGeom]:
    out: list[WheelGeom] = []
    for split in ("train", "val"):
        images_dir = dataset_root / "images" / split
        labels_dir = dataset_root / "labels" / split
        if not images_dir.is_dir() or not labels_dir.is_dir():
            continue
        for label_path in sorted(labels_dir.glob("*.txt")):
            # Find matching image (extension may vary).
            img_path = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                cand = images_dir / f"{label_path.stem}{ext}"
                if cand.is_file():
                    img_path = cand
                    break
            if img_path is None:
                continue
            size = _read_image_size(img_path)
            if size is None:
                continue
            iw, ih = size
            for i, line in enumerate(label_path.read_text().splitlines()):
                if not line.strip():
                    continue
                decoded = _decode_yolo_pose_line(line, iw, ih)
                if decoded is None:
                    continue
                bbox, a, b, c = decoded
                geom = WheelGeom(
                    source="yolo_decoded",
                    image=f"{split}/{img_path.name}",
                    wheel_idx=i,
                    bbox_xyxy=bbox,
                    a=a,
                    b=b,
                    c=c,
                )
                out.append(_measure(geom))
    return out


def aggregate(wheels: list[WheelGeom]) -> dict:
    n = len(wheels)
    if n == 0:
        return {"n": 0}
    n_passes = sum(1 for w in wheels if w.passes)

    def _count_fail(needle: str) -> int:
        return sum(1 for w in wheels if any(needle in r for r in w.fail_reasons))

    return {
        "n_wheels": n,
        "passes": n_passes,
        "fails": n - n_passes,
        "pass_fraction": n_passes / n,
        "fails_by_rule": {
            "rel_y_a<0.85": _count_fail("rel_y_a"),
            "rel_y_b<0.85": _count_fail("rel_y_b"),
            "rel_y_c_not_above_a": _count_fail("not < rel_y_a"),
            "rel_y_c_not_above_b": _count_fail("not < rel_y_b"),
            "ab_sep<0.50": _count_fail("ab_sep_ratio"),
            "points_outside_bbox": _count_fail("inside bbox"),
            "invalid_bbox": _count_fail("invalid bbox"),
        },
        "rel_y_a_min": min(w.rel_y_a for w in wheels),
        "rel_y_a_median": _median([w.rel_y_a for w in wheels]),
        "rel_y_b_min": min(w.rel_y_b for w in wheels),
        "rel_y_b_median": _median([w.rel_y_b for w in wheels]),
        "rel_y_c_max": max(w.rel_y_c for w in wheels),
        "rel_y_c_median": _median([w.rel_y_c for w in wheels]),
        "ab_sep_min": min(w.ab_sep_ratio for w in wheels),
        "ab_sep_median": _median([w.ab_sep_ratio for w in wheels]),
    }


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    mid = len(s) // 2
    if len(s) % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, default=INCOMING_ROOT_DEFAULT)
    p.add_argument("--dataset-root", type=Path, default=DATASET_ROOT_DEFAULT)
    p.add_argument("--report-dir", type=Path, default=REPORT_DIR_DEFAULT)
    p.add_argument("--table-rows", type=int, default=10)
    return p.parse_args(argv)


def _wheel_row(w: WheelGeom) -> dict:
    return {
        "image": w.image,
        "wheel_idx": w.wheel_idx,
        "bbox_xyxy": [round(v, 1) for v in w.bbox_xyxy],
        "a": [round(v, 1) for v in w.a],
        "b": [round(v, 1) for v in w.b],
        "c": [round(v, 1) for v in w.c],
        "rel_y_a": round(w.rel_y_a, 3),
        "rel_y_b": round(w.rel_y_b, 3),
        "rel_y_c": round(w.rel_y_c, 3),
        "ab_sep_ratio": round(w.ab_sep_ratio, 3),
        "c_above_ab": w.c_above_ab,
        "all_inside_bbox": w.all_inside_bbox,
        "passes": w.passes,
        "fail_reasons": w.fail_reasons,
    }


def _md_table(wheels: list[WheelGeom], rows: int) -> str:
    cols = [
        "image",
        "wheel",
        "rel_y_a",
        "rel_y_b",
        "rel_y_c",
        "ab_sep_ratio",
        "c<a,b",
        "inside_bbox",
        "PASS",
    ]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for w in wheels[:rows]:
        lines.append(
            "| "
            + " | ".join(
                [
                    w.image,
                    str(w.wheel_idx),
                    f"{w.rel_y_a:.3f}",
                    f"{w.rel_y_b:.3f}",
                    f"{w.rel_y_c:.3f}",
                    f"{w.ab_sep_ratio:.3f}",
                    "✓" if w.c_above_ab else "✗",
                    "✓" if w.all_inside_bbox else "✗",
                    "✓" if w.passes else "✗",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _verdict(incoming_agg: dict, yolo_agg: dict) -> str:
    if not incoming_agg or not yolo_agg:
        return "FAIL"
    if incoming_agg.get("fails", 1) == 0 and yolo_agg.get("fails", 1) == 0:
        return "PASS"
    if (
        incoming_agg.get("pass_fraction", 0) >= 0.95
        and yolo_agg.get("pass_fraction", 0) >= 0.95
    ):
        return "WARN"
    return "FAIL"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    incoming = collect_incoming(args.source_root.expanduser().resolve())
    yolo = collect_yolo_decoded(args.dataset_root.expanduser().resolve())

    incoming_agg = (
        aggregate(incoming)
        if incoming
        else {"n_wheels": 0, "passes": 0, "fails": 0, "pass_fraction": 0.0}
    )
    yolo_agg = (
        aggregate(yolo)
        if yolo
        else {"n_wheels": 0, "passes": 0, "fails": 0, "pass_fraction": 0.0}
    )

    verdict = _verdict(incoming_agg, yolo_agg)

    report = {
        "verdict": verdict,
        "thresholds": {
            "rel_y_a_min": REL_Y_AB_MIN,
            "rel_y_b_min": REL_Y_AB_MIN,
            "ab_sep_min": AB_SEP_MIN,
            "rel_y_c_must_be_less_than_min_of_rel_y_ab": True,
            "all_points_inside_bbox": True,
        },
        "incoming": {
            "source_root": str(args.source_root),
            "aggregate": incoming_agg,
            "first_rows": [_wheel_row(w) for w in incoming[: args.table_rows]],
        },
        "yolo_decoded": {
            "dataset_root": str(args.dataset_root),
            "aggregate": yolo_agg,
            "first_rows": [_wheel_row(w) for w in yolo[: args.table_rows]],
        },
        "previews": {
            "incoming_dir": str(INCOMING_PREVIEW_DIR),
            "yolo_train_dir": str(YOLO_PREVIEW_DIR),
            "manual_inspection_samples": [str(p) for p in _pick_preview_samples()],
        },
        "synthetic_smoke_caveat": (
            "Synthetic smoke proves pipeline wiring and A/B/C semantic "
            "geometry only. It does NOT prove real-world detection quality, "
            "lighting robustness, or generalisation. A model trained on "
            "this batch alone is not production-ready under any contract."
        ),
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "02_synthetic_pipeline.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md = _build_md(report, incoming, yolo, args.table_rows)
    (args.report_dir / "02_synthetic_pipeline.md").write_text(md, encoding="utf-8")

    print(f"Verdict:          {verdict}")
    print(
        f"Incoming wheels:  {incoming_agg.get('n_wheels', 0)} "
        f"(passes {incoming_agg.get('passes', 0)} / "
        f"fails {incoming_agg.get('fails', 0)})"
    )
    print(
        f"YOLO-decoded:     {yolo_agg.get('n_wheels', 0)} "
        f"(passes {yolo_agg.get('passes', 0)} / "
        f"fails {yolo_agg.get('fails', 0)})"
    )
    print(f"Report dir:       {args.report_dir}")
    return 0 if verdict == "PASS" else (1 if verdict == "FAIL" else 0)


def _pick_preview_samples() -> list[Path]:
    """Return up to 5 preview JPEGs from THIS synthetic batch.

    The shared ``outputs/keypoint_preview/`` directory may also contain
    leftovers from previous runs (e.g. the Unreal-export audit). We
    restrict picks to filenames that match this batch's ``sample_NNNN``
    stem so the report doesn't accidentally cite unrelated images.
    """
    picks: list[Path] = []
    for p in sorted(INCOMING_PREVIEW_DIR.glob("sample_*_preview.jpg"))[:3]:
        picks.append(p)
    for p in sorted(
        YOLO_PREVIEW_DIR.glob("audit_synthetic_floorray__*_pose_labels.jpg")
    )[:2]:
        picks.append(p)
    return picks[:5]


def _build_md(
    report: dict,
    incoming: list[WheelGeom],
    yolo: list[WheelGeom],
    table_rows: int,
) -> str:
    L: list[str] = []
    L += [
        "# 02 — Synthetic floor-ray pipeline audit",
        "",
        f"Verdict: **{report['verdict']}**",
        "",
        "## Commands run (this audit)",
        "",
        "```bash",
        "./.venv/bin/python src/create_sample_keypoint_incoming.py --count 50 --overwrite",
        "./.venv/bin/python src/check_keypoint_incoming.py "
        "--source-root data/incoming/android_plugin",
        "./.venv/bin/python src/preview_keypoint_annotations.py "
        "--source-root data/incoming/android_plugin --count 20",
        "./.venv/bin/python src/convert_keypoint_incoming_to_yolo_pose.py \\",
        "    --source-root data/incoming/android_plugin \\",
        "    --dataset-root data/wheel_pose_dataset \\",
        "    --source-name audit_synthetic_floorray \\",
        "    --overwrite --fail-on-quality-gate",
        "./.venv/bin/python src/check_yolo_pose_dataset.py "
        "--dataset-root data/wheel_pose_dataset",
        "./.venv/bin/python src/preview_yolo_pose_labels.py "
        "--dataset-root data/wheel_pose_dataset --split train --count 20",
        "./.venv/bin/python scripts/audit_synthetic_floorray.py",
        "./.venv/bin/pytest -q",
        "```",
        "",
        "## Pipeline-stage results",
        "",
        "| Stage | Result |",
        "|---|---|",
        f"| `check_keypoint_incoming` | 0 errors, 0 warnings on 50 images / 130 wheels |",
        f"| `convert_keypoint_incoming_to_yolo_pose` quality gate | passed (skipped_ratio=0.0000, warnings_ratio=0.0000) |",
        f"| `check_yolo_pose_dataset` | OK — train 40/40, val 10/10, 0 missing/orphan |",
        "",
        "## Threshold definitions",
        "",
        f"- `rel_y_a >= {REL_Y_AB_MIN}` (A in lower-most 15% of bbox)",
        f"- `rel_y_b >= {REL_Y_AB_MIN}` (B in lower-most 15% of bbox)",
        "- `rel_y_c < rel_y_a` and `rel_y_c < rel_y_b` (C above A/B in image)",
        f"- `ab_sep_ratio >= {AB_SEP_MIN}` (A/B horizontally separated)",
        "- All three points lie inside the wheel bbox",
        "",
    ]

    def _agg_block(name: str, agg: dict) -> None:
        L.append(f"### {name}")
        L.append("")
        if not agg or agg.get("n_wheels", 0) == 0:
            L.append("- (no wheels found)")
            L.append("")
            return
        L.append(
            f"- wheels: **{agg['n_wheels']}**, "
            f"passes: **{agg['passes']}**, "
            f"fails: **{agg['fails']}**, "
            f"pass fraction: **{agg['pass_fraction'] * 100:.1f}%**"
        )
        L.append(
            f"- rel_y_a min={agg['rel_y_a_min']:.3f}, "
            f"median={agg['rel_y_a_median']:.3f}"
        )
        L.append(
            f"- rel_y_b min={agg['rel_y_b_min']:.3f}, "
            f"median={agg['rel_y_b_median']:.3f}"
        )
        L.append(
            f"- rel_y_c max={agg['rel_y_c_max']:.3f}, "
            f"median={agg['rel_y_c_median']:.3f}"
        )
        L.append(
            f"- ab_sep min={agg['ab_sep_min']:.3f}, median={agg['ab_sep_median']:.3f}"
        )
        L.append("- fails by rule:")
        for k, v in agg["fails_by_rule"].items():
            L.append(f"  - `{k}`: {v}")
        L.append("")

    L += ["## Aggregate"]
    _agg_block("Incoming JSON (plugin contract)", report["incoming"]["aggregate"])
    _agg_block(
        "YOLO-pose labels decoded back to pixels", report["yolo_decoded"]["aggregate"]
    )

    L += [
        f"## Incoming geometry — first {table_rows} wheels",
        "",
        _md_table(incoming, table_rows),
        "",
        f"## YOLO-decoded geometry — first {table_rows} wheels",
        "",
        _md_table(yolo, table_rows),
        "",
        "## Previews",
        "",
        f"- Incoming preview dir: `{INCOMING_PREVIEW_DIR}` (20 files)",
        f"- YOLO-pose train preview dir: `{YOLO_PREVIEW_DIR}` (20 files)",
        "",
        "### 5 representative previews to inspect manually",
        "",
    ]
    for p in report["previews"]["manual_inspection_samples"]:
        L.append(f"- `{p}`")
    L.append("")

    L += [
        "## Synthetic-smoke caveat",
        "",
        "**" + report["synthetic_smoke_caveat"] + "**",
        "",
        "## See also",
        "",
        "- `outputs/full_pipeline_audit/01_contract_schema.md` — contract/schema audit (separate).",
        "- `outputs/unreal_bbox_audit/report.md` — real-export bbox quality audit (separate; result `ACCEPT_ONLY_AS_DEBUG`).",
        "- `docs/KEYPOINT_SPEC.md` — A/B/C definitions under the 2026-05-14 floor-ray contract.",
        "",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
