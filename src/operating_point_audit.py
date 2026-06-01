"""Select a production confidence operating point from real-val sweeps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_REPORT_GLOB = "outputs/production_audit/threshold_conf*_real_val.json"
DEFAULT_JSON_OUT = Path("outputs/production_audit/operating_point_audit.json")
DEFAULT_MD_OUT = Path("docs/OPERATING_POINT_AUDIT.md")
MIN_MAP50 = 0.85
MIN_OKS = 0.80
MAX_FN = 0.10
MAX_FP = 0.15


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def metric(report: dict[str, Any], *keys: str, default: float | None = None) -> float | None:
    cur: Any = report
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    if isinstance(cur, bool):
        return default
    try:
        value = float(cur)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def candidate_from_report(path: Path) -> dict[str, Any]:
    report = read_json(path)
    conf = metric(report, "thresholds", "conf", default=None)
    map50 = metric(report, "metrics_bbox", "mAP50", default=0.0) or 0.0
    oks = metric(report, "oks", "mean", default=0.0) or 0.0
    fn = metric(report, "rates", "false_negative_rate", default=1.0) or 1.0
    fp = metric(report, "rates", "false_positive_rate", default=1.0) or 1.0
    failures: list[str] = []
    if conf is None:
        failures.append("missing_conf_threshold")
    if map50 < MIN_MAP50:
        failures.append("bbox_mAP50_below_minimum")
    if oks < MIN_OKS:
        failures.append("oks_below_minimum")
    if fn > MAX_FN:
        failures.append("false_negative_rate_above_maximum")
    if fp > MAX_FP:
        failures.append("false_positive_rate_above_maximum")
    counts = report.get("counts", {}) if isinstance(report.get("counts"), dict) else {}
    return {
        "path": str(path),
        "conf": conf,
        "ok": not failures,
        "failures": failures,
        "bbox_mAP50": map50,
        "oks_mean": oks,
        "false_negative_rate": fn,
        "false_positive_rate": fp,
        "gt_wheels": counts.get("gt_wheels"),
        "pred_wheels_above_conf": counts.get("pred_wheels_above_conf"),
        "matched": counts.get("matched"),
        "false_positives": counts.get("false_positives"),
        "false_negatives": counts.get("false_negatives"),
    }


def _selection_key(candidate: dict[str, Any]) -> tuple[float, float, float]:
    conf = candidate.get("conf")
    conf_value = float(conf) if isinstance(conf, (int, float)) else float("inf")
    return (
        conf_value,
        -float(candidate.get("bbox_mAP50") or 0.0),
        float(candidate.get("false_negative_rate") or 1.0),
    )


def build_audit(report_paths: list[Path]) -> dict[str, Any]:
    candidates = [candidate_from_report(path) for path in sorted(report_paths)]
    passing = [candidate for candidate in candidates if candidate["ok"]]
    selected = sorted(passing, key=_selection_key)[0] if passing else None
    failures = [] if selected else ["no_threshold_candidate_meets_quality_gates"]
    return {
        "ok": selected is not None,
        "selection_policy": "lowest_confidence_threshold_that_meets_all_quality_gates",
        "thresholds": {
            "min_bbox_mAP50": MIN_MAP50,
            "min_oks_mean": MIN_OKS,
            "max_false_negative_rate": MAX_FN,
            "max_false_positive_rate": MAX_FP,
        },
        "selected": selected,
        "failures": failures,
        "counts": {
            "candidate_reports": len(candidates),
            "passing_candidates": len(passing),
        },
        "candidates": candidates,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    selected = audit.get("selected")
    lines = [
        "# Operating Point Audit",
        "",
        "Real-validation confidence threshold selection for the current champion.",
        "",
        f"- OK: {audit.get('ok')}",
        f"- Policy: {audit.get('selection_policy')}",
    ]
    if isinstance(selected, dict):
        lines.extend(
            [
                f"- Selected report: `{selected.get('path')}`",
                f"- Selected conf: {float(selected.get('conf') or 0.0):.3f}",
                (
                    f"- Selected metrics: mAP50={float(selected.get('bbox_mAP50') or 0.0):.3f}, "
                    f"OKS={float(selected.get('oks_mean') or 0.0):.3f}, "
                    f"FN={float(selected.get('false_negative_rate') or 0.0):.3f}, "
                    f"FP={float(selected.get('false_positive_rate') or 0.0):.3f}"
                ),
            ]
        )
    else:
        lines.append(f"- Failures: {', '.join(audit.get('failures', [])) or 'none'}")
    lines.extend(
        [
            "",
            "| Report | Conf | OK | mAP50 | OKS | FN | FP | Failures |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in audit.get("candidates", []):
        conf = row.get("conf")
        conf_text = f"{float(conf):.3f}" if isinstance(conf, (int, float)) else "n/a"
        lines.append(
            "| "
            f"`{row.get('path')}` | "
            f"{conf_text} | "
            f"{row.get('ok')} | "
            f"{float(row.get('bbox_mAP50') or 0.0):.3f} | "
            f"{float(row.get('oks_mean') or 0.0):.3f} | "
            f"{float(row.get('false_negative_rate') or 0.0):.3f} | "
            f"{float(row.get('false_positive_rate') or 0.0):.3f} | "
            f"{', '.join(row.get('failures', [])) or 'none'} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-glob", default=DEFAULT_REPORT_GLOB)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = sorted(Path(".").glob(args.report_glob))
    audit = build_audit(reports)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(audit), encoding="utf-8")
    selected = audit.get("selected")
    if isinstance(selected, dict):
        print(
            f"ok=True selected_conf={float(selected.get('conf') or 0.0):.3f} "
            f"fp={float(selected.get('false_positive_rate') or 0.0):.3f}"
        )
    else:
        print(f"ok=False failures={audit['failures']}")
    print(f"json={args.json_out}")
    print(f"markdown={args.md_out}")
    return 0 if audit["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
