"""Audit champion selection and candidate promotion decisions.

The model inventory proves lineage exists. This audit adds the missing
selection policy: every local PyTorch model with an anchor validation
report is compared against the configured champion, and the suite fails
if a better unpromoted candidate appears.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from promotion_gate_3d import evaluate_3d_acceptance

DEFAULT_INVENTORY = Path("outputs/production_audit/model_inventory.json")
DEFAULT_JSON_OUT = Path("outputs/production_audit/model_selection_audit.json")
DEFAULT_MD_OUT = Path("docs/MODEL_SELECTION_AUDIT.md")
DEFAULT_CHAMPION = "runs/pose/wheel_real_v1_self_plus_ue_synthetic_s/weights/best.pt"
DEFAULT_ANCHOR_DATA = "configs/pose_dataset_real_v1_self_plus_ue_synthetic.yaml"
DEFAULT_REAL_DATA = "configs/pose_dataset_real_v1_self.yaml"
REAL_VALIDATION_MINIMUMS = {
    "bbox_mAP50": 0.85,
    "oks_mean": 0.80,
    "fn_rate": 0.10,
}


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _norm_path(value: Any) -> str:
    return str(value or "").replace("\\", "/")


def _metric(
    report: dict[str, Any], key: str, default: float | None = None
) -> float | None:
    try:
        value = report[key]
    except KeyError:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _metrics(report: dict[str, Any] | None) -> dict[str, float | int | None]:
    report = report or {}
    return {
        "bbox_mAP50": _metric(report, "bbox_mAP50"),
        "bbox_mAP50_95": _metric(report, "bbox_mAP50_95"),
        "oks_mean": _metric(report, "oks_mean"),
        "fn_rate": _metric(report, "fn_rate"),
        "fp_rate": _metric(report, "fp_rate"),
        "gt_wheels": report.get("gt_wheels"),
        "pred_wheels_above_conf": report.get("pred_wheels_above_conf"),
        "matched": report.get("matched"),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _find_eval(
    eval_reports: list[dict[str, Any]],
    *,
    model: str,
    data: str,
) -> dict[str, Any] | None:
    model = _norm_path(model)
    data = _norm_path(data)
    candidates = [
        report
        for report in eval_reports
        if _norm_path(report.get("model")) == model
        and _norm_path(report.get("data")) == data
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda report: (
            _metric(report, "bbox_mAP50", -1.0) or -1.0,
            _metric(report, "oks_mean", -1.0) or -1.0,
            -(_metric(report, "fn_rate", 1.0) or 1.0),
        ),
    )


def _model_to_run(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for run in inventory.get("runs", []):
        if not isinstance(run, dict):
            continue
        for artifact in run.get("artifacts", []):
            if isinstance(artifact, dict) and artifact.get("path"):
                mapping[_norm_path(artifact["path"])] = run
    return mapping


def _passes_real_validation(report: dict[str, Any] | None) -> tuple[bool, list[str]]:
    if report is None:
        return False, ["missing_real_only_eval_for_promotion"]
    metrics = _metrics(report)
    failures: list[str] = []
    map50 = metrics["bbox_mAP50"] if isinstance(metrics["bbox_mAP50"], float) else 0.0
    oks = metrics["oks_mean"] if isinstance(metrics["oks_mean"], float) else 0.0
    fn = metrics["fn_rate"] if isinstance(metrics["fn_rate"], float) else 1.0
    if map50 < REAL_VALIDATION_MINIMUMS["bbox_mAP50"]:
        failures.append("real_bbox_mAP50_below_minimum")
    if oks < REAL_VALIDATION_MINIMUMS["oks_mean"]:
        failures.append("real_oks_below_minimum")
    if fn > REAL_VALIDATION_MINIMUMS["fn_rate"]:
        failures.append("real_fn_rate_above_maximum")
    return not failures, failures


def _compare_to_champion(
    candidate: dict[str, Any],
    champion: dict[str, Any],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    comparable = True
    for key, direction in (
        ("bbox_mAP50", "higher"),
        ("oks_mean", "higher"),
        ("fn_rate", "lower"),
        ("fp_rate", "lower"),
        ("matched", "higher"),
    ):
        cand_value = candidate.get(key)
        champ_value = champion.get(key)
        if cand_value is None or champ_value is None:
            failures.append(f"{key}_missing")
            comparable = False
            continue
        if direction == "higher" and cand_value < champ_value:
            failures.append(f"{key}_below_champion")
            comparable = False
        if direction == "lower" and cand_value > champ_value:
            failures.append(f"{key}_above_champion")
            comparable = False
    return comparable, failures


def _build_3d_acceptance(eval3d_report: dict[str, Any] | None) -> dict[str, Any]:
    """The disc-height (3D) acceptance dimension, next to the 2D KPIs.

    This is the executable "done" spec for goal item #1's 3D slice: a
    candidate may only be promoted to production when its disc-height
    reconstruction meets the budget (< 3 cm) AND the report is a trusted
    real *model* gate. Delegates the verdict to
    ``promotion_gate_3d.evaluate_3d_acceptance`` so the criterion lives in
    exactly one place.

    Default (no report) is the steady state today: data-blocked, so
    ``insufficient_evidence``. This NEVER turns a clean 2D selection audit
    red on its own — it records the 3D status so a reviewer (and CI, via
    ``promotion_gate_3d``) sees both dimensions. The load-bearing
    invariant (synthetic / GT-2D / unverified reports can never pass) is
    inherited from ``evaluate_3d_acceptance``.
    """
    if not eval3d_report:
        return {
            "ok": False,
            "status": "insufficient_evidence",
            "severity": "insufficient_evidence",
            "detail": (
                "no eval3d report supplied — data-blocked: needs a floor-ray "
                "correct export + model-predicted A/B/C (see "
                "docs/EVAL3D_AND_3D_LOSS_STATUS.md, docs/EXPORT_PARITY_AUDIT.md)"
            ),
            "gate_status": None,
            "points_source": None,
        }
    item = evaluate_3d_acceptance(eval3d_report)
    return {
        "ok": item.ok,
        "status": "pass" if item.ok else item.severity,
        "severity": item.severity,
        "detail": item.detail,
        "gate_status": eval3d_report.get("gate_status"),
        "points_source": eval3d_report.get("points_source"),
        "sigma_cm": eval3d_report.get("sigma_cm"),
        "height_error_cm": eval3d_report.get("height_error_cm"),
    }


def build_audit(
    inventory: dict[str, Any],
    *,
    champion: str = DEFAULT_CHAMPION,
    anchor_data: str = DEFAULT_ANCHOR_DATA,
    real_data: str = DEFAULT_REAL_DATA,
    eval3d_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    eval_reports = [
        report
        for report in inventory.get("eval_reports", [])
        if isinstance(report, dict)
    ]
    champion = _norm_path(champion)
    anchor_data = _norm_path(anchor_data)
    real_data = _norm_path(real_data)
    model_to_run = _model_to_run(inventory)
    champion_anchor = _find_eval(eval_reports, model=champion, data=anchor_data)
    champion_real = _find_eval(eval_reports, model=champion, data=real_data)
    failures: list[str] = []
    if not inventory:
        failures.append("missing_or_invalid_model_inventory")
    if champion_anchor is None:
        failures.append("missing_champion_anchor_eval")
    if champion_real is None:
        failures.append("missing_champion_real_eval")

    champion_anchor_metrics = _metrics(champion_anchor)
    anchor_reports = [
        report
        for report in eval_reports
        if _norm_path(report.get("data")) == anchor_data
        and _norm_path(report.get("model")).endswith(".pt")
    ]

    candidates: list[dict[str, Any]] = []
    promotion_recommended: list[str] = []
    for report in sorted(
        anchor_reports, key=lambda item: _norm_path(item.get("model"))
    ):
        model = _norm_path(report.get("model"))
        run = model_to_run.get(model, {})
        metrics = _metrics(report)
        real_eval = _find_eval(eval_reports, model=model, data=real_data)
        real_ok, real_failures = _passes_real_validation(real_eval)
        decision_reasons: list[str] = []
        is_champion = model == champion
        beats_champion = False
        if is_champion:
            status = "selected_champion"
            decision_reasons.append("configured_champion")
        elif champion_anchor is None:
            status = "not_evaluable"
            decision_reasons.append("champion_anchor_eval_missing")
        else:
            beats_champion, metric_failures = _compare_to_champion(
                metrics,
                champion_anchor_metrics,
            )
            decision_reasons.extend(metric_failures)
            if not real_ok:
                decision_reasons.extend(real_failures)
            if beats_champion and real_ok:
                status = "promotion_required"
                promotion_recommended.append(model)
                failures.append(f"unpromoted_better_candidate:{model}")
            else:
                status = "not_promoted"
        candidates.append(
            {
                "model": model,
                "run": run.get("name", "n/a"),
                "run_dir": run.get("run_dir", "n/a"),
                "trained_data": run.get("data", "n/a"),
                "anchor_eval": _norm_path(report.get("path")),
                "real_eval": _norm_path(real_eval.get("path")) if real_eval else None,
                "status": status,
                "promotion_ready": status == "promotion_required",
                "anchor_metrics": metrics,
                "real_metrics": _metrics(real_eval) if real_eval else None,
                "decision_reasons": decision_reasons,
            }
        )

    if champion_anchor is not None and not any(
        c["model"] == champion for c in candidates
    ):
        failures.append("champion_not_in_anchor_candidate_set")

    disc_height_3d = _build_3d_acceptance(eval3d_report)
    # A 2D-promotable candidate must also clear the 3D gate before a
    # production promotion is justified. Only enforced when a report is
    # actually supplied, so the data-blocked default never flips the audit.
    promotion_blocked_on_3d = (
        list(promotion_recommended)
        if (eval3d_report is not None and not disc_height_3d["ok"])
        else []
    )

    return {
        "schema_version": 2,
        "ok": not failures,
        "selection_ok": not failures,
        "disc_height_3d": disc_height_3d,
        "promotion_blocked_on_3d": promotion_blocked_on_3d,
        "selected_champion": champion,
        "anchor_data": anchor_data,
        "real_validation_data": real_data,
        "policy": {
            "candidate_scope": "PyTorch .pt models with anchor validation on the configured real+self+UE validation split",
            "promotion_rule": "candidate must be no worse than champion on bbox_mAP50, OKS, FN, FP, and matched count",
            "real_validation_required": True,
            "real_validation_minimums": REAL_VALIDATION_MINIMUMS,
            "three_d_acceptance": (
                "disc-height sigma < 3 cm (target < 1 cm) AND, when 3D GT is "
                "present, disc-height error < 3 cm, from a trusted real model "
                "gate (promotion_gate_3d); synthetic / GT-2D / unverified "
                "reports can never satisfy it"
            ),
            "auto_promote": False,
        },
        "champion_evidence": {
            "anchor_eval": _norm_path(champion_anchor.get("path"))
            if champion_anchor
            else None,
            "real_eval": _norm_path(champion_real.get("path"))
            if champion_real
            else None,
            "anchor_metrics": champion_anchor_metrics if champion_anchor else None,
            "real_metrics": _metrics(champion_real) if champion_real else None,
        },
        "counts": {
            "anchor_candidates": len(candidates),
            "promotion_required": len(promotion_recommended),
        },
        "promotion_recommended": promotion_recommended,
        "failures": failures,
        "candidates": candidates,
    }


def render_markdown(audit: dict[str, Any]) -> str:
    champion = audit.get("selected_champion", "n/a")
    evidence = audit.get("champion_evidence", {})
    anchor_metrics = evidence.get("anchor_metrics") or {}
    real_metrics = evidence.get("real_metrics") or {}
    lines = [
        "# Model Selection Audit",
        "",
        "Machine-readable champion retention and candidate promotion guard.",
        "",
        f"- OK: {audit.get('ok')}",
        f"- Selected champion: `{champion}`",
        f"- Anchor data: `{audit.get('anchor_data', 'n/a')}`",
        f"- Real validation data: `{audit.get('real_validation_data', 'n/a')}`",
        f"- Promotion required: {audit.get('counts', {}).get('promotion_required', 'n/a')}",
        f"- Failures: {', '.join(audit.get('failures', [])) if audit.get('failures') else 'none'}",
        "",
        "## 3D Disc-Height Acceptance",
        "",
        f"- Status: `{(audit.get('disc_height_3d') or {}).get('status', 'n/a')}`",
        f"- Detail: {(audit.get('disc_height_3d') or {}).get('detail', 'n/a')}",
        f"- Promotion blocked on 3D: {', '.join(audit.get('promotion_blocked_on_3d', [])) or 'none'}",
        "",
        "## Champion Evidence",
        "",
        f"- Anchor eval: `{evidence.get('anchor_eval') or 'missing'}`",
        f"- Anchor metrics: mAP50={_fmt(anchor_metrics.get('bbox_mAP50'))}, OKS={_fmt(anchor_metrics.get('oks_mean'))}, FN={_fmt(anchor_metrics.get('fn_rate'))}, FP={_fmt(anchor_metrics.get('fp_rate'))}, matched={_fmt(anchor_metrics.get('matched'))}",
        f"- Real eval: `{evidence.get('real_eval') or 'missing'}`",
        f"- Real metrics: mAP50={_fmt(real_metrics.get('bbox_mAP50'))}, OKS={_fmt(real_metrics.get('oks_mean'))}, FN={_fmt(real_metrics.get('fn_rate'))}, FP={_fmt(real_metrics.get('fp_rate'))}, matched={_fmt(real_metrics.get('matched'))}",
        "",
        "## Candidates",
        "",
        "| Status | Run | Model | Anchor mAP50 | Anchor OKS | FN | FP | Matched | Real eval | Reasons |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for candidate in audit.get("candidates", []):
        metrics = candidate.get("anchor_metrics") or {}
        reasons = ", ".join(candidate.get("decision_reasons") or []) or "none"
        lines.append(
            "| "
            f"{candidate.get('status', 'n/a')} | "
            f"`{candidate.get('run', 'n/a')}` | "
            f"`{candidate.get('model', 'n/a')}` | "
            f"{_fmt(metrics.get('bbox_mAP50'))} | "
            f"{_fmt(metrics.get('oks_mean'))} | "
            f"{_fmt(metrics.get('fn_rate'))} | "
            f"{_fmt(metrics.get('fp_rate'))} | "
            f"{_fmt(metrics.get('matched'))} | "
            f"`{candidate.get('real_eval') or 'missing'}` | "
            f"{reasons} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--champion", default=DEFAULT_CHAMPION)
    parser.add_argument("--anchor-data", default=DEFAULT_ANCHOR_DATA)
    parser.add_argument("--real-data", default=DEFAULT_REAL_DATA)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    parser.add_argument(
        "--eval3d-report",
        type=Path,
        default=None,
        help="optional eval3d disc-height report json (adds the 3D acceptance dimension)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval3d_report = read_json(args.eval3d_report) if args.eval3d_report else None
    audit = build_audit(
        read_json(args.inventory),
        champion=args.champion,
        anchor_data=args.anchor_data,
        real_data=args.real_data,
        eval3d_report=eval3d_report,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(audit), encoding="utf-8")
    print(
        f"ok={audit['ok']} champion={audit['selected_champion']} "
        f"anchor_candidates={audit['counts']['anchor_candidates']} "
        f"promotion_required={audit['counts']['promotion_required']} "
        f"3d={audit['disc_height_3d']['status']}"
    )
    print(f"json={args.json_out}")
    print(f"markdown={args.md_out}")
    return 0 if audit["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
