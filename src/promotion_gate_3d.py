"""3D promotion gate — the disc-height acceptance slice of goal item #1.

Bridges the new 3D-eval harness (``src/eval3d_report.py``) to the model
promotion decision (alongside the 2D KPIs in
``src/model_selection_audit.py``). A candidate model may be promoted to
production only if its disc-height reconstruction meets the budget
(< 3 cm accept, < 1 cm target — 3D error budget still open,
``docs/OPEN_QUESTIONS_AR_SPEC.md`` §9) AND the report comes from a
trusted real capture.

LOAD-BEARING INVARIANT: a report whose ``gate_status`` is not ``"gate"``
(synthetic round-trip, or a real capture whose UE pose convention has
not been verified) returns **insufficient_evidence** — it can never
produce a green production pass. This is what keeps "validates plumbing
on synthetic data" from ever masquerading as "model is good" (repo rule;
[[feedback_stop_hook_impossible_goal]]).

This module defines and enforces the gate. Executing it on a real
candidate still requires real labelled data + a clean UE export (the
upstream blockers, ``docs/EXPORT_PARITY_AUDIT.md``); until then the gate
correctly reports insufficient evidence for every synthetic report.

Usage::

    python src/promotion_gate_3d.py \\
        --candidate outputs/eval3d/candidate_report.json \\
        --champion  outputs/eval3d/champion_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from production_gate import GateItem, read_json

ACCEPT_CM = 3.0
TARGET_CM = 1.0


def evaluate_3d_acceptance(
    report: dict,
    accept_cm: float = ACCEPT_CM,
    target_cm: float = TARGET_CM,
) -> GateItem:
    """Score one eval3d report against the disc-height budget.

    Returns a :class:`production_gate.GateItem`. Never passes on a
    synthetic / unverified-pose report.
    """
    name = "eval3d_disc_height"
    if not report:
        return GateItem(name, False, "fail", "missing or empty eval3d report")

    gate_status = report.get("gate_status")
    if gate_status != "gate":
        return GateItem(
            name,
            False,
            "insufficient_evidence",
            f"source not a trusted real gate (gate_status={gate_status!r}, "
            f"source={report.get('source')!r}); synthetic/unverified reports "
            f"validate plumbing only, never model quality",
        )

    n_est = int(report.get("n_sigma_estimable", 0))
    if n_est <= 0:
        return GateItem(
            name,
            False,
            "production_fail",
            "no sigma-estimable scenes (need >= 2 frames/scene)",
        )

    median = float(report.get("sigma_cm", {}).get("median", float("inf")))
    pass_accept = report.get("acceptance", {}).get("pass_accept") is True
    pass_target = report.get("acceptance", {}).get("pass_target") is True
    ok = pass_accept and median < accept_cm
    detail = (
        f"median_sigma={median:.3f}cm accept(<{accept_cm})={ok} "
        f"target(<{target_cm})={pass_target} scenes_estimable={n_est}"
    )
    return GateItem(name, ok, "production_fail" if not ok else "pass", detail)


def compare_candidate_vs_champion(
    candidate: dict,
    champion: dict,
    accept_cm: float = ACCEPT_CM,
) -> dict:
    """Decide whether a candidate may be promoted over the champion on 3D.

    Promote only if the candidate passes 3D acceptance (which requires a
    trusted real gate) AND its median disc-height sigma is no worse than
    the champion's. 2D KPI no-regression stays the responsibility of
    ``src/model_selection_audit.py``; this gate adds the 3D dimension.
    """
    cand_item = evaluate_3d_acceptance(candidate, accept_cm=accept_cm)
    champ_item = evaluate_3d_acceptance(champion, accept_cm=accept_cm)

    cand_sigma = float(candidate.get("sigma_cm", {}).get("median", float("inf")))
    champ_sigma = float(champion.get("sigma_cm", {}).get("median", float("inf")))

    if not cand_item.ok:
        promote = False
        reason = (
            f"candidate fails 3D acceptance: {cand_item.severity} — {cand_item.detail}"
        )
    elif cand_sigma > champ_sigma:
        promote = False
        reason = (
            f"candidate disc-height sigma {cand_sigma:.3f}cm worse than "
            f"champion {champ_sigma:.3f}cm"
        )
    else:
        promote = True
        reason = (
            f"candidate sigma {cand_sigma:.3f}cm <= champion {champ_sigma:.3f}cm "
            f"and meets budget"
        )

    return {
        "promote": promote,
        "reason": reason,
        "candidate_sigma": cand_sigma,
        "champion_sigma": champ_sigma,
        "candidate_ok": cand_item.ok,
        "champion_ok": champ_item.ok,
        "candidate_item": cand_item.__dict__,
        "champion_item": champ_item.__dict__,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--candidate", required=True, type=Path, help="candidate eval3d report json"
    )
    p.add_argument(
        "--champion", type=Path, help="champion eval3d report json (optional)"
    )
    p.add_argument(
        "--out", type=Path, default=Path("outputs/eval3d/promotion_gate_3d.json")
    )
    p.add_argument("--accept-cm", type=float, default=ACCEPT_CM)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    candidate = read_json(args.candidate)
    cand_item = evaluate_3d_acceptance(candidate, accept_cm=args.accept_cm)
    out: dict = {"candidate": cand_item.__dict__}
    if args.champion:
        champion = read_json(args.champion)
        out["comparison"] = compare_candidate_vs_champion(
            candidate, champion, accept_cm=args.accept_cm
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    verdict = "PASS" if cand_item.ok else cand_item.severity.upper()
    print(f"[promotion_gate_3d] {verdict}: {cand_item.detail} -> {args.out}")
    # exit non-zero when the candidate does not pass, so CI can gate on it
    return 0 if cand_item.ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
