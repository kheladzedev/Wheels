"""Tests for the 3D promotion gate (``src/promotion_gate_3d.py``).

Defines and *enforces* the 3D-acceptance slice of goal item #1 (promote
skipless to production): a candidate may only be promoted if its
disc-height reconstruction, scored by the new harness
(``src/eval3d_report.py``), meets the budget AND comes from a trusted
real capture. The load-bearing invariant: a synthetic / unverified-pose
report (``gate_status != "gate"``) can NEVER pass — it returns
insufficient-evidence, never a green production gate. This makes #1
executable the moment real data + a clean UE export land, without ever
letting synthetic round-trip masquerade as a quality pass.
"""

from __future__ import annotations

import promotion_gate_3d as pg


def _report(gate_status="gate", median=1.5, pass_accept=True, n_est=6, source="real"):
    return {
        "units": "cm",
        "source": source,
        "gate_status": gate_status,
        "n_sigma_estimable": n_est,
        "sigma_cm": {"median": median, "p95": median, "max": median},
        "acceptance": {
            "sigma_accept_cm": 3.0,
            "sigma_target_cm": 1.0,
            "pass_accept": pass_accept,
            "pass_target": median < 1.0,
        },
    }


def test_synthetic_report_can_never_pass():
    item = pg.evaluate_3d_acceptance(
        _report(gate_status="informational", source="synthetic")
    )
    assert item.ok is False
    assert item.severity == "insufficient_evidence"
    assert "synthetic" in item.detail.lower() or "informational" in item.detail.lower()


def test_real_report_meeting_budget_passes():
    item = pg.evaluate_3d_acceptance(_report(median=1.5, pass_accept=True))
    assert item.ok is True


def test_real_report_failing_budget_fails():
    item = pg.evaluate_3d_acceptance(_report(median=4.0, pass_accept=False))
    assert item.ok is False
    assert item.severity == "production_fail"


def test_no_estimable_scenes_fails():
    item = pg.evaluate_3d_acceptance(_report(n_est=0))
    assert item.ok is False


def test_missing_report_fails():
    item = pg.evaluate_3d_acceptance({})
    assert item.ok is False


def test_candidate_beating_champion_promotes():
    champ = _report(median=2.5)
    cand = _report(median=1.2)
    res = pg.compare_candidate_vs_champion(cand, champ)
    assert res["promote"] is True
    assert res["candidate_sigma"] < res["champion_sigma"]


def test_candidate_worse_than_champion_does_not_promote():
    champ = _report(median=1.2)
    cand = _report(median=2.5)
    res = pg.compare_candidate_vs_champion(cand, champ)
    assert res["promote"] is False


def test_synthetic_candidate_never_promotes_even_if_sigma_lower():
    champ = _report(median=2.5)
    cand = _report(gate_status="informational", source="synthetic", median=0.1)
    res = pg.compare_candidate_vs_champion(cand, champ)
    assert res["promote"] is False
    assert (
        "insufficient" in res["reason"].lower() or "synthetic" in res["reason"].lower()
    )
