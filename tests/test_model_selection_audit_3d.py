"""3D disc-height acceptance dimension of the model-selection audit.

These pin goal item #1's 3D "done" spec as wired into
``build_audit``: the criterion is delegated to
``promotion_gate_3d.evaluate_3d_acceptance`` (single source of truth) and
the load-bearing invariant holds — synthetic / GT-2D / unverified reports
can never satisfy it, only a trusted real *model* gate can.
"""

from __future__ import annotations

from src.model_selection_audit import build_audit, render_markdown


CHAMPION = "runs/pose/champion/weights/best.pt"
ANCHOR_DATA = "configs/anchor.yaml"
REAL_DATA = "configs/real.yaml"


def eval_report(*, path, model, data, map50, oks, fn, fp, matched) -> dict:
    return {
        "path": path,
        "model": model,
        "data": data,
        "bbox_mAP50": map50,
        "bbox_mAP50_95": map50 - 0.1,
        "oks_mean": oks,
        "fn_rate": fn,
        "fp_rate": fp,
        "gt_wheels": 20,
        "pred_wheels_above_conf": 20,
        "matched": matched,
    }


def inventory(*reports: dict) -> dict:
    return {
        "champion": CHAMPION,
        "champion_run": {"name": "champion"},
        "runs": [
            {
                "name": "champion",
                "run_dir": "runs/pose/champion",
                "data": ANCHOR_DATA,
                "artifacts": [{"path": CHAMPION, "kind": "pt"}],
            },
            {
                "name": "candidate",
                "run_dir": "runs/pose/candidate",
                "data": "configs/candidate.yaml",
                "artifacts": [
                    {"path": "runs/pose/candidate/weights/best.pt", "kind": "pt"}
                ],
            },
        ],
        "eval_reports": list(reports),
    }


def champion_anchor() -> dict:
    return eval_report(
        path="outputs/eval/champion_anchor.json",
        model=CHAMPION,
        data=ANCHOR_DATA,
        map50=0.70,
        oks=0.88,
        fn=0.20,
        fp=0.10,
        matched=16,
    )


def champion_real() -> dict:
    return eval_report(
        path="outputs/eval/champion_real.json",
        model=CHAMPION,
        data=REAL_DATA,
        map50=0.90,
        oks=0.88,
        fn=0.05,
        fp=0.10,
        matched=18,
    )


def _clean_inventory() -> dict:
    return inventory(champion_anchor(), champion_real())


def _passing_real_gate_report() -> dict:
    # what a real model gate would look like once data + a floor-ray export
    # land: real source, sigma and GT-error both inside budget.
    return {
        "source": "real",
        "gate_status": "gate",
        "points_source": "model_prediction",
        "n_sigma_estimable": 4,
        "sigma_cm": {"median": 0.8, "p95": 1.2, "max": 1.5},
        "height_error_cm": {"median": 1.1, "p95": 2.0, "max": 2.4},
        "acceptance": {"pass_accept": True, "pass_target": True},
    }


def test_no_report_is_insufficient_evidence_and_does_not_fail_2d_audit():
    audit = build_audit(_clean_inventory(), champion=CHAMPION, anchor_data=ANCHOR_DATA, real_data=REAL_DATA)
    assert audit["ok"] is True  # clean 2D selection unaffected
    block = audit["disc_height_3d"]
    assert block["ok"] is False
    assert block["status"] == "insufficient_evidence"
    assert audit["promotion_blocked_on_3d"] == []
    assert audit.get("schema_version", 0) >= 2


def test_informational_real_geometry_gt2d_report_cannot_pass():
    # exactly the v0_2 real-geometry-but-GT-2D report: real camera, but the
    # points are ground truth, not model output -> never a model gate.
    report = {
        "source": "real_geometry_gt2d",
        "gate_status": "informational",
        "points_source": "ue_ground_truth",
        "n_sigma_estimable": 4,
        "sigma_cm": {"median": 0.55},
        "height_error_cm": {"median": 27.0},
        "acceptance": {"pass_accept": False, "pass_target": False},
    }
    audit = build_audit(_clean_inventory(), champion=CHAMPION, anchor_data=ANCHOR_DATA, real_data=REAL_DATA, eval3d_report=report)
    block = audit["disc_height_3d"]
    assert block["ok"] is False
    assert block["status"] == "insufficient_evidence"
    assert block["gate_status"] == "informational"


def test_passing_real_model_gate_satisfies_3d_dimension():
    audit = build_audit(_clean_inventory(), champion=CHAMPION, anchor_data=ANCHOR_DATA, real_data=REAL_DATA, eval3d_report=_passing_real_gate_report())
    block = audit["disc_height_3d"]
    assert block["ok"] is True
    assert block["status"] == "pass"
    assert audit["promotion_blocked_on_3d"] == []
    assert "3D Disc-Height Acceptance" in render_markdown(audit)


def test_failing_real_model_gate_blocks_2d_promotion_candidate():
    # a candidate that beats the champion on 2D, but the supplied 3D report
    # fails -> the candidate is listed as promotion_blocked_on_3d.
    candidate_model = "runs/pose/candidate/weights/best.pt"
    cand_anchor = eval_report(
        path="outputs/eval/cand_anchor.json",
        model=candidate_model,
        data=ANCHOR_DATA,
        map50=0.71,
        oks=0.89,
        fn=0.19,
        fp=0.09,
        matched=17,
    )
    cand_real = eval_report(
        path="outputs/eval/cand_real.json",
        model=candidate_model,
        data=REAL_DATA,
        map50=0.91,
        oks=0.89,
        fn=0.05,
        fp=0.09,
        matched=18,
    )
    inv = inventory(champion_anchor(), champion_real(), cand_anchor, cand_real)

    failing = {
        "source": "real",
        "gate_status": "gate",
        "points_source": "model_prediction",
        "n_sigma_estimable": 4,
        "sigma_cm": {"median": 0.5},
        "height_error_cm": {"median": 9.0},
        "acceptance": {"pass_accept": False, "pass_target": False},
    }
    audit = build_audit(inv, champion=CHAMPION, anchor_data=ANCHOR_DATA, real_data=REAL_DATA, eval3d_report=failing)
    assert candidate_model in audit["promotion_recommended"]  # 2D says promote
    assert audit["disc_height_3d"]["ok"] is False
    assert candidate_model in audit["promotion_blocked_on_3d"]  # 3D blocks it
