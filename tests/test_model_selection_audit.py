from __future__ import annotations

from src.model_selection_audit import build_audit, render_markdown


CHAMPION = "runs/pose/champion/weights/best.pt"
ANCHOR_DATA = "configs/anchor.yaml"
REAL_DATA = "configs/real.yaml"


def eval_report(
    *,
    path: str,
    model: str,
    data: str,
    map50: float,
    oks: float,
    fn: float,
    fp: float,
    matched: int,
) -> dict:
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
                "artifacts": [{"path": "runs/pose/candidate/weights/best.pt", "kind": "pt"}],
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


def test_model_selection_rejects_candidate_below_champion():
    candidate = eval_report(
        path="outputs/eval/candidate_anchor.json",
        model="runs/pose/candidate/weights/best.pt",
        data=ANCHOR_DATA,
        map50=0.69,
        oks=0.87,
        fn=0.21,
        fp=0.11,
        matched=15,
    )

    audit = build_audit(
        inventory(champion_anchor(), champion_real(), candidate),
        champion=CHAMPION,
        anchor_data=ANCHOR_DATA,
        real_data=REAL_DATA,
    )

    assert audit["ok"] is True
    assert audit["counts"]["promotion_required"] == 0
    rejected = [c for c in audit["candidates"] if c["model"].endswith("candidate/weights/best.pt")][0]
    assert rejected["status"] == "not_promoted"
    assert "bbox_mAP50_below_champion" in rejected["decision_reasons"]
    assert "missing_real_only_eval_for_promotion" in rejected["decision_reasons"]


def test_model_selection_fails_when_champion_anchor_eval_missing():
    audit = build_audit(
        inventory(champion_real()),
        champion=CHAMPION,
        anchor_data=ANCHOR_DATA,
        real_data=REAL_DATA,
    )

    assert audit["ok"] is False
    assert "missing_champion_anchor_eval" in audit["failures"]


def test_model_selection_flags_unpromoted_better_candidate():
    candidate_model = "runs/pose/candidate/weights/best.pt"
    candidate_anchor = eval_report(
        path="outputs/eval/candidate_anchor.json",
        model=candidate_model,
        data=ANCHOR_DATA,
        map50=0.71,
        oks=0.89,
        fn=0.19,
        fp=0.09,
        matched=17,
    )
    candidate_real = eval_report(
        path="outputs/eval/candidate_real.json",
        model=candidate_model,
        data=REAL_DATA,
        map50=0.91,
        oks=0.89,
        fn=0.05,
        fp=0.09,
        matched=18,
    )

    audit = build_audit(
        inventory(champion_anchor(), champion_real(), candidate_anchor, candidate_real),
        champion=CHAMPION,
        anchor_data=ANCHOR_DATA,
        real_data=REAL_DATA,
    )

    assert audit["ok"] is False
    assert audit["promotion_recommended"] == [candidate_model]
    assert f"unpromoted_better_candidate:{candidate_model}" in audit["failures"]
    promoted = [c for c in audit["candidates"] if c["model"] == candidate_model][0]
    assert promoted["status"] == "promotion_required"
    assert "Model Selection Audit" in render_markdown(audit)
