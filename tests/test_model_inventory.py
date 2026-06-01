from __future__ import annotations

import json

from src.model_inventory import build_inventory, render_markdown


def test_model_inventory_links_eval_reports_to_artifacts(tmp_path):
    runs_root = tmp_path / "runs" / "pose"
    run = runs_root / "wheel_a"
    weights = run / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"pt")
    (weights / "best.onnx").write_bytes(b"onnx")
    data_config = tmp_path / "configs" / "data.yaml"
    data_config.parent.mkdir()
    data_config.write_text("path: data\n", encoding="utf-8")
    (run / "args.yaml").write_text(
        "\n".join(
            [
                "task: pose",
                "mode: train",
                "name: wheel_a",
                "model: yolo11n-pose.pt",
                f"data: {data_config}",
                "epochs: 5",
                "batch: 2",
                "imgsz: 640",
            ]
        ),
        encoding="utf-8",
    )

    eval_root = tmp_path / "outputs" / "eval"
    eval_root.mkdir(parents=True)
    model_path = weights / "best.pt"
    (eval_root / "wheel_a.json").write_text(
        json.dumps(
            {
                "model": str(model_path),
                "data": str(data_config),
                "metrics_bbox": {"mAP50": 0.9, "mAP50_95": 0.8},
                "oks": {"mean": 0.85},
                "rates": {"false_negative_rate": 0.1, "false_positive_rate": 0.2},
                "counts": {"gt_wheels": 10, "pred_wheels_above_conf": 11, "matched": 9},
            }
        ),
        encoding="utf-8",
    )

    deployment_root = tmp_path / "exports"
    deployment_root.mkdir()
    (deployment_root / "model.tflite").write_bytes(b"tflite")
    (deployment_root / "model.mlmodel").write_bytes(b"coreml")

    inventory = build_inventory(runs_root, eval_root, model_path, deployment_root)

    assert inventory["counts"]["train_runs"] == 1
    assert inventory["counts"]["run_artifacts"] == 2
    assert inventory["counts"]["deployment_artifacts"] == 2
    assert inventory["counts"]["artifacts"] == 4
    assert inventory["counts"]["tflite_artifacts"] == 1
    assert inventory["counts"]["coreml_artifacts"] == 1
    assert inventory["counts"]["eval_reports"] == 1
    assert inventory["runs"][0]["eval_reports"][0]["bbox_mAP50"] == 0.9
    assert inventory["champion_run"]["name"] == "wheel_a"


def test_model_inventory_reports_missing_lineage_warning(tmp_path):
    runs_root = tmp_path / "runs"
    run = runs_root / "wheel_b"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"pt")
    (run / "args.yaml").write_text(
        "task: pose\nmode: train\nname: wheel_b\nmodel: missing.pt\ndata: missing.yaml\n",
        encoding="utf-8",
    )
    eval_root = tmp_path / "eval"
    eval_root.mkdir()

    inventory = build_inventory(runs_root, eval_root, run / "weights" / "best.pt", tmp_path / "missing_exports")
    markdown = render_markdown(inventory)

    warnings = inventory["runs"][0]["warnings"]
    assert "source_model_missing:missing.pt" in warnings
    assert "data_config_missing:missing.yaml" in warnings
    assert "source_model_missing:missing.pt" in markdown
