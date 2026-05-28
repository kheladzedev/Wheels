from __future__ import annotations

import argparse
import json

from src.evaluate_ar_holdout import (
    build_converter_cmd,
    build_eval_cmd,
    evaluate_holdout_metrics,
    load_source_provenance,
    source_manifest_sha256,
    validate_production_annotations,
    validate_production_provenance,
    write_dataset_config,
)


def test_evaluate_holdout_metrics_passes_thresholds():
    report = {
        "counts": {"images": 60, "gt_wheels": 100},
        "metrics_bbox": {"mAP50": 0.9},
        "oks": {"mean": 0.85},
        "rates": {"false_negative_rate": 0.05},
    }

    result = evaluate_holdout_metrics(
        report,
        min_map50=0.85,
        min_oks=0.8,
        max_fn=0.1,
        min_images=50,
        min_gt_wheels=80,
    )

    assert result["ok"] is True
    assert result["failures"] == []


def test_evaluate_holdout_metrics_rejects_non_integer_counts():
    report = {
        "counts": {"images": 60.5, "gt_wheels": True},
        "metrics_bbox": {"mAP50": 0.9},
        "oks": {"mean": 0.85},
        "rates": {"false_negative_rate": 0.05},
    }

    result = evaluate_holdout_metrics(
        report,
        min_map50=0.85,
        min_oks=0.8,
        max_fn=0.1,
        min_images=50,
        min_gt_wheels=80,
    )

    assert result["ok"] is False
    assert "count_not_integer:images:60.5" in result["failures"]
    assert "count_not_integer:gt_wheels:True" in result["failures"]


def test_evaluate_holdout_metrics_reports_all_failures():
    report = {
        "counts": {"images": 10, "gt_wheels": 12},
        "metrics_bbox": {"mAP50": 0.5},
        "oks": {"mean": 0.4},
        "rates": {"false_negative_rate": 0.3},
    }

    result = evaluate_holdout_metrics(
        report,
        min_map50=0.85,
        min_oks=0.8,
        max_fn=0.1,
        min_images=50,
        min_gt_wheels=80,
    )

    assert result["ok"] is False
    assert len(result["failures"]) == 5
    assert any(failure.startswith("images:") for failure in result["failures"])
    assert any(failure.startswith("gt_wheels:") for failure in result["failures"])


def test_write_dataset_config_points_to_dataset_root(tmp_path):
    config = tmp_path / "holdout.yaml"
    dataset = tmp_path / "dataset"

    write_dataset_config(config, dataset)

    text = config.read_text(encoding="utf-8")
    assert f"path: {dataset}" in text
    assert "kpt_shape: [3, 3]" in text
    assert "flip_idx: [1, 0, 2]" in text


def test_commands_use_validation_only_holdout(tmp_path):
    args = argparse.Namespace(
        source_root=tmp_path / "incoming",
        dataset_root=tmp_path / "dataset",
        source_name="ar_device_holdout",
        seed=7,
        max_skip_ratio=0.02,
        max_warning_ratio=0.05,
        model=tmp_path / "best.pt",
        config_out=tmp_path / "holdout.yaml",
        device="cpu",
        conf=0.5,
        iou=0.45,
        max_det=20,
        eval_out=tmp_path / "eval.json",
        worst_n=20,
    )

    converter_cmd = build_converter_cmd(args)
    eval_cmd = build_eval_cmd(args)

    assert "--val-ratio" in converter_cmd
    assert "1.0" in converter_cmd
    assert "--fail-on-quality-gate" in converter_cmd
    assert str(args.eval_out) in eval_cmd
    assert "--split" in eval_cmd
    assert "val" in eval_cmd


def test_validate_production_provenance_requires_human_ar_device_source():
    failures = validate_production_provenance(
        {
            "source_name": "synthetic_keypoint_sample",
            "notes": "Not real training data.",
        }
    )

    assert any("source_type" in failure for failure in failures)
    assert any("label_type" in failure for failure in failures)
    assert any("capture_device" in failure for failure in failures)


def test_validate_production_provenance_accepts_reviewed_ar_holdout():
    failures = validate_production_provenance(
        {
            "schema_version": 1,
            "source_type": "android_ar_device_human_labelled",
            "label_type": "human_reviewed",
            "capture_device": "Pixel test device",
            "capture_app_version": "1.2.3",
            "capture_date_utc": "2026-05-27",
            "annotator": "labeler_a",
            "reviewer": "reviewer_b",
            "review_status": "accepted",
        }
    )

    assert failures == []


def test_validate_production_provenance_rejects_placeholder_device():
    failures = validate_production_provenance(
        {
            "schema_version": 1,
            "source_type": "android_ar_device_human_labelled",
            "label_type": "human_reviewed",
            "capture_device": "FILL_ME",
            "capture_app_version": "FILL_ME",
            "capture_date_utc": "FILL_ME_YYYY-MM-DD",
            "annotator": "FILL_ME",
            "reviewer": "FILL_ME",
            "review_status": "accepted",
        }
    )

    assert any("capture_device" in failure for failure in failures)
    assert any("capture_app_version" in failure for failure in failures)
    assert any("capture_date_utc" in failure for failure in failures)
    assert any("annotator" in failure for failure in failures)
    assert any("reviewer" in failure for failure in failures)


def test_validate_production_provenance_rejects_impossible_capture_date():
    failures = validate_production_provenance(
        {
            "schema_version": 1,
            "source_type": "android_ar_device_human_labelled",
            "label_type": "human_reviewed",
            "capture_device": "Pixel test device",
            "capture_app_version": "1.2.3",
            "capture_date_utc": "2026-99-99",
            "annotator": "labeler_a",
            "reviewer": "reviewer_b",
            "review_status": "accepted",
        }
    )

    assert "capture_date_utc must be a real UTC date in YYYY-MM-DD format" in failures


def test_validate_production_provenance_rejects_future_capture_date():
    failures = validate_production_provenance(
        {
            "schema_version": 1,
            "source_type": "android_ar_device_human_labelled",
            "label_type": "human_reviewed",
            "capture_device": "Pixel test device",
            "capture_app_version": "1.2.3",
            "capture_date_utc": "2999-01-01",
            "annotator": "labeler_a",
            "reviewer": "reviewer_b",
            "review_status": "accepted",
        }
    )

    assert "capture_date_utc must be a real UTC date in YYYY-MM-DD format" in failures


def test_validate_production_provenance_requires_independent_review():
    failures = validate_production_provenance(
        {
            "schema_version": 1,
            "source_type": "android_ar_device_human_labelled",
            "label_type": "human_reviewed",
            "capture_device": "Pixel test device",
            "capture_app_version": "1.2.3",
            "capture_date_utc": "2026-05-27",
            "annotator": "same_person",
            "reviewer": "same_person",
            "review_status": "accepted",
        }
    )

    assert any("annotator and reviewer" in failure for failure in failures)


def test_validate_production_provenance_requires_accepted_review_status():
    failures = validate_production_provenance(
        {
            "schema_version": 1,
            "source_type": "android_ar_device_human_labelled",
            "label_type": "human_reviewed",
            "capture_device": "Pixel test device",
            "capture_app_version": "1.2.3",
            "capture_date_utc": "2026-05-27",
            "annotator": "labeler_a",
            "reviewer": "reviewer_b",
        }
    )

    assert "review_status must be accepted, got missing" in failures

    failures = validate_production_provenance(
        {
            "schema_version": 1,
            "source_type": "android_ar_device_human_labelled",
            "label_type": "human_reviewed",
            "capture_device": "Pixel test device",
            "capture_app_version": "1.2.3",
            "capture_date_utc": "2026-05-27",
            "annotator": "labeler_a",
            "reviewer": "reviewer_b",
            "review_status": "approved",
        }
    )

    assert "review_status must be accepted, got approved" in failures


def test_validate_production_provenance_requires_human_reviewed_label_type():
    failures = validate_production_provenance(
        {
            "schema_version": 1,
            "source_type": "android_ar_device_human_labelled",
            "label_type": "human_labelled",
            "capture_device": "Pixel test device",
            "capture_app_version": "1.2.3",
            "capture_date_utc": "2026-05-27",
            "annotator": "labeler_a",
            "reviewer": "reviewer_b",
            "review_status": "accepted",
        }
    )

    assert "label_type must be human_reviewed, got human_labelled" in failures


def test_validate_production_provenance_requires_schema_version():
    failures = validate_production_provenance(
        {
            "source_type": "android_ar_device_human_labelled",
            "label_type": "human_reviewed",
            "capture_device": "Pixel test device",
            "capture_app_version": "1.2.3",
            "capture_date_utc": "2026-05-27",
            "annotator": "labeler_a",
            "reviewer": "reviewer_b",
            "review_status": "accepted",
        }
    )

    assert "schema_version must be 1, got missing" in failures


def test_validate_production_provenance_rejects_boolean_schema_version():
    failures = validate_production_provenance(
        {
            "schema_version": True,
            "source_type": "android_ar_device_human_labelled",
            "label_type": "human_reviewed",
            "capture_device": "Pixel test device",
            "capture_app_version": "1.2.3",
            "capture_date_utc": "2026-05-27",
            "annotator": "labeler_a",
            "reviewer": "reviewer_b",
            "review_status": "accepted",
        }
    )

    assert "schema_version must be 1, got True" in failures


def test_validate_production_annotations_accepts_versioned_holdout_annotation(tmp_path):
    root = tmp_path / "incoming"
    annotations = root / "annotations"
    annotations.mkdir(parents=True)
    (annotations / "frame_0001.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frame_id": "frame_0001",
                "image": "frame_0001.jpg",
                "wheels": [
                    {
                        "bbox_xyxy": [10, 10, 30, 30],
                        "points": {
                            "a": [12, 28],
                            "b": [28, 28],
                            "c_disc_bottom": [20, 24],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert validate_production_annotations(root) == []


def test_validate_production_annotations_requires_schema_version(tmp_path):
    root = tmp_path / "incoming"
    annotations = root / "annotations"
    annotations.mkdir(parents=True)
    (annotations / "frame_0001.json").write_text(
        '{"frame_id":"frame_0001","image":"frame_0001.jpg","wheels":[]}',
        encoding="utf-8",
    )

    failures = validate_production_annotations(root)

    assert "frame_0001.json: schema_version must be 1, got missing" in failures


def test_validate_production_annotations_rejects_boolean_schema_version(tmp_path):
    root = tmp_path / "incoming"
    annotations = root / "annotations"
    annotations.mkdir(parents=True)
    (annotations / "frame_0001.json").write_text(
        json.dumps(
            {
                "schema_version": True,
                "frame_id": "frame_0001",
                "image": "frame_0001.jpg",
                "wheels": [],
            }
        ),
        encoding="utf-8",
    )

    failures = validate_production_annotations(root)

    assert "frame_0001.json: schema_version must be 1, got True" in failures


def test_validate_production_annotations_rejects_invalid_wheel_schema(tmp_path):
    root = tmp_path / "incoming"
    annotations = root / "annotations"
    annotations.mkdir(parents=True)
    (annotations / "frame_0001.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frame_id": "frame_0001",
                "image": "frame_0001.jpg",
                "wheels": [
                    {
                        "bbox_xyxy": [30, 10, 10, 30],
                        "points": {"a": [12, 28], "b": [28, 28], "c_disc_bottom": ["bad", 24]},
                    },
                    "not-a-wheel",
                ],
            }
        ),
        encoding="utf-8",
    )

    failures = validate_production_annotations(root)

    assert "frame_0001.json:wheel[0]: bbox_xyxy must have positive area" in failures
    assert (
        "frame_0001.json:wheel[0]: point c_disc_bottom must be [x, y] finite numbers"
        in failures
    )
    assert "frame_0001.json:wheel[1]: wheel must be an object" in failures


def test_validate_production_annotations_rejects_points_outside_bbox(tmp_path):
    root = tmp_path / "incoming"
    annotations = root / "annotations"
    annotations.mkdir(parents=True)
    (annotations / "frame_0001.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frame_id": "frame_0001",
                "image": "frame_0001.jpg",
                "wheels": [
                    {
                        "bbox_xyxy": [10, 10, 30, 30],
                        "points": {
                            "a": [9, 20],
                            "b": [28, 28],
                            "c_disc_bottom": [20, 31],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    failures = validate_production_annotations(root)

    assert "frame_0001.json:wheel[0]: point a must lie inside bbox_xyxy" in failures
    assert "frame_0001.json:wheel[0]: point c_disc_bottom must lie inside bbox_xyxy" in failures


def test_validate_production_annotations_requires_wheels_array(tmp_path):
    root = tmp_path / "incoming"
    annotations = root / "annotations"
    annotations.mkdir(parents=True)
    (annotations / "frame_0001.json").write_text(
        '{"schema_version":1,"frame_id":"frame_0001","image":"frame_0001.jpg","wheels":{}}',
        encoding="utf-8",
    )

    failures = validate_production_annotations(root)

    assert "frame_0001.json: wheels must be an array" in failures


def test_validate_production_annotations_rejects_frame_and_image_stem_mismatch(tmp_path):
    root = tmp_path / "incoming"
    images = root / "images"
    annotations = root / "annotations"
    images.mkdir(parents=True)
    annotations.mkdir()
    (images / "frame_0001.jpg").write_bytes(b"jpg")
    (images / "frame_0002.jpg").write_bytes(b"jpg")
    (annotations / "frame_0001.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "frame_id": "frame_0002",
                "image": "frame_0002.jpg",
                "wheels": [],
            }
        ),
        encoding="utf-8",
    )

    failures = validate_production_annotations(root)

    assert "frame_0001.json: frame_id must match annotation stem, got frame_0002" in failures
    assert "frame_0001.json: image stem must match annotation stem, got frame_0002.jpg" in failures


def test_load_source_provenance_prefers_explicit_provenance(tmp_path):
    root = tmp_path / "incoming"
    metadata = root / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "source_info.json").write_text('{"source_type":"legacy"}', encoding="utf-8")
    explicit = metadata / "provenance.json"
    explicit.write_text('{"source_type":"android_ar_device_human_labelled"}', encoding="utf-8")

    path, payload = load_source_provenance(root)

    assert path == explicit
    assert payload["source_type"] == "android_ar_device_human_labelled"


def test_source_manifest_sha256_changes_when_holdout_content_changes(tmp_path):
    root = tmp_path / "incoming"
    (root / "images").mkdir(parents=True)
    (root / "annotations").mkdir()
    (root / "metadata").mkdir()
    (root / "images" / "frame.jpg").write_bytes(b"jpg")
    (root / "annotations" / "frame.json").write_text("{}", encoding="utf-8")
    (root / "metadata" / "provenance.json").write_text(
        '{"source_type":"android_ar_device_human_labelled"}',
        encoding="utf-8",
    )

    before = source_manifest_sha256(root)
    (root / "annotations" / "frame.json").write_text('{"changed":true}', encoding="utf-8")
    after = source_manifest_sha256(root)

    assert before != after
