from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from scripts.build_awe_demo_pack import (
    BADGE_TEXT,
    DemoItem,
    PROVENANCE_LABELS,
    annotate_with_badge,
    build_demo_pack,
    compose_demo_summary,
    draw_ar_mock_overlay,
    iter_model_prediction_frames,
)


def _make_grey_jpg(path: Path, h: int = 480, w: int = 640) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _make_wheel_payload(frame_id: str) -> dict:
    return {
        "frame_id": frame_id,
        "wheels": [
            {
                "bbox_xyxy": [100.0, 300.0, 220.0, 420.0],
                "confidence": 0.9,
                "points": {
                    "a": [110.0, 410.0],
                    "b": [210.0, 410.0],
                    "c_disc_bottom": [160.0, 360.0],
                },
            }
        ],
    }


def test_annotate_with_badge_burns_text_and_changes_pixels():
    image = np.full((200, 600, 3), 200, dtype=np.uint8)
    untouched = image.copy()

    out = annotate_with_badge(image, "model_prediction")

    assert out is image
    assert not np.array_equal(out, untouched)


def test_annotate_with_badge_rejects_unknown_provenance():
    image = np.full((100, 100, 3), 0, dtype=np.uint8)
    try:
        annotate_with_badge(image, "unknown_label")
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("expected ValueError on unknown provenance")


def test_draw_ar_mock_overlay_marks_image_with_badge():
    image = np.full((480, 640, 3), 90, dtype=np.uint8)
    wheels = _make_wheel_payload("frame_00")["wheels"]

    out = draw_ar_mock_overlay(image, wheels)

    assert out.shape == image.shape
    assert not np.array_equal(out, image)


def test_iter_model_prediction_frames_filters_to_complete_triples(tmp_path):
    pred_dir = tmp_path / "demo"
    pred_dir.mkdir()
    json_dir = pred_dir / "json"
    json_dir.mkdir()
    source_dir = tmp_path / "src"
    source_dir.mkdir()

    _make_grey_jpg(pred_dir / "frame_a_pred.jpg")
    (json_dir / "frame_a.json").write_text(
        json.dumps(_make_wheel_payload("frame_a")), encoding="utf-8"
    )
    _make_grey_jpg(source_dir / "frame_a.jpg")

    _make_grey_jpg(pred_dir / "frame_b_pred.jpg")
    (json_dir / "frame_b.json").write_text(
        json.dumps({"frame_id": "frame_b", "wheels": []}), encoding="utf-8"
    )
    _make_grey_jpg(source_dir / "frame_b.jpg")

    _make_grey_jpg(pred_dir / "frame_c_pred.jpg")
    (json_dir / "frame_c.json").write_text(
        json.dumps(_make_wheel_payload("frame_c")), encoding="utf-8"
    )

    triples = iter_model_prediction_frames(pred_dir, json_dir, source_dir, limit=5)

    stems = [tri[0].stem for tri in triples]
    assert stems == ["frame_a_pred"]


def test_compose_demo_summary_has_safety_flags(tmp_path):
    items = [
        DemoItem(
            stem="frame_a",
            provenance="model_prediction",
            overlay_relpath="overlays/frame_a__model_prediction.jpg",
            source_relpath="data/manual_real/images/frame_a.jpg",
            json_relpath="json/frame_a.json",
        ),
    ]

    summary = compose_demo_summary(
        items,
        out_dir=tmp_path,
        contact_sheet=None,
        demo_video=None,
        notes=["test note"],
    )

    assert summary["production_claim"] is False
    assert summary["ar_ready_claim"] is False
    assert summary["schema_changed"] is False
    assert summary["training_performed"] is False
    assert "test note" in summary["notes"]
    assert summary["provenance_labels"] == list(PROVENANCE_LABELS)
    assert summary["items"][0]["provenance"] == "model_prediction"


def test_badge_text_covers_every_provenance_label():
    for label in PROVENANCE_LABELS:
        assert label in BADGE_TEXT
        assert BADGE_TEXT[label]


def test_build_demo_pack_writes_expected_artifacts_end_to_end(tmp_path):
    pred_dir = tmp_path / "demo"
    json_dir = pred_dir / "json"
    source_dir = tmp_path / "src_images"
    ann_dir = tmp_path / "annotations"
    syn_kp_dir = tmp_path / "syn_kp"
    syn_pose_dir = tmp_path / "syn_pose"
    out_dir = tmp_path / "awe_demo"

    for d in (pred_dir, json_dir, source_dir, ann_dir, syn_kp_dir, syn_pose_dir):
        d.mkdir(parents=True, exist_ok=True)

    _make_grey_jpg(pred_dir / "scene_42_pred.jpg")
    (json_dir / "scene_42.json").write_text(
        json.dumps(_make_wheel_payload("scene_42")), encoding="utf-8"
    )
    _make_grey_jpg(source_dir / "scene_42.jpg")
    _make_grey_jpg(ann_dir / "scene_42_preview.jpg")
    _make_grey_jpg(syn_kp_dir / "sample_0000_preview.jpg")
    _make_grey_jpg(syn_pose_dir / "sample_0000_pose_labels.jpg")

    summary = build_demo_pack(
        out_dir=out_dir,
        pred_overlay_dir=pred_dir,
        pred_json_dir=json_dir,
        source_image_dir=source_dir,
        annotation_preview_dir=ann_dir,
        synthetic_keypoint_dir=syn_kp_dir,
        synthetic_pose_dir=syn_pose_dir,
        count=5,
        annotation_count=1,
        synthetic_count=2,
        write_video=False,
    )

    assert (out_dir / "demo_summary.json").is_file()
    assert (out_dir / "overlays" / "scene_42__model_prediction.jpg").is_file()
    assert (out_dir / "overlays" / "scene_42__ar_mock.jpg").is_file()
    assert (out_dir / "json" / "scene_42.json").is_file()
    assert (out_dir / "report" / "contact_sheet.jpg").is_file()

    provenances = [item["provenance"] for item in summary["items"]]
    assert "model_prediction" in provenances
    assert "ar_mock_visualization" in provenances
    assert "annotation_preview" in provenances
    assert "synthetic_smoke" in provenances
    assert summary["production_claim"] is False
    assert summary["ar_ready_claim"] is False

    on_disk = json.loads((out_dir / "demo_summary.json").read_text(encoding="utf-8"))
    assert on_disk["items"] == summary["items"]
