from __future__ import annotations

from pathlib import Path

from scripts.build_external_evidence_handoff_bundle import DEFAULT_BUNDLE_ARTIFACTS
from src.release_integrity import DEFAULT_REQUIRED_ARTIFACTS


HARNESS_README = Path("ar_holdout_harness/README.md")
HARNESS_WRITER = Path("ar_holdout_harness/ArHoldoutAnnotationWriter.kt")


def test_ar_holdout_harness_documents_production_holdout_flow():
    text = HARNESS_README.read_text(encoding="utf-8")

    assert "data/incoming/ar_device_holdout/" in text
    assert "images/<frame_id>.jpg" in text
    assert "annotations/<frame_id>.json" in text
    assert "metadata/provenance.json" in text
    assert "src/evaluate_ar_holdout.py" in text
    assert "human_reviewed" in text


def test_ar_holdout_writer_emits_keypoint_dataset_contract_keys():
    text = HARNESS_WRITER.read_text(encoding="utf-8")

    assert "class ArHoldoutAnnotationWriter" in text
    assert 'DEFAULT_ROOT_NAME = "ar_device_holdout"' in text
    assert 'SOURCE_TYPE_ANDROID_AR_DEVICE_HUMAN_LABELLED = "android_ar_device_human_labelled"' in text
    assert 'LABEL_TYPE_HUMAN_REVIEWED = "human_reviewed"' in text
    assert '"schema_version"' in text
    assert '"frame_id"' in text
    assert '"image"' in text
    assert '"wheels"' in text
    assert '"bbox_xyxy"' in text
    assert '"c_disc_bottom"' in text
    assert '"provenance.json"' in text
    assert "annotator and reviewer must be different" in text
    assert "captureDateUtc must be a real UTC date" in text
    assert "must not be in the future" in text
    assert "LocalDate.now(ZoneOffset.UTC)" in text
    assert "LocalDate.parse(value)" in text
    assert "imageFileName must be a filename, not a path" in text
    assert "SUPPORTED_IMAGE_EXTENSIONS" in text


def test_ar_holdout_harness_is_in_handoff_and_release_sets():
    assert str(HARNESS_README) in DEFAULT_BUNDLE_ARTIFACTS
    assert str(HARNESS_WRITER) in DEFAULT_BUNDLE_ARTIFACTS
    assert str(HARNESS_README) in DEFAULT_REQUIRED_ARTIFACTS
    assert str(HARNESS_WRITER) in DEFAULT_REQUIRED_ARTIFACTS
