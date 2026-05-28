from __future__ import annotations

from scripts.create_ar_holdout_provenance_template import build_template
from src.evaluate_ar_holdout import validate_production_provenance


def test_ar_holdout_provenance_template_has_expected_contract():
    template = build_template()

    assert template["source_type"] == "android_ar_device_human_labelled"
    assert template["label_type"] == "human_reviewed"
    assert template["review_status"] == "accepted"
    assert "provenance.json" in template["notes"]


def test_ar_holdout_provenance_template_is_not_production_ready():
    template = build_template()

    failures = validate_production_provenance(template)

    assert any("capture_device" in failure for failure in failures)
