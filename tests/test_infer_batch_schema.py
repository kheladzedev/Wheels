"""Schema tests for the per-frame writer in src/infer_batch.py.

These pin the contract:
  - primary <stem>__frame_XXX.json keys are EXACTLY {frame_id, wheels}
  - per-wheel keys are EXACTLY {bbox_xyxy, confidence, points}
  - points keys are EXACTLY {a, b, c_disc_bottom}
  - legacy companion, when opted in, is named *_legacy.json
  - batch_summary.frame_index entries point `json` at the confirmed
    primary and `legacy_json` at the legacy companion (or null)
  - no forbidden fields leak into the confirmed primary

YOLO is not invoked here: we drive `_build_payloads` + `_write_per_frame`
directly with synthetic detections. That keeps the test fast and lets it
run on any CI without GPU / weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from infer_batch import (
    CONFIRMED_FORBIDDEN_TOP_LEVEL,
    CONFIRMED_FORBIDDEN_WHEEL_KEYS,
    _assert_no_forbidden,
    _build_payloads,
    _write_per_frame,
)

CONFIRMED_TOP_KEYS = {"frame_id", "wheels"}
CONFIRMED_WHEEL_KEYS = {"bbox_xyxy", "confidence", "points"}
CONFIRMED_POINT_KEYS = {"a", "b", "c_disc_bottom"}


def _detection(
    *,
    bbox: tuple[float, float, float, float] = (100.0, 200.0, 300.0, 400.0),
    conf: float = 0.9,
) -> dict:
    """Minimal valid detection compatible with build_ar_payload.

    Three visible keypoints — name/visibility shape matches what
    `detections_from_result` produces in production.
    """
    return {
        "class_name": "wheel",
        "bbox": list(bbox),
        "confidence": conf,
        "keypoints": [
            {"xy": [110.0, 380.0], "visibility": 2, "confidence": 0.95},
            {"xy": [290.0, 380.0], "visibility": 2, "confidence": 0.94},
            {"xy": [200.0, 340.0], "visibility": 2, "confidence": 0.93},
        ],
    }


# ---------------------------------------------------------------------------
# _build_payloads — confirmed primary shape
# ---------------------------------------------------------------------------


def test_build_payloads_default_returns_confirmed_only():
    """Without --emit-legacy the second tuple element is None."""
    confirmed, legacy = _build_payloads(
        [_detection()],
        conf=0.25,
        frame_id="frame_0042",
        timestamp=1.5,
        img_size=[640, 480],
        thresholds={"conf": 0.25, "iou": 0.45, "max_det": 20},
        image_field="data/foo.jpg",
        want_legacy=False,
    )
    assert legacy is None
    assert set(confirmed.keys()) == CONFIRMED_TOP_KEYS
    assert confirmed["frame_id"] == "frame_0042"
    assert isinstance(confirmed["wheels"], list)
    assert len(confirmed["wheels"]) == 1


def test_build_payloads_wheel_keys_are_exactly_the_contract():
    confirmed, _ = _build_payloads(
        [_detection()],
        conf=0.25,
        frame_id="f0",
        timestamp=0.0,
        img_size=[640, 480],
        thresholds={"conf": 0.25, "iou": 0.45, "max_det": 20},
        image_field="data/foo.jpg",
        want_legacy=False,
    )
    w = confirmed["wheels"][0]
    assert set(w.keys()) == CONFIRMED_WHEEL_KEYS
    assert set(w["points"].keys()) == CONFIRMED_POINT_KEYS
    assert isinstance(w["bbox_xyxy"], list) and len(w["bbox_xyxy"]) == 4
    assert isinstance(w["confidence"], float)
    for kp_name in CONFIRMED_POINT_KEYS:
        assert isinstance(w["points"][kp_name], list) and len(w["points"][kp_name]) == 2


def test_build_payloads_no_forbidden_fields_in_confirmed():
    """No forbidden field may slip in — top-level or per-wheel."""
    confirmed, _ = _build_payloads(
        [_detection(), _detection(bbox=(500.0, 200.0, 700.0, 400.0), conf=0.8)],
        conf=0.25,
        frame_id="f1",
        timestamp=2.0,
        img_size=[1024, 768],
        thresholds={"conf": 0.25, "iou": 0.45, "max_det": 20},
        image_field="data/bar.jpg",
        want_legacy=False,
    )
    for k in CONFIRMED_FORBIDDEN_TOP_LEVEL:
        assert k not in confirmed
    for w in confirmed["wheels"]:
        for k in CONFIRMED_FORBIDDEN_WHEEL_KEYS:
            assert k not in w


def test_build_payloads_emit_legacy_returns_legacy_with_meta():
    """When want_legacy=True the legacy companion carries the meta."""
    confirmed, legacy = _build_payloads(
        [_detection()],
        conf=0.25,
        frame_id="f-legacy",
        timestamp=3.0,
        img_size=[1920, 1080],
        thresholds={"conf": 0.25, "iou": 0.45, "max_det": 20},
        image_field="data/abc.jpg",
        want_legacy=True,
    )
    assert legacy is not None
    assert legacy["image"] == "data/abc.jpg"
    assert legacy["image_size"] == [1920, 1080]
    assert legacy["thresholds"]["conf"] == 0.25
    # Confirmed payload still untouched and contract-shaped.
    assert set(confirmed.keys()) == CONFIRMED_TOP_KEYS


# ---------------------------------------------------------------------------
# _assert_no_forbidden — the runtime guard
# ---------------------------------------------------------------------------


def test_assert_no_forbidden_passes_clean_payload():
    payload = {
        "frame_id": "f",
        "wheels": [
            {
                "bbox_xyxy": [0.0, 0.0, 1.0, 1.0],
                "confidence": 0.5,
                "points": {"a": [0, 0], "b": [1, 1], "c_disc_bottom": [0.5, 0.9]},
            }
        ],
    }
    # No raise.
    _assert_no_forbidden(payload, source_label="test")


@pytest.mark.parametrize("bad_key", CONFIRMED_FORBIDDEN_TOP_LEVEL)
def test_assert_no_forbidden_top_level(bad_key: str):
    payload = {"frame_id": "f", "wheels": [], bad_key: "leak"}
    with pytest.raises(AssertionError, match="forbidden"):
        _assert_no_forbidden(payload, source_label="test")


@pytest.mark.parametrize("bad_key", CONFIRMED_FORBIDDEN_WHEEL_KEYS)
def test_assert_no_forbidden_wheel_level(bad_key: str):
    payload = {
        "frame_id": "f",
        "wheels": [
            {
                "bbox_xyxy": [0, 0, 1, 1],
                "confidence": 0.5,
                "points": {"a": [0, 0], "b": [1, 1], "c_disc_bottom": [0.5, 0.9]},
                bad_key: "leak",
            }
        ],
    }
    with pytest.raises(AssertionError, match="forbidden"):
        _assert_no_forbidden(payload, source_label="test")


def test_assert_no_forbidden_rejects_forbidden_substrings_at_wheel_level():
    payload = {
        "frame_id": "f",
        "wheels": [
            {
                "bbox_xyxy": [0, 0, 1, 1],
                "confidence": 0.5,
                "points": {"a": [0, 0], "b": [1, 1], "c_disc_bottom": [0.5, 0.9]},
                "ransac_residual": 0.1,
            }
        ],
    }
    with pytest.raises(AssertionError, match="ransac"):
        _assert_no_forbidden(payload, source_label="test")


def test_assert_no_forbidden_rejects_forbidden_substrings_nested_in_points():
    payload = {
        "frame_id": "f",
        "wheels": [
            {
                "bbox_xyxy": [0, 0, 1, 1],
                "confidence": 0.5,
                "points": {
                    "a": {"xy": [0, 0], "depth": 1.0},
                    "b": [1, 1],
                    "c_disc_bottom": [0.5, 0.9],
                },
            }
        ],
    }
    with pytest.raises(AssertionError, match="depth"):
        _assert_no_forbidden(payload, source_label="test")


# ---------------------------------------------------------------------------
# _write_per_frame — file layout
# ---------------------------------------------------------------------------


def _make_confirmed() -> dict:
    return {
        "frame_id": "frame_0007",
        "wheels": [
            {
                "bbox_xyxy": [10.0, 20.0, 110.0, 120.0],
                "confidence": 0.91,
                "points": {
                    "a": [15.0, 110.0],
                    "b": [105.0, 110.0],
                    "c_disc_bottom": [60.0, 115.0],
                },
            }
        ],
    }


def _make_legacy_with_meta() -> dict:
    return {
        "frame_id": "frame_0007",
        "wheels": [
            {
                "wheel_bbox": [10.0, 20.0, 110.0, 120.0],
                "confidence": 0.91,
                "keypoints": [
                    {
                        "xy": [15.0, 110.0],
                        "visibility": 2,
                        "confidence": 0.9,
                        "name": "rim_left",
                    },
                    {
                        "xy": [105.0, 110.0],
                        "visibility": 2,
                        "confidence": 0.9,
                        "name": "rim_right",
                    },
                    {
                        "xy": [60.0, 115.0],
                        "visibility": 2,
                        "confidence": 0.9,
                        "name": "disc_bottom",
                    },
                ],
                "warnings": [],
            }
        ],
        "stats": {"n_wheels": 1},
        "image": "data/foo.jpg",
        "image_size": [640, 480],
        "thresholds": {"conf": 0.25, "iou": 0.45, "max_det": 20},
    }


def test_write_per_frame_primary_only(tmp_path: Path):
    """Without legacy, only the confirmed primary file is written."""
    primary, legacy_path = _write_per_frame(
        tmp_path,
        stem="src",
        frame_index=7,
        confirmed_payload=_make_confirmed(),
        legacy_payload=None,
    )
    assert primary.name == "src__frame_000007.json"
    assert legacy_path is None
    assert (tmp_path / "src__frame_000007.json").exists()
    assert not (tmp_path / "src__frame_000007_legacy.json").exists()
    assert not (tmp_path / "src__frame_000007_target.json").exists()


def test_write_per_frame_with_legacy_companion(tmp_path: Path):
    """With legacy companion enabled, both files exist with the right names."""
    primary, legacy_path = _write_per_frame(
        tmp_path,
        stem="src",
        frame_index=7,
        confirmed_payload=_make_confirmed(),
        legacy_payload=_make_legacy_with_meta(),
    )
    assert primary.name == "src__frame_000007.json"
    assert legacy_path is not None
    assert legacy_path.name == "src__frame_000007_legacy.json"
    # No _target.json is ever produced under the new policy.
    assert not (tmp_path / "src__frame_000007_target.json").exists()


def test_write_per_frame_primary_file_content_matches_contract(tmp_path: Path):
    """Round-trip the confirmed payload through disk; keys stay exact."""
    primary, _ = _write_per_frame(
        tmp_path,
        stem="src",
        frame_index=0,
        confirmed_payload=_make_confirmed(),
        legacy_payload=_make_legacy_with_meta(),
    )
    loaded = json.loads(primary.read_text(encoding="utf-8"))
    assert set(loaded.keys()) == CONFIRMED_TOP_KEYS
    for w in loaded["wheels"]:
        assert set(w.keys()) == CONFIRMED_WHEEL_KEYS
        assert set(w["points"].keys()) == CONFIRMED_POINT_KEYS


def test_write_per_frame_legacy_file_carries_meta(tmp_path: Path):
    """The legacy companion is allowed to carry image/image_size/thresholds —
    that's the whole point of the debug file."""
    _, legacy_path = _write_per_frame(
        tmp_path,
        stem="src",
        frame_index=0,
        confirmed_payload=_make_confirmed(),
        legacy_payload=_make_legacy_with_meta(),
    )
    assert legacy_path is not None
    loaded = json.loads(legacy_path.read_text(encoding="utf-8"))
    assert "image" in loaded
    assert "image_size" in loaded
    assert "thresholds" in loaded
    # Wheel shape inside the legacy file is the *legacy* shape.
    assert "wheel_bbox" in loaded["wheels"][0]
    assert "keypoints" in loaded["wheels"][0]


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_emit_legacy_flag_exists_on_parser(monkeypatch):
    """Sanity check: argparse exposes --emit-legacy."""
    from infer_batch import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "infer_batch.py",
            "--source",
            "x",
            "--model",
            "y.pt",
            "--out-dir",
            "z",
            "--emit-legacy",
        ],
    )
    ns = parse_args()
    assert ns.emit_legacy is True


def test_target_schema_flag_is_removed_from_parser(monkeypatch):
    from infer_batch import parse_args

    monkeypatch.setattr(
        "sys.argv",
        [
            "infer_batch.py",
            "--source",
            "x",
            "--model",
            "y.pt",
            "--out-dir",
            "z",
            "--target-schema",
        ],
    )
    with pytest.raises(SystemExit):
        parse_args()
