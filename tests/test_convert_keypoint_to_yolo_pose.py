"""Smoke + targeted tests for the Android-plugin → YOLO-pose converter."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from convert_keypoint_incoming_to_yolo_pose import (
    DEFAULT_MAX_SKIP_RATIO,
    DEFAULT_MAX_WARNING_RATIO,
    DROP_REASON_KEYS,
    KEYPOINT_KEYS,
    KEYPOINT_VISIBILITY,
    N_KEYPOINTS,
    assign_splits,
    evaluate_quality_gate,
    format_label_line,
    main as convert_main,
    validate_and_convert_bbox,
    validate_and_convert_points,
)


# ---- bbox validation ----------------------------------------------------


def test_bbox_normalized_to_unit_square():
    yolo, warn = validate_and_convert_bbox([10, 20, 110, 220], 200, 400, "x.jpg")
    assert warn is None
    cx, cy, w, h = yolo
    assert cx == 60 / 200 and cy == 120 / 400
    assert w == 100 / 200 and h == 200 / 400


def test_bbox_outside_image_is_clipped_with_warning():
    yolo, warn = validate_and_convert_bbox([-10, -20, 300, 500], 200, 400, "x.jpg")
    assert yolo is not None
    assert warn and "clipped" in warn


def test_bbox_with_inverted_corners_rejected():
    yolo, warn = validate_and_convert_bbox([100, 100, 50, 50], 200, 400, "x.jpg")
    assert yolo is None
    assert "invalid bbox order" in warn


def test_bbox_non_list_rejected():
    yolo, warn = validate_and_convert_bbox({"x1": 0}, 200, 400, "x.jpg")
    assert yolo is None
    assert "must be a list of 4" in warn


# ---- points validation --------------------------------------------------


def _good_points():
    return {
        "a": [50, 60],
        "b": [150, 60],
        "c_disc_bottom": [100, 120],
    }


def test_points_normalized_and_visibility_two():
    out, warns = validate_and_convert_points(_good_points(), 200, 400, "x.jpg")
    assert warns == []
    assert len(out) == N_KEYPOINTS
    for kx, ky, v in out:
        assert v == KEYPOINT_VISIBILITY == 2
        assert 0.0 <= kx <= 1.0
        assert 0.0 <= ky <= 1.0
    assert out[0] == (50 / 200, 60 / 400, 2)


def test_points_missing_key_rejected():
    bad = _good_points()
    del bad["c_disc_bottom"]
    out, warns = validate_and_convert_points(bad, 200, 400, "x.jpg")
    assert out is None
    assert any("missing keys" in w for w in warns)


def test_points_extra_key_is_warned_not_rejected():
    bad = _good_points()
    bad["d_extra"] = [5, 5]
    out, warns = validate_and_convert_points(bad, 200, 400, "x.jpg")
    assert out is not None
    assert any("unexpected keys" in w for w in warns)


def test_points_outside_image_clipped_with_warning():
    bad = _good_points()
    bad["c_disc_bottom"] = [500, 800]
    out, warns = validate_and_convert_points(bad, 200, 400, "x.jpg")
    assert out is not None
    assert out[2] == (200 / 200, 400 / 400, 2)
    assert any("clipped" in w for w in warns)


def test_points_non_dict_rejected():
    out, warns = validate_and_convert_points([[1, 2]], 200, 400, "x.jpg")
    assert out is None
    assert any("must be a dict" in w for w in warns)


def test_points_xy_not_list_rejected():
    bad = _good_points()
    bad["a"] = "not a list"
    out, warns = validate_and_convert_points(bad, 200, 400, "x.jpg")
    assert out is None
    assert any("must be a list of 2 numbers" in w for w in warns)


def test_points_xy_non_numeric_rejected():
    bad = _good_points()
    bad["b"] = ["x", "y"]
    out, warns = validate_and_convert_points(bad, 200, 400, "x.jpg")
    assert out is None
    assert any("non-numeric" in w for w in warns)


# ---- label line formatting ---------------------------------------------


def test_label_line_has_expected_field_count():
    bbox = (0.5, 0.5, 0.1, 0.1)
    kps = [(0.5, 0.4, 2), (0.5, 0.6, 2), (0.5, 0.65, 2)]
    line = format_label_line(0, bbox, kps)
    parts = line.split()
    assert len(parts) == 5 + N_KEYPOINTS * 3
    assert parts[0] == "0"
    # Each visibility entry must serialize as an int, not a float.
    assert parts[7] == "2" and parts[10] == "2" and parts[13] == "2"


# ---- split assignment ---------------------------------------------------


def test_split_respects_val_ratio():
    images = [Path(f"img_{i:03d}.jpg") for i in range(100)]
    assignment = assign_splits(images, val_ratio=0.2, seed=42)
    n_val = sum(1 for v in assignment.values() if v == "val")
    assert n_val == 20


def test_split_is_deterministic_under_seed():
    images = [Path(f"img_{i:03d}.jpg") for i in range(50)]
    a1 = assign_splits(images, val_ratio=0.2, seed=7)
    a2 = assign_splits(images, val_ratio=0.2, seed=7)
    assert a1 == a2


# ---- end-to-end via main() ---------------------------------------------


def _wheel_with_points(img_w: int, img_h: int) -> dict:
    return {
        "bbox_xyxy": [10, 10, img_w - 10, img_h - 10],
        "points": {
            "a": [20, 30],
            "b": [40, 30],
            "c_disc_bottom": [30, 60],
        },
    }


def _good_anno(stem: str, img_w: int, img_h: int) -> dict:
    return {
        "frame_id": stem,
        "image": f"{stem}.jpg",
        "wheels": [_wheel_with_points(img_w, img_h)],
    }


def _write_image(path: Path, w: int, h: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    ok = cv2.imwrite(str(path), img)
    assert ok, f"cv2.imwrite failed for {path}"


def _write_anno(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_source(tmp_path: Path, sources: dict[str, tuple[int, int]]) -> Path:
    source_root = tmp_path / "incoming" / "android_plugin"
    images_dir = source_root / "images"
    annos_dir = source_root / "annotations"
    for stem, (w, h) in sources.items():
        _write_image(images_dir / f"{stem}.jpg", w, h)
        _write_anno(annos_dir / f"{stem}.json", _good_anno(stem, w, h))
    (source_root / "metadata").mkdir(parents=True, exist_ok=True)
    return source_root


def _run_main(source_root: Path, dataset_root: Path, *extra: str) -> int:
    argv = [
        "convert_keypoint_incoming_to_yolo_pose.py",
        "--source-root",
        str(source_root),
        "--dataset-root",
        str(dataset_root),
        "--overwrite",
        *extra,
    ]
    with patch.object(sys, "argv", argv):
        return convert_main()


def _load_report(dataset_root: Path) -> dict:
    return json.loads(
        (dataset_root / "metadata" / "conversion_report.json").read_text(
            encoding="utf-8"
        )
    )


def test_end_to_end_writes_expected_layout(tmp_path: Path):
    source_root = _make_source(
        tmp_path,
        {f"frame_{i:03d}": (640, 480) for i in range(5)},
    )
    dataset_root = tmp_path / "dataset"

    rc = _run_main(source_root, dataset_root)
    assert rc == 0

    for split in ("train", "val"):
        assert (dataset_root / "images" / split).is_dir()
        assert (dataset_root / "labels" / split).is_dir()
    assert (dataset_root / "metadata" / "split_manifest.json").is_file()
    assert (dataset_root / "metadata" / "conversion_report.json").is_file()

    # Every image must have a sibling label file with one wheel line of 14 fields.
    for img in (dataset_root / "images").rglob("*.jpg"):
        label = dataset_root / "labels" / img.parent.name / f"{img.stem}.txt"
        assert label.exists(), f"missing label for {img}"
        lines = [l for l in label.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        parts = lines[0].split()
        assert len(parts) == 5 + N_KEYPOINTS * 3
        assert parts[0] == "0"
        # bbox in [0,1]
        for v in parts[1:5]:
            f = float(v)
            assert 0.0 <= f <= 1.0
        # visibility flag for A/B/C must be 2
        assert parts[7] == "2" and parts[10] == "2" and parts[13] == "2"


def test_drop_reasons_dict_covers_all_known_keys(tmp_path: Path):
    source_root = _make_source(
        tmp_path,
        {f"frame_{i:03d}": (640, 480) for i in range(3)},
    )
    dataset_root = tmp_path / "dataset"
    rc = _run_main(source_root, dataset_root)
    assert rc == 0

    report = _load_report(dataset_root)
    assert set(report["drop_reasons"].keys()) == set(DROP_REASON_KEYS)
    assert all(v == 0 for v in report["drop_reasons"].values())


def test_missing_annotation_recorded(tmp_path: Path):
    source_root = tmp_path / "incoming" / "android_plugin"
    _write_image(source_root / "images" / "lonely.jpg", 640, 480)
    _write_image(source_root / "images" / "good.jpg", 640, 480)
    _write_anno(source_root / "annotations" / "good.json", _good_anno("good", 640, 480))

    dataset_root = tmp_path / "dataset"
    rc = _run_main(source_root, dataset_root)
    assert rc == 0

    report = _load_report(dataset_root)
    assert report["drop_reasons"]["missing_annotation"] == 1
    assert report["converted"] == 1


def test_invalid_json_recorded(tmp_path: Path):
    source_root = tmp_path / "incoming" / "android_plugin"
    _write_image(source_root / "images" / "broken.jpg", 640, 480)
    _write_image(source_root / "images" / "good.jpg", 640, 480)
    (source_root / "annotations").mkdir(parents=True, exist_ok=True)
    (source_root / "annotations" / "broken.json").write_text(
        "{this is not json", encoding="utf-8"
    )
    _write_anno(source_root / "annotations" / "good.json", _good_anno("good", 640, 480))

    dataset_root = tmp_path / "dataset"
    rc = _run_main(source_root, dataset_root)
    assert rc == 0

    report = _load_report(dataset_root)
    assert report["drop_reasons"]["invalid_json"] == 1
    assert report["converted"] == 1


def test_wheels_not_list_recorded(tmp_path: Path):
    source_root = tmp_path / "incoming" / "android_plugin"
    _write_image(source_root / "images" / "bad.jpg", 640, 480)
    _write_image(source_root / "images" / "good.jpg", 640, 480)
    _write_anno(
        source_root / "annotations" / "bad.json",
        {"frame_id": "bad", "image": "bad.jpg", "wheels": "not a list"},
    )
    _write_anno(source_root / "annotations" / "good.json", _good_anno("good", 640, 480))

    dataset_root = tmp_path / "dataset"
    rc = _run_main(source_root, dataset_root)
    assert rc == 0

    report = _load_report(dataset_root)
    assert report["drop_reasons"]["wheels_not_list"] == 1
    assert report["converted"] == 1


def test_invalid_wheel_is_warned_image_still_emitted(tmp_path: Path):
    """A wheel with malformed bbox is logged as a warning, but the image still
    gets a (possibly empty) label and stays in the split — no silent data loss
    at the image level.
    """
    source_root = tmp_path / "incoming" / "android_plugin"
    _write_image(source_root / "images" / "frame_001.jpg", 640, 480)
    _write_anno(
        source_root / "annotations" / "frame_001.json",
        {
            "frame_id": "frame_001",
            "image": "frame_001.jpg",
            "wheels": [
                # Invalid bbox order — wheel dropped but image kept.
                {
                    "bbox_xyxy": [100, 100, 50, 50],
                    "points": _good_points(),
                }
            ],
        },
    )

    dataset_root = tmp_path / "dataset"
    rc = _run_main(source_root, dataset_root)
    assert rc == 0

    report = _load_report(dataset_root)
    assert report["converted"] == 1
    assert report["wheels"] == 0
    assert any("invalid bbox order" in w for w in report["warnings"])

    # Label file exists but is empty.
    labels = list((dataset_root / "labels").rglob("*.txt"))
    assert len(labels) == 1
    assert labels[0].read_text(encoding="utf-8") == ""


def test_keypoint_keys_constant_is_authoritative():
    """KEYPOINT_KEYS is part of the contract — guard against accidental
    renames that would silently change YOLO label order.
    """
    assert KEYPOINT_KEYS == ("a", "b", "c_disc_bottom")


# ---- edge case: empty wheels list ---------------------------------------


def test_empty_wheels_list_produces_empty_label_file(tmp_path: Path):
    """If an annotation has `wheels: []` (frame with no detections), the
    converter must still copy the image into the split and write an empty
    label file. Empty labels are valid YOLO — they signal "no objects".
    """
    source_root = tmp_path / "incoming" / "android_plugin"
    _write_image(source_root / "images" / "frame_empty.jpg", 640, 480)
    _write_anno(
        source_root / "annotations" / "frame_empty.json",
        {"frame_id": "frame_empty", "image": "frame_empty.jpg", "wheels": []},
    )

    dataset_root = tmp_path / "dataset"
    rc = _run_main(source_root, dataset_root)
    assert rc == 0

    report = _load_report(dataset_root)
    assert report["converted"] == 1
    assert report["wheels"] == 0
    assert report["skipped"] == 0

    images = list((dataset_root / "images").rglob("*.jpg"))
    labels = list((dataset_root / "labels").rglob("*.txt"))
    assert len(images) == 1
    assert len(labels) == 1
    # Empty file (no lines, no trailing newline) — Ultralytics treats this
    # as "image with zero objects" and uses it as a negative sample.
    assert labels[0].read_text(encoding="utf-8") == ""


def test_wheel_missing_points_key_dropped_with_reason(tmp_path: Path):
    """If a wheel lacks the `points` dict entirely, it must be dropped and
    the reason logged in the conversion report (per goal §4).
    """
    source_root = tmp_path / "incoming" / "android_plugin"
    _write_image(source_root / "images" / "frame_001.jpg", 640, 480)
    _write_anno(
        source_root / "annotations" / "frame_001.json",
        {
            "frame_id": "frame_001",
            "image": "frame_001.jpg",
            "wheels": [{"bbox_xyxy": [10, 10, 100, 100]}],  # no `points`
        },
    )

    dataset_root = tmp_path / "dataset"
    rc = _run_main(source_root, dataset_root)
    assert rc == 0

    report = _load_report(dataset_root)
    assert report["converted"] == 1
    assert report["wheels"] == 0
    # The reason must be human-readable AND point at this image.
    assert any(
        "frame_001.jpg" in w and ("points" in w.lower()) for w in report["warnings"]
    ), f"No drop reason for the points-missing wheel: {report['warnings']!r}"


def test_wheel_with_missing_individual_point_key_dropped_with_reason(tmp_path: Path):
    """A wheel with `points` present but missing one of {a, b, c_disc_bottom}
    must be dropped with a reason — partial keypoints would silently
    corrupt training otherwise.
    """
    source_root = tmp_path / "incoming" / "android_plugin"
    _write_image(source_root / "images" / "frame_002.jpg", 640, 480)
    _write_anno(
        source_root / "annotations" / "frame_002.json",
        {
            "frame_id": "frame_002",
            "image": "frame_002.jpg",
            "wheels": [
                {
                    "bbox_xyxy": [10, 10, 100, 100],
                    "points": {"a": [20, 30], "b": [40, 30]},  # c missing
                }
            ],
        },
    )

    dataset_root = tmp_path / "dataset"
    rc = _run_main(source_root, dataset_root)
    assert rc == 0

    report = _load_report(dataset_root)
    assert report["converted"] == 1
    assert report["wheels"] == 0
    assert any(
        "frame_002.jpg" in w and "missing keys" in w for w in report["warnings"]
    ), f"No drop reason for the missing-c wheel: {report['warnings']!r}"


# ---- cross-module regression: AR confirmed schema is unchanged ----------


def test_confirmed_schema_shape_unchanged_after_adapter_work(tmp_path: Path):
    """The dataset adapter has no business touching the AR response schema.
    This regression test pins both shapes side-by-side: run the adapter
    end-to-end on synthetic data, then independently exercise
    `to_confirmed_schema` and confirm its output still matches the
    AR-team-confirmed contract.

    If a future refactor of the adapter accidentally rewires
    `postprocess_wheels.to_confirmed_schema` or its keypoint mapping,
    this test fails first — before we ship a contract-breaking change.
    """
    # 1. Exercise the adapter end-to-end (should not raise, returns 0).
    source_root = _make_source(
        tmp_path, {f"frame_{i:03d}": (640, 480) for i in range(3)}
    )
    dataset_root = tmp_path / "dataset"
    rc = _run_main(source_root, dataset_root)
    assert rc == 0

    # 2. Independently verify `to_confirmed_schema` still produces the
    #    AR-team-confirmed shape. Import is deliberately inside the test
    #    to keep the converter module dependency-free.
    from postprocess_wheels import build_ar_payload, to_confirmed_schema

    detection = {
        "class_name": "wheel",
        "bbox": [10, 20, 110, 220],
        "confidence": 0.91,
        "keypoints": [
            {"xy": [20.0, 200.0], "visibility": 2, "confidence": 0.9},
            {"xy": [100.0, 202.0], "visibility": 2, "confidence": 0.9},
            {"xy": [60.0, 170.0], "visibility": 2, "confidence": 0.88},
        ],
    }
    legacy = build_ar_payload([detection], frame_id="frame_42")
    confirmed = to_confirmed_schema(legacy)

    # Top-level keyset is exactly {frame_id, wheels} — no track_id,
    # timestamp, stats, image, image_size, thresholds, warnings.
    assert set(confirmed.keys()) == {"frame_id", "wheels"}
    assert confirmed["frame_id"] == "frame_42"
    assert len(confirmed["wheels"]) == 1

    # Per-wheel keyset is exactly {bbox_xyxy, confidence, points} —
    # no wheel_bbox / bbox_xywh / keypoints / visibility / per-kp conf.
    w = confirmed["wheels"][0]
    assert set(w.keys()) == {"bbox_xyxy", "confidence", "points"}
    assert w["bbox_xyxy"] == [10.0, 20.0, 110.0, 220.0]
    assert isinstance(w["confidence"], float)

    # Points are exactly {a, b, c_disc_bottom} with pure [x, y] floats.
    assert set(w["points"].keys()) == {"a", "b", "c_disc_bottom"}
    for name in ("a", "b", "c_disc_bottom"):
        xy = w["points"][name]
        assert isinstance(xy, list)
        assert len(xy) == 2
        assert all(isinstance(v, float) for v in xy)


def test_converter_module_does_not_import_ar_response_code():
    """Architectural guard: the dataset converter must not depend on the
    AR response code path (postprocess_wheels / infer_image). If it does,
    a refactor of the adapter could silently break the AR schema.

    Inspecting the source rather than the imported module avoids
    triggering the import-time side effects of the inference modules.
    """
    import convert_keypoint_incoming_to_yolo_pose as conv_mod

    src = Path(conv_mod.__file__).read_text(encoding="utf-8")
    forbidden_imports = (
        "from postprocess_wheels",
        "import postprocess_wheels",
        "from infer_image",
        "import infer_image",
    )
    for forbidden in forbidden_imports:
        assert forbidden not in src, (
            f"convert_keypoint_incoming_to_yolo_pose.py imports "
            f"AR-response code ({forbidden!r}) — keep the adapter "
            f"independent of the inference path."
        )


# ---- quality gate -------------------------------------------------------


def test_quality_gate_defaults_are_documented():
    """Pin the defaults — the goal specifies 5% / 10% and downstream
    callers may rely on these without passing the flags.
    """
    assert DEFAULT_MAX_SKIP_RATIO == 0.05
    assert DEFAULT_MAX_WARNING_RATIO == 0.10


def test_evaluate_quality_gate_passes_when_under_thresholds():
    block = evaluate_quality_gate(
        source_images=100,
        skipped_images=2,
        warnings_count=3,
        max_skip_ratio=0.05,
        max_warning_ratio=0.10,
    )
    assert block["quality_gate"]["passed"] is True
    assert block["quality_gate"]["reasons"] == []
    assert block["skipped_ratio"] == 0.02
    assert block["warnings_ratio"] == 0.03


def test_evaluate_quality_gate_fails_on_skip_ratio():
    block = evaluate_quality_gate(
        source_images=100,
        skipped_images=10,
        warnings_count=0,
        max_skip_ratio=0.05,
        max_warning_ratio=0.10,
    )
    qg = block["quality_gate"]
    assert qg["passed"] is False
    assert any("skipped_ratio" in r for r in qg["reasons"])
    assert all("warnings_ratio" not in r for r in qg["reasons"])


def test_evaluate_quality_gate_fails_on_warning_ratio():
    block = evaluate_quality_gate(
        source_images=100,
        skipped_images=0,
        warnings_count=20,
        max_skip_ratio=0.05,
        max_warning_ratio=0.10,
    )
    qg = block["quality_gate"]
    assert qg["passed"] is False
    assert any("warnings_ratio" in r for r in qg["reasons"])
    assert all("skipped_ratio" not in r for r in qg["reasons"])


def test_evaluate_quality_gate_zero_source_images_is_vacuously_passed():
    """Defensive: source_images=0 must not divide-by-zero. Converter would
    have errored earlier; this test pins the safe-fallback contract.
    """
    block = evaluate_quality_gate(
        source_images=0,
        skipped_images=0,
        warnings_count=0,
        max_skip_ratio=0.05,
        max_warning_ratio=0.10,
    )
    assert block["skipped_ratio"] == 0.0
    assert block["warnings_ratio"] == 0.0
    assert block["quality_gate"]["passed"] is True


# ---- quality gate end-to-end via main() ---------------------------------


def test_good_batch_passes_quality_gate_with_flag(tmp_path: Path):
    """A clean synthetic batch must pass the gate even with
    --fail-on-quality-gate set (no false positives on healthy data).
    """
    source_root = _make_source(
        tmp_path, {f"frame_{i:03d}": (640, 480) for i in range(20)}
    )
    dataset_root = tmp_path / "dataset"
    rc = _run_main(source_root, dataset_root, "--fail-on-quality-gate")
    assert rc == 0

    report = _load_report(dataset_root)
    assert "quality_gate" in report
    assert report["quality_gate"]["passed"] is True
    assert report["quality_gate"]["reasons"] == []
    assert report["skipped_ratio"] == 0.0
    assert report["warnings_ratio"] == 0.0
    # New report keys present alongside legacy ones.
    assert report["source_images"] == 20
    assert report["converted_images"] == 20
    assert report["skipped_images"] == 0
    assert report["warnings_count"] == 0


def test_too_many_skipped_fails_with_fail_on_quality_gate_flag(tmp_path: Path):
    """Build a batch where most images lack annotations → high skip ratio.
    With the flag, converter must exit 1 and still write the report.
    """
    source_root = tmp_path / "incoming" / "android_plugin"
    # 1 good pair, 9 images with no annotations → skip ratio 9/10 = 0.9.
    _write_image(source_root / "images" / "good.jpg", 640, 480)
    _write_anno(source_root / "annotations" / "good.json", _good_anno("good", 640, 480))
    for i in range(9):
        _write_image(source_root / "images" / f"lonely_{i:02d}.jpg", 640, 480)

    dataset_root = tmp_path / "dataset"
    rc = _run_main(source_root, dataset_root, "--fail-on-quality-gate")
    assert rc == 1

    report = _load_report(dataset_root)
    assert report["quality_gate"]["passed"] is False
    assert any("skipped_ratio" in r for r in report["quality_gate"]["reasons"])
    assert report["skipped_images"] == 9
    assert report["source_images"] == 10
    assert report["skipped_ratio"] == 0.9


def test_too_many_warnings_fails_with_fail_on_quality_gate_flag(tmp_path: Path):
    """Build a batch where every wheel has an invalid bbox order →
    every image emits a warning. Warnings ratio = 1.0, gate must fail.
    """
    source_root = tmp_path / "incoming" / "android_plugin"
    # Use 5 images, each with one wheel that has an invalid bbox.
    bad_wheel = {
        "bbox_xyxy": [100, 100, 50, 50],  # invalid order → warning
        "points": _good_points(),
    }
    for i in range(5):
        stem = f"bad_{i:02d}"
        _write_image(source_root / "images" / f"{stem}.jpg", 640, 480)
        _write_anno(
            source_root / "annotations" / f"{stem}.json",
            {"frame_id": stem, "image": f"{stem}.jpg", "wheels": [bad_wheel]},
        )

    dataset_root = tmp_path / "dataset"
    rc = _run_main(source_root, dataset_root, "--fail-on-quality-gate")
    assert rc == 1

    report = _load_report(dataset_root)
    assert report["quality_gate"]["passed"] is False
    assert any("warnings_ratio" in r for r in report["quality_gate"]["reasons"])
    assert report["warnings_count"] >= 5
    assert report["warnings_ratio"] >= 1.0


def test_quality_gate_failure_without_flag_returns_zero(tmp_path: Path):
    """Same broken batch as the skip-ratio test, but without the flag.
    Converter must exit 0 (the report still says gate didn't pass).
    """
    source_root = tmp_path / "incoming" / "android_plugin"
    _write_image(source_root / "images" / "good.jpg", 640, 480)
    _write_anno(source_root / "annotations" / "good.json", _good_anno("good", 640, 480))
    for i in range(9):
        _write_image(source_root / "images" / f"lonely_{i:02d}.jpg", 640, 480)

    dataset_root = tmp_path / "dataset"
    rc = _run_main(source_root, dataset_root)  # no --fail-on-quality-gate
    assert rc == 0
    report = _load_report(dataset_root)
    assert report["quality_gate"]["passed"] is False


def test_report_always_contains_quality_gate_section(tmp_path: Path):
    """Whether the gate passes or fails, whether the flag is set or not —
    `quality_gate` must always be present in the report.
    """
    source_root = _make_source(
        tmp_path, {f"frame_{i:03d}": (640, 480) for i in range(5)}
    )
    dataset_root = tmp_path / "dataset"
    rc = _run_main(source_root, dataset_root)
    assert rc == 0

    report = _load_report(dataset_root)
    assert "quality_gate" in report
    qg = report["quality_gate"]
    assert set(qg.keys()) == {
        "max_skip_ratio",
        "max_warning_ratio",
        "passed",
        "reasons",
    }
    assert isinstance(qg["passed"], bool)
    assert isinstance(qg["reasons"], list)


def test_report_has_all_new_field_names(tmp_path: Path):
    """Goal §2: explicit top-level field names. Legacy aliases coexist."""
    source_root = _make_source(
        tmp_path, {f"frame_{i:03d}": (640, 480) for i in range(5)}
    )
    dataset_root = tmp_path / "dataset"
    rc = _run_main(source_root, dataset_root)
    assert rc == 0

    report = _load_report(dataset_root)
    expected_new = {
        "source_images",
        "converted_images",
        "skipped_images",
        "skipped_ratio",
        "warnings_count",
        "warnings_ratio",
        "quality_gate",
    }
    missing = expected_new - report.keys()
    assert not missing, f"Missing required report fields: {missing}"
    # Legacy aliases still present — backward compat with downstream
    # consumers that expect them.
    assert "converted" in report
    assert "skipped" in report


def test_custom_quality_gate_thresholds_via_cli(tmp_path: Path):
    """If a caller relaxes the gate (e.g. for an exploratory batch),
    --max-skip-ratio=1.0 must allow any skip ratio to pass.
    """
    source_root = tmp_path / "incoming" / "android_plugin"
    _write_image(source_root / "images" / "good.jpg", 640, 480)
    _write_anno(source_root / "annotations" / "good.json", _good_anno("good", 640, 480))
    for i in range(9):
        _write_image(source_root / "images" / f"lonely_{i:02d}.jpg", 640, 480)

    dataset_root = tmp_path / "dataset"
    rc = _run_main(
        source_root,
        dataset_root,
        "--fail-on-quality-gate",
        "--max-skip-ratio",
        "1.0",
        "--max-warning-ratio",
        "1.0",
    )
    assert rc == 0
    report = _load_report(dataset_root)
    assert report["quality_gate"]["passed"] is True
    assert report["quality_gate"]["max_skip_ratio"] == 1.0
    assert report["quality_gate"]["max_warning_ratio"] == 1.0
