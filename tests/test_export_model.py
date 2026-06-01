"""Tests for the pure helpers in src/export_model.py.

Covers ``compare_detections`` tolerance behaviour, greedy IoU matching
on shuffled detections, and ``pick_sample_image`` fallback to the val
split. Deliberately does NOT call ``YOLO(...)``, ``model.predict(...)``,
or ``model.export(...)`` — the export step is the integration smoke test
in the CLI, not in unit tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from export_model import (
    VALID_FORMATS,
    compare_detections,
    infer_one,
    pick_sample_image,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _det(bbox, conf=0.9, kps=((10, 10), (20, 20), (30, 30))):
    """Build a single detection dict in the comparison-report shape."""
    return {
        "bbox": list(bbox),
        "conf": conf,
        "keypoints": [list(p) for p in kps],
    }


def _result(*detections):
    return {"detections": list(detections)}


def test_export_model_accepts_legacy_coreml_mlmodel_format():
    assert "coreml" in VALID_FORMATS
    assert "mlmodel" in VALID_FORMATS


# ---------------------------------------------------------------------------
# Fakes for infer_one — mirror only the attributes the code touches.
# ---------------------------------------------------------------------------


class _FakeTensor:
    """Mimics the torch tensor API ``infer_one`` uses: ``.cpu().numpy()``.

    ``infer_one`` reads ``result.keypoints.xy[i].cpu().numpy()``; we hand it
    a numpy array dressed up to satisfy that chain without pulling torch in.
    """

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._arr

    def tolist(self) -> list:
        return self._arr.tolist()


class _FakeBox:
    """Mimics one Ultralytics box: ``.xyxy[0].tolist()`` and ``.conf.item()``.

    ``xyxy`` is shaped ``(1, 4)`` per Ultralytics, so we expose a list with one
    entry. ``conf`` is a scalar with an ``.item()`` method.
    """

    def __init__(self, xyxy: list[float], conf: float) -> None:
        self.xyxy = [_FakeTensor(np.asarray(xyxy, dtype=np.float32))]

        class _Scalar:
            def __init__(self, v: float) -> None:
                self._v = v

            def item(self) -> float:
                return float(self._v)

        self.conf = _Scalar(conf)


class _FakeBoxes:
    """Indexable container for ``_FakeBox`` with ``len()`` support."""

    def __init__(self, boxes: list[_FakeBox]) -> None:
        self._boxes = boxes

    def __len__(self) -> int:
        return len(self._boxes)

    def __getitem__(self, i: int) -> _FakeBox:
        return self._boxes[i]


class _FakeKeypoints:
    """Mimics ``result.keypoints``: ``.xy[i]`` returns a ``_FakeTensor``."""

    def __init__(self, xy: np.ndarray) -> None:
        # xy shape: (N, K, 2). Wrap each detection's keypoints as a tensor.
        self.xy = [_FakeTensor(xy[i]) for i in range(xy.shape[0])]


class _FakeResult:
    def __init__(
        self, boxes: _FakeBoxes | None, keypoints: _FakeKeypoints | None
    ) -> None:
        self.boxes = boxes
        self.keypoints = keypoints


class _FakeModel:
    """Returns a pre-built list of fake results regardless of inputs.

    ``infer_one`` only reads ``results[0]``, so a one-element list is enough.
    """

    def __init__(self, results: list[_FakeResult]) -> None:
        self._results = results

    def predict(self, **_kwargs) -> list[_FakeResult]:
        return self._results


# ---------------------------------------------------------------------------
# compare_detections — perfect match + count mismatch
# ---------------------------------------------------------------------------


def test_compare_detections_perfect_match():
    # Identical dicts must produce matched=True with all drifts at 0.
    pt = _result(_det((0, 0, 100, 100), conf=0.9, kps=((10, 10), (90, 50), (50, 90))))
    ex = _result(_det((0, 0, 100, 100), conf=0.9, kps=((10, 10), (90, 50), (50, 90))))

    report = compare_detections(pt, ex)

    assert report["matched"] is True
    assert report["n_pt"] == 1
    assert report["n_exported"] == 1
    assert report["max_bbox_drift_px"] == 0.0
    assert report["max_kp_drift_px"] == 0.0
    assert report["max_conf_drift"] == 0.0
    assert report["pair_diagnostics"][0]["coordinate_scale_warning"] is False
    assert report["failures"] == []


def test_compare_detections_count_mismatch_flagged():
    # 2 PT detections vs 1 exported -- quantization shouldn't drop a wheel.
    pt = _result(
        _det((0, 0, 50, 50), conf=0.9),
        _det((100, 100, 150, 150), conf=0.85),
    )
    ex = _result(_det((0, 0, 50, 50), conf=0.9))

    report = compare_detections(pt, ex)

    assert report["matched"] is False
    assert report["n_pt"] == 2
    assert report["n_exported"] == 1
    assert any("count differs" in f for f in report["failures"])


# ---------------------------------------------------------------------------
# compare_detections — bbox tolerance
# ---------------------------------------------------------------------------


def test_compare_detections_bbox_drift_within_tol():
    # Exported bbox shifted by 1.5 px (< 2.0 default atol).
    pt = _result(_det((0, 0, 100, 100)))
    ex = _result(_det((1.5, 1.5, 101.5, 101.5)))

    report = compare_detections(pt, ex)

    assert report["matched"] is True
    assert report["max_bbox_drift_px"] == pytest.approx(1.5)
    assert report["failures"] == []


def test_compare_detections_bbox_drift_exceeds_tol():
    # Exported bbox shifted by 5 px (> 2.0 default atol).
    pt = _result(_det((0, 0, 100, 100)))
    ex = _result(_det((5, 5, 105, 105)))

    report = compare_detections(pt, ex)

    assert report["matched"] is False
    assert report["max_bbox_drift_px"] == pytest.approx(5.0)
    assert any("bbox drift" in f for f in report["failures"])


def test_compare_detections_flags_normalized_exported_coordinates():
    pt = _result(_det((100, 100, 200, 200), conf=0.9))
    ex = _result(_det((0.1, 0.1, 0.2, 0.2), conf=0.9))

    report = compare_detections(pt, ex)

    assert report["matched"] is False
    assert report["pair_diagnostics"][0]["coordinate_scale_warning"] is True
    assert report["pair_diagnostics"][0]["exported_bbox_max_coord"] <= 2.0


# ---------------------------------------------------------------------------
# compare_detections — keypoint tolerance
# ---------------------------------------------------------------------------


def test_compare_detections_kp_drift_within_tol():
    # Each keypoint shifted by exactly 2 px (< 3.0 default atol).
    pt = _result(_det((0, 0, 100, 100), kps=((10, 10), (20, 20), (30, 30))))
    ex = _result(_det((0, 0, 100, 100), kps=((12, 10), (22, 20), (32, 30))))

    report = compare_detections(pt, ex)

    assert report["matched"] is True
    assert report["max_kp_drift_px"] == pytest.approx(2.0)
    assert report["failures"] == []


def test_compare_detections_kp_drift_exceeds_tol():
    # Each keypoint shifted by 5 px (> 3.0 default atol).
    pt = _result(_det((0, 0, 100, 100), kps=((10, 10), (20, 20), (30, 30))))
    ex = _result(_det((0, 0, 100, 100), kps=((15, 10), (25, 20), (35, 30))))

    report = compare_detections(pt, ex)

    assert report["matched"] is False
    assert report["max_kp_drift_px"] == pytest.approx(5.0)
    assert any("keypoint drift" in f for f in report["failures"])


# ---------------------------------------------------------------------------
# compare_detections — confidence tolerance
# ---------------------------------------------------------------------------


def test_compare_detections_conf_drift_flagged():
    # 0.9 vs 0.5 is way outside the 0.05 abs tolerance.
    pt = _result(_det((0, 0, 100, 100), conf=0.9))
    ex = _result(_det((0, 0, 100, 100), conf=0.5))

    report = compare_detections(pt, ex)

    assert report["matched"] is False
    assert report["max_conf_drift"] == pytest.approx(0.4)
    assert any("conf drift" in f for f in report["failures"])


# ---------------------------------------------------------------------------
# compare_detections — greedy IoU matching
# ---------------------------------------------------------------------------


def test_compare_detections_uses_greedy_iou_matching():
    # Three PT wheels at distinct locations. The "exported" list contains
    # the same three wheels in a shuffled order. A naive index-based
    # comparator would flag drift; the greedy IoU matcher must pair them
    # correctly and report matched=True with zero drift.
    pt = _result(
        _det((0, 0, 50, 50), conf=0.9, kps=((10, 10), (40, 10), (25, 45))),
        _det((100, 0, 150, 50), conf=0.8, kps=((110, 10), (140, 10), (125, 45))),
        _det((200, 0, 250, 50), conf=0.7, kps=((210, 10), (240, 10), (225, 45))),
    )
    # Shuffled order: third, first, second -- but the bboxes themselves
    # are exactly the same as their PT counterparts.
    ex = _result(
        _det((200, 0, 250, 50), conf=0.7, kps=((210, 10), (240, 10), (225, 45))),
        _det((0, 0, 50, 50), conf=0.9, kps=((10, 10), (40, 10), (25, 45))),
        _det((100, 0, 150, 50), conf=0.8, kps=((110, 10), (140, 10), (125, 45))),
    )

    report = compare_detections(pt, ex)

    assert report["matched"] is True
    assert report["max_bbox_drift_px"] == 0.0
    assert report["max_kp_drift_px"] == 0.0
    assert report["max_conf_drift"] == 0.0
    assert report["failures"] == []


# ---------------------------------------------------------------------------
# pick_sample_image
# ---------------------------------------------------------------------------


def test_pick_sample_image_prefers_arg_when_exists(tmp_path: Path):
    # Real file path on disk -- pick_sample_image must just hand it back.
    arg_image = tmp_path / "manual.jpg"
    arg_image.write_bytes(b"\xff\xd8\xff")  # not a valid JPEG, but exists

    out = pick_sample_image(arg_image, dataset_root=tmp_path / "irrelevant")

    assert out == arg_image


def test_pick_sample_image_falls_back_to_val_split(tmp_path: Path):
    # No arg given, val dir has two images -- expect the lexicographically
    # first one back (deterministic across runs).
    val_dir = tmp_path / "val"
    val_dir.mkdir()
    img_b = val_dir / "b.jpg"
    img_a = val_dir / "a.jpg"
    img_b.write_bytes(b"\xff\xd8\xff")
    img_a.write_bytes(b"\xff\xd8\xff")

    out = pick_sample_image(None, dataset_root=val_dir)

    assert out == img_a


def test_pick_sample_image_raises_when_nothing_found(tmp_path: Path):
    # Empty val dir AND no arg -- must raise FileNotFoundError so the
    # sanity check doesn't silently skip.
    val_dir = tmp_path / "val"
    val_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        pick_sample_image(None, dataset_root=val_dir)


def test_pick_sample_image_raises_on_missing_arg(tmp_path: Path):
    # Explicit --sample-image that doesn't exist on disk -- must raise
    # with a message that points at the flag, not silently fall back to
    # the val dir.
    missing = tmp_path / "does_not_exist.jpg"

    with pytest.raises(FileNotFoundError, match="--sample-image"):
        pick_sample_image(missing, dataset_root=tmp_path)


# ---------------------------------------------------------------------------
# compare_detections — shuffled order with sub-tolerance drift (defends
# against an index-based regression in the matcher).
# ---------------------------------------------------------------------------


def test_compare_detections_greedy_match_survives_shuffle_with_drift():
    # 3 PT detections at distinct x-locations. Exported list is the SAME
    # three detections, REVERSED, each bbox shifted by exactly 1.0 px on
    # every coordinate. Greedy IoU pairs detection-0 with the last
    # exported entry (highest overlap), etc., so the bbox drift is the
    # 1 px shift -- under the 2 px atol.
    # An index-based matcher would pair pt[0] with ex[0] (200..210 vs
    # 0..10), which have ZERO IoU and a >>2 px drift, blowing the test.
    pt = _result(
        _det((0, 0, 10, 10), conf=0.9, kps=((1, 1), (5, 1), (3, 9))),
        _det((100, 0, 110, 10), conf=0.8, kps=((101, 1), (105, 1), (103, 9))),
        _det((200, 0, 210, 10), conf=0.7, kps=((201, 1), (205, 1), (203, 9))),
    )
    ex = _result(
        _det(
            (201, 1, 211, 11),
            conf=0.7,
            kps=((202, 2), (206, 2), (204, 10)),
        ),
        _det(
            (101, 1, 111, 11),
            conf=0.8,
            kps=((102, 2), (106, 2), (104, 10)),
        ),
        _det(
            (1, 1, 11, 11),
            conf=0.9,
            kps=((2, 2), (6, 2), (4, 10)),
        ),
    )

    report = compare_detections(pt, ex)

    assert report["matched"] is True
    assert report["max_bbox_drift_px"] == pytest.approx(1.0)
    assert report["max_bbox_drift_px"] <= 1.5
    assert report["failures"] == []


# ---------------------------------------------------------------------------
# infer_one — uses fake YOLO results so we never touch ultralytics.
# ---------------------------------------------------------------------------


def test_infer_one_no_keypoints_returns_empty_kp_list(tmp_path: Path):
    # Detect-only model: result has boxes but result.keypoints is None.
    # The detection must still come through, with `keypoints: []`.
    boxes = _FakeBoxes([_FakeBox([10.0, 20.0, 30.0, 40.0], conf=0.9)])
    result = _FakeResult(boxes=boxes, keypoints=None)
    model = _FakeModel([result])

    out = infer_one(model, tmp_path / "anything.jpg")

    assert out == {
        "detections": [
            {
                "bbox": [10.0, 20.0, 30.0, 40.0],
                "conf": pytest.approx(0.9),
                "keypoints": [],
            }
        ]
    }


def test_infer_one_zero_detections_returns_empty_list(tmp_path: Path):
    # Either no boxes at all (boxes=None) or an empty boxes container --
    # infer_one must short-circuit to `detections: []` so the comparator
    # sees a real count of 0, not a synthetic placeholder.
    model_none = _FakeModel([_FakeResult(boxes=None, keypoints=None)])
    out_none = infer_one(model_none, tmp_path / "anything.jpg")
    assert out_none == {"detections": []}

    model_empty = _FakeModel([_FakeResult(boxes=_FakeBoxes([]), keypoints=None)])
    out_empty = infer_one(model_empty, tmp_path / "anything.jpg")
    assert out_empty == {"detections": []}


def test_infer_one_extracts_bbox_conf_kp_from_result(tmp_path: Path):
    # One detection with one keypoint triplet -- the bbox, conf, and
    # keypoint xy floats must all flow through unchanged (typed float).
    boxes = _FakeBoxes([_FakeBox([10.0, 20.0, 30.0, 40.0], conf=0.9)])
    kp_arr = np.array([[[11.0, 21.0], [22.0, 32.0], [25.0, 35.0]]], dtype=np.float32)
    keypoints = _FakeKeypoints(kp_arr)
    result = _FakeResult(boxes=boxes, keypoints=keypoints)
    model = _FakeModel([result])

    out = infer_one(model, tmp_path / "anything.jpg")

    assert len(out["detections"]) == 1
    det = out["detections"][0]
    assert det["bbox"] == [10.0, 20.0, 30.0, 40.0]
    assert det["conf"] == pytest.approx(0.9)
    assert det["keypoints"] == [
        [pytest.approx(11.0), pytest.approx(21.0)],
        [pytest.approx(22.0), pytest.approx(32.0)],
        [pytest.approx(25.0), pytest.approx(35.0)],
    ]
