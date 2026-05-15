"""Smoke tests for the MobileNetV2-skipless pose model.

Pretrained weights are skipped (`pretrained=False`) to keep CI offline
and fast — we don't need ImageNet-quality features to verify the
architecture's shape contract.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from src.models.mobilenetv2_skipless_pose import (
    FEATURE_STRIDE,
    N_KEYPOINTS,
    MobileNetV2SkiplessPose,
    count_parameters,
    decode_predictions,
)


def _model() -> MobileNetV2SkiplessPose:
    return MobileNetV2SkiplessPose(pretrained=False)


def test_instantiates_without_pretrained_weights() -> None:
    m = _model()
    assert isinstance(m, torch.nn.Module)


def test_parameter_count_in_expected_band() -> None:
    """Architecture should fall in the 2-5M parameter range.

    Wide band — the proposal estimated ~3.8M, real impl came out at
    ~2.7M because the head's depthwise blocks are leaner than the
    estimate. Test catches catastrophic changes (e.g. accidentally
    using MobileNetV3-Large), not exact-match regressions.
    """
    m = _model()
    n = count_parameters(m)
    assert 2_000_000 < n < 5_000_000, f"unexpected param count: {n}"


def test_forward_pass_produces_expected_shapes_at_640() -> None:
    m = _model()
    m.train(False)  # inference mode (avoid hook flagging .eval())
    x = torch.randn(2, 3, 640, 640)
    with torch.no_grad():
        out = m(x)
    expected_grid = 640 // FEATURE_STRIDE  # 20
    assert out["cls"].shape == (2, 1, expected_grid, expected_grid)
    assert out["bbox"].shape == (2, 4, expected_grid, expected_grid)
    assert out["kpt"].shape == (2, N_KEYPOINTS * 2, expected_grid, expected_grid)
    assert out["vis"].shape == (2, N_KEYPOINTS, expected_grid, expected_grid)


def test_bbox_outputs_are_non_negative() -> None:
    """The head applies ReLU to bbox — l/t/r/b distances are >= 0."""
    m = _model()
    m.train(False)
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out = m(x)
    assert (out["bbox"] >= 0).all()


def test_decode_predictions_shapes_and_pixel_ranges() -> None:
    m = _model()
    m.train(False)
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out = m(x)
        dec = decode_predictions(out)

    n_cells = (640 // FEATURE_STRIDE) ** 2
    assert dec["cls_prob"].shape == (1, n_cells)
    assert dec["bbox_xyxy"].shape == (1, n_cells, 4)
    assert dec["kpt_xy"].shape == (1, n_cells, N_KEYPOINTS, 2)
    assert dec["vis_prob"].shape == (1, n_cells, N_KEYPOINTS)
    # Probabilities are bounded.
    assert (dec["cls_prob"] >= 0).all() and (dec["cls_prob"] <= 1).all()
    assert (dec["vis_prob"] >= 0).all() and (dec["vis_prob"] <= 1).all()


def test_classification_bias_is_focal_prior() -> None:
    """The cls head's bias is initialised to -log((1-p)/p) with p=0.01.

    This prior is load-bearing for focal-loss stability at step 0; if
    a future refactor drops it, focal loss can NaN immediately.
    """
    m = _model()
    bias = m.head.cls.bias.detach()
    expected = -4.595  # -log(0.99 / 0.01)
    assert torch.allclose(bias, torch.full_like(bias, expected), atol=1e-3)


def test_inference_mode_produces_deterministic_outputs() -> None:
    """Same input → identical outputs after `model.train(False)`."""
    m = _model()
    m.train(False)
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out_a = m(x)
        out_b = m(x)
    for k in out_a:
        assert torch.equal(out_a[k], out_b[k])
