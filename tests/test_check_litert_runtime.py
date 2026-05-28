from __future__ import annotations

import numpy as np
import pytest

from src.check_litert_runtime import _tensor_summary


def test_tensor_summary_reports_shape_dtype_and_finiteness():
    value = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    summary = _tensor_summary(value)

    assert summary["shape"] == [2, 2]
    assert summary["dtype"] == "float32"
    assert summary["finite"] is True
    assert summary["min"] == pytest.approx(1.0)
    assert summary["max"] == pytest.approx(4.0)
    assert summary["mean"] == pytest.approx(2.5)


def test_tensor_summary_detects_nonfinite_values():
    summary = _tensor_summary(np.asarray([1.0, np.nan], dtype=np.float32))

    assert summary["finite"] is False
