"""Guards for AR-facing inference overlay semantics."""

from __future__ import annotations

import numpy as np

from infer_image import COLOR_KP, DISPLAY_KP_NAMES, draw_final_overlay


def test_infer_overlay_uses_ar_mock_spec_keypoint_colours() -> None:
    # OpenCV BGR tuples: A red, B blue, C green.
    assert COLOR_KP == ((0, 0, 255), (255, 0, 0), (0, 255, 0))
    assert DISPLAY_KP_NAMES == ("A", "B", "C")


def test_final_overlay_renders_confirmed_schema_points() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    wheels = [
        {
            "bbox_xyxy": [10.0, 20.0, 110.0, 110.0],
            "confidence": 0.9,
            "points": {
                "a": [20.0, 100.0],
                "b": [95.0, 100.0],
                "c_disc_bottom": [60.0, 78.0],
            },
        }
    ]

    out = draw_final_overlay(image, wheels)

    assert tuple(int(v) for v in out[100, 20]) == COLOR_KP[0]
    assert tuple(int(v) for v in out[100, 95]) == COLOR_KP[1]
    assert tuple(int(v) for v in out[78, 60]) == COLOR_KP[2]
