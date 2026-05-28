"""Offline tests for render-image pseudo-label helpers."""

from __future__ import annotations

import numpy as np

import pseudo_label_images_to_incoming as pli


def test_build_wheel_annotation_maps_three_pose_keypoints():
    wheel = pli.build_wheel_annotation(
        np.array([10.0, 20.0, 50.0, 70.0]),
        np.array([[12.0, 65.0], [48.0, 65.0], [30.0, 68.0]]),
        0.87654,
        image_w=100,
        image_h=80,
        min_bbox_side=8,
    )

    assert wheel is not None
    assert wheel["bbox_xyxy"] == [10.0, 20.0, 50.0, 70.0]
    assert wheel["points"] == {
        "a": [12.0, 65.0],
        "b": [48.0, 65.0],
        "c_disc_bottom": [30.0, 68.0],
    }
    assert wheel["_needs_review"] is True
    assert wheel["_pseudo_conf"] == 0.8765


def test_build_wheel_annotation_drops_tiny_or_degenerate_detections():
    assert (
        pli.build_wheel_annotation(
            np.array([10.0, 20.0, 12.0, 23.0]),
            np.array([[11.0, 21.0], [11.5, 21.0], [11.0, 22.0]]),
            0.9,
            image_w=100,
            image_h=80,
            min_bbox_side=8,
        )
        is None
    )
    assert (
        pli.build_wheel_annotation(
            np.array([10.0, 20.0, 50.0, 70.0]),
            np.array([[0.0, 65.0], [48.0, 65.0], [30.0, 68.0]]),
            0.9,
            image_w=100,
            image_h=80,
            min_bbox_side=8,
        )
        is None
    )


def test_clip_bbox_rejects_inverted_bbox_after_clipping():
    assert (
        pli.build_wheel_annotation(
            np.array([50.0, 20.0, 10.0, 70.0]),
            np.array([[12.0, 65.0], [48.0, 65.0], [30.0, 68.0]]),
            0.9,
            image_w=100,
            image_h=80,
            min_bbox_side=8,
        )
        is None
    )
