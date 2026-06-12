"""Postprocess helpers for web floor model outputs.

This module keeps runtime decoding lightweight: raw floor tensor plus decoded
wheel candidates go through the Phase 2 contract validator. It does not run
runtime depth, segmentation, RANSAC, or backend geometry.
"""

from __future__ import annotations

from typing import Any, Sequence

from web_floor_contract import RUNTIME_SCOPE, floor_from_tensor, validate_web_floor_payload


def wheels_from_target(target: dict[str, Any]) -> list[dict[str, Any]]:
    """Build fixture wheel candidates from dataset targets.

    This is a fixture/eval proxy until a production web pose decoder is wired.
    It preserves the public A/B/C names and keeps the readiness report honest.
    """
    boxes = target["boxes"].detach().cpu().tolist()
    keypoints = target["keypoints"].detach().cpu().tolist()
    wheels: list[dict[str, Any]] = []
    for bbox, points in zip(boxes, keypoints):
        wheels.append(
            {
                "bbox_xyxy": [float(v) for v in bbox],
                "confidence": 1.0,
                "points": {
                    "a": [float(v) for v in points[0]],
                    "b": [float(v) for v in points[1]],
                    "c_disc_bottom": [float(v) for v in points[2]],
                },
            }
        )
    return wheels


def decode_web_floor_payload(
    *,
    frame_id: str,
    floor_values: Sequence[float],
    wheels: list[dict[str, Any]],
    distance_mode: str = "scale_relative",
    fov_mode: str = "unknown",
    runtime_scope: str = RUNTIME_SCOPE,
) -> dict[str, Any]:
    """Decode a raw floor vector and wheel candidates into the web contract."""
    payload = {
        "frame_id": frame_id,
        "runtime_scope": runtime_scope,
        "floor": floor_from_tensor(
            floor_values,
            distance_mode=distance_mode,
            fov_mode=fov_mode,
        ),
        "wheels": wheels,
    }
    return validate_web_floor_payload(payload, require_frame_id=True)
