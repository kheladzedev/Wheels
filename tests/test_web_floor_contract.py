from __future__ import annotations

import math

import pytest

from web_floor_contract import (
    POINT_KEYS,
    RUNTIME_SCOPE,
    WebFloorContractError,
    floor_from_tensor,
    validate_web_floor_payload,
)


def _valid_payload() -> dict:
    return {
        "frame_id": "web-frame-0001",
        "runtime_scope": RUNTIME_SCOPE,
        "floor": {
            "pitch": 0.04,
            "roll": -0.01,
            "distance": 1.6,
            "distance_mode": "scale_relative",
            "fov_mode": "unknown",
        },
        "wheels": [
            {
                "bbox_xyxy": [100, 210, 220, 350],
                "confidence": 0.91,
                "points": {
                    "a": [112, 340],
                    "b": [208, 341],
                    "c_disc_bottom": [160, 305],
                },
            }
        ],
    }


def test_valid_payload_is_normalized() -> None:
    out = validate_web_floor_payload(_valid_payload(), require_frame_id=True)

    assert out["frame_id"] == "web-frame-0001"
    assert out["runtime_scope"] == RUNTIME_SCOPE
    assert out["floor"] == {
        "pitch": 0.04,
        "roll": -0.01,
        "distance": 1.6,
        "distance_mode": "scale_relative",
        "fov_mode": "unknown",
    }
    assert set(out["wheels"][0]["points"]) == set(POINT_KEYS)
    assert all(isinstance(v, float) for v in out["wheels"][0]["bbox_xyxy"])


def test_floor_from_tensor_maps_internal_delta_z_slot_to_public_distance() -> None:
    floor = floor_from_tensor([0.1, -0.2, 1.4], distance_mode="normalized")

    assert floor["pitch"] == 0.1
    assert floor["roll"] == -0.2
    assert floor["distance"] == 1.4
    assert floor["distance_mode"] == "normalized"
    assert "delta_z" not in floor


def test_missing_floor_is_rejected() -> None:
    payload = _valid_payload()
    payload.pop("floor")

    with pytest.raises(WebFloorContractError, match="missing floor"):
        validate_web_floor_payload(payload)


def test_invalid_distance_mode_is_rejected() -> None:
    payload = _valid_payload()
    payload["floor"]["distance_mode"] = "metric_because_i_said_so"

    with pytest.raises(WebFloorContractError, match="distance_mode"):
        validate_web_floor_payload(payload)


def test_non_finite_floor_values_are_rejected() -> None:
    payload = _valid_payload()
    payload["floor"]["pitch"] = math.nan

    with pytest.raises(WebFloorContractError, match="finite"):
        validate_web_floor_payload(payload)


@pytest.mark.parametrize("field", ["a", "b", "c_disc_bottom"])
def test_missing_wheel_point_is_rejected(field: str) -> None:
    payload = _valid_payload()
    payload["wheels"][0]["points"].pop(field)

    with pytest.raises(WebFloorContractError, match="expected keys"):
        validate_web_floor_payload(payload)


def test_rim_semantics_are_not_allowed_in_web_points() -> None:
    payload = _valid_payload()
    payload["wheels"][0]["points"]["rim_left"] = payload["wheels"][0]["points"].pop("a")

    with pytest.raises(WebFloorContractError, match="expected keys"):
        validate_web_floor_payload(payload)


def test_invalid_a_b_geometry_is_rejected() -> None:
    payload = _valid_payload()
    payload["wheels"][0]["points"]["a"] = [210, 340]
    payload["wheels"][0]["points"]["b"] = [112, 341]

    with pytest.raises(WebFloorContractError, match="a.x"):
        validate_web_floor_payload(payload)


def test_invalid_c_geometry_is_rejected() -> None:
    payload = _valid_payload()
    payload["wheels"][0]["points"]["c_disc_bottom"] = [160, 346]

    with pytest.raises(WebFloorContractError, match="c_disc_bottom"):
        validate_web_floor_payload(payload)


def test_point_outside_bbox_is_rejected() -> None:
    payload = _valid_payload()
    payload["wheels"][0]["points"]["a"] = [99, 340]

    with pytest.raises(WebFloorContractError, match="inside bbox"):
        validate_web_floor_payload(payload)


def test_empty_wheels_frame_is_valid() -> None:
    payload = _valid_payload()
    payload["wheels"] = []

    out = validate_web_floor_payload(payload)

    assert out["wheels"] == []
    assert out["floor"]["distance_mode"] == "scale_relative"


def test_heavy_runtime_fields_are_rejected() -> None:
    payload = _valid_payload()
    payload["depth_map"] = "not part of this runtime"

    with pytest.raises(WebFloorContractError, match="forbidden runtime"):
        validate_web_floor_payload(payload)
