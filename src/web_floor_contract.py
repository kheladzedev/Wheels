"""Validation helpers for the web floor + wheels output contract.

The web contract is separate from the frozen mobile AR contract. It adds a
lightweight direct floor output, but still keeps wheel points in the confirmed
``a`` / ``b`` / ``c_disc_bottom`` naming and avoids runtime depth/RANSAC fields.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping, Sequence
from typing import Any

RUNTIME_SCOPE = "single_forward_no_depth_no_ransac"
POINT_KEYS = ("a", "b", "c_disc_bottom")
DISTANCE_MODES = frozenset({"scale_relative", "metric_anchor", "normalized", "unknown"})
FOV_MODES = frozenset({"unknown", "fixed", "provided", "predicted"})

FORBIDDEN_RUNTIME_KEY_SUBSTRINGS = (
    "depth",
    "segmentation",
    "mask",
    "point_cloud",
    "ransac",
    "raycast",
    "world_xyz",
    "backend_postprocess",
)


class WebFloorContractError(ValueError):
    """Raised when a decoded web-floor payload violates the contract."""


@dataclass(frozen=True)
class WebFloor:
    pitch: float
    roll: float
    distance: float
    distance_mode: str = "scale_relative"
    fov_mode: str = "unknown"

    def to_json(self) -> dict[str, float | str]:
        return {
            "pitch": self.pitch,
            "roll": self.roll,
            "distance": self.distance,
            "distance_mode": self.distance_mode,
            "fov_mode": self.fov_mode,
        }


@dataclass(frozen=True)
class WebWheel:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    points: dict[str, tuple[float, float]]

    def to_json(self) -> dict[str, Any]:
        return {
            "bbox_xyxy": [float(v) for v in self.bbox_xyxy],
            "confidence": float(self.confidence),
            "points": {k: [float(x), float(y)] for k, (x, y) in self.points.items()},
        }


@dataclass(frozen=True)
class WebFloorPayload:
    floor: WebFloor
    wheels: tuple[WebWheel, ...]
    frame_id: str | None = None
    runtime_scope: str = RUNTIME_SCOPE

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "runtime_scope": self.runtime_scope,
            "floor": self.floor.to_json(),
            "wheels": [wheel.to_json() for wheel in self.wheels],
        }
        if self.frame_id is not None:
            payload["frame_id"] = self.frame_id
        return payload


def _fail(path: str, message: str) -> None:
    raise WebFloorContractError(f"{path}: {message}")


def _as_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(path, "expected object")
    return value


def _finite_float(value: Any, path: str) -> float:
    if isinstance(value, bool):
        _fail(path, "expected finite number, got bool")
    try:
        out = float(value)
    except (TypeError, ValueError):
        _fail(path, "expected finite number")
    if not math.isfinite(out):
        _fail(path, "expected finite number")
    return out


def _unit_float(value: Any, path: str) -> float:
    out = _finite_float(value, path)
    if not 0.0 <= out <= 1.0:
        _fail(path, "expected value in [0, 1]")
    return out


def _point(value: Any, path: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail(path, "expected [x, y]")
    if len(value) != 2:
        _fail(path, "expected [x, y]")
    return (_finite_float(value[0], f"{path}[0]"), _finite_float(value[1], f"{path}[1]"))


def _bbox(value: Any, path: str) -> tuple[float, float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail(path, "expected [x1, y1, x2, y2]")
    if len(value) != 4:
        _fail(path, "expected [x1, y1, x2, y2]")
    x1, y1, x2, y2 = (_finite_float(v, f"{path}[{i}]") for i, v in enumerate(value))
    if not (x1 < x2 and y1 < y2):
        _fail(path, "expected non-degenerate xyxy bbox")
    return (x1, y1, x2, y2)


def _check_forbidden_keys(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_str = str(key)
            lowered = key_str.lower()
            for needle in FORBIDDEN_RUNTIME_KEY_SUBSTRINGS:
                if needle in lowered:
                    _fail(f"{path}.{key_str}", f"forbidden runtime field contains {needle!r}")
            _check_forbidden_keys(child, f"{path}.{key_str}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_forbidden_keys(child, f"{path}[{index}]")


def floor_from_tensor(
    values: Sequence[float],
    *,
    distance_mode: str = "scale_relative",
    fov_mode: str = "unknown",
) -> dict[str, float | str]:
    """Decode the model's ``[pitch, roll, distance]`` floor tensor.

    The current model code still calls the third scalar ``delta_z`` internally.
    This helper is the bridge to the public web contract, where the field is
    named ``distance`` and must carry a ``distance_mode``.
    """
    if len(values) != 3:
        _fail("floor", "expected [pitch, roll, distance]")
    floor = _validate_floor(
        {
            "pitch": values[0],
            "roll": values[1],
            "distance": values[2],
            "distance_mode": distance_mode,
            "fov_mode": fov_mode,
        },
        path="floor",
    )
    return floor.to_json()


def _validate_floor(value: Any, path: str) -> WebFloor:
    floor = _as_mapping(value, path)
    missing = {"pitch", "roll", "distance", "distance_mode"} - set(floor)
    if missing:
        _fail(path, f"missing field(s): {sorted(missing)}")

    distance_mode = floor["distance_mode"]
    if distance_mode not in DISTANCE_MODES:
        _fail(f"{path}.distance_mode", f"expected one of {sorted(DISTANCE_MODES)}")

    fov_mode = floor.get("fov_mode", "unknown")
    if fov_mode not in FOV_MODES:
        _fail(f"{path}.fov_mode", f"expected one of {sorted(FOV_MODES)}")

    return WebFloor(
        pitch=_finite_float(floor["pitch"], f"{path}.pitch"),
        roll=_finite_float(floor["roll"], f"{path}.roll"),
        distance=_finite_float(floor["distance"], f"{path}.distance"),
        distance_mode=str(distance_mode),
        fov_mode=str(fov_mode),
    )


def _validate_wheel(value: Any, path: str) -> WebWheel:
    wheel = _as_mapping(value, path)
    allowed = {"bbox_xyxy", "confidence", "points"}
    if set(wheel) != allowed:
        _fail(path, f"expected keys {sorted(allowed)}, got {sorted(wheel)}")
    bbox = _bbox(wheel["bbox_xyxy"], f"{path}.bbox_xyxy")
    confidence = _unit_float(wheel["confidence"], f"{path}.confidence")
    points_raw = _as_mapping(wheel["points"], f"{path}.points")
    if set(points_raw) != set(POINT_KEYS):
        _fail(f"{path}.points", f"expected keys {list(POINT_KEYS)}")

    points = {key: _point(points_raw[key], f"{path}.points.{key}") for key in POINT_KEYS}
    x1, y1, x2, y2 = bbox
    for key, (px, py) in points.items():
        if not (x1 <= px <= x2 and y1 <= py <= y2):
            _fail(f"{path}.points.{key}", "point must lie inside bbox")

    a = points["a"]
    b = points["b"]
    c = points["c_disc_bottom"]
    if not a[0] < b[0]:
        _fail(f"{path}.points", "a.x must be left of b.x")
    if c[1] > max(a[1], b[1]):
        _fail(f"{path}.points.c_disc_bottom", "c_disc_bottom must not sit below A/B")

    return WebWheel(bbox_xyxy=bbox, confidence=confidence, points=points)


def validate_web_floor_payload(payload: Any, *, require_frame_id: bool = False) -> dict[str, Any]:
    """Validate and normalize a decoded web-floor payload.

    Returns a JSON-serializable dict with floats normalized and optional fields
    filled. Raises :class:`WebFloorContractError` on contract drift.
    """
    root = _as_mapping(payload, "$")
    _check_forbidden_keys(root)

    if "floor" not in root:
        _fail("$", "missing floor")
    if "wheels" not in root:
        _fail("$", "missing wheels")

    runtime_scope = root.get("runtime_scope", RUNTIME_SCOPE)
    if runtime_scope != RUNTIME_SCOPE:
        _fail("$.runtime_scope", f"expected {RUNTIME_SCOPE!r}")

    frame_id = root.get("frame_id")
    if frame_id is not None and not isinstance(frame_id, str):
        _fail("$.frame_id", "expected string when present")
    if require_frame_id and not frame_id:
        _fail("$.frame_id", "missing or empty frame_id")

    wheels_raw = root["wheels"]
    if not isinstance(wheels_raw, list):
        _fail("$.wheels", "expected list")
    wheels = tuple(
        _validate_wheel(wheel, f"$.wheels[{index}]")
        for index, wheel in enumerate(wheels_raw)
    )

    decoded = WebFloorPayload(
        floor=_validate_floor(root["floor"], path="$.floor"),
        wheels=wheels,
        frame_id=frame_id,
        runtime_scope=RUNTIME_SCOPE,
    )
    return decoded.to_json()
