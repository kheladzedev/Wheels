from __future__ import annotations

from src.ue_wheel_asset_filter import classify_wheel_asset_paths, is_wheel_asset_path


def test_wheel_asset_filter_accepts_common_terms():
    assert is_wheel_asset_path("/Game/Cars/staticmeshes/front_left_wheel_0")
    assert is_wheel_asset_path("/Game/Cars/staticmeshes/rear_tire_mesh")
    assert is_wheel_asset_path("/Game/Cars/staticmeshes/roue_arriere")


def test_wheel_asset_filter_accepts_position_abbreviations():
    assert is_wheel_asset_path("/Game/Cars/staticmeshes/flw_vehicle_0.flw_vehicle_0")
    assert is_wheel_asset_path("/Game/Cars/staticmeshes/frw_001_vehicle_0.frw_001_vehicle_0")
    assert is_wheel_asset_path("/Game/Cars/staticmeshes/rlw_honda_t360_0.rlw_honda_t360_0")
    assert is_wheel_asset_path("/Game/Cars/staticmeshes/rrw_vehicle_0.rrw_vehicle_0")


def test_wheel_asset_filter_rejects_non_wheel_words():
    assert not is_wheel_asset_path("/Game/Cars/staticmeshes/steering_wheel_0")
    assert not is_wheel_asset_path("/Game/Cars/staticmeshes/flywheel_cover_0")
    assert not is_wheel_asset_path("/Game/Cars/staticmeshes/polysurface1083_w_bmw_body_0")


def test_group_classifier_uses_conservative_rubber_fallback():
    paths = [
        "/Game/Cars/staticmeshes/polysurface11_blackrubber_0",
        "/Game/Cars/staticmeshes/polysurface13_blackrubber_0",
        "/Game/Cars/staticmeshes/polysurface16_blackrubber_0",
        "/Game/Cars/staticmeshes/polysurface18_blackrubber_0",
        "/Game/Cars/staticmeshes/body_paint_0",
    ]

    assert classify_wheel_asset_paths(paths) == [True, True, True, True, False]


def test_group_classifier_does_not_mark_large_rubber_groups():
    paths = [f"/Game/Cars/staticmeshes/object{i:03d}_rubber_0" for i in range(12)]

    assert classify_wheel_asset_paths(paths) == [False] * 12
