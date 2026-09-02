import importlib.util
from pathlib import Path


_path = (
    Path(__file__).resolve().parents[2]
    / "internnav"
    / "utils"
    / "stage75_route_prompt.py"
)
_spec = importlib.util.spec_from_file_location("stage75_route_prompt", _path)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)

build_dualvln_route_recovery_card = _module.build_dualvln_route_recovery_card
natural_route_direction = _module.natural_route_direction
route_guidance_from_bridge = _module.route_guidance_from_bridge


def test_direction_phrases_follow_sparseocc_left_positive_convention():
    assert natural_route_direction(0.0) == "ahead"
    assert natural_route_direction(45.0) == "ahead-left"
    assert natural_route_direction(90.0) == "left"
    assert natural_route_direction(135.0) == "behind-left"
    assert natural_route_direction(-45.0) == "ahead-right"
    assert natural_route_direction(-90.0) == "right"
    assert natural_route_direction(-135.0) == "behind-right"
    assert natural_route_direction(180.0) == "behind"


def test_bridge_guidance_is_coarse_and_detects_arrival():
    guidance = route_guidance_from_bridge(
        {
            "path_reachable": True,
            "start_grid": [100, 100],
            "anchor_grid": [90, 100],
            "path_cell_count": 21,
            "path_m": 1.03,
            "initial_direction_angle_deg": 61.0,
            "initial_direction_bucket": "left",
        }
    )
    assert guidance["valid"] is True
    assert guidance["arrived"] is False
    assert guidance["natural_direction"] == "ahead-left"
    assert guidance["quantized_bearing_deg"] == 60
    assert guidance["quantized_distance_m"] == 1.0

    arrived = route_guidance_from_bridge(
        {
            "path_reachable": True,
            "start_grid": [100, 100],
            "anchor_grid": [100, 100],
            "path_cell_count": 1,
            "path_m": 0.0,
            "initial_direction_angle_deg": 0.0,
        }
    )
    assert arrived["valid"] is True
    assert arrived["arrived"] is True
    assert arrived["reason"] == "anchor_reached"


def test_card_keeps_dualvln_image_waypoint_and_observation_protocol():
    guidance = route_guidance_from_bridge(
        {
            "path_reachable": True,
            "path_m": 0.76,
            "initial_direction_angle_deg": -92.0,
        }
    )
    card = build_dualvln_route_recovery_card(guidance)
    assert "previously visited place" in card
    assert "right (about 90 degrees)" in card
    assert "0.75 meters" in card
    assert "current image" in card
    assert "TURN LEFT (←)" in card
    assert "TURN RIGHT (→)" in card
    assert "STOP only when the original task is complete" in card
    assert "grid" not in card
    assert "XYZ" not in card


def test_unreachable_route_uses_visual_fallback_without_fake_direction():
    guidance = route_guidance_from_bridge(
        {"path_reachable": False, "reason": "no_known_free_path_to_anchor"}
    )
    card = build_dualvln_route_recovery_card(guidance)
    assert guidance["valid"] is False
    assert "about" not in card
    assert "Compare that reference with the current view" in card
