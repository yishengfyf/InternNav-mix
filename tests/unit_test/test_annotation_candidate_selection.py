import importlib.util
from pathlib import Path

from PIL import Image


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "dualvln_mainline"
    / "build_annotation_pilot.py"
)
SPEC = importlib.util.spec_from_file_location("build_annotation_pilot", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def entry(score, frame, distance, yaw_deg):
    return (
        score,
        frame,
        {
            "distance": distance,
            "dyaw_rad": yaw_deg * 3.141592653589793 / 180.0,
            "dyaw_deg": yaw_deg,
        },
    )


def test_phase_diverse_selects_recent_turn_and_spatial_anchor():
    current = 20
    pool = [
        entry(3.1, 18, 0.5, 0),
        entry(3.1, 17, 0.75, 0),
        entry(2.1, 13, 1.5, 80),
        entry(2.1, 8, 4.5, 20),
        entry(3.1, 9, 0.0, 160),
    ]
    actions = ["forward"] * 21
    actions[13] = "turn_left"

    selected = MODULE.select_history_candidates(
        current, pool, actions, "phase_diverse"
    )

    assert [item[1] for item in selected] == [18, 13, 8]


def test_phase_diverse_falls_back_without_non_degenerate_anchor():
    current = 10
    pool = [entry(3.1, 8, 0.0, 30), entry(2.1, 5, 0.2, 75)]
    actions = ["forward"] * 11

    selected = MODULE.select_history_candidates(
        current, pool, actions, "phase_diverse"
    )

    assert len(selected) == 2
    assert {item[1] for item in selected} == {5, 8}


def test_black_pixel_fraction_detects_simulator_void(tmp_path):
    image = Image.new("RGB", (20, 10), "black")
    for x in range(10, 20):
        for y in range(10):
            image.putpixel((x, y), (120, 100, 80))
    path = tmp_path / "half_void.png"
    image.save(path)

    assert MODULE.black_pixel_fraction(path) == 0.5
