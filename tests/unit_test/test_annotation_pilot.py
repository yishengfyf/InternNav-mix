import importlib.util
import math
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "dualvln_mainline" / "build_annotation_pilot.py"
SPEC = importlib.util.spec_from_file_location("build_annotation_pilot", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def transform(x, y, yaw=0.0):
    import math

    return [
        [math.cos(yaw), -math.sin(yaw), 0.0, x],
        [math.sin(yaw), math.cos(yaw), 0.0, y],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_discrete_action_mapping_matches_dataset_protocol():
    assert MODULE.action_name(0) == "stop"
    assert MODULE.action_name(1) == "forward"
    assert MODULE.action_name(2) == "turn_left"
    assert MODULE.action_name(3) == "turn_right"
    assert MODULE.action_name(5) == "look_down"


def test_invalid_pixel_goal_is_not_presented_as_target():
    assert MODULE.local_goal([-1, -1]) is None
    assert MODULE.local_goal([320, 240]) == [320.0, 240.0]


def test_decision_alignment_matches_dataset_protocol():
    actions = [1, 2, 3, 0]
    goals = [[-1, -1], [320, 240], [-1, -1], [-1, -1]]
    relative_ids = [-1, 4, -1, -1]
    assert MODULE.decision_for_frame(0, actions, goals, relative_ids) == ("turn_left", None)
    assert MODULE.decision_for_frame(1, actions, goals, relative_ids) == ("pixel_goal", [320.0, 240.0])


def test_matrix_relative_pose_uses_translation_column_and_current_frame():
    current = transform(1.0, 2.0, 0.0)
    history = transform(3.0, 5.0, 0.0)
    relative = MODULE.rel_pose(current, history)
    assert relative["dx"] == 2.0
    assert relative["dy"] == 3.0
    assert math.isclose(relative["distance"], 13 ** 0.5, rel_tol=1e-6)
