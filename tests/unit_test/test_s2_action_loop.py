from internnav.s2_action_loop import (
    init_s2_action_loop_state,
    normalize_direct_turn_output,
    observe_s2_action_query,
)


def test_normalize_direct_turn_output_rejects_waypoints_and_mixed_turns():
    assert normalize_direct_turn_output("←←←←")["direction"] == "left"
    assert normalize_direct_turn_output("→")["turn_action_count"] == 1
    assert normalize_direct_turn_output("120 318") is None
    assert normalize_direct_turn_output("←→") is None
    assert normalize_direct_turn_output("STOP") is None


def test_cross_query_loop_requires_repeated_generations_and_low_translation():
    state = init_s2_action_loop_state()
    event = None
    for step, output in ((38, "←←←←"), (42, "←←←←"), (46, "←←←←"), (50, "←←←←"), (54, "←←←←")):
        event = observe_s2_action_query(
            state,
            output=output,
            step_id=step,
            gps=[0.0, 0.0],
            compass=[0.0],
            config={"enable": True},
        )

    assert event["transition"] == "start"
    assert event["step_id"] == 54
    assert event["same_turn_generation_streak"] == 5
    assert event["cumulative_turn_actions"] == 20
    assert event["low_translation"] is True


def test_translation_and_non_turn_output_break_the_loop():
    state = init_s2_action_loop_state()
    for index, step in enumerate((38, 42, 46, 50, 54)):
        event = observe_s2_action_query(
            state,
            output="←←←←",
            step_id=step,
            gps=[float(index), 0.0],
            compass=[0.0],
            config={"enable": True},
        )
    assert event is None

    assert observe_s2_action_query(
        state,
        output="120 318",
        step_id=58,
        gps=[5.0, 0.0],
        compass=[0.0],
        config={"enable": True},
    ) is None
    assert state["generation_streak"] == 0
