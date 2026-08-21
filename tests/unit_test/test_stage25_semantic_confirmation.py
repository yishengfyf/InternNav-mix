from internnav.utils.stage25_semantic_confirmation import (
    select_causal_window, summarize_semantic_window,
)


def test_causal_window_never_selects_future_observations():
    observations = [{"step_id": step} for step in range(20)]
    selected = select_causal_window(observations, event_step=12, window_steps=4)
    assert [item["step_id"] for item in selected] == [8, 9, 10, 11, 12]


def test_causal_window_can_keep_only_latest_frames():
    observations = [{"step_id": step} for step in range(20)]
    selected = select_causal_window(
        observations, event_step=12, window_steps=8, max_frames=4
    )
    assert [item["step_id"] for item in selected] == [9, 10, 11, 12]


def test_semantic_recurrence_confirms_existing_suspicion():
    cells = ["0:1:2:3", "8:2:2:3"]
    frames = [
        {
            "valid": True,
            "spatial_semantic_cells": cells,
            "class_surface_counts": {"door": 12},
        }
        for _ in range(4)
    ]
    summary = summarize_semantic_window(frames)
    assert summary["spatial_stagnation"] is True
    assert summary["recent_spatial_recurrence"] == 1.0
    assert summary["classes_with_multiframe_support"] == ["door"]


def test_semantic_change_does_not_confirm_stagnation():
    frames = [
        {"valid": True, "spatial_semantic_cells": [f"0:{step}:0:0"]}
        for step in range(4)
    ]
    summary = summarize_semantic_window(frames)
    assert summary["spatial_stagnation"] is False
    assert summary["recent_semantic_novelty"] == 1.0
