import numpy as np

from internnav.utils.lseg_replay_frequency import (
    evaluate_frequency_gate, select_causal_keyframes, short_lived_labels,
)


def _observation(index, x=0.0, heading=0.0, pitch=0.0, height=0.0):
    return {
        "record_index": index, "observation_key": f"{index}:{index}",
        "camera_pitch_deg": pitch,
        "pose": {
            "gps": [x, 0.0], "compass": [heading],
            "stage23a_gt_relative_height_m": height,
        },
    }


def test_causal_keyframes_keep_queries_and_motion_events():
    observations = [
        _observation(0), _observation(1, x=0.25), _observation(2, x=0.55),
        _observation(3, x=0.55, heading=np.pi / 6),
        _observation(4, x=0.55, heading=np.pi / 3),
    ]
    selected = select_causal_keyframes(
        observations, {"0:0", "3:3"}, min_gap=2, max_gap=4, visual_change=1.0
    )
    assert "s2_query" in selected[0]
    assert "translation" in selected[2]
    assert "s2_query" in selected[3]
    assert 4 not in selected


def test_short_lived_labels_count_frames_not_points():
    assert short_lived_labels([
        {"floor": 100, "door": 2}, {"floor": 90},
        {"floor": 80, "chair": 1},
    ]) == {"door", "chair"}


def test_frequency_gate_requires_quality_cost_and_conflicts():
    q = {
        "gt_hit_rate": 0.75, "class_count": 10, "short_lived_class_count": 4,
        "call_count": 20, "conflicts_per_100_nodes": 2.0,
    }
    all_frames = {
        "gt_hit_rate": 0.80, "class_count": 10, "short_lived_class_count": 4,
        "call_count": 100, "conflicts_per_100_nodes": 2.5,
    }
    good = dict(all_frames, call_count=60, conflicts_per_100_nodes=1.5)
    assert evaluate_frequency_gate(q, good, all_frames)["passed"]
    bad = dict(good, call_count=70)
    assert not evaluate_frequency_gate(q, bad, all_frames)["passed"]


def test_frequency_gate_accepts_no_short_lived_classes():
    base = {
        "gt_hit_rate": 0.8, "class_count": 2, "short_lived_class_count": 0,
        "call_count": 10, "conflicts_per_100_nodes": 0.0,
    }
    qk = dict(base, call_count=6)
    assert evaluate_frequency_gate(base, qk, base)["passed"]
