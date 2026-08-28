import copy
import importlib.util
from pathlib import Path


_path = (
    Path(__file__).resolve().parents[2]
    / "internnav"
    / "utils"
    / "stage55_occ_2p5d_audit.py"
)
_spec = importlib.util.spec_from_file_location("stage55_occ_2p5d_audit", _path)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)


class _Memory:
    cs = 0.05

    def __init__(self):
        self.occ_counts = {
            (10, 10, 0): 8,
            (10, 11, 2): 6,
            (10, 12, 4): 5,
        }
        self.occ3d_frame_counts = {
            (10, 10, 0): 3,
            (10, 11, 2): 3,
            (10, 12, 4): 2,
        }

    def validation_floor_aligned_cell_evidence(
        self, row, col, floor_z_m, *, height_max_m
    ):
        blocked = (int(row), int(col)) in {(10, 10), (10, 11), (10, 12)}
        consensus = "blocked" if blocked else "unknown"
        return {
            "state": "blocked" if blocked else "unknown",
            "occupied_hits": 2 if blocked else 0,
            "free_hits": 0,
            "frame_aware_cell_evidence": {
                "frame_consensus_state": consensus,
                "occupied_frame_count": 3 if blocked else 0,
                "free_frame_count": 0,
                "last_occupied_observation": 7 if blocked else None,
                "last_free_observation": None,
                "current_frame_occupied": False,
            },
        }


def test_occ_2p5d_audit_is_read_only_and_tracks_continuous_support():
    memory = _Memory()
    candidate = {
        "candidate_id": "route",
        "path_cells": [[10, 10], [10, 11], [10, 12]],
        "floor_z_m": 0.0,
        "footprint_radius_m": 0.05,
        "floor_aligned_height_max_m": 1.5,
    }
    before = copy.deepcopy(candidate)
    report = _module.audit_candidate_occ_2p5d(memory, candidate)
    assert candidate == before
    assert report["decision_applied"] is False
    assert report["unknown_is_free"] is False
    assert report["support_2p5d"]["support_coverage"] == 1.0
    assert report["support_2p5d"]["continuous_support_shadow"] is True
    assert report["support_2p5d"]["max_abs_support_step_m"] == 0.1
    assert report["corridor_summary_is_complete"] is True
    assert report["frame_consensus_scope"] == "all_height_2d_cell"


def test_occ_2p5d_audit_summarizes_full_corridor_with_bounded_records():
    memory = _Memory()
    candidate = {
        "candidate_id": "long_route",
        "path_cells": [[10, col] for col in range(10, 74)],
        "floor_z_m": 0.0,
        "footprint_radius_m": 0.18,
        "floor_aligned_height_max_m": 1.5,
    }
    report = _module.audit_candidate_occ_2p5d(
        memory,
        candidate,
        max_corridor_cells=64,
    )
    assert report["corridor_cell_count"] > 64
    assert report["corridor_cell_record_count"] == 64
    assert report["cell_records_are_bounded_sample"] is True
    assert sum(report["legacy_floor_state_counts"].values()) == report[
        "corridor_cell_count"
    ]
    assert sum(report["frame_consensus_state_counts"].values()) == report[
        "corridor_cell_count"
    ]


def test_post_turn_guard_requires_real_collided_s2_forward_and_budget():
    base = {
        "enabled": True,
        "armed": True,
        "previous_action": 1,
        "forward_action": 1,
        "previous_action_source": "system2_action_queue",
        "collision_delta": 1.0,
        "guard_age_steps": 12,
        "horizon_steps": 400,
        "requery_count": 0,
        "requery_budget": 8,
    }
    assert _module.should_post_turn_collision_guard(**base)
    assert not _module.should_post_turn_collision_guard(
        **{**base, "collision_delta": 0.0}
    )
    assert not _module.should_post_turn_collision_guard(
        **{**base, "previous_action_source": "vlmap_recovery_queue"}
    )
    assert not _module.should_post_turn_collision_guard(
        **{**base, "requery_count": 8}
    )
    assert not _module.should_post_turn_collision_guard(
        **{**base, "guard_age_steps": 401}
    )
