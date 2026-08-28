import importlib.util
from pathlib import Path


_path = Path(__file__).resolve().parents[2] / "internnav" / "utils" / "stage46_active_recovery.py"
_spec = importlib.util.spec_from_file_location("stage46_active_recovery", _path)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)


def _candidate(candidate_id, *, support, openness, path, **extra):
    value = {
        "candidate_id": candidate_id,
        "route_support_edge_count": support,
        "local_free_fraction": openness,
        "path_length_m": path,
        "source_step": 10,
        "floor_aligned_known_free": True,
        "unknown_fraction": 0.0,
        "occupied_fraction": 0.0,
        "route_occ_conflict": False,
        "shadow_only": True,
        "action_applied": False,
        "gt_fields_used": [],
    }
    value.update(extra)
    return value


def _event(candidates):
    return {
        "ablation": {
            "route_occ_clearance_frontier": {"candidates": candidates}
        }
    }


def test_selector_uses_frozen_composite_and_never_trains_ranker():
    selected, report = _module.select_frozen_m3_candidate(
        _event(
            [
                _candidate("open", support=2, openness=0.99, path=0.5),
                _candidate("supported", support=4, openness=0.70, path=0.8),
            ]
        )
    )
    assert selected["candidate_id"] == "supported"
    assert report["ranking_rule"] == "stage40_composite_fixed_heuristic"
    assert report["ranker_trained"] is False
    assert report["unknown_is_free"] is False


def test_selector_fails_closed_on_unknown_or_occ_conflict():
    selected, report = _module.select_frozen_m3_candidate(
        _event(
            [
                _candidate("unknown", support=5, openness=1, path=0.2, unknown_fraction=0.1),
                _candidate("occupied", support=4, openness=1, path=0.2, route_occ_conflict=True),
            ]
        )
    )
    assert selected is None
    assert report["reason"] == "zero_safe_m3_candidate"


def test_binding_marks_only_existing_safe_candidate_executable():
    result = _module.bind_candidate_to_loop_event(
        {"scene_id": "scene", "episode_id": 1, "triage_tier": "hold"},
        _event([_candidate("safe", support=2, openness=0.8, path=0.6)]),
    )
    assert result["candidate_source"] == "stage27_frozen_m3"
    assert result["candidate"]["geometry_safe"] is True
    assert result["candidate"]["active_gate_safe"] is True
    assert result["triage_tier"] == "strict_intervention"


def test_selector_report_keeps_pool_accounting_for_post_selection_gates():
    selected, report = _module.select_frozen_m3_candidate(
        _event(
            [
                _candidate("safe", support=2, openness=0.8, path=0.6),
                _candidate("unsafe", support=8, openness=1.0, path=0.2, occupied_fraction=0.1),
            ]
        )
    )
    assert selected["candidate_id"] == "safe"
    assert report["pool_count"] == 2
    assert report["safe_count"] == 1
    assert report["unsafe_record_count"] == 1


def test_route_occ_turn_only_keeps_clearance_failure_auditable():
    candidate = _candidate(
        "clearance-failed",
        support=3,
        openness=0.8,
        path=0.6,
        floor_aligned_known_free=False,
    )
    event = {"ablation": {"route_occ": {"candidates": [candidate]}}}
    selected, report = _module.select_frozen_m3_candidate(
        event,
        candidate_stage="route_occ",
        safety_mode="route_occ_turn_only",
    )
    assert selected["candidate_id"] == "clearance-failed"
    assert report["turn_only_relaxation"] is True
    assert report["translation_allowed"] is False


def test_route_only_turn_binding_marks_relaxation_without_granting_translation():
    candidate = _candidate(
        "route-conflict",
        support=3,
        openness=0.8,
        path=0.5,
        route_occ_conflict=True,
        occupied_fraction=0.2,
    )
    event = {"ablation": {"route_only": {"candidates": [candidate]}}}
    result = _module.bind_candidate_to_loop_event(
        {"scene_id": "scene", "episode_id": 1},
        event,
        candidate_stage="route_only",
        safety_mode="route_only_turn_only",
    )
    assert result["candidate"]["stage54_turn_only_relaxation"] is True
    assert result["candidate"]["stage54_translation_allowed"] is False
    assert result["candidate"]["stage46_safety_derivation"] == "route_only"


def test_active_path_bound_is_local_and_zero_disables_it():
    assert _module.active_path_within_bound(0.95, 1.0)
    assert not _module.active_path_within_bound(1.65, 1.0)
    assert _module.active_path_within_bound(1.65, 0.0)


def test_iterative_reorientation_requires_monotonic_bearing_improvement():
    decision = _module.iterative_reorientation_decision(
        -100.0,
        -85.0,
        primitive_count=1,
        max_primitives=4,
        deadband_deg=7.5,
    )
    assert decision["continue_reorientation"] is True
    assert decision["turn_direction"] == "right"

    stalled = _module.iterative_reorientation_decision(
        -85.0,
        -86.0,
        primitive_count=2,
        max_primitives=4,
        deadband_deg=7.5,
    )
    assert stalled["continue_reorientation"] is False
    assert stalled["reason"] == "iterative_reorient_not_converging"


def test_iterative_reorientation_stops_at_alignment_or_budget():
    aligned = _module.iterative_reorientation_decision(
        16.0,
        1.0,
        primitive_count=1,
        max_primitives=4,
        deadband_deg=7.5,
    )
    assert aligned["continue_reorientation"] is False
    assert aligned["reason"] == "path_aligned_no_visible_proxy"

    exhausted = _module.iterative_reorientation_decision(
        -55.0,
        -40.0,
        primitive_count=4,
        max_primitives=4,
        deadband_deg=7.5,
    )
    assert exhausted["continue_reorientation"] is False
    assert exhausted["reason"] == "iterative_reorient_budget_exhausted"
