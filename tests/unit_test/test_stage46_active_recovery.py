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
