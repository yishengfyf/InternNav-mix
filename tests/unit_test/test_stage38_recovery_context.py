import importlib.util
from pathlib import Path


_path = Path(__file__).resolve().parents[2] / "internnav" / "utils" / "stage38_recovery_context.py"
_spec = importlib.util.spec_from_file_location("stage38_recovery_context", _path)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)

attach_current_reaudit = _module.attach_current_reaudit
build_recovery_anchor = _module.build_recovery_anchor
build_recovery_bev_digest = _module.build_recovery_bev_digest
recovery_contract_ok = _module.recovery_contract_ok


def _event():
    return {"scene_id": "scene", "episode_id": 7, "step_id": 12}


def _candidate(**extra):
    value = {
        "candidate_id": "route_node:4",
        "source_type": "R-route-open",
        "path_cells": [[10, 10], [10, 11]],
        "floor_aligned_known_free": True,
        "unknown_fraction": 0.0,
        "occupied_fraction": 0.0,
        "route_occ_conflict": False,
    }
    value.update(extra)
    return value


def test_anchor_requires_current_reaudit_and_bev_is_non_authoritative():
    anchor = build_recovery_anchor(_event(), _candidate(), capture_semantic={"top": "chair"})
    current = attach_current_reaudit(anchor, _candidate())
    digest = build_recovery_bev_digest(current, semantic_relevance=0.9, reobserve_gain=0.4)
    assert current["current_safety"]["status"] == "safe"
    assert digest["safety_authority"] == "current_sparseocc_reaudit"
    assert recovery_contract_ok(current, digest)


def test_unknown_and_semantic_never_promote_anchor():
    anchor = build_recovery_anchor(_event(), _candidate(), capture_semantic={"top": "stairs"})
    current = attach_current_reaudit(anchor, _candidate(unknown_fraction=0.1))
    digest = build_recovery_bev_digest(current, semantic_relevance=1.0)
    assert current["current_safety"]["status"] == "rejected"
    assert digest["semantic_can_override_safety"] is False
    assert digest["unknown_is_free"] is False
    assert recovery_contract_ok(current, digest)


def test_missing_current_candidate_abstains():
    anchor = build_recovery_anchor(_event(), _candidate())
    current = attach_current_reaudit(anchor, None)
    digest = build_recovery_bev_digest(current)
    assert current["current_safety"]["status"] == "no_current_candidate"
    assert recovery_contract_ok(current, digest)
