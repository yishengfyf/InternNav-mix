from internnav.semantic_recovery_triage import classify_semantic_recovery_triage


def _candidate(**overrides):
    candidate = {
        "geometry_safe": True,
        "active_gate_safe": True,
        "target_frontier_intent_safe": True,
        "target_frontier_escape_candidate": True,
        "semantic_resilience_open_score": 0.82,
        "target_frontier_score": 0.30,
        "target_frontier_doorway_like_score": 0.70,
        "completed_landmark_penalty": 0.0,
        "semantic_resilience_step_gap": 24,
        "semantic_resilience_backtrack_distance_m": 2.5,
        "nearby_visit_count": 4,
        "direction_bucket": "left",
        "semantic_resilience_obstacle_term_count": 3,
    }
    candidate.update(overrides)
    return candidate


def _classify(candidate, **overrides):
    kwargs = {
        "failure_type": "stuck_collision",
        "recommended_primitive": "reorient_reobserve",
        "trigger_reasons": ["local_trap", "current_waypoint_occupied"],
        "context_tags": ["spatial_constriction", "semantic_obstacle_context"],
    }
    kwargs.update(overrides)
    return classify_semantic_recovery_triage(candidate, {}, **kwargs)


def test_consistent_multi_evidence_is_strict_intervention():
    result = _classify(_candidate())

    assert result["tier"] == "strict_intervention"
    assert result["evidence_vote_count"] == 6


def test_safe_but_nonpersistent_candidate_is_held_for_adapter():
    result = _classify(
        _candidate(semantic_resilience_step_gap=5, nearby_visit_count=1),
        trigger_reasons=["current_waypoint_occupied"],
        context_tags=["semantic_obstacle_context"],
    )

    assert result["tier"] == "adapter_candidate"
    assert result["persistence"] is False


def test_semantic_only_signal_abstains():
    result = _classify(
        _candidate(semantic_resilience_obstacle_term_count=0),
        failure_type="semantic_stagnation",
        recommended_primitive="reobserve",
        trigger_reasons=["semantic_dead_zone", "semantic_stagnation"],
        context_tags=["semantic_uncertainty_or_stagnation"],
    )

    assert result["tier"] == "abstain"
    assert "semantic_only_no_spatial_conflict" in result["hard_abstain_reasons"]


def test_completed_landmark_candidate_abstains():
    result = _classify(_candidate(completed_landmark_penalty=1.0))

    assert result["tier"] == "abstain"
    assert "completed_landmark_penalty" in result["hard_abstain_reasons"]


def test_stale_anchor_cannot_enter_strict_or_adapter_tier():
    result = _classify(_candidate(semantic_resilience_step_gap=245), step_id=285)

    assert result["tier"] == "abstain"
    assert "stale_backtrack_anchor" in result["hard_abstain_reasons"]


def test_limited_frontier_tag_does_not_replace_trap_evidence():
    result = _classify(
        _candidate(),
        trigger_reasons=["semantic_stagnation", "current_waypoint_occupied"],
        context_tags=[
            "limited_frontier_escape",
            "policy_memory_conflict",
            "semantic_obstacle_context",
        ],
    )

    assert result["tier"] == "adapter_candidate"
    assert result["spatial_constriction"] is False
