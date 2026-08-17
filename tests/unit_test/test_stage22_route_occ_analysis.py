import json

from scripts.eval.analyze_stage22a_executed_route_occ_audit import analyze


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_stage22_analysis_verifies_replay_and_summarizes_route_gap(tmp_path):
    run_root = tmp_path / "run"
    reference_root = tmp_path / "reference"
    manifest = tmp_path / "manifest.json"
    progress = {
        "scene_id": "scene",
        "episode_id": 7,
        "episode_eval_seed": 42,
        "success": 0,
        "spl": 0.0,
        "ne": 5.0,
        "steps": 80,
        "collision_count": 2,
        "s2_loop_strict_active_applied_count": 0,
        "s2_loop_path_reobserve_applied_count": 0,
        "stage19_semantic_resilience_active_applied_count": 0,
    }
    loop = {
        "scene_id": "scene",
        "episode_id": 7,
        "transition": "start",
        "step_id": 60,
        "start_step": 52,
        "turn_direction": "left",
        "triage_tier": "strict_intervention",
    }
    audit = {
        "event_type": "s2_loop_executed_route_occ_audit",
        "scene_id": "scene",
        "episode_id": 7,
        "episode_eval_seed": 42,
        "step_id": 60,
        "triage_tier": "strict_intervention",
        "shadow_only": True,
        "action_applied": False,
        "output_rewritten": False,
        "gt_fields_used": [],
        "audit": {
            "valid": True,
            "reason": "ok",
            "shadow_only": True,
            "action_applied": False,
            "output_rewritten": False,
            "gt_fields_used": [],
            "source_step": 20,
            "anchor_grid": [10, 10],
            "trigger_pose_grid": [10, 15],
            "source_anchor_pose_match": True,
            "route_raw_pose_count": 9,
            "route_translation_node_count": 6,
            "route_movement_edge_count": 5,
            "route_length_m": 1.25,
            "route_max_edge_m": 0.25,
            "route_chain_continuous": True,
            "route_cell_state_counts": {"free": 3, "unknown": 3, "occupied": 0},
            "route_cell_state_ratios": {"free": 0.5, "unknown": 0.5, "occupied": 0.0},
            "longest_unknown_gap_m": 0.75,
            "first_occupied_conflict": None,
            "first_unknown_gap": {"route_cell_index": 3, "grid": [10, 13]},
            "known_free_connectivity": {
                "reachable": False,
                "reason": "no_known_free_path_to_anchor",
            },
            "continuous_but_ray_disconnected": True,
        },
    }
    _write_jsonl(run_root / "progress.json", [progress])
    _write_jsonl(reference_root / "progress.json", [progress])
    _write_jsonl(
        run_root / "vlmap_safety_debug" / "rank0_run_001" / "s2_action_loop_events.jsonl",
        [loop],
    )
    _write_jsonl(
        reference_root
        / "vlmap_safety_debug"
        / "rank0_run_001"
        / "s2_action_loop_events.jsonl",
        [loop],
    )
    _write_jsonl(
        run_root
        / "vlmap_safety_debug"
        / "rank0_run_001"
        / "s2_loop_executed_route_occ_audit_events.jsonl",
        [audit],
    )
    manifest.write_text(
        json.dumps(
            [{"scene_id": "scene", "episode_id": 7, "episode_eval_seed": 42}]
        ),
        encoding="utf-8",
    )

    summary = analyze(run_root, 1, manifest, reference_root)

    assert summary["integrity_passed"] is True
    assert summary["route_audit_event_count"] == 1
    assert summary["route_chain_continuous_count"] == 1
    assert summary["ray_free_reachable_count"] == 0
    assert summary["continuous_but_ray_disconnected_count"] == 1
    assert summary["route_cell_unknown_ratio_mean"] == 0.5
