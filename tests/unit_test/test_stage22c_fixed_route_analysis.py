import json

from scripts.eval.analyze_stage22c_fixed_route_pitch_occ import analyze


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _audit(occupied_ratio, reachable, pitch_aware):
    return {
        "valid": True,
        "reason": "ok",
        "shadow_only": True,
        "action_applied": False,
        "output_rewritten": False,
        "gt_fields_used": [],
        "mapping_camera_pitch_aware": pitch_aware,
        "source_step": 10,
        "anchor_grid": [8, 8],
        "source_pose_grid": [8, 8],
        "trigger_pose_grid": [8, 10],
        "source_anchor_pose_match": True,
        "route_cells": [[8, 8], [8, 9], [8, 10]],
        "route_chain_continuous": True,
        "route_cell_state_ratios": {
            "free": 1.0 - occupied_ratio,
            "unknown": 0.0,
            "occupied": occupied_ratio,
        },
        "known_free_connectivity": {
            "reachable": reachable,
            "reason": "ok" if reachable else "no_known_free_path_to_anchor",
        },
        "route_occupied_height_diagnostics": {
            "occupied_route_cell_count": 1,
            "body_obstacle_conflict_cell_count": 1,
        },
    }


def test_fixed_route_analysis_allows_candidate_and_triage_changes(tmp_path):
    run = tmp_path / "run"
    nav = tmp_path / "nav"
    stage22a = tmp_path / "stage22a"
    seed_manifest = tmp_path / "seeds.json"
    fixed_manifest = tmp_path / "fixed.json"
    progress = {
        "scene_id": "scene",
        "episode_id": 1,
        "episode_eval_seed": 9,
        "success": 0,
        "spl": 0.0,
        "ne": 4.0,
        "steps": 60,
        "collision_count": 1,
    }
    baseline_loop = {
        "scene_id": "scene",
        "episode_id": 1,
        "transition": "start",
        "step_id": 50,
        "start_step": 40,
        "turn_direction": "left",
        "failure_type": "local_trap",
        "triage_tier": "strict_intervention",
        "candidate": {
            "candidate_id": "A",
            "semantic_resilience_source_step_id": 10,
            "grid": [8, 8],
        },
    }
    current_loop = {
        **baseline_loop,
        "triage_tier": "adapter_candidate",
        "candidate": {
            "candidate_id": "B",
            "semantic_resilience_source_step_id": 12,
            "grid": [9, 8],
        },
    }
    for root in (run, nav):
        _write_jsonl(root / "progress.json", [progress])
    _write_jsonl(
        stage22a
        / "vlmap_safety_debug"
        / "rank0_run_001"
        / "s2_action_loop_events.jsonl",
        [baseline_loop],
    )
    _write_jsonl(
        run
        / "vlmap_safety_debug"
        / "rank0_run_001"
        / "s2_action_loop_events.jsonl",
        [current_loop],
    )
    baseline_wrapper = {
        "scene_id": "scene",
        "episode_id": 1,
        "step_id": 50,
        "audit": _audit(0.5, False, False),
    }
    fixed_reference = {
        "scene_id": "scene",
        "episode_id": 1,
        "step_id": 50,
        "source_step": 10,
        "anchor_grid": [8, 8],
        "candidate_id": "A",
    }
    current_wrapper = {
        "scene_id": "scene",
        "episode_id": 1,
        "step_id": 50,
        "fixed_reference": fixed_reference,
        "current_selected_candidate": {
            "candidate_id": "B",
            "source_step": 12,
            "grid": [9, 8],
        },
        "current_triage_tier": "adapter_candidate",
        "shadow_only": True,
        "action_applied": False,
        "output_rewritten": False,
        "gt_fields_used": [],
        "audit": _audit(0.25, True, True),
    }
    _write_jsonl(
        stage22a
        / "vlmap_safety_debug"
        / "rank0_run_001"
        / "s2_loop_executed_route_occ_audit_events.jsonl",
        [baseline_wrapper],
    )
    _write_jsonl(
        run
        / "vlmap_safety_debug"
        / "rank0_run_001"
        / "s2_loop_fixed_route_occ_audit_events.jsonl",
        [current_wrapper],
    )
    _write_jsonl(
        run
        / "vlmap_safety_debug"
        / "rank0_run_001"
        / "occ_memory"
        / "memory_events.jsonl",
        [
            {
                "event_type": "occ_memory_update",
                "valid": True,
                "requested_camera_pitch_deg": 30.0,
                "applied_camera_pitch_deg": 30.0,
            }
        ],
    )
    seed_manifest.write_text(
        json.dumps(
            [{"scene_id": "scene", "episode_id": 1, "episode_eval_seed": 9}]
        ),
        encoding="utf-8",
    )
    fixed_manifest.write_text(json.dumps([fixed_reference]), encoding="utf-8")

    summary = analyze(
        run,
        1,
        seed_manifest,
        nav,
        stage22a,
        fixed_manifest,
        expected_fixed_routes=1,
    )

    assert summary["integrity_passed"] is True, summary["violations"]
    assert summary["route_identity_verified_count"] == 1
    assert summary["occupied_ratio_delta_mean"] == -0.25
    assert summary["ray_reachability_gain_count"] == 1
    assert summary["candidate_identity_changed_count"] == 1
    assert summary["triage_transition_counts"] == {
        "strict_intervention->adapter_candidate": 1
    }
