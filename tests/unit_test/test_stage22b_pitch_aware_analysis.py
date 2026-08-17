import json

from scripts.eval.analyze_stage22b_pitch_aware_occ import analyze


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _audit(occupied_ratio, reachable, pitch_aware=False):
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
        "route_cell_state_counts": {"free": 2, "unknown": 0, "occupied": 1},
        "route_cell_state_ratios": {
            "free": 1.0 - occupied_ratio,
            "unknown": 0.0,
            "occupied": occupied_ratio,
        },
        "known_free_connectivity": {
            "reachable": reachable,
            "reason": "ok" if reachable else "no_known_free_path_to_anchor",
        },
        "route_pitch_observation_count": 1,
        "route_max_camera_pitch_deg": 30.0,
        "route_occupied_height_diagnostics": {
            "occupied_route_cell_count": 1,
            "obstacle_band_conflict_cell_count": 1,
        },
    }


def test_stage22b_analysis_pairs_map_events_and_preserves_navigation(tmp_path):
    run = tmp_path / "run"
    nav = tmp_path / "nav"
    stage22a = tmp_path / "stage22a"
    manifest = tmp_path / "manifest.json"
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
    loop = {
        "scene_id": "scene",
        "episode_id": 1,
        "transition": "start",
        "step_id": 50,
        "start_step": 40,
        "turn_direction": "left",
        "triage_tier": "strict_intervention",
    }
    for root in (run, nav):
        _write_jsonl(root / "progress.json", [progress])
        _write_jsonl(
            root / "vlmap_safety_debug" / "rank0_run_001" / "s2_action_loop_events.jsonl",
            [loop],
        )
    wrapper = {
        **loop,
        "event_type": "s2_loop_executed_route_occ_audit",
        "episode_eval_seed": 9,
        "shadow_only": True,
        "action_applied": False,
        "output_rewritten": False,
        "gt_fields_used": [],
    }
    _write_jsonl(
        stage22a / "vlmap_safety_debug" / "rank0_run_001" / "s2_loop_executed_route_occ_audit_events.jsonl",
        [{**wrapper, "audit": _audit(0.5, False)}],
    )
    _write_jsonl(
        run / "vlmap_safety_debug" / "rank0_run_001" / "s2_loop_executed_route_occ_audit_events.jsonl",
        [{**wrapper, "audit": _audit(0.25, True, True)}],
    )
    _write_jsonl(
        run / "vlmap_safety_debug" / "rank0_run_001" / "occ_memory" / "memory_events.jsonl",
        [
            {
                "event_type": "occ_memory_update",
                "valid": True,
                "requested_camera_pitch_deg": 30.0,
                "applied_camera_pitch_deg": 30.0,
            }
        ],
    )
    manifest.write_text(
        json.dumps([{"scene_id": "scene", "episode_id": 1, "episode_eval_seed": 9}]),
        encoding="utf-8",
    )

    summary = analyze(run, 1, manifest, nav, stage22a)

    assert summary["comparison_integrity_passed"] is True, {
        "integrity": summary.get("integrity_passed"),
        "violations": summary.get("violations"),
        "missing": (summary.get("missing_current_event_keys"), summary.get("missing_baseline_event_keys")),
        "identity": summary.get("route_identity_mismatches"),
        "pitch": (summary.get("non_pitch_aware_audit_count"), summary.get("valid_occ_update_count"), summary.get("pitched_occ_update_count"), summary.get("pitch_application_mismatch_count")),
    }
    assert summary["occupied_ratio_delta_mean"] == -0.25
    assert summary["ray_reachability_gain_count"] == 1
    assert summary["pitch_application_mismatch_count"] == 0
