import json

import numpy as np

from scripts.eval.analyze_stage26_hsgm_semantic_filter import analyze


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _make_ledger(root, scene_id, episode_id):
    ledger = root / "replay_ledger" / f"{scene_id}_{episode_id}_r0"
    _write_json(ledger / "episode_meta.json", {
        "scene_id": scene_id, "episode_id": episode_id,
    })
    _write_jsonl(ledger / "queries.jsonl", [{
        "step_id": 0, "output": "forward", "pixel_goal": [1, 2],
        "input_steps": [0],
    }])
    _write_jsonl(ledger / "actions.jsonl", [{
        "step_id": 0, "action": 1, "action_source": "s2",
        "pre_safety_action": 1, "action_applied": True,
    }])
    _write_jsonl(ledger / "observations.jsonl", [{
        "step_id": 0, "previous_action": 1,
        "previous_action_source": "s2", "previous_action_applied": True,
        "pose": {
            "gps": [0.0, 0.0], "compass": [0.0],
            "stage23_gt_camera_pose_map": np.eye(4).tolist(),
        },
    }])


def test_stage26_analyzer_checks_exact_subset_and_aggregates_gate(tmp_path):
    scene_id, episode_id, seed = "scene", 7, 440007
    run_root = tmp_path / "run"
    baseline_root = tmp_path / "baseline"
    _make_ledger(run_root, scene_id, episode_id)
    _make_ledger(baseline_root, scene_id, episode_id)
    semantic = run_root / "online_lseg_shadow" / f"{scene_id}_{episode_id}_r0"
    _write_json(semantic / "episode_meta.json", {
        "scene_id": scene_id, "episode_id": episode_id,
        "episode_eval_seed": seed,
    })
    raw_nodes = [
        {
            "label": "door", "evidence_tier": "strong",
            "gt_surface_distance_m": distance,
        }
        for distance in (0.0, 0.1, 0.2, 3.0)
    ]
    filtered_nodes = raw_nodes[:3]
    _write_json(semantic / "nodes.json", raw_nodes)
    _write_json(semantic / "nodes_filtered.json", filtered_nodes)
    _write_json(semantic / "summary.json", {
        "node_count": 4,
        "decision_status": "audit_only_not_navigation_ready",
        "action_applied_count": 0,
        "error_count": 0,
        "valid_frame_count": 1,
        "cross_label_conflict_audit": {
            "severe_count": 5, "strong_severe_count": 2,
        },
        "component_filter": {
            "enabled": True,
            "filtered_node_count": 3,
            "stored_surface_retention_rate": 0.75,
            "filtered_cross_label_conflict_audit": {
                "severe_count": 2, "strong_severe_count": 1,
            },
        },
    })
    _write_jsonl(semantic / "events.jsonl", [{
        "valid": True,
        "component_filter": {
            "raw_sample_count": 4, "retained_sample_count": 3,
            "small_component_rejected_sample_count": 1,
            "density_rejected_sample_count": 0,
            "component_count": 2, "edge_touch_component_count": 1,
        },
    }])
    points = np.asarray([
        [0.0, 0.0, 1.0], [0.1, 0.0, 1.0],
        [0.2, 0.0, 1.0], [3.0, 0.0, 1.0],
    ], dtype=np.float32)
    surface = {
        "map_xyz": points,
        "class_id": np.zeros(4, dtype=np.int16),
        "confidence": np.full(4, 0.8, dtype=np.float32),
        "occ_state": np.full(4, 2, dtype=np.int8),
    }
    semantic.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(semantic / "semantic_surface_memory.npz", **surface)
    np.savez_compressed(
        semantic / "semantic_surface_memory_filtered.npz",
        **{key: value[:3] for key, value in surface.items()},
    )
    baseline_semantic = (
        baseline_root / "online_lseg_shadow" / f"{scene_id}_{episode_id}_r0"
    )
    _write_json(baseline_semantic / "episode_meta.json", {
        "scene_id": scene_id, "episode_id": episode_id,
    })
    _write_json(baseline_semantic / "nodes.json", raw_nodes)
    np.savez_compressed(
        baseline_semantic / "semantic_surface_memory.npz", **surface
    )
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, [{
        "scene_id": scene_id, "episode_id": episode_id,
        "episode_eval_seed": seed,
    }])

    result = analyze(
        run_root, baseline_root, manifest, tmp_path / "stage26_audit.json"
    )

    assert result["integrity_passed"]
    assert result["all_trajectories_exact_match"]
    assert result["all_raw_semantics_exact_match"]
    assert result["frame_filter"]["sample_retention_rate"] == 0.75
    assert result["delta"]["gt_hit_count_retention"] == 1.0
    assert result["delta"]["strong_node_retention"] == 0.75
    assert result["delta"]["severe_conflict_reduction_rate"] == 0.6
    assert result["evidence_gate"]["passed"]
