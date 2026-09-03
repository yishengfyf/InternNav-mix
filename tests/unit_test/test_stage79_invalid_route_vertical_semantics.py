import json
from pathlib import Path

from scripts.eval.analyze_stage79_invalid_route_vertical_semantics import analyze


def _jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_stage79_attributes_vertical_false_block_without_releasing_semantics(tmp_path):
    root = tmp_path / "return"
    events = root / "semantic_debug" / "rank0" / "s2_recovery_context_events.jsonl"
    native = {
        "event_type": "stage65_native_recovery_set", "scene_id": "scene", "episode_id": 1,
        "stage59_productive_onset": {"anchors": [{
            "anchor": "last_productive_pre_loop", "grid": [10, 12],
            "route_state_counts": {"occupied": 2}, "current_sparseocc_connectivity": False,
            "image_bridge_reason": "anchor_not_free",
            "offline_primitive_truth": {"valid": True, "primitive_safe": True},
            "stage58_support_policy": {"arms": []},
        }]},
    }
    spatial = {
        "channels": {
            "known_free": [[10, 10]], "occupied": [[10, 12]], "unknown": [[10, 11]],
            "semantic_nodes": [{"label": "stairs", "grid": [10, 12],
                                "centroid": [0.0, 0.0, 1.0], "evidence_tier": "strong"}],
        },
        "executed_route": [[10, 9], [10, 10]],
        "current_pose": {"z": 0.0, "gt_relative_height_m": 1.0,
                         "pose_height_source": "gps_compass_2d"},
    }
    route = {
        "event_type": "stage75_route_guidance", "scene_id": "scene", "episode_id": 1,
        "current_query_step": 5, "valid": False, "reason": "anchor_not_free",
        "start_grid": [10, 10], "anchor_grid": [10, 12],
        "stage78_recovery_bev_spatial": spatial,
    }
    _jsonl(events, [native, route])
    (root / "stage78_semantic_attachment_audit.json").write_text(
        json.dumps({"stable_route_landmark_episode_keys": []}), encoding="utf-8"
    )
    report = analyze(input_root=root, output=tmp_path / "audit.json", viz_dir=tmp_path / "viz")
    assert report["integrity_passed"] is True
    assert report["vertical_pose_omission_suspect_episode_count"] == 1
    assert report["event_reports"][0]["anchor_state"] == "occupied"
    assert report["event_reports"][0]["attribution"] == "vertical_pose_omission_false_block_suspect"
    assert report["event_reports"][0]["nearest_semantic_to_anchor_m"] == 0.0
    assert report["semantic_prompt_release_gate_passed"] is False
    assert report["contract"]["action_applied"] is False
    assert (tmp_path / "viz" / "scene_1_invalid_route.png").is_file()
