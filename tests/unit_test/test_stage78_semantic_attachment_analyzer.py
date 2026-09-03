import json
from pathlib import Path

from scripts.eval.analyze_stage78_semantic_attachment import analyze


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_stage78_analyzer_requires_shadow_contract(tmp_path):
    run = tmp_path / "run"
    baseline = tmp_path / "baseline"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps([{"scene_id": "scene", "episode_id": 1, "episode_eval_seed": 7}]),
        encoding="utf-8",
    )
    progress = [{"scene_id": "scene", "episode_id": 1}]
    _write_jsonl(run / "progress.json", progress)
    query = {"scene_id": "scene", "episode_id": 1, "query_id": 0, "output": "←"}
    action = {"scene_id": "scene", "episode_id": 1, "step_id": 0, "action": 2}
    for root in (run, baseline):
        episode = root / "x" / "replay_ledger" / "scene_1_r0"
        (episode).mkdir(parents=True, exist_ok=True)
        (episode / "episode_meta.json").write_text(
            json.dumps({"scene_id": "scene", "episode_id": 1}), encoding="utf-8"
        )
        _write_jsonl(episode / "queries.jsonl", [query])
        _write_jsonl(episode / "actions.jsonl", [action])
        _write_jsonl(episode / "observations.jsonl", [])
    summary = {
        "scene_id": "scene", "episode_id": 1, "error_count": 0,
        "decision_status": "audit_only_not_navigation_ready",
        "action_applied_count": 0,
    }
    summary_path = run / "x" / "online_lseg_shadow" / "scene_1_r0" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    spatial = {
        "center_grid": [10, 10],
        "channels": {"known_free": [[10, 10]], "occupied": [], "unknown": [], "semantic": [], "semantic_nodes": []},
        "executed_route": [], "current_pose": {"grid": [10, 10], "yaw": 0.0},
        "candidate_path": [[10, 10]], "unknown_is_free": False,
        "semantic_can_override_safety": False,
    }
    attachment = {
        "valid": True, "semantic_node_count": 1, "route_bound_node_count": 1,
        "stable_route_bound_node_count": 1, "structural_route_bound_node_count": 1,
        "label_counts": {"door": 1}, "stable_label_counts": {"door": 1},
        "occ_state_at_centroid_counts": {"occupied": 1},
        "route_bound_nodes": [{"label": "door", "structural_label": True}],
        "unknown_is_free": False, "semantic_can_override_safety": False,
        "prompt_injected": False,
    }
    _write_jsonl(
        run / "x" / "s2_recovery_context_events.jsonl",
        [{"event_type": "stage75_route_guidance", "scene_id": "scene", "episode_id": 1,
          "current_query_step": 3, "valid": True, "start_grid": [10, 10],
          "anchor_grid": [10, 11], "path_preview": [[10, 10]],
          "stage78_semantic_route_attachment": attachment,
          "stage78_recovery_bev_spatial": spatial}],
    )
    report = analyze(
        run_root=run, baseline_root=baseline, manifest=manifest,
        output=tmp_path / "report.json", bev_dir=tmp_path / "bev",
    )
    assert report["integrity_passed"] is True
    assert report["trajectory_exact_match"] is True
    assert report["stable_route_landmark_episode_count"] == 1
    assert report["semantic_centroid_occ_state_counts"] == {"occupied": 1}
    assert report["bev_render_count"] == 1
