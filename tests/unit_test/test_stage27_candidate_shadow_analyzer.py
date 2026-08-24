import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "eval"
    / "analyze_stage27_m3_candidate_shadow.py"
)
SPEC = importlib.util.spec_from_file_location("stage27_candidate_analyzer", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _candidate(candidate_id: str, grid: list[int]) -> dict:
    return {
        "candidate_id": candidate_id,
        "source_type": "R-route-near",
        "grid": grid,
        "shadow_only": True,
        "action_applied": False,
        "gt_fields_used": [],
        "floor_aligned_known_free": True,
    }


def _event(scene: str, episode: int, step: int, stage_counts: dict[str, int]) -> dict:
    pools = {}
    for stage, count in stage_counts.items():
        candidates = [_candidate(f"{stage}:{index}", [index + 1, index + 1]) for index in range(count)]
        pools[stage] = {"candidates": candidates, "event_has_candidate": bool(candidates)}
    return {
        "scene_id": scene,
        "episode_id": episode,
        "step_id": step,
        "trigger_grid": [0, 0],
        "event_schema_version": "stage27_m3_candidate_generation_v5",
        "candidate_pool_contract": "R-route-near_union_R-route-open",
        "frontier_pool_contract": "F-local-known-safe-frontier",
        "frontier_path_mode": "known_free_geodesic",
        "frontier_candidates": [],
        "shadow_only": True,
        "action_applied": False,
        "gt_fields_used": [],
        "ablation": pools,
    }


def test_manifest_coverage_counts_missing_events_as_zero(tmp_path: Path) -> None:
    stages = (
        "route_only",
        "route_occ",
        "route_occ_clearance",
        "route_occ_clearance_frontier",
    )
    events = [
        _event("scene-a", 1, 10, {stage: 2 for stage in stages}),
        _event("scene-b", 2, 20, {
            "route_only": 1,
            "route_occ": 1,
            "route_occ_clearance": 0,
            "route_occ_clearance_frontier": 1,
        }),
    ]
    event_dir = tmp_path / "run" / "rank0"
    event_dir.mkdir(parents=True)
    (event_dir / "stage27_m3_candidate_events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events), encoding="utf-8"
    )
    manifest = [
        {"scene_id": "scene-a", "episode_id": 1, "step_id": 10, "gt_state": "true_trap", "gt_split": "dev"},
        {"scene_id": "scene-c", "episode_id": 3, "step_id": 30, "gt_state": "true_trap", "gt_split": "holdout"},
        {"scene_id": "scene-b", "episode_id": 2, "step_id": 20, "gt_state": "wrong_way_progress", "gt_split": "dev"},
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = MODULE.analyze(tmp_path / "run", manifest_path)
    coverage = report["manifest_candidate_coverage"]

    assert coverage["unique_expected_event_count"] == 3
    assert coverage["observed_exact_event_count"] == 2
    assert coverage["missing_expected_event_count"] == 1
    assert coverage["all"]["emitted_event_recall"] == 2 / 3
    assert coverage["all"]["reports"]["route_only"]["event_coverage"] == 2 / 3
    assert coverage["all"]["reports"]["route_occ_clearance"]["event_coverage"] == 1 / 3
    assert coverage["by_gt_state"]["true_trap"]["reports"]["route_only"]["event_coverage"] == 1 / 2
    assert coverage["by_gt_state"]["wrong_way_progress"]["reports"]["route_only"]["event_coverage"] == 1.0
    assert coverage["by_gt_split"]["holdout"]["observed_exact_event_count"] == 0
