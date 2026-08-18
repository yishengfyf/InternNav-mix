#!/usr/bin/env python3
"""Audit complete Habitat sensor-pose endpoint and observed-volume OCC metrics."""

import argparse
import json
from pathlib import Path


def _read_jsonl(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _key(row):
    return str(row.get("scene_id")), int(row.get("episode_id"))


def _load_unique(paths, event_type=None):
    rows = {}
    for path in paths:
        for row in _read_jsonl(path):
            if event_type and row.get("event_type") != event_type:
                continue
            rows[_key(row)] = row
    return rows


def _load_manifest(path):
    return {_key(row): row for row in json.loads(path.read_text(encoding="utf-8"))}


def _find_comparison(run_root, key, comparison_dir):
    for path in run_root.glob(
        f"vlmap_safety_debug/*run_*/{comparison_dir}/*.json"
    ):
        if path.stem == f"{key[0]}_{key[1]}_comparison":
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def analyze(
    run_root,
    manifest,
    output,
    require_all,
    require_height_ablation,
    require_mesh_raycast,
    require_signed_mesh,
    require_navmesh_traversability,
):
    expected = _load_manifest(manifest)
    progress = _load_unique(run_root.glob("vlmap_safety_debug/*run_*/progress.json"))
    current = _load_unique(
        run_root.glob("vlmap_safety_debug/*run_*/occ_memory/memory_episode_summary.jsonl"),
        "occ_memory_episode_summary",
    )
    oracle = _load_unique(
        run_root.glob(
            "vlmap_safety_debug/*run_*/stage23a_oracle_sensor_pose/occ_memory/memory_episode_summary.jsonl"
        ),
        "occ_memory_episode_summary",
    )
    oracle_height = _load_unique(
        run_root.glob(
            "vlmap_safety_debug/*run_*/stage23a_oracle_pose/occ_memory/memory_episode_summary.jsonl"
        ),
        "occ_memory_episode_summary",
    )
    errors = []
    episodes = []
    action_fields = (
        "occ_memory_stage21_multitask_shadow_action_applied_count",
        "s2_loop_path_reobserve_applied_count",
        "s2_loop_path_reobserve_intervention_count",
        "s2_loop_path_reobserve_pixel_rewrite_count",
        "s2_loop_strict_active_applied_count",
        "s2_loop_strict_active_rewrite_count",
        "stage19_semantic_resilience_active_applied_count",
    )
    for key, manifest_row in expected.items():
        p, c, o = progress.get(key), current.get(key), oracle.get(key)
        if p is None or c is None or o is None:
            errors.append(f"missing_rows:{key}")
            continue
        h = oracle_height.get(key)
        if require_height_ablation and h is None:
            errors.append(f"missing_oracle_height_row:{key}")
            continue
        if int(p.get("episode_eval_seed", -1)) != int(manifest_row["episode_eval_seed"]):
            errors.append(f"seed_mismatch:{key}")
        if int(c.get("update_count", -1)) != int(o.get("update_count", -2)):
            errors.append(f"branch_update_mismatch:{key}")
        if require_height_ablation and int(c.get("update_count", -1)) != int(
            h.get("update_count", -2)
        ):
            errors.append(f"height_branch_update_mismatch:{key}")
        if not p.get("stage23a_oracle_sensor_pose_audit_enabled"):
            errors.append(f"sensor_branch_disabled:{key}")
        if p.get("stage23a_gt_fields_used_for_navigation"):
            errors.append(f"gt_navigation_leakage:{key}")
        violations = sum(int(p.get(name, 0) or 0) for name in action_fields)
        if violations:
            errors.append(f"shadow_action_violation:{key}:{violations}")
        comparison = _find_comparison(
            run_root, key, "stage23a_sensor_occ_comparison"
        )
        if comparison is None:
            errors.append(f"missing_sensor_comparison:{key}")
        height_comparison = _find_comparison(
            run_root, key, "stage23a_pose_occ_comparison"
        )
        if require_height_ablation and height_comparison is None:
            errors.append(f"missing_height_comparison:{key}")
        mesh_raycast = p.get("stage23a_mesh_raycast") or {}
        if require_mesh_raycast and (
            not mesh_raycast.get("enabled")
            or int(mesh_raycast.get("total_rays", 0) or 0) <= 0
            or int(mesh_raycast.get("hit_count", 0) or 0) <= 0
            or int(mesh_raycast.get("endpoint_error_count", 0) or 0) <= 0
        ):
            errors.append(f"missing_mesh_raycast:{key}")
        if require_signed_mesh and (
            int(mesh_raycast.get("signed_error_count", 0) or 0) <= 0
            or mesh_raycast.get("surface_match_abs_le_0_05m_rate") is None
            or mesh_raycast.get("potential_false_free_gt_0_05m_rate") is None
            or mesh_raycast.get("potential_false_occupied_lt_neg_0_05m_rate") is None
        ):
            errors.append(f"missing_signed_mesh_raycast:{key}")
        navmesh_current = p.get("stage23b_navmesh_traversability_current") or {}
        navmesh_oracle = (
            p.get("stage23b_navmesh_traversability_oracle_sensor") or {}
        )
        if require_navmesh_traversability:
            for branch, navmesh in (
                ("current", navmesh_current),
                ("oracle_sensor", navmesh_oracle),
            ):
                if (
                    not navmesh.get("enabled")
                    or not navmesh.get("valid")
                    or int(navmesh.get("sampled_cell_count", 0) or 0) <= 0
                    or int(navmesh.get("executed_route_cell_count", 0) or 0) <= 0
                    or navmesh.get("executed_route_navmesh_free_recall") is None
                    or int(navmesh.get("pair_count", 0) or 0) <= 0
                    or navmesh.get("reachability_agreement") is None
                ):
                    errors.append(f"missing_navmesh_traversability:{branch}:{key}")
        endpoint = c.get("validation_endpoint_gt_error_stats") or {}
        if not endpoint.get("count"):
            errors.append(f"missing_endpoint_gt_stats:{key}")
        episodes.append(
            {
                "scene_id": key[0],
                "episode_id": key[1],
                "audit_role": manifest_row.get("audit_role"),
                "episode_eval_seed": p.get("episode_eval_seed"),
                "steps": p.get("steps"),
                "success": p.get("success"),
                "current_update_count": c.get("update_count"),
                "oracle_sensor_update_count": o.get("update_count"),
                "oracle_height_update_count": (
                    h.get("update_count") if h is not None else None
                ),
                "pose_xy_error_stats": c.get("validation_pose_gt_xy_error_stats"),
                "pose_z_error_stats": c.get("validation_pose_gt_z_error_stats"),
                "pose_yaw_error_stats": c.get("validation_pose_gt_yaw_error_deg_stats"),
                "endpoint_error_stats": endpoint,
                "endpoint_error_groups": c.get("validation_endpoint_gt_error_groups"),
                "endpoint_mapped_ratio": c.get("validation_endpoint_mapped_ratio"),
                "comparison": comparison,
                "current_to_oracle_height_comparison": height_comparison,
                "shadow_action_violation_count": violations,
                "mesh_raycast": mesh_raycast,
                "navmesh_traversability_current": navmesh_current,
                "navmesh_traversability_oracle_sensor": navmesh_oracle,
            }
        )
    report = {
        "audit_name": "stage23a_sensor_pose_occ",
        "integrity_passed": not errors,
        "expected_episode_count": len(expected),
        "completed_episode_count": len(episodes),
        "gt_navigation_leakage_count": sum(
            bool(row.get("stage23a_gt_fields_used_for_navigation"))
            for row in progress.values()
        ),
        "shadow_action_violation_count": sum(
            int(row["shadow_action_violation_count"]) for row in episodes
        ),
        "height_ablation_required": bool(require_height_ablation),
        "mesh_raycast_required": bool(require_mesh_raycast),
        "signed_mesh_raycast_required": bool(require_signed_mesh),
        "navmesh_traversability_required": bool(require_navmesh_traversability),
        "episodes": episodes,
        "errors": errors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if require_all and errors:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--require-height-ablation", action="store_true")
    parser.add_argument("--require-mesh-raycast", action="store_true")
    parser.add_argument("--require-signed-mesh", action="store_true")
    parser.add_argument("--require-navmesh-traversability", action="store_true")
    args = parser.parse_args()
    analyze(
        args.run_root,
        args.manifest,
        args.output,
        args.require_all,
        args.require_height_ablation,
        args.require_mesh_raycast,
        args.require_signed_mesh,
        args.require_navmesh_traversability,
    )


if __name__ == "__main__":
    main()
