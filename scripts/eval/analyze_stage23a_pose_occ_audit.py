#!/usr/bin/env python3
"""Audit Stage23A current-pose and GT-height SparseOcc outputs."""

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
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {_key(row): row for row in rows}


def _nonempty(path, minimum=64):
    return path.is_file() and path.stat().st_size >= minimum


def _combine_views(run_root, keys):
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return [], ["PIL_unavailable"]
    out_dir = run_root / "stage23a_visual_comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    errors = []
    for scene_id, episode_id in keys:
        rank_dirs = list(run_root.glob("vlmap_safety_debug/rank*_run_*"))
        owner = None
        for rank_dir in rank_dirs:
            summary = rank_dir / "occ_memory" / "memory_episode_summary.jsonl"
            if summary.is_file() and any(
                _key(row) == (scene_id, episode_id) for row in _read_jsonl(summary)
            ):
                owner = rank_dir
                break
        if owner is None:
            errors.append(f"missing_owner:{scene_id}/{episode_id}")
            continue
        current_dir = owner / "occ_memory" / "validation"
        oracle_dir = owner / "stage23a_oracle_pose" / "occ_memory" / "validation"
        for view in (
            "bev_xy",
            "side_xz",
            "side_yz",
            "surface_bev_xy",
            "surface_side_xz",
            "surface_side_yz",
        ):
            current = list(current_dir.glob(f"ep{episode_id}_step*_final_{view}.png"))
            oracle = list(oracle_dir.glob(f"ep{episode_id}_step*_final_{view}.png"))
            if len(current) != 1 or len(oracle) != 1:
                errors.append(f"missing_pair:{scene_id}/{episode_id}:{view}")
                continue
            left = Image.open(current[0]).convert("RGB")
            right = Image.open(oracle[0]).convert("RGB")
            width = left.width + right.width
            height = max(left.height, right.height) + 30
            canvas = Image.new("RGB", (width, height), (18, 18, 18))
            canvas.paste(left, (0, 30))
            canvas.paste(right, (left.width, 30))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 8), "current GPS+compass (z=0)", fill=(245, 245, 245))
            draw.text((left.width + 8, 8), "same pose + Habitat relative height (GT-only)", fill=(245, 245, 245))
            output = out_dir / f"{scene_id}_{episode_id}_{view}_current_vs_oracle.png"
            canvas.save(output)
            outputs.append(str(output))
    return outputs, errors


def analyze(
    run_root,
    manifest_path,
    reference_root,
    output,
    require_all,
    min_flat_episodes=1,
    min_height_change_episodes=1,
):
    expected = _load_manifest(manifest_path)
    progress = _load_unique(run_root.glob("vlmap_safety_debug/rank*_run_*/progress.json"))
    current = _load_unique(
        run_root.glob("vlmap_safety_debug/rank*_run_*/occ_memory/memory_episode_summary.jsonl"),
        "occ_memory_episode_summary",
    )
    oracle = _load_unique(
        run_root.glob("vlmap_safety_debug/rank*_run_*/stage23a_oracle_pose/occ_memory/memory_episode_summary.jsonl"),
        "occ_memory_episode_summary",
    )
    reference = _load_unique(reference_root.glob("rank*_run_*/progress.json"))
    if not reference:
        reference = _load_unique(reference_root.glob("progress.json"))

    errors = []
    episode_rows = []
    action_violation_fields = (
        "occ_memory_stage21_multitask_shadow_action_applied_count",
        "s2_loop_path_reobserve_applied_count",
        "s2_loop_path_reobserve_intervention_count",
        "s2_loop_path_reobserve_pixel_rewrite_count",
        "s2_loop_strict_active_applied_count",
        "s2_loop_strict_active_rewrite_count",
        "stage19_semantic_resilience_active_applied_count",
    )
    required_suffixes = (
        "_accumulated_rgb_surface.ply",
        "_occupied_only.ply",
        "_free_only.ply",
        "_trajectory.ply",
        "_bev_xy.png",
        "_side_xz.png",
        "_side_yz.png",
        "_surface_bev_xy.png",
        "_surface_side_xz.png",
        "_surface_side_yz.png",
        "_pose_height_audit.json",
    )
    for key, manifest_row in expected.items():
        p = progress.get(key)
        c = current.get(key)
        o = oracle.get(key)
        if p is None or c is None or o is None:
            errors.append(f"missing_rows:{key}")
            continue
        if int(p.get("episode_eval_seed", -1)) != int(manifest_row["episode_eval_seed"]):
            errors.append(f"seed_mismatch:{key}")
        if int(c.get("update_count", -1)) != int(o.get("update_count", -2)):
            errors.append(f"branch_update_mismatch:{key}")
        if int(o.get("update_count", 0)) <= 0:
            errors.append(f"empty_oracle_branch:{key}")
        if p.get("stage23a_gt_fields_used_for_navigation"):
            errors.append(f"gt_navigation_leakage:{key}")
        action_violation_count = sum(
            int(p.get(name, 0) or 0) for name in action_violation_fields
        )
        if action_violation_count:
            errors.append(f"shadow_action_violation:{key}:{action_violation_count}")
        ref = reference.get(key)
        navigation_match = None
        if ref is not None:
            navigation_match = all(
                abs(float(p.get(name, 0.0)) - float(ref.get(name, 0.0))) <= 1e-6
                for name in ("success", "spl", "ne", "steps")
            )
            if not navigation_match:
                errors.append(f"navigation_reference_mismatch:{key}")
        height_range = float(o.get("validation_gt_relative_height_range_m") or 0.0)
        current_height_range = float(c.get("validation_pose_height_range_m") or 0.0)
        oracle_height_range = float(o.get("validation_pose_height_range_m") or 0.0)
        if abs(current_height_range) > 1e-4:
            errors.append(f"current_branch_height_not_zero:{key}")
        if abs(oracle_height_range - height_range) > 1e-3:
            errors.append(f"oracle_height_not_applied:{key}")

        owner_candidates = []
        for rank_dir in run_root.glob("vlmap_safety_debug/rank*_run_*"):
            summary_path = rank_dir / "occ_memory" / "memory_episode_summary.jsonl"
            if summary_path.is_file() and any(_key(row) == key for row in _read_jsonl(summary_path)):
                owner_candidates.append(rank_dir)
        if len(owner_candidates) != 1:
            errors.append(f"owner_count:{key}:{len(owner_candidates)}")
            files_ok = False
        else:
            files_ok = True
            for branch_dir in (
                owner_candidates[0] / "occ_memory" / "validation",
                owner_candidates[0] / "stage23a_oracle_pose" / "occ_memory" / "validation",
            ):
                for suffix in required_suffixes:
                    matches = list(branch_dir.glob(f"ep{key[1]}_step*_final{suffix}"))
                    if len(matches) != 1 or not _nonempty(matches[0]):
                        errors.append(f"missing_visual:{key}:{branch_dir}:{suffix}")
                        files_ok = False
        episode_rows.append(
            {
                "scene_id": key[0],
                "episode_id": key[1],
                "audit_role": manifest_row.get("audit_role"),
                "reference_path_height_range_m": manifest_row.get(
                    "reference_path_height_range_m"
                ),
                "episode_eval_seed": p.get("episode_eval_seed"),
                "steps": p.get("steps"),
                "success": p.get("success"),
                "navigation_reference_match": navigation_match,
                "current_update_count": c.get("update_count"),
                "oracle_update_count": o.get("update_count"),
                "gt_relative_height_range_m": height_range,
                "current_pose_height_range_m": current_height_range,
                "oracle_pose_height_range_m": oracle_height_range,
                "current_height_error_p95_m": c.get("validation_height_abs_error_p95_m"),
                "oracle_height_error_p95_m": o.get("validation_height_abs_error_p95_m"),
                "current_endpoint_mapped_ratio": c.get(
                    "validation_endpoint_mapped_ratio"
                ),
                "oracle_endpoint_mapped_ratio": o.get(
                    "validation_endpoint_mapped_ratio"
                ),
                "current_endpoint_below_volume_ratio": c.get(
                    "validation_endpoint_below_volume_ratio"
                ),
                "oracle_endpoint_below_volume_ratio": o.get(
                    "validation_endpoint_below_volume_ratio"
                ),
                "current_endpoint_above_volume_ratio": c.get(
                    "validation_endpoint_above_volume_ratio"
                ),
                "oracle_endpoint_above_volume_ratio": o.get(
                    "validation_endpoint_above_volume_ratio"
                ),
                "current_endpoint_negative_z_ratio": c.get(
                    "validation_endpoint_negative_z_ratio"
                ),
                "oracle_endpoint_negative_z_ratio": o.get(
                    "validation_endpoint_negative_z_ratio"
                ),
                "current_endpoint_negative_z_mapped_ratio": c.get(
                    "validation_endpoint_negative_z_mapped_ratio"
                ),
                "oracle_endpoint_negative_z_mapped_ratio": o.get(
                    "validation_endpoint_negative_z_mapped_ratio"
                ),
                "visual_bundle_complete": files_ok,
                "shadow_action_violation_count": action_violation_count,
            }
        )

    ranges = [row["gt_relative_height_range_m"] for row in episode_rows]
    flat_count = sum(value <= 0.20 for value in ranges)
    height_change_count = sum(value >= 0.40 for value in ranges)
    if flat_count < int(min_flat_episodes):
        errors.append("missing_measured_flat_episode")
    if height_change_count < int(min_height_change_episodes):
        errors.append("missing_measured_height_change_episode")
    comparison_paths, comparison_errors = _combine_views(run_root, expected.keys())
    errors.extend(comparison_errors)
    report = {
        "audit_name": "stage23a_pose_occ_audit",
        "integrity_passed": not errors,
        "expected_episode_count": len(expected),
        "completed_episode_count": len(episode_rows),
        "flat_episode_count": flat_count,
        "height_change_episode_count": height_change_count,
        "gt_navigation_leakage_count": sum(
            bool(row.get("stage23a_gt_fields_used_for_navigation"))
            for row in progress.values()
        ),
        "shadow_action_violation_count": sum(
            row.get("shadow_action_violation_count", 0) for row in episode_rows
        ),
        "comparison_images": comparison_paths,
        "episodes": episode_rows,
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
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--min-flat-episodes", type=int, default=1)
    parser.add_argument("--min-height-change-episodes", type=int, default=1)
    args = parser.parse_args()
    analyze(
        args.run_root,
        args.manifest,
        args.reference_root,
        args.output,
        args.require_all,
        min_flat_episodes=args.min_flat_episodes,
        min_height_change_episodes=args.min_height_change_episodes,
    )


if __name__ == "__main__":
    main()
