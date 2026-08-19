"""Check Stage24A replay files for per-episode alignment and shadow invariants."""

import argparse
import json
from pathlib import Path


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def analyze(run_root: Path, output: Path):
    errors = []
    episodes_by_key = {}
    progress_paths = list(run_root.glob("**/progress.json"))
    for progress_path in progress_paths:
        for row in _jsonl(progress_path):
            if not row.get("replay_ledger_enabled"):
                continue
            scene_id = str(row.get("scene_id"))
            episode_id = str(row.get("episode_id"))
            ledger_root = row.get("replay_ledger_dir")
            if not ledger_root:
                errors.append(f"missing_ledger_root:{scene_id}/{episode_id}")
                continue
            candidates = sorted(Path(ledger_root).glob(f"{scene_id}_{episode_id}_r*/summary.json"))
            if not candidates:
                errors.append(f"missing_summary:{scene_id}/{episode_id}")
                continue
            summary_path = candidates[-1]
            episode_dir = summary_path.parent
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            metadata = json.loads((episode_dir / "episode_meta.json").read_text(encoding="utf-8"))
            observations = _jsonl(episode_dir / "observations.jsonl")
            queries = _jsonl(episode_dir / "queries.jsonl")
            actions = _jsonl(episode_dir / "actions.jsonl")
            keys = [item.get("observation_key") for item in observations]
            key_set = set(keys)
            if len(keys) != len(set(keys)):
                errors.append(f"duplicate_observation_key:{scene_id}/{episode_id}")
            if len(observations) != int(summary.get("observation_count", -1)):
                errors.append(f"observation_count_mismatch:{scene_id}/{episode_id}")
            if len(queries) != int(summary.get("query_count", -1)):
                errors.append(f"query_count_mismatch:{scene_id}/{episode_id}")
            if len(actions) != int(summary.get("action_count", -1)):
                errors.append(f"action_count_mismatch:{scene_id}/{episode_id}")
            if any("action_applied" not in item for item in actions):
                errors.append(f"action_applied_missing:{scene_id}/{episode_id}")
            camera_model = metadata.get("camera_model") or {}
            intrinsic = camera_model.get("intrinsic")
            if not (
                isinstance(intrinsic, list)
                and len(intrinsic) == 3
                and all(isinstance(row, list) and len(row) == 3 for row in intrinsic)
            ):
                errors.append(f"camera_intrinsic_missing:{scene_id}/{episode_id}")
            semantic_gt = metadata.get("semantic_scene_gt") or {}
            if not semantic_gt.get("available"):
                errors.append(f"semantic_scene_gt_missing:{scene_id}/{episode_id}")
            transforms = metadata.get("coordinate_transforms") or {}
            map_to_gt = transforms.get("map_to_habitat_world")
            if not (
                isinstance(map_to_gt, list)
                and len(map_to_gt) == 4
                and all(isinstance(row, list) and len(row) == 4 for row in map_to_gt)
            ):
                errors.append(f"map_to_gt_transform_missing:{scene_id}/{episode_id}")
            for item in observations:
                pose = item.get("pose") or {}
                if pose.get("stage23a_sensor_position") is None:
                    errors.append(f"sensor_position_missing:{scene_id}/{episode_id}:{item.get('observation_key')}")
                if pose.get("stage23a_sensor_rotation_wxyz") is None:
                    errors.append(f"sensor_rotation_missing:{scene_id}/{episode_id}:{item.get('observation_key')}")
                camera_pose = pose.get("stage23_gt_camera_pose_map")
                if not (
                    isinstance(camera_pose, list)
                    and len(camera_pose) == 4
                    and all(isinstance(row, list) and len(row) == 4 for row in camera_pose)
                ):
                    errors.append(f"camera_pose_map_missing:{scene_id}/{episode_id}:{item.get('observation_key')}")
                for path_key in ("rgb_path", "depth_path"):
                    relative = item.get(path_key)
                    if not relative or not (episode_dir / relative).is_file():
                        errors.append(f"{path_key}_missing:{scene_id}/{episode_id}:{item.get('observation_key')}")
            for kind, items in (("query", queries), ("action", actions)):
                for item in items:
                    if item.get("observation_key") not in key_set:
                        errors.append(
                            f"{kind}_observation_reference_invalid:{scene_id}/{episode_id}:"
                            f"{item.get('observation_key')}"
                        )
            if int(row.get("s2_loop_strict_active_applied_count", 0) or 0):
                errors.append(f"strict_active_action_violation:{scene_id}/{episode_id}")
            if int(row.get("s2_loop_path_reobserve_applied_count", 0) or 0):
                errors.append(f"path_reobserve_action_violation:{scene_id}/{episode_id}")
            rank = int(summary.get("rank", row.get("rank", 0)) or 0)
            key = (scene_id, episode_id, rank)
            episodes_by_key[key] = {
                "scene_id": scene_id,
                "episode_id": episode_id,
                "rank": rank,
                "success": row.get("success"),
                "steps": row.get("steps"),
                "observation_count": len(observations),
                "query_count": len(queries),
                "action_count": len(actions),
                "applied_action_count": sum(bool(item.get("action_applied")) for item in actions),
                "discarded_action_count": sum(not bool(item.get("action_applied")) for item in actions),
                "semantic_gt_object_count": len(semantic_gt.get("objects") or []),
                "semantic_gt_region_count": len(semantic_gt.get("regions") or []),
                "ledger_dir": str(episode_dir),
            }
    episodes = [episodes_by_key[key] for key in sorted(episodes_by_key)]
    report = {
        "audit_name": "stage24a_replay_ledger",
        "integrity_passed": not errors,
        "episode_count": len(episodes),
        "episodes": episodes,
        "errors": errors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run_root, args.output)


if __name__ == "__main__":
    main()
