#!/usr/bin/env python3
"""Build causal expert-evidence annotation cards from R2R LeRobot episodes.

This tool only mines candidates. Heuristic scores are ranking signals, never labels.
It is intentionally dependency-light and compatible with Python 3.9.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path


def read_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def scalar(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1:
        return scalar(value[0])
    return value


def as_float_list(value):
    value = scalar(value)
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = [x for row in value for x in row]
    return [float(x) for x in value] if isinstance(value, (list, tuple)) else []


def yaw_from_pose(pose):
    pose = as_float_list(pose)
    if len(pose) == 16:
        return math.atan2(pose[4], pose[0])
    if len(pose) >= 3:
        return float(pose[2])
    if len(pose) >= 7:
        x, y, z, qx, qy, qz, qw = pose[:7]
        return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return 0.0


def rel_pose(current, history):
    c, h = as_float_list(current), as_float_list(history)
    if len(c) < 2 or len(h) < 2:
        return None
    dx, dy = h[0] - c[0], h[1] - c[1]
    dyaw = (yaw_from_pose(history) - yaw_from_pose(current) + math.pi) % (2 * math.pi) - math.pi
    return {"dx": dx, "dy": dy, "distance": math.hypot(dx, dy),
            "dyaw_rad": dyaw, "dyaw_deg": math.degrees(dyaw)}


def action_name(action):
    a = as_float_list(action)
    if not a:
        return "unknown"
    if len(a) == 1:
        return {0: "forward", 1: "turn_left", 2: "turn_right", 3: "look_down", 4: "stop"}.get(int(a[0]), "unknown")
    if len(a) >= 2:
        if abs(a[1]) < 0.05 and abs(a[0]) < 0.05:
            return "stop"
        if a[1] > 0.15:
            return "turn_left"
        if a[1] < -0.15:
            return "turn_right"
    return "forward"


def find_column(table, names):
    for name in names:
        if name in table.column_names:
            return name
    return None


def image_path(root, episode_id, frame_id, rgb_dir):
    candidates = [
        root / "videos" / rgb_dir / f"episode_{episode_id:06d}.mp4",
        root / "video" / rgb_dir / f"episode_{episode_id:06d}.mp4",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def candidate_score(i, j, poses, actions):
    rp = rel_pose(poses[i], poses[j])
    if rp is None:
        return -1.0, rp
    age = i - j
    score = 0.0
    if 2 <= age <= 30:
        score += 1.0
    if age >= 8:
        score += 0.4
    if rp["distance"] < 2.5:
        score += 1.0
    if action_name(actions[i]) != action_name(actions[j]):
        score += 0.7
    return score, rp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--max-episodes", type=int, default=0)
    args = ap.parse_args()
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("需要 pyarrow 才能读取 LeRobot parquet: %s" % exc)
    meta = args.data_root / "meta"
    episodes = read_jsonl(meta / "episodes.jsonl")
    by_ep = {int(e.get("episode_index", e.get("episode_id", -1))): e for e in episodes}
    tasks = {int(e.get("task_index", -1)): str(e.get("task", "")).strip() for e in read_jsonl(meta / "tasks.jsonl")}
    parquet_paths = sorted((args.data_root / "data").glob("chunk-*/*.parquet"))
    if args.max_episodes:
        parquet_paths = parquet_paths[:args.max_episodes]
    rng = random.Random(args.seed)
    candidates = []
    audit = {"parquet_files": len(parquet_paths), "episodes": 0, "frames": 0,
             "causal_violations": 0, "missing_rgb": 0, "candidate_pool": 0}
    for path in parquet_paths:
        table = pq.read_table(path)
        n = table.num_rows
        if not n:
            continue
        cols = {c: table[c].to_pylist() for c in table.column_names}
        ep_ids = cols.get("episode_index", [path.stem] * n)
        ep_id = int(scalar(ep_ids[0]))
        episode = by_ep.get(ep_id, {})
        task_ids = cols.get("task_index", [0])
        instruction = tasks.get(int(scalar(task_ids[0])), "")
        pose_col = find_column(table, ["pose.125cm_0deg", "pose.rgb", "pose.rgb_front", "pose"])
        action_col = find_column(table, ["action", "actions"])
        goal_col = find_column(table, ["goal.125cm_0deg", "goal.rgb", "goal"])
        if not pose_col:
            continue
        poses = cols[pose_col]
        actions = cols.get(action_col, [None] * n)
        audit["episodes"] += 1
        audit["frames"] += n
        for i in range(2, n):
            pool = []
            for j in range(max(0, i - 40), i):
                score, rp = candidate_score(i, j, poses, actions)
                if score < 0 or rp is None:
                    continue
                pool.append((score, j, rp))
            if len(pool) < 2:
                continue
            pool.sort(reverse=True)
            selected = pool[:3]
            if len(selected) < 2:
                continue
            rng.shuffle(selected)
            labels = [chr(ord("A") + k) for k in range(len(selected))]
            annotation_id = "r2r_train_ep%06d_t%04d_%s" % (ep_id, i, hashlib.sha1((str(ep_id)+":"+str(i)).encode()).hexdigest()[:8])
            future_ok = all(j < i for _, j, _ in selected)
            if not future_ok:
                audit["causal_violations"] += 1
                continue
            item = {
                "schema_version": 1, "annotation_id": annotation_id, "split": "train",
                "scene_id": episode.get("scene_id", episode.get("scene", "unknown")),
                "episode_id": str(ep_id), "current_frame_id": i,
                "instruction": instruction, "candidate_frame_ids": [j for _, j, _ in selected],
                "candidate_display_order": labels,
                "causal_check": {"all_history_before_current": True, "future_input_exposed": False},
                "context": {"expert_action": action_name(actions[i]),
                            "expert_local_goal": as_float_list(cols[goal_col][i]) if goal_col else [],
                            "relative_pose_available": True},
                "candidates": [{"label": label, "frame_id": j, "age": i-j, "relative_pose": rp,
                                "image_path": None} for label, (_, j, rp) in zip(labels, selected)],
                "annotation": {"need_history": "", "preferred_evidence": [], "evidence_role": "",
                               "misleading_evidence": [], "confidence": "", "short_reason": ""},
                "annotator_id": "", "created_at": ""
            }
            candidates.append(item)
            audit["candidate_pool"] += 1
            if len(candidates) >= args.limit:
                break
        if len(candidates) >= args.limit:
            break
    if not candidates:
        raise SystemExit("未生成候选：请检查 parquet 字段、pose 列和数据路径")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "annotations_template.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in candidates) + "\n")
    (args.output / "candidate_manifest.jsonl").write_text("\n".join(json.dumps({k:v for k,v in x.items() if k != 'annotation'}, ensure_ascii=False) for x in candidates) + "\n")
    (args.output / "candidate_stats.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    (args.output / "source_manifest.json").write_text(json.dumps({"data_root": str(args.data_root), "seed": args.seed, "limit": args.limit}, ensure_ascii=False, indent=2) + "\n")
    (args.output / "schema.json").write_text(json.dumps({"schema_version": 1, "annotation_fields": ["need_history", "preferred_evidence", "evidence_role", "misleading_evidence", "confidence", "short_reason"]}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"generated": len(candidates), **audit}, ensure_ascii=False))


if __name__ == "__main__":
    main()
