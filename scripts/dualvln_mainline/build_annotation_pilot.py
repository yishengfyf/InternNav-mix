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
import re
import shutil
from pathlib import Path

from PIL import Image


def read_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    if len(c) == 16 and len(h) == 16:
        world_dx, world_dy = h[3] - c[3], h[7] - c[7]
        current_yaw = yaw_from_pose(current)
        dx = math.cos(current_yaw) * world_dx + math.sin(current_yaw) * world_dy
        dy = -math.sin(current_yaw) * world_dx + math.cos(current_yaw) * world_dy
    else:
        dx, dy = h[0] - c[0], h[1] - c[1]
    dyaw = (yaw_from_pose(history) - yaw_from_pose(current) + math.pi) % (2 * math.pi) - math.pi
    return {"dx": dx, "dy": dy, "distance": math.hypot(dx, dy),
            "dyaw_rad": dyaw, "dyaw_deg": math.degrees(dyaw)}


def action_name(action):
    action = scalar(action)
    if isinstance(action, (int, float)):
        return {0: "stop", 1: "forward", 2: "turn_left", 3: "turn_right", 5: "look_down"}.get(int(action), "unknown")
    a = as_float_list(action)
    if not a:
        return "unknown"
    if len(a) == 1:
        return {0: "stop", 1: "forward", 2: "turn_left", 3: "turn_right", 5: "look_down"}.get(int(a[0]), "unknown")
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


def image_path(root, episode_id, frame_id, rgb_dir="observation.images.rgb.60cm_15deg"):
    candidates = [
        root / "videos" / "chunk-000" / rgb_dir / f"episode_{episode_id:06d}_{frame_id}.jpg",
        root / "videos" / "chunk-000" / rgb_dir / f"episode_{episode_id:06d}_{frame_id:03d}.jpg",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def black_pixel_fraction(path, sample_size=80):
    """Estimate simulator void pixels without loading a full-resolution tensor."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((sample_size, sample_size))
        pixels = list(image.getdata())
    return sum(max(pixel) <= 4 for pixel in pixels) / max(1, len(pixels))


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


def select_history_candidates(i, pool, actions, strategy):
    """Pick causal frames with complementary temporal and spatial roles."""
    if strategy == "legacy":
        pool = sorted(pool, reverse=True)
        strata = [[], [], []]
        for entry in pool:
            age = i - entry[1]
            strata[0 if age <= 3 else 1 if age <= 10 else 2].append(entry)
        selected = [bucket[0] for bucket in strata if bucket]
        selected.extend(entry for entry in pool if entry not in selected)
        return selected[:3]

    selected = []

    def add_best(entries, key):
        remaining = [entry for entry in entries if entry not in selected]
        if remaining:
            selected.append(max(remaining, key=key))

    recent = [entry for entry in pool if 2 <= i - entry[1] <= 4]
    add_best(recent, lambda entry: (-(i - entry[1]), entry[0]))

    transition = [
        entry for entry in pool
        if 4 <= i - entry[1] <= 24 and entry[2]["distance"] >= 0.35
    ]
    add_best(
        transition,
        lambda entry: (
            abs(entry[2]["dyaw_rad"]),
            action_name(actions[i]) != action_name(actions[entry[1]]),
            min(entry[2]["distance"], 4.0),
            i - entry[1],
        ),
    )

    anchor = [
        entry for entry in pool
        if i - entry[1] >= 8 and entry[2]["distance"] >= 0.75
    ]
    add_best(
        anchor,
        lambda entry: (
            min(entry[2]["distance"], 6.0),
            i - entry[1],
            abs(entry[2]["dyaw_rad"]),
        ),
    )

    fallback = sorted(
        pool,
        key=lambda entry: (
            entry[2]["distance"] >= 0.35,
            min(entry[2]["distance"], 6.0),
            abs(entry[2]["dyaw_rad"]),
            i - entry[1],
            entry[0],
        ),
        reverse=True,
    )
    selected.extend(entry for entry in fallback if entry not in selected)
    return selected[:3]


def local_goal(value):
    goal = as_float_list(value)
    if len(goal) >= 2 and goal[0] >= 0 and goal[1] >= 0:
        return goal[:2]
    return None


def decision_for_frame(frame_id, actions, goals, relative_goal_ids):
    shifted_action = actions[frame_id + 1] if frame_id + 1 < len(actions) else 0
    relative_goal_id = int(scalar(relative_goal_ids[frame_id])) if relative_goal_ids else -1
    goal = local_goal(goals[frame_id]) if goals else None
    if relative_goal_id >= 3:
        return ("pixel_goal", goal) if goal is not None else (None, None)
    if relative_goal_id != -1:
        return None, None
    action = action_name(shifted_action)
    if relative_goal_id == -1 and action == "forward":
        return None, None
    return action, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument(
        "--selection-strategy",
        choices=("legacy", "phase_diverse"),
        default="phase_diverse",
    )
    ap.add_argument("--max-black-fraction", type=float, default=0.25)
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
             "causal_violations": 0, "missing_rgb": 0, "candidate_pool": 0,
             "invalid_current_rgb": 0, "invalid_history_rgb": 0}
    quality_cache = {}

    def image_quality(path):
        if path not in quality_cache:
            quality_cache[path] = 1.0 - black_pixel_fraction(path)
        return quality_cache[path]
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
        instruction = (episode.get("tasks") or [tasks.get(int(scalar(task_ids[0])), "")])[0]
        pose_col = find_column(table, ["pose.60cm_15deg"])
        action_col = find_column(table, ["action", "actions"])
        goal_col = find_column(table, ["goal.60cm_15deg"])
        relative_goal_col = find_column(table, ["relative_goal_frame_id.60cm_15deg"])
        if not pose_col:
            continue
        poses = cols[pose_col]
        actions = cols.get(action_col, [None] * n)
        decision_actions = actions[1:] + [0]
        goals = cols.get(goal_col, [])
        relative_goal_ids = cols.get(relative_goal_col, [])
        audit["episodes"] += 1
        audit["frames"] += n
        episode_added = 0
        frame_candidates = sorted(set([max(2, n // 3), max(2, n // 2), max(2, (2 * n) // 3), max(2, n - 2)]))
        for i in frame_candidates:
            expert_action, expert_goal = decision_for_frame(i, actions, goals, relative_goal_ids)
            if expert_action is None:
                continue
            current_image = image_path(args.data_root, ep_id, i)
            if not current_image:
                audit["missing_rgb"] += 1
                continue
            if image_quality(current_image) < 1.0 - args.max_black_fraction:
                audit["invalid_current_rgb"] += 1
                continue
            pool = []
            for j in range(max(0, i - 40), i):
                score, rp = candidate_score(i, j, poses, decision_actions)
                if score < 0 or rp is None:
                    continue
                history_image = image_path(args.data_root, ep_id, j)
                if not history_image:
                    audit["missing_rgb"] += 1
                    continue
                if image_quality(history_image) < 1.0 - args.max_black_fraction:
                    audit["invalid_history_rgb"] += 1
                    continue
                pool.append((score, j, rp))
            if len(pool) < 2:
                continue
            selected = select_history_candidates(
                i, pool, decision_actions, args.selection_strategy
            )
            if len(selected) < 2:
                continue
            rng.shuffle(selected)
            labels = [chr(ord("A") + k) for k in range(len(selected))]
            scene_id = str(episode.get("scene_id", episode.get("scene", args.data_root.name)))
            scene_slug = re.sub(r"[^A-Za-z0-9_-]", "_", Path(scene_id).stem)
            identity = f"{scene_slug}:{ep_id}:{i}"
            annotation_id = "r2r_train_%s_ep%06d_t%04d_%s" % (
                scene_slug,
                ep_id,
                i,
                hashlib.sha1(identity.encode()).hexdigest()[:8],
            )
            future_ok = all(j < i for _, j, _ in selected)
            if not future_ok:
                audit["causal_violations"] += 1
                continue
            history_items = [{"label": label, "frame_id": j, "age": i-j, "relative_pose": rp,
                              "quality": {"pose": 1.0, "observation": image_quality(image_path(args.data_root, ep_id, j))},
                              "image_path": image_path(args.data_root, ep_id, j)}
                             for label, (_, j, rp) in zip(labels, selected)]
            if not current_image or any(not x["image_path"] for x in history_items):
                audit["missing_rgb"] += 1
                continue
            item = {
                "schema_version": 1, "annotation_id": annotation_id, "split": "train",
                "scene_id": scene_id,
                "episode_id": str(ep_id), "current_frame_id": i,
                "instruction": instruction, "candidate_frame_ids": [j for _, j, _ in selected],
                "candidate_display_order": labels,
                "causal_check": {"all_history_before_current": True, "future_input_exposed": False},
                "context": {"expert_action": expert_action,
                            "expert_local_goal": expert_goal,
                            "relative_pose_available": True},
                "current_image_path": current_image, "candidates": history_items,
                "annotation": {"need_history": "", "preferred_evidence": [], "evidence_role": "",
                               "misleading_evidence": [], "confidence": "", "short_reason": ""},
                "annotator_id": "", "created_at": ""
            }
            candidates.append(item)
            audit["candidate_pool"] += 1
            episode_added += 1
            if episode_added >= 2:
                break
        if len(candidates) >= args.limit:
            break
    if not candidates:
        raise SystemExit("未生成候选：请检查 parquet 字段、pose 列和数据路径")
    args.output.mkdir(parents=True, exist_ok=True)
    images = args.output / "images"
    images.mkdir(exist_ok=True)
    for item in candidates:
        paths = [item["current_image_path"]] + [x["image_path"] for x in item["candidates"]]
        for source in paths:
            target = images / (item["annotation_id"] + "_" + Path(source).name)
            shutil.copy2(source, target)
        item["current_image_path"] = "images/" + (item["annotation_id"] + "_" + Path(paths[0]).name)
        for x, source in zip(item["candidates"], paths[1:]):
            x["image_path"] = "images/" + (item["annotation_id"] + "_" + Path(source).name)
    (args.output / "annotations_template.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in candidates) + "\n")
    (args.output / "candidate_manifest.jsonl").write_text("\n".join(json.dumps({k:v for k,v in x.items() if k != 'annotation'}, ensure_ascii=False) for x in candidates) + "\n")
    (args.output / "candidate_stats.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    script_path = Path(__file__).resolve()
    sources = {str(path.relative_to(args.data_root)): sha256(path) for path in
               [meta / "episodes.jsonl", meta / "tasks.jsonl", meta / "info.json"]}
    (args.output / "source_manifest.json").write_text(json.dumps({"data_root": str(args.data_root), "seed": args.seed, "limit": args.limit,
                                                                   "script_path": str(script_path), "script_sha256": sha256(script_path),
                                                                   "source_sha256": sources}, ensure_ascii=False, indent=2) + "\n")
    (args.output / "schema.json").write_text(json.dumps({"schema_version": 1, "annotation_fields": ["need_history", "preferred_evidence", "evidence_role", "misleading_evidence", "confidence", "short_reason"]}, ensure_ascii=False, indent=2) + "\n")
    ui_source = Path(__file__).with_name("annotation_ui.html")
    if not ui_source.exists():
        raise SystemExit("缺少标注页面模板: %s" % ui_source)
    shutil.copy2(ui_source, args.output / "index.html")
    cards = args.output / "cards"
    cards.mkdir(exist_ok=True)
    for item in candidates:
        rows = ["<h2>%s</h2><p>指令：%s</p><p>当前动作：%s；当前帧：%s</p>" % (item["annotation_id"], item["instruction"], item["context"]["expert_action"], item["current_frame_id"]),
                '<h3>当前帧</h3><img src="../%s" width="480">' % item["current_image_path"]]
        for c in item["candidates"]:
            rp = c["relative_pose"]
            rows.append('<h3>历史 %s（frame %s，age %s，距当前 %.2fm，朝向 %.1f°）</h3><img src="../%s" width="320">' % (c["label"], c["frame_id"], c["age"], rp["distance"], rp["dyaw_deg"], c["image_path"]))
        rows.append("<p>人工填写：need_history / preferred_evidence / evidence_role / misleading_evidence / confidence / short_reason</p>")
        (cards / (item["annotation_id"] + ".html")).write_text("<html><meta charset='utf-8'><body>%s</body></html>" % "\n".join(rows), encoding="utf-8")
    selected_candidates = [c for item in candidates for c in item["candidates"]]
    selected_count = max(1, len(selected_candidates))
    metrics = {"candidate_count": len(candidates), "episodes_with_candidates": len(set(x["episode_id"] for x in candidates)),
               "causal_all_history_before_current": audit["causal_violations"] == 0,
               "future_input_exposed": False, "missing_rgb_candidates": audit["missing_rgb"],
               "selection_strategy": args.selection_strategy,
               "max_black_fraction": args.max_black_fraction,
               "invalid_current_rgb": audit["invalid_current_rgb"],
               "invalid_history_rgb": audit["invalid_history_rgb"],
               "selected_age_mean": sum(c["age"] for c in selected_candidates) / selected_count,
               "selected_distance_mean_m": sum(c["relative_pose"]["distance"] for c in selected_candidates) / selected_count,
               "selected_abs_yaw_mean_deg": sum(abs(c["relative_pose"]["dyaw_deg"]) for c in selected_candidates) / selected_count,
               "degenerate_pose_rate": sum(c["relative_pose"]["distance"] < 0.05 for c in selected_candidates) / selected_count}
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    summary = """# DualVLN 人工证据标注 Pilot 自动挖掘结果

本目录只包含程序生成的候选，不包含人工标签。候选排序启发式不是监督标签，后续必须由人工填写 `annotation` 字段。

- 候选数：%d
- 覆盖 episode 数：%d
- 因果前缀检查：%s
- 未来输入暴露：否
- RGB 缺失候选：%d
- 选择策略：`%s`
- 历史平均年龄：`%.2f` 帧；平均位移：`%.2f` 米；平均朝向差：`%.1f` 度
- 退化位姿比例（位移 < 0.05 米）：`%.1f%%`
- 黑区质量过滤：当前帧 `%d` 次、历史池 `%d` 次被拒绝（阈值 `%.0f%%`）

下一步：先抽查 HTML 卡片和图像，再由第一位标注者填写 30--40 张；其中约 20--30%% 交给第二位标注者复核。只有一致性和 relevance probe 达标后，才扩展到 100--200 张。
""" % (metrics["candidate_count"], metrics["episodes_with_candidates"], "通过" if metrics["causal_all_history_before_current"] else "失败", metrics["missing_rgb_candidates"], metrics["selection_strategy"], metrics["selected_age_mean"], metrics["selected_distance_mean_m"], metrics["selected_abs_yaw_mean_deg"], 100 * metrics["degenerate_pose_rate"], metrics["invalid_current_rgb"], metrics["invalid_history_rgb"], 100 * metrics["max_black_fraction"])
    (args.output / "summary.md").write_text(summary, encoding="utf-8")
    (args.output / "README.md").write_text("# 标注目录使用说明\n\n推荐启动本地静态服务后打开 `index.html`：页面支持逐张填写、浏览器自动暂存、导入已有 JSONL 和导出标注 JSONL。也可以打开 `cards/` 中的只读 HTML 抽查候选。禁止修改帧号、位姿、动作和因果检查字段；人工只填写 `annotation` 字段。\n", encoding="utf-8")
    (args.output / "run.log").write_text(json.dumps({"status": "completed", **metrics}, ensure_ascii=False) + "\n", encoding="utf-8")
    svg = "<svg xmlns='http://www.w3.org/2000/svg' width='640' height='220'><rect width='100%%' height='100%%' fill='white'/><text x='24' y='32' font-size='20'>Annotation pilot candidate audit</text><text x='24' y='78'>候选数</text><rect x='140' y='60' width='%d' height='24' fill='#3878c7'/><text x='150' y='78' fill='white'>%d</text><text x='24' y='124'>覆盖 episode</text><rect x='140' y='106' width='%d' height='24' fill='#4b9e62'/><text x='150' y='124' fill='white'>%d</text><text x='24' y='170'>因果检查：%s</text></svg>" % (min(420, metrics["candidate_count"] * 10), metrics["candidate_count"], min(420, metrics["episodes_with_candidates"] * 10), metrics["episodes_with_candidates"], "通过" if metrics["causal_all_history_before_current"] else "失败")
    (args.output / "metrics.svg").write_text(svg, encoding="utf-8")
    print(json.dumps({"generated": len(candidates), **audit}, ensure_ascii=False))


if __name__ == "__main__":
    main()
