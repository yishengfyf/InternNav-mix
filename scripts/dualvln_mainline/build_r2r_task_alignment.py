#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.raw, "rt", encoding="utf-8") as stream:
        raw = json.load(stream)["episodes"]
    converted = [
        json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    scene_episodes = [episode for episode in raw if args.scene in episode["scene_id"]]
    by_text = defaultdict(list)
    by_trajectory = defaultdict(list)
    for episode in scene_episodes:
        text = normalize(episode["instruction"]["instruction_text"])
        by_text[text].append(episode)
        by_trajectory[episode["trajectory_id"]].append(text)
    rows, errors = [], []
    converted_texts = [normalize(item["task"]) for item in converted]
    duplicate_texts = sorted({text for text in converted_texts if converted_texts.count(text) > 1})
    task_indices = [item["task_index"] for item in converted]
    duplicate_indices = sorted({index for index in task_indices if task_indices.count(index) > 1})
    if duplicate_texts:
        errors.append({"kind": "duplicate_converted_instructions", "values": duplicate_texts})
    if duplicate_indices:
        errors.append({"kind": "duplicate_task_indices", "values": duplicate_indices})
    for item in converted:
        text = normalize(item["task"])
        matches = by_text[text]
        if len(matches) != 1:
            errors.append(
                {"kind": "raw_match_count", "task_index": item["task_index"], "matches": len(matches), "instruction": text}
            )
            continue
        trajectory_id = matches[0]["trajectory_id"]
        rows.append(
            {
                "task_index": item["task_index"],
                "instruction": text,
                "trajectory_id": trajectory_id,
                "paraphrases": sorted(set(by_trajectory[trajectory_id])),
            }
        )
    passed = not errors and len(rows) == len(converted) and all(len(row["paraphrases"]) >= 2 for row in rows)
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "scene": args.scene,
        "raw_path": str(args.raw),
        "raw_sha256": sha256(args.raw),
        "tasks_path": str(args.tasks),
        "tasks_sha256": sha256(args.tasks),
        "converted_instructions": len(converted),
        "matched_instructions": len(rows),
        "unique_trajectories": len({row["trajectory_id"] for row in rows}),
        "errors": errors,
        "instructions": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "matched_instructions", "unique_trajectories")}, ensure_ascii=False))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
