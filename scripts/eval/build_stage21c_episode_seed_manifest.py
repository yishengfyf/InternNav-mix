"""Attach deterministic per-episode seeds to an existing episode manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_rows(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"episode manifest must be a non-empty list: {path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overrides", type=Path)
    parser.add_argument(
        "--include",
        type=Path,
        help="Manifest rows that must be included before filling from --input.",
    )
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--base-seed", type=int, default=300000)
    args = parser.parse_args()

    input_rows = _load_rows(args.input)
    required_rows = _load_rows(args.include) if args.include else []
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive")
    rows = []
    combined_keys = set()
    for row in required_rows + input_rows:
        key = f"{row['scene_id']}/{int(row['episode_id'])}"
        if key in combined_keys:
            continue
        combined_keys.add(key)
        rows.append(row)
        if args.max_episodes is not None and len(rows) >= args.max_episodes:
            break
    if args.max_episodes is not None and len(rows) != args.max_episodes:
        raise ValueError(
            f"not enough unique rows for --max-episodes={args.max_episodes}: {len(rows)}"
        )
    overrides = {}
    if args.overrides:
        for row in _load_rows(args.overrides):
            key = f"{row['scene_id']}/{int(row['episode_id'])}"
            overrides[key] = int(row["episode_eval_seed"])

    output = []
    seen_keys = set()
    seen_seeds = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"manifest row {index} is not an object")
        scene_id = row.get("scene_id")
        episode_id = row.get("episode_id")
        if scene_id is None or episode_id is None:
            raise ValueError(f"manifest row {index} lacks scene_id/episode_id")
        key = f"{scene_id}/{int(episode_id)}"
        if key in seen_keys:
            raise ValueError(f"duplicate episode key: {key}")
        seen_keys.add(key)
        seed = overrides.get(key, row.get("episode_eval_seed", args.base_seed + index))
        seed = int(seed)
        if seed in seen_seeds:
            raise ValueError(f"duplicate episode_eval_seed {seed} at {key}")
        seen_seeds.add(seed)
        output.append({
            "scene_id": str(scene_id),
            "episode_id": int(episode_id),
            "episode_eval_seed": seed,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "input": str(args.input),
        "output": str(args.output),
        "episode_count": len(output),
        "override_count": sum(
            f"{row['scene_id']}/{int(row['episode_id'])}" in overrides
            for row in output
        ),
        "required_count": len(required_rows),
        "max_episodes": args.max_episodes,
        "base_seed": args.base_seed,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
