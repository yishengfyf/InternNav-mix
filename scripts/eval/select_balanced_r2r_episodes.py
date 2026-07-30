"""Select a scene-balanced subset of R2R episodes for Stage17 data collection."""

from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _read_text(path: Path) -> str:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return f.read()
    return path.read_text(encoding="utf-8")


def _scene_token(scene_id: Any) -> str:
    text = str(scene_id or "")
    parts = [part for part in text.replace("\\", "/").split("/") if part]
    if len(parts) >= 2 and parts[-1].endswith((".glb", ".basis", ".navmesh")):
        return parts[-2]
    return Path(parts[-1]).stem if parts else text


def _read_episodes(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(_read_text(path))
    episodes = data.get("episodes") if isinstance(data, dict) else data
    if not isinstance(episodes, list):
        raise ValueError(f"Expected an episode list in {path}")
    return episodes


def _round_robin(groups: Dict[str, List[Dict[str, Any]]], max_episodes: int) -> Iterable[Dict[str, Any]]:
    scene_names = sorted(groups)
    emitted = 0
    index = 0
    while emitted < max_episodes:
        progressed = False
        for scene_name in scene_names:
            items = groups[scene_name]
            if index >= len(items):
                continue
            yield items[index]
            emitted += 1
            progressed = True
            if emitted >= max_episodes:
                break
        if not progressed:
            break
        index += 1


def select_balanced_episodes(
    episodes: List[Dict[str, Any]],
    *,
    max_episodes: int,
    seed: int,
    shuffle_within_scene: bool,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        scene_id = _scene_token(episode.get("scene_id"))
        groups[scene_id].append(episode)

    for scene_id, items in groups.items():
        if shuffle_within_scene:
            rng.shuffle(items)
        else:
            items.sort(key=lambda item: int(item.get("episode_id", 0)))
        for item in items:
            item["_stage17_scene_token"] = scene_id

    selected = []
    for episode in _round_robin(groups, max_episodes):
        selected.append(
            {
                "scene_id": episode["_stage17_scene_token"],
                "episode_id": int(episode["episode_id"]),
            }
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--max-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--shuffle-within-scene", action="store_true")
    args = parser.parse_args()

    if args.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive")

    episodes = _read_episodes(args.episodes_file)
    selected = select_balanced_episodes(
        episodes,
        max_episodes=args.max_episodes,
        seed=args.seed,
        shuffle_within_scene=args.shuffle_within_scene,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    counts = Counter(item["scene_id"] for item in selected)
    summary = {
        "episodes_file": str(args.episodes_file),
        "output": str(args.output),
        "max_episodes": args.max_episodes,
        "selected_episodes": len(selected),
        "selected_scenes": len(counts),
        "seed": args.seed,
        "shuffle_within_scene": args.shuffle_within_scene,
        "scene_counts": dict(sorted(counts.items())),
    }
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
