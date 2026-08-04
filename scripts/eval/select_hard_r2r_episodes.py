"""Select scene-balanced hard R2R episodes for Stage18 data collection.

The selector uses only episode annotations available before rollout:
reference-path length, turn count, path point count, and instruction length.
It does not use model success/failure, candidate labels, or validation data.

The output format is compatible with ``STAGE17_EPISODE_IDS`` used by the
Stage17/18 balanced eval configs:

[
  {"scene_id": "17DRP5sb8fy", "episode_id": 123},
  ...
]
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


Point2D = Tuple[float, float]


def _read_text(path: Path) -> str:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8")


def _read_episodes(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(_read_text(path))
    episodes = data.get("episodes") if isinstance(data, dict) else data
    if not isinstance(episodes, list):
        raise ValueError(f"Expected an episode list in {path}")
    return [item for item in episodes if isinstance(item, dict)]


def _scene_token(scene_id: Any) -> str:
    text = str(scene_id or "")
    parts = [part for part in text.replace("\\", "/").split("/") if part]
    if len(parts) >= 2 and parts[-1].endswith((".glb", ".basis", ".navmesh")):
        return parts[-2]
    return Path(parts[-1]).stem if parts else text


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _extract_instruction_text(episode: Mapping[str, Any]) -> str:
    instruction = episode.get("instruction")
    if isinstance(instruction, str):
        return instruction
    if isinstance(instruction, Mapping):
        for key in ("instruction_text", "text", "instruction"):
            value = instruction.get(key)
            if isinstance(value, str):
                return value
    return ""


def _extract_reference_path(episode: Mapping[str, Any]) -> List[Any]:
    path = episode.get("reference_path") or episode.get("reference_paths")
    if isinstance(path, list) and path and isinstance(path[0], list):
        if path[0] and isinstance(path[0][0], list):
            return path[0]
        return path
    return []


def _point_xy(point: Any, coordinate_mode: str) -> Optional[Point2D]:
    if not isinstance(point, (list, tuple)) or len(point) < 3:
        return None
    x = _safe_float(point[0], float("nan"))
    y = _safe_float(point[1], float("nan"))
    z = _safe_float(point[2], float("nan"))
    if any(math.isnan(value) for value in (x, y, z)):
        return None
    if coordinate_mode == "xz":
        return x, z
    if coordinate_mode == "x_neg_z":
        return x, -z
    if coordinate_mode == "xy":
        return x, y
    if coordinate_mode == "x_neg_y":
        return x, -y
    raise ValueError(f"Unsupported coordinate mode: {coordinate_mode}")


def _path_points(reference_path: Sequence[Any], coordinate_mode: str) -> List[Point2D]:
    points = []
    for item in reference_path:
        point = _point_xy(item, coordinate_mode)
        if point is not None:
            points.append(point)
    return points


def _path_length(points: Sequence[Point2D]) -> float:
    return sum(
        math.hypot(points[index][0] - points[index - 1][0], points[index][1] - points[index - 1][1])
        for index in range(1, len(points))
    )


def _turn_count(points: Sequence[Point2D], *, turn_angle_deg: float, min_segment_m: float) -> int:
    count = 0
    for index in range(1, len(points) - 1):
        prev_vec = (
            points[index][0] - points[index - 1][0],
            points[index][1] - points[index - 1][1],
        )
        next_vec = (
            points[index + 1][0] - points[index][0],
            points[index + 1][1] - points[index][1],
        )
        prev_norm = math.hypot(*prev_vec)
        next_norm = math.hypot(*next_vec)
        if prev_norm < min_segment_m or next_norm < min_segment_m:
            continue
        cosine = (
            (prev_vec[0] * next_vec[0] + prev_vec[1] * next_vec[1])
            / max(1e-9, prev_norm * next_norm)
        )
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        if angle >= turn_angle_deg:
            count += 1
    return count


def _load_excluded_episode_keys(paths: Sequence[Path]) -> set[str]:
    excluded = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Exclude file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list in exclude file: {path}")
        for item in data:
            if isinstance(item, Mapping):
                scene_id = item.get("scene_id")
                episode_id = item.get("episode_id")
                if episode_id is None:
                    continue
                if scene_id is None:
                    excluded.add(f"|{int(episode_id)}")
                else:
                    excluded.add(f"{_scene_token(scene_id)}|{int(episode_id)}")
            else:
                excluded.add(f"|{int(item)}")
    return excluded


def _episode_key(record: Mapping[str, Any]) -> str:
    return f"{record.get('scene_token')}|{int(record.get('episode_id'))}"


def _is_excluded(record: Mapping[str, Any], excluded: set[str]) -> bool:
    episode_id = int(record.get("episode_id"))
    return _episode_key(record) in excluded or f"|{episode_id}" in excluded


def _normalize(value: float, maximum: float) -> float:
    if maximum <= 0.0:
        return 0.0
    return float(value) / float(maximum)


def _score_records(
    episodes: Sequence[Mapping[str, Any]],
    *,
    coordinate_mode: str,
    turn_angle_deg: float,
    min_turn_segment_m: float,
    length_weight: float,
    turn_weight: float,
    instruction_weight: float,
    point_weight: float,
    seed: int,
) -> List[Dict[str, Any]]:
    raw_records: List[Dict[str, Any]] = []
    rng = random.Random(seed)
    for episode in episodes:
        if episode.get("episode_id") is None:
            continue
        reference_path = _extract_reference_path(episode)
        points = _path_points(reference_path, coordinate_mode)
        instruction_text = _extract_instruction_text(episode)
        record = {
            "scene_token": _scene_token(episode.get("scene_id")),
            "episode_id": int(episode.get("episode_id")),
            "reference_path_points": len(points),
            "path_length_m": _path_length(points),
            "turn_count": _turn_count(
                points,
                turn_angle_deg=turn_angle_deg,
                min_segment_m=min_turn_segment_m,
            ),
            "instruction_word_count": len(instruction_text.split()),
            "random_tiebreaker": rng.random() * 1e-6,
        }
        raw_records.append(record)

    max_length = max((item["path_length_m"] for item in raw_records), default=0.0)
    max_turns = max((item["turn_count"] for item in raw_records), default=0)
    max_words = max((item["instruction_word_count"] for item in raw_records), default=0)
    max_points = max((item["reference_path_points"] for item in raw_records), default=0)
    for item in raw_records:
        item["difficulty_score"] = (
            length_weight * _normalize(item["path_length_m"], max_length)
            + turn_weight * _normalize(item["turn_count"], float(max_turns))
            + instruction_weight * _normalize(item["instruction_word_count"], float(max_words))
            + point_weight * _normalize(item["reference_path_points"], float(max_points))
            + item["random_tiebreaker"]
        )
    return raw_records


def _round_robin_hard(
    records: Sequence[Dict[str, Any]],
    *,
    max_episodes: int,
    excluded: set[str],
) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        if _is_excluded(record, excluded):
            continue
        groups[str(record["scene_token"])].append(record)
    for items in groups.values():
        items.sort(
            key=lambda item: (
                item["difficulty_score"],
                item["path_length_m"],
                item["turn_count"],
                item["instruction_word_count"],
            ),
            reverse=True,
        )

    selected = []
    index = 0
    scene_names = sorted(groups)
    while len(selected) < max_episodes:
        progressed = False
        for scene_name in scene_names:
            items = groups[scene_name]
            if index >= len(items):
                continue
            selected.append(items[index])
            progressed = True
            if len(selected) >= max_episodes:
                break
        if not progressed:
            break
        index += 1
    return selected


def _stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--max-episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument(
        "--coordinate-mode",
        choices=("xz", "x_neg_z", "xy", "x_neg_y"),
        default="xz",
        help="2D plane used only for hard episode annotation scoring.",
    )
    parser.add_argument("--turn-angle-deg", type=float, default=45.0)
    parser.add_argument("--min-turn-segment-m", type=float, default=0.25)
    parser.add_argument("--length-weight", type=float, default=0.45)
    parser.add_argument("--turn-weight", type=float, default=0.35)
    parser.add_argument("--instruction-weight", type=float, default=0.15)
    parser.add_argument("--point-weight", type=float, default=0.05)
    parser.add_argument("--exclude-episode-ids", type=Path, nargs="*", default=[])
    args = parser.parse_args()

    if args.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive")

    episodes = _read_episodes(args.episodes_file)
    excluded = _load_excluded_episode_keys(args.exclude_episode_ids)
    scored = _score_records(
        episodes,
        coordinate_mode=args.coordinate_mode,
        turn_angle_deg=args.turn_angle_deg,
        min_turn_segment_m=args.min_turn_segment_m,
        length_weight=args.length_weight,
        turn_weight=args.turn_weight,
        instruction_weight=args.instruction_weight,
        point_weight=args.point_weight,
        seed=args.seed,
    )
    selected = _round_robin_hard(scored, max_episodes=args.max_episodes, excluded=excluded)
    output_rows = [
        {"scene_id": item["scene_token"], "episode_id": int(item["episode_id"])}
        for item in selected
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    scene_counts = Counter(item["scene_id"] for item in output_rows)
    summary = {
        "episodes_file": str(args.episodes_file),
        "output": str(args.output),
        "max_episodes": int(args.max_episodes),
        "selected_episodes": len(output_rows),
        "selected_scenes": len(scene_counts),
        "excluded_episode_keys": len(excluded),
        "seed": int(args.seed),
        "coordinate_mode": args.coordinate_mode,
        "turn_angle_deg": float(args.turn_angle_deg),
        "min_turn_segment_m": float(args.min_turn_segment_m),
        "weights": {
            "length": float(args.length_weight),
            "turn": float(args.turn_weight),
            "instruction": float(args.instruction_weight),
            "point": float(args.point_weight),
        },
        "selected_stats": {
            "difficulty_score": _stats([item["difficulty_score"] for item in selected]),
            "path_length_m": _stats([item["path_length_m"] for item in selected]),
            "turn_count": _stats([float(item["turn_count"]) for item in selected]),
            "instruction_word_count": _stats(
                [float(item["instruction_word_count"]) for item in selected]
            ),
            "reference_path_points": _stats(
                [float(item["reference_path_points"]) for item in selected]
            ),
        },
        "scene_counts": dict(sorted(scene_counts.items())),
        "top_examples": [
            {
                "scene_id": item["scene_token"],
                "episode_id": int(item["episode_id"]),
                "difficulty_score": float(item["difficulty_score"]),
                "path_length_m": float(item["path_length_m"]),
                "turn_count": int(item["turn_count"]),
                "instruction_word_count": int(item["instruction_word_count"]),
                "reference_path_points": int(item["reference_path_points"]),
            }
            for item in sorted(selected, key=lambda row: row["difficulty_score"], reverse=True)[:20]
        ],
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
