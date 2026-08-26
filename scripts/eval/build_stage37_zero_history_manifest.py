#!/usr/bin/env python3
"""Build a minimal Stage37 smoke manifest from frozen E1 zero events.

Selection is deterministic and offline-only.  It requires an observed route
universe and path-eligible history while the frozen route+OCC+clearance pool
is empty.  GT-lite fields are copied only for exact audit alignment.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple


EventKey = Tuple[str, int, int]
EpisodeKey = Tuple[str, int]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _scene(value: Any) -> str:
    return Path(str(value)).stem


def _event_key(row: Mapping[str, Any]) -> EventKey:
    return _scene(row["scene_id"]), int(row["episode_id"]), int(row["step_id"])


def _episode_key(row: Mapping[str, Any]) -> EpisodeKey:
    return _scene(row["scene_id"]), int(row["episode_id"])


def _load_events(root: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for path in sorted(root.glob("**/stage27_m3_candidate_events.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    unique = {_event_key(row): row for row in rows}
    return [unique[key] for key in sorted(unique)]


def build(
    *,
    e1_root: Path,
    episode_manifest: Path,
    event_count: int = 4,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], Dict[str, Any]]:
    events = _load_events(e1_root)
    episodes = _load_json(episode_manifest)
    episodes_by_key = {_episode_key(row): dict(row) for row in episodes}
    eligible = []
    for row in events:
        route = row.get("ablation", {}).get("route_occ_clearance", {})
        if (
            int(row.get("route_candidate_universe_count", 0) or 0) > 0
            and int(row.get("route_path_eligible_candidate_count", 0) or 0) > 0
            and not list(route.get("candidates") or [])
            and _episode_key(row) in episodes_by_key
        ):
            eligible.append(row)

    # Prefer distinct scenes and include a holdout event when available.
    chosen: list[Dict[str, Any]] = []
    used_episodes: set[EpisodeKey] = set()
    used_scenes: set[str] = set()
    ordered = sorted(
        eligible,
        key=lambda row: (
            0 if row.get("audit_selection", {}).get("gt_split") == "holdout" else 1,
            int(row.get("episode_eval_seed", 0)),
            int(row.get("step_id", 0)),
        ),
    )
    for row in ordered:
        key = _episode_key(row)
        if key in used_episodes or _scene(row["scene_id"]) in used_scenes:
            continue
        chosen.append(row)
        used_episodes.add(key)
        used_scenes.add(_scene(row["scene_id"]))
        if len(chosen) >= event_count:
            break
    if len(chosen) < event_count:
        for row in ordered:
            key = _episode_key(row)
            if key in used_episodes:
                continue
            chosen.append(row)
            used_episodes.add(key)
            if len(chosen) >= event_count:
                break
    if len(chosen) != event_count:
        raise ValueError(f"requested {event_count} episodes, selected {len(chosen)}")

    event_rows = []
    for row in chosen:
        event = dict(row.get("audit_selection") or {})
        event.update({
            "scene_id": _scene(row["scene_id"]),
            "episode_id": int(row["episode_id"]),
            "step_id": int(row["step_id"]),
            "episode_eval_seed": int(row["episode_eval_seed"]),
            "stage37_selection": "e1_route_history_zero_clearance",
            "e1_route_candidate_universe_count": int(row.get("route_candidate_universe_count", 0) or 0),
            "e1_route_path_eligible_candidate_count": int(row.get("route_path_eligible_candidate_count", 0) or 0),
        })
        event_rows.append(event)
    episode_rows = sorted(
        (episodes_by_key[_episode_key(row)] for row in chosen),
        key=lambda row: int(row["episode_eval_seed"]),
    )
    audit = {
        "task": "stage37_zero_history_fallback_smoke_manifest",
        "selection_inputs": "frozen_stage27_e1_event_logs_and_fresh500_episode_seed_manifest",
        "selection_contract": "route_universe_gt0_and_path_eligible_gt0_and_route_occ_clearance_zero",
        "gt_fields_online": [],
        "event_count": len(event_rows),
        "episode_count": len(episode_rows),
        "scene_count": len({_scene(row["scene_id"]) for row in episode_rows}),
        "events": event_rows,
        "episodes": episode_rows,
    }
    return episode_rows, event_rows, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1-root", type=Path, required=True)
    parser.add_argument("--episode-manifest", type=Path, required=True)
    parser.add_argument("--episode-output", type=Path, required=True)
    parser.add_argument("--event-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    episodes, events, audit = build(
        e1_root=args.e1_root,
        episode_manifest=args.episode_manifest,
    )
    for path, payload in (
        (args.episode_output, episodes),
        (args.event_output, events),
        (args.audit_output, audit),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
