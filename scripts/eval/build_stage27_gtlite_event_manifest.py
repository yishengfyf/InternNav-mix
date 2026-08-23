#!/usr/bin/env python3
"""Build a Stage27 audit manifest from the frozen Stage25 GT-lite review pool.

This is an offline manifest adapter. It does not alter detector labels, use
future fields online, or change the Stage27 candidate generator. Events are
scheduled at the adjudicated onset, while the original interval and label
metadata remain audit-only fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_seed_map(path: Path | None) -> Dict[tuple[str, int], int]:
    if path is None:
        return {}
    payload = _load(path)
    return {
        (str(row["scene_id"]), int(row["episode_id"])): int(row["episode_eval_seed"])
        for row in payload
        if row.get("episode_eval_seed") is not None
    }


def build(
    review_manifest: Iterable[Mapping[str, Any]],
    *,
    episode_seeds: Mapping[tuple[str, int], int] | None = None,
    states: set[str] | None = None,
    split: str = "all",
) -> list[Dict[str, Any]]:
    episode_seeds = episode_seeds or {}
    states = states or {"true_trap", "wrong_way_progress"}
    result: list[Dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for row in review_manifest:
        annotation = dict(row.get("annotation") or {})
        state = annotation.get("state")
        row_split = str(row.get("split") or "unspecified")
        if state not in states or (split != "all" and row_split != split):
            continue
        onset = annotation.get("onset_step", row.get("onset_step"))
        if onset is None:
            continue
        scene_id = str(row["scene_id"])
        episode_id = int(row["episode_id"])
        key = (scene_id, episode_id, int(onset))
        if key in seen:
            continue
        seen.add(key)
        item: Dict[str, Any] = {
            "scene_id": scene_id,
            "episode_id": episode_id,
            "step_id": int(onset),
            "audit_role": f"gt_lite_{state}",
            "gt_state": state,
            "gt_type": annotation.get("type"),
            "gt_split": row_split,
            "gt_end_step": row.get("end_step"),
            "gt_recoverability": annotation.get("recoverability"),
            "gt_intervention_likely_needed": annotation.get(
                "intervention_likely_needed"
            ),
            "gt_fields_used": [],
            "future_used_for_review_only": bool(
                row.get("uses_future_for_review_only")
            ),
        }
        seed = episode_seeds.get((scene_id, episode_id))
        if seed is not None:
            item["episode_eval_seed"] = seed
        result.append(item)
    return sorted(result, key=lambda item: (
        item["gt_split"], item["scene_id"], item["episode_id"], item["step_id"]
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--episode-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--states", nargs="+", default=["true_trap", "wrong_way_progress"]
    )
    parser.add_argument("--split", choices=("all", "dev", "holdout"), default="all")
    args = parser.parse_args()
    result = build(
        _load(args.review_manifest),
        episode_seeds=_episode_seed_map(args.episode_manifest),
        states=set(args.states),
        split=args.split,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"count": len(result), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
