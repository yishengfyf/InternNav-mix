#!/usr/bin/env python3
"""Run compact offline Stage25 detector threshold sweeps on fixed ledgers."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from scripts.eval.analyze_stage25_gt_detector import (
    discover_episodes,
    episode_key,
    jsonl,
    loop_events_by_episode,
    lseg_events,
    mine_events,
    progress_by_episode,
)


def csv_values(value: str, cast: Any) -> List[Any]:
    return [cast(item) for item in value.split(",") if item.strip()]


def load_dataset(
    run_root: Path, manifest_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    progress = progress_by_episode(run_root)
    loops = loop_events_by_episode(run_root)
    manifest = {
        episode_key(row): row
        for row in json.loads(manifest_path.read_text(encoding="utf-8"))
    }
    episodes = []
    for episode_dir in discover_episodes(run_root):
        meta = json.loads(
            (episode_dir / "episode_meta.json").read_text(encoding="utf-8")
        )
        key = episode_key(meta)
        episodes.append({
            "key": key,
            "observations": jsonl(episode_dir / "observations.jsonl"),
            "loops": loops.get(key, []),
            "semantic": lseg_events(episode_dir),
            "outcome": progress.get(key, {}),
            "role": (manifest.get(key) or {}).get("audit_role"),
        })
    return episodes, manifest


def summarize(
    episodes: Sequence[Mapping[str, Any]], options: Mapping[str, Any],
) -> Dict[str, Any]:
    detected = []
    failures = {row["key"] for row in episodes if row["outcome"].get("success") == 0}
    successes = {row["key"] for row in episodes if row["outcome"].get("success") == 1}
    hard_negatives = {
        row["key"] for row in episodes if "hard_negative" in str(row.get("role"))
    }
    leads = []
    persistent_events = 0
    recovered_events = 0
    event_count = 0
    for row in episodes:
        events = mine_events(
            row["observations"], row["loops"], row["semantic"], **options
        )["D1"]
        if not events:
            continue
        key = str(row["key"])
        detected.append(key)
        event_count += len(events)
        persistent_events += sum(
            event["recoverability_proxy"].startswith("persistent")
            for event in events
        )
        recovered_events += sum(
            event["recoverability_proxy"].startswith("self_recovered")
            for event in events
        )
        if key in failures:
            episode_steps = int(row["outcome"].get("steps") or 0)
            leads.append(max(0, episode_steps - min(int(e["step_id"]) for e in events)))
    detected_set = set(detected)
    failure_detected = failures & detected_set
    success_detected = successes & detected_set
    return {
        "event_count": event_count,
        "detected_episode_count": len(detected_set),
        "failure_detected": len(failure_detected),
        "failure_total": len(failures),
        "success_detected": len(success_detected),
        "success_total": len(successes),
        "hard_negative_detected": len(hard_negatives & detected_set),
        "hard_negative_total": len(hard_negatives),
        "persistent_event_count": persistent_events,
        "self_recovered_event_count": recovered_events,
        "median_failure_lead_steps": None if not leads else float(median(leads)),
        "detected_failures": sorted(failure_detected),
        "detected_successes": sorted(success_detected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs=3, action="append", metavar=("NAME", "RUN_ROOT", "MANIFEST"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collision-burst-min", default="1,2,3")
    parser.add_argument("--geometry-max-displacement-m", default="0.15,0.25,0.35")
    parser.add_argument("--forward-min-count", default="2,3,4")
    parser.add_argument("--forward-max-displacement-m", default="0.10,0.15,0.25")
    args = parser.parse_args()

    datasets = {}
    for name, run_root, manifest in args.dataset:
        episodes, expected = load_dataset(Path(run_root), Path(manifest))
        if len(episodes) != len(expected):
            raise SystemExit(
                f"{name}: expected {len(expected)} episodes, found {len(episodes)}"
            )
        datasets[name] = episodes

    grids = itertools.product(
        csv_values(args.collision_burst_min, float),
        csv_values(args.geometry_max_displacement_m, float),
        csv_values(args.forward_min_count, int),
        csv_values(args.forward_max_displacement_m, float),
    )
    rows = []
    for collision_min, geometry_disp, forward_min, forward_disp in grids:
        options = {
            "collision_burst_min": collision_min,
            "geometry_max_displacement_m": geometry_disp,
            "forward_min_count": forward_min,
            "forward_max_displacement_m": forward_disp,
        }
        rows.append({
            "options": options,
            "datasets": {
                name: summarize(episodes, options)
                for name, episodes in datasets.items()
            },
        })
    default_options = {
        "collision_burst_min": 2.0,
        "geometry_max_displacement_m": 0.25,
        "forward_min_count": 3,
        "forward_max_displacement_m": 0.15,
    }
    default = next(row for row in rows if row["options"] == default_options)
    report = {
        "task": "stage25_geometry_threshold_sweep",
        "outcome_is_event_gt": False,
        "future_used_by_detector": False,
        "combination_count": len(rows),
        "default": default,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
