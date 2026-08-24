#!/usr/bin/env python3
"""Build the preregistered Stage28 semantic-candidate pilot manifests.

Selection uses only the frozen Stage27 E1 geometry result, GT-lite split/state,
and an existing hard-negative control manifest. Semantic scores and Stage28
outputs are deliberately unavailable to this builder.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


EventKey = Tuple[str, int, int]
EpisodeKey = Tuple[str, int]


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scene_id(value: Any) -> str:
    return Path(str(value)).stem


def _episode_key(row: Mapping[str, Any]) -> EpisodeKey:
    return _scene_id(row["scene_id"]), int(row["episode_id"])


def _event_key(row: Mapping[str, Any]) -> EventKey:
    scene_id, episode_id = _episode_key(row)
    return scene_id, episode_id, int(row["step_id"])


def _load_jsonl(paths: Iterable[Path]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def _final_candidate_count(event: Mapping[str, Any]) -> int:
    final = (event.get("ablation") or {}).get(
        "route_occ_clearance_frontier", {}
    )
    return len(final.get("candidates") or [])


def _pick_scene_diverse(
    episode_keys: Sequence[EpisodeKey],
    *,
    count: int,
    excluded: set[EpisodeKey],
) -> list[EpisodeKey]:
    available = [key for key in episode_keys if key not in excluded]
    chosen: list[EpisodeKey] = []
    used_scenes = {scene_id for scene_id, _ in excluded}
    for key in available:
        if key[0] not in used_scenes:
            chosen.append(key)
            used_scenes.add(key[0])
            if len(chosen) == count:
                return chosen
    for key in available:
        if key not in chosen:
            chosen.append(key)
            if len(chosen) == count:
                return chosen
    raise ValueError(f"requested {count} episodes but only found {len(chosen)}")


def build(
    *,
    all_events: Sequence[Mapping[str, Any]],
    e1_events: Sequence[Mapping[str, Any]],
    full_episodes: Sequence[Mapping[str, Any]],
    control_episodes: Sequence[Mapping[str, Any]],
    progress_rows: Sequence[Mapping[str, Any]],
    episode_count: int = 48,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], Dict[str, Any]]:
    e1_by_key = {_event_key(row): row for row in e1_events}
    full_by_key = {_episode_key(row): dict(row) for row in full_episodes}
    progress_by_key = {_episode_key(row): row for row in progress_rows}
    all_event_episode_keys = {_episode_key(row) for row in all_events}

    missing = sorted(_event_key(row) for row in all_events if _event_key(row) not in e1_by_key)
    if missing:
        raise ValueError(f"E1 output is missing {len(missing)} GT-lite events")

    selected: Dict[EpisodeKey, Dict[str, Any]] = {}
    selection_reason: Dict[EpisodeKey, str] = {}

    def add(key: EpisodeKey, reason: str) -> None:
        if key not in full_by_key:
            raise ValueError(f"episode is absent from full manifest: {key}")
        if key not in selected:
            selected[key] = full_by_key[key]
            selection_reason[key] = reason

    # Scene-disjoint holdout is the primary acceptance stratum, so retain all
    # of its event episodes before sampling any development examples.
    for row in sorted(all_events, key=lambda item: int(item["episode_eval_seed"])):
        if row.get("gt_split") == "holdout":
            add(_episode_key(row), "all_holdout_event_episode")

    strata = (
        ("true_trap", True, 6),
        ("wrong_way_progress", True, 6),
        ("true_trap", False, 4),
        ("wrong_way_progress", False, 4),
    )
    for state, require_zero, count in strata:
        rows = [
            row
            for row in all_events
            if row.get("gt_split") == "dev"
            and row.get("gt_state") == state
            and ((_final_candidate_count(e1_by_key[_event_key(row)]) == 0) == require_zero)
        ]
        ordered_keys: list[EpisodeKey] = []
        for row in sorted(rows, key=lambda item: int(item["episode_eval_seed"])):
            key = _episode_key(row)
            if key not in ordered_keys:
                ordered_keys.append(key)
        chosen = _pick_scene_diverse(
            ordered_keys, count=count, excluded=set(selected)
        )
        label = f"dev_geometry_{'zero' if require_zero else 'nonzero'}_{state}"
        for key in chosen:
            add(key, label)

    # Controls are inherited from the already preregistered Stage26 fresh48.
    # Success/collision are used only to define normal hard negatives, never
    # as event GT or candidate labels.
    for row in control_episodes:
        if len(selected) >= episode_count:
            break
        key = _episode_key(row)
        progress = progress_by_key.get(key, {})
        if (
            key not in selected
            and key not in all_event_episode_keys
            and float(progress.get("success", 0.0)) == 1.0
            and float(progress.get("collision_count", -1.0)) == 0.0
        ):
            add(key, "preregistered_success_collision_free_no_gtlite_event_control")

    if len(selected) != episode_count:
        raise ValueError(f"expected {episode_count} episodes, selected {len(selected)}")

    episode_manifest = sorted(
        selected.values(), key=lambda row: int(row["episode_eval_seed"])
    )
    selected_keys = set(selected)
    event_manifest = [
        dict(row) for row in all_events if _episode_key(row) in selected_keys
    ]

    event_audit: list[Dict[str, Any]] = []
    for row in event_manifest:
        candidate_count = _final_candidate_count(e1_by_key[_event_key(row)])
        event_audit.append({
            "scene_id": _scene_id(row["scene_id"]),
            "episode_id": int(row["episode_id"]),
            "step_id": int(row["step_id"]),
            "gt_split": row.get("gt_split"),
            "gt_state": row.get("gt_state"),
            "e1_final_candidate_count": candidate_count,
            "e1_candidate_stratum": "zero" if candidate_count == 0 else "nonzero",
        })

    cross = Counter(
        (
            row["gt_split"],
            row["gt_state"],
            row["e1_candidate_stratum"],
        )
        for row in event_audit
    )
    audit = {
        "task": "stage28_stratified_semantic_candidate_pilot_manifest",
        "selection_inputs": "frozen_stage27_e1_geometry_and_gtlite_split_only",
        "semantic_fields_used": [],
        "future_success_used_as_event_gt": False,
        "episode_count": len(episode_manifest),
        "scene_count": len({_episode_key(row)[0] for row in episode_manifest}),
        "event_count": len(event_manifest),
        "selection_reason_counts": dict(sorted(Counter(selection_reason.values()).items())),
        "event_strata": {
            "/".join(key): value for key, value in sorted(cross.items())
        },
        "e1_zero_event_count": sum(
            row["e1_candidate_stratum"] == "zero" for row in event_audit
        ),
        "e1_nonzero_event_count": sum(
            row["e1_candidate_stratum"] == "nonzero" for row in event_audit
        ),
        "holdout_zero_event_count": sum(
            row["gt_split"] == "holdout"
            and row["e1_candidate_stratum"] == "zero"
            for row in event_audit
        ),
        "episodes": [
            {
                **dict(row),
                "selection_reason": selection_reason[_episode_key(row)],
            }
            for row in episode_manifest
        ],
        "events": event_audit,
    }
    return episode_manifest, event_manifest, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-event-manifest", type=Path, required=True)
    parser.add_argument("--e1-run-root", type=Path, required=True)
    parser.add_argument("--full-episode-manifest", type=Path, required=True)
    parser.add_argument("--control-episode-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-count", type=int, default=48)
    args = parser.parse_args()

    event_paths = sorted(
        Path(path)
        for path in glob.glob(
            str(args.e1_run_root / "vlmap_safety_debug" / "*" / "stage27_m3_candidate_events.jsonl")
        )
    )
    if not event_paths:
        raise ValueError("no Stage27 E1 event logs found")
    progress_path = args.e1_run_root / "progress.json"
    episode_manifest, event_manifest, audit = build(
        all_events=_load(args.all_event_manifest),
        e1_events=_load_jsonl(event_paths),
        full_episodes=_load(args.full_episode_manifest),
        control_episodes=_load(args.control_episode_manifest),
        progress_rows=_load_jsonl([progress_path]),
        episode_count=args.episode_count,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_path = args.output_dir / "stage28_semantic_candidate_stratified48_episode_seed_replay.json"
    event_path = args.output_dir / "stage28_semantic_candidate_stratified48_gtlite_events.json"
    audit_path = args.output_dir / "stage28_semantic_candidate_stratified48_manifest_audit.json"
    episode_path.write_text(json.dumps(episode_manifest, indent=2) + "\n", encoding="utf-8")
    event_path.write_text(json.dumps(event_manifest, indent=2) + "\n", encoding="utf-8")
    audit.update({
        "input_sha256": {
            "all_event_manifest": _sha256(args.all_event_manifest),
            "full_episode_manifest": _sha256(args.full_episode_manifest),
            "control_episode_manifest": _sha256(args.control_episode_manifest),
            "progress": _sha256(progress_path),
        },
        "output_sha256": {
            "episode_manifest": _sha256(episode_path),
            "event_manifest": _sha256(event_path),
        },
    })
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "episode_manifest": str(episode_path),
        "event_manifest": str(event_path),
        "audit": str(audit_path),
        "episode_count": len(episode_manifest),
        "event_count": len(event_manifest),
    }))


if __name__ == "__main__":
    main()
