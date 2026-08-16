"""Summarize Stage21c path-reobserve active events for automatic review."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _events(root: Path):
    for path in sorted(root.rglob("s2_loop_path_reobserve_active_events.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row["_source"] = str(path)
                yield row


def _key(row):
    return f"{row.get('scene_id')}/{row.get('episode_id')}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = list(_events(args.active_root))
    trigger_rows = [
        row for row in rows if row.get("event_type") == "s2_loop_path_reobserve_active"
    ]
    strict_rows = [
        row for row in trigger_rows if row.get("triage_tier") == "strict_intervention"
    ]
    strict_keys = sorted({_key(row) for row in strict_rows})
    strict_trigger_keys = sorted(
        {f"{_key(row)}@{row.get('trigger_step')}" for row in strict_rows}
    )
    no_path_rows = [
        row for row in strict_rows if row.get("reason") == "no_known_free_path_to_anchor"
    ]
    bridge_rows = [
        row for row in strict_rows if isinstance(row.get("path_bridge"), dict)
    ]
    reachable_rows = [
        row for row in bridge_rows if row["path_bridge"].get("path_reachable") is True
    ]
    post_rows = [
        row
        for row in rows
        if row.get("event_type") == "s2_loop_path_reobserve_post_observation"
    ]

    visited = [
        int(row["path_bridge"]["path_visited_cell_count"])
        for row in bridge_rows
        if row["path_bridge"].get("path_visited_cell_count") is not None
    ]
    path_lengths = [
        float(row["path_bridge"]["path_m"])
        for row in bridge_rows
        if row["path_bridge"].get("path_m") is not None
    ]
    anchor_steps = sorted(
        {
            int(row["candidate"]["semantic_resilience_source_step_id"])
            for row in strict_rows
            if isinstance(row.get("candidate"), dict)
            and row["candidate"].get("semantic_resilience_source_step_id") is not None
        }
    )

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    paired = audit.get("paired_aggregate", {})
    result = {
        "task": "stage21c_path_reobserve_diagnostics",
        "active_event_count": len(rows),
        "trigger_event_count": len(trigger_rows),
        "event_reason_counts": dict(Counter(str(row.get("reason")) for row in rows)),
        "strict_trigger_count": len(strict_rows),
        "strict_episode_count": len(strict_keys),
        "strict_episodes": strict_keys,
        "strict_trigger_keys": strict_trigger_keys,
        "no_known_free_path_to_anchor_count": len(no_path_rows),
        "path_bridge_attempt_count": len(bridge_rows),
        "path_bridge_reachable_count": len(reachable_rows),
        "path_bridge_reachable_rate": (
            len(reachable_rows) / len(bridge_rows) if bridge_rows else None
        ),
        "path_visited_cell_counts": visited,
        "path_length_m": path_lengths,
        "anchor_source_steps": anchor_steps,
        "post_reobserve_event_count": len(post_rows),
        "post_reobserve_state_changed_count": audit.get(
            "post_reobserve_state_changed_count", 0
        ),
        "reorient_completed_event_count": audit.get(
            "reorient_completed_event_count", 0
        ),
        "path_pixel_applied_event_count": audit.get(
            "path_pixel_applied_event_count", 0
        ),
        "applied_intervention_count": audit.get("applied_intervention_count", 0),
        "failed_to_success_count": audit.get("failed_to_success_count", 0),
        "success_to_failed_count": audit.get("success_to_failed_count", 0),
        "paired_aggregate": paired,
        "interpretation_guard": (
            "Diagnostics describe event coverage and paired outcomes; they do not "
            "establish navigation causality or generalization."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
