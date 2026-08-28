#!/usr/bin/env python3
"""Aggregate Stage50 depth-short-lookahead shadow events."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def rows(root: Path) -> list[dict]:
    paths = sorted(root.glob("**/stage50_depth_short_lookahead_events.jsonl"))
    result = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                result.append(json.loads(line))
    return result


def analyze(root: Path) -> dict:
    events = rows(root)
    eligible = [row for row in events if int(row.get("eligible_count", 0) or 0) > 0]
    records = [item for row in events for item in row.get("lookahead_records", [])]
    blockers = [
        ((item.get("floor_footprint_audit") or {}).get("first_blocker") or {}).get("reason")
        for item in records
    ]
    return {
        "task": "stage50_depth_short_lookahead_shadow",
        "event_count": len(events),
        "eligible_event_count": len(eligible),
        "eligible_record_count": sum(int(row.get("eligible_count", 0) or 0) for row in events),
        "lookahead_record_count": len(records),
        "reason_counts": dict(Counter(str(row.get("reason")) for row in events)),
        "path_state_counts": dict(Counter(str(row.get("path_state")) for row in records)),
        "floor_footprint_state_counts": dict(Counter(str(row.get("floor_footprint_state")) for row in records)),
        "path_source_counts": dict(Counter(str(row.get("path_source")) for row in records)),
        "first_blocker_reason_counts": dict(Counter(str(value) for value in blockers if value)),
        "out_of_bounds_projection_count": sum(row.get("lookahead_pixel_in_bounds") is False for row in records),
        "local_search_attempt_count": sum(int(row.get("local_search_attempt_count", 0) or 0) for row in records),
        "local_search_eligible_count": sum(
            row.get("eligible") is True and row.get("path_source") == "local_8n_depth_supported"
            for row in records
        ),
        "occupied_hit_count": sum(
            int((row.get("floor_footprint_audit") or {}).get("occupied_hit_count", 0) or 0)
            for row in records
        ),
        "unknown_is_free": False,
        "semantic_can_override_safety": False,
        "shadow_only": all(row.get("shadow_only") is True for row in events),
        "action_applied": any(row.get("action_applied") for row in events),
        "gt_fields_used": sorted({field for row in events for field in row.get("gt_fields_used", [])}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["action_applied"] or report["gt_fields_used"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
