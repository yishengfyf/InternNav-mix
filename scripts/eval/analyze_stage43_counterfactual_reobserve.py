#!/usr/bin/env python3
"""Audit Stage43 no-action counterfactual re-observation records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


def _key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return str(row.get("scene_id")), int(row.get("episode_id", -1)), int(row.get("step_id", -1))


def _load(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in root.glob("**/stage43_counterfactual_reobserve_events.jsonl"):
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return list({_key(row): row for row in rows}.values())


def _manifest(path: Path) -> set[tuple[str, int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("selected_detector_events", [])
    return {_key(row) for row in rows}


def analyze(root: Path, manifest_path: Path) -> dict[str, Any]:
    rows = _load(root)
    expected = _manifest(manifest_path)
    observed = {_key(row): row for row in rows}
    probes = [probe for row in rows for probe in row.get("probes", [])]
    post_contracts = [
        item.get("contract", {})
        for probe in probes
        for item in probe.get("post_contracts", [])
    ]
    post_edges = [
        edge
        for probe in probes
        for item in probe.get("post_contracts", [])
        for edge in item.get("edge_audits", [])
    ]
    reasons = Counter(str(probe.get("reason") or "none") for probe in probes)
    zero_rows = [row for row in rows if int(row.get("pre_candidate_count", 0)) == 0]
    return {
        "task": "stage43_counterfactual_reobserve_shadow_audit",
        "expected_event_count": len(expected),
        "observed_exact_event_count": len(expected & set(observed)),
        "missing_event_count": len(expected - set(observed)),
        "unexpected_event_count": len(set(observed) - expected),
        "event_count": len(rows),
        "probe_count": len(probes),
        "counterfactual_observation_readable_count": sum(bool(row.get("observation_readable")) for row in probes),
        "sim_pose_restored_count": sum(bool(row.get("sim_pose_restored")) for row in probes),
        "official_memory_mutated_count": sum(bool(row.get("official_memory_mutated")) for row in probes),
        "pre_zero_event_count": len(zero_rows),
        "zero_to_nonzero_event_count": sum(int(row.get("post_candidate_count_max", 0)) > 0 for row in zero_rows),
        "post_candidate_count_max": max((int(row.get("post_candidate_count_max", 0)) for row in rows), default=0),
        "post_contract_count": len(post_contracts),
        "post_first_edge_checked_count": sum(bool(row.get("first_edge_depth_checked")) for row in post_contracts),
        "post_first_edge_clear_count": sum(bool(row.get("first_edge_depth_clear")) for row in post_contracts),
        "post_executor_eligible_count": sum(bool(row.get("executor_eligible")) for row in post_contracts),
        "post_unsafe_edge_count": sum(not bool(row.get("sparseocc_safe")) for row in post_edges),
        "post_unknown_edge_count": sum(bool(row.get("unknown")) for row in post_edges),
        "post_occupied_edge_count": sum(bool(row.get("occupied")) for row in post_edges),
        "probe_reason_counts": dict(sorted(reasons.items())),
        "action_emitted_count": sum(bool(row.get("action_emitted")) for row in probes),
        "action_applied_count": sum(bool(row.get("action_applied")) for row in rows),
        "unknown_is_free": False,
        "gt_fields_used": [],
        "shadow_only": True,
        "integrity_passed": bool(rows)
        and bool(probes)
        and not (expected - set(observed))
        and not (set(observed) - expected)
        and all(bool(row.get("shadow_only")) for row in rows)
        and all(not row.get("action_applied") and not row.get("gt_fields_used") for row in rows)
        and all(row.get("unknown_is_free") is False for row in rows)
        and all(bool(probe.get("sim_pose_restored")) for probe in probes)
        and all(probe.get("official_memory_mutated") is False for probe in probes)
        and not any(bool(probe.get("action_emitted")) for probe in probes)
        and all(bool(row.get("sparseocc_safe")) for row in post_edges)
        and not any(bool(row.get("unknown")) or bool(row.get("occupied")) for row in post_edges)
        and all(
            row.get("unknown_is_free") is False
            and not row.get("action_emitted")
            and not row.get("action_applied")
            and not row.get("gt_fields_used")
            for row in post_contracts
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.run_root, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
