#!/usr/bin/env python3
"""Audit Stage41 real-depth, no-action executor contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return str(row.get("scene_id")), int(row.get("episode_id", -1)), int(row.get("step_id", -1))


def _load_jsonl(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in root.glob("**/stage41_executor_contract_events.jsonl"):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return list({_key(row): row for row in rows}.values())


def analyze(root: Path, manifest: Path) -> dict[str, Any]:
    rows = _load_jsonl(root)
    expected = json.loads(manifest.read_text(encoding="utf-8"))
    expected_keys = {_key(row) for row in expected}
    observed = {_key(row): row for row in rows}
    contracts = [item for row in rows for item in row.get("contracts", [])]
    reports = [item.get("contract", {}) for item in contracts]
    edge_audits = [edge for item in contracts for edge in item.get("edge_audits", [])]
    return {
        "task": "stage41_real_depth_executor_contract_shadow_audit",
        "expected_event_count": len(expected_keys),
        "observed_exact_event_count": len(expected_keys & set(observed)),
        "missing_event_count": len(expected_keys - set(observed)),
        "unexpected_event_count": len(set(observed) - expected_keys),
        "candidate_contract_count": len(contracts),
        "executor_eligible_count": sum(bool(row.get("executor_eligible")) for row in reports),
        "abstain_count": sum(not bool(row.get("executor_eligible")) for row in reports),
        "depth_readable_count": sum(bool(row.get("depth_readable")) for row in reports),
        "sensor_hfov_values": sorted({row.get("sensor_hfov_deg") for row in reports}),
        "first_edge_depth_checked_count": sum(bool(row.get("first_edge_depth_checked")) for row in reports),
        "first_edge_depth_clear_count": sum(bool(row.get("first_edge_depth_clear")) for row in reports),
        "all_edges_sparseocc_reaudited_count": sum(bool(row.get("all_edges_sparseocc_reaudited")) for row in reports),
        "unsafe_edge_count": sum(not bool(edge.get("sparseocc_safe")) for edge in edge_audits),
        "unknown_edge_count": sum(bool(edge.get("unknown")) for edge in edge_audits),
        "occupied_edge_count": sum(bool(edge.get("occupied")) for edge in edge_audits),
        "action_emitted_count": sum(bool(row.get("action_emitted")) for row in reports),
        "action_applied_count": sum(bool(row.get("action_applied")) for row in rows),
        "unknown_is_free": False,
        "gt_fields_used": [],
        "shadow_only": True,
        "integrity_passed": bool(rows) and not (expected_keys - set(observed)) and not (set(observed) - expected_keys)
        and all(bool(row.get("shadow_only")) and not row.get("action_applied") and not row.get("gt_fields_used") for row in rows)
        and all(row.get("unknown_is_free") is False for row in rows)
        and all(float(row.get("sensor_hfov_deg")) == 79.0 for row in reports)
        and all(bool(row.get("depth_readable")) for row in reports)
        and all(bool(row.get("first_edge_depth_checked")) for row in reports)
        and all(bool(row.get("all_edges_sparseocc_reaudited")) for row in reports)
        and not any(bool(row.get("action_emitted")) for row in reports)
        and not any(not bool(edge.get("sparseocc_safe")) for edge in edge_audits),
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
