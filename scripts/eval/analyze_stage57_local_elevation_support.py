#!/usr/bin/env python3
"""Aggregate Stage57 local elevation-support graph audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

VERTICAL_ROLES = {"stairs_height_change", "mixed_height_change"}

def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def _key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("scene_id")), int(row.get("episode_id", -1))

def analyze(run_root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = {_key(row): row for row in json.loads(manifest_path.read_text(encoding="utf-8"))}
    progress = {}
    for path in run_root.glob("vlmap_safety_debug/*run_*/progress.json"):
        for row in _jsonl(path):
            progress[_key(row)] = row
    errors = []
    records = []
    for key, expected in manifest.items():
        row = progress.get(key)
        if row is None:
            errors.append(f"missing_progress:{key}")
            continue
        if int(row.get("episode_eval_seed", -1)) != int(expected["episode_eval_seed"]):
            errors.append(f"seed_mismatch:{key}")
        branches = {}
        for name in ("stage23b_navmesh_traversability_current_clearance", "stage23b_navmesh_traversability_oracle_sensor_clearance"):
            audit = row.get(name) or {}
            graph = audit.get("stage57_local_elevation_support") or {}
            if not audit.get("valid"):
                errors.append(f"audit_invalid:{name}:{key}")
            if graph and (graph.get("decision_applied") is not False or graph.get("unknown_is_free") is not False):
                errors.append(f"graph_mutation:{name}:{key}")
            branches[name] = graph
        records.append({"scene_id": key[0], "episode_id": key[1], "episode_eval_seed": row.get("episode_eval_seed"), "audit_role": expected.get("audit_role"), "current": branches["stage23b_navmesh_traversability_current_clearance"], "oracle_sensor": branches["stage23b_navmesh_traversability_oracle_sensor_clearance"]})

    def summary(items: list[dict[str, Any]], branch: str) -> dict[str, Any]:
        graphs = [item.get(branch) or {} for item in items]
        def mean(name: str) -> float | None:
            values = [float(g[name]) for g in graphs if g.get(name) is not None]
            return sum(values) / len(values) if values else None
        return {"episode_count": len(items), "graph_valid_count": sum(bool(g.get("reason") == "ok") for g in graphs), "mean_centerline_support_coverage": mean("centerline_support_coverage"), "mean_corridor_support_coverage": mean("corridor_support_coverage"), "mean_headroom_blocked_count": mean("headroom_blocked_count"), "continuous_support_centerline_count": sum(bool(g.get("continuous_support_centerline")) for g in graphs), "full_footprint_support_count": sum(bool(g.get("full_footprint_support")) for g in graphs), "full_footprint_safe_corridor_count": sum(bool(g.get("full_footprint_safe_corridor")) for g in graphs)}

    flat = [r for r in records if r.get("audit_role") not in VERTICAL_ROLES]
    vertical = [r for r in records if r.get("audit_role") in VERTICAL_ROLES]
    groups = {label: {"current": summary(items, "current"), "oracle_sensor": summary(items, "oracle_sensor")} for label, items in (("all", records), ("flat", flat), ("vertical", vertical))}
    integrity = len(records) == len(manifest) and not errors
    safe_count = groups["vertical"]["oracle_sensor"]["full_footprint_safe_corridor_count"]
    return {"task": "stage57_local_elevation_support", "schema_version": "stage57_local_elevation_support_audit_v1", "expected_episode_count": len(manifest), "completed_episode_count": len(records), "integrity_passed": integrity, "shadow_only": True, "decision_applied": False, "unknown_is_free": False, "pixel_translation_allowed": False, "groups": groups, "gate": {"integrity_passed": integrity, "vertical_full_footprint_safe_corridor_count": int(safe_count), "stage58_pixel_active_ready": bool(integrity and safe_count > 0)}, "errors": errors, "records": records}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    report = analyze(args.run_root, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require_all and not report["integrity_passed"]:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
