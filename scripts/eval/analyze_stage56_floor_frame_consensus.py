#!/usr/bin/env python3
"""Compare Stage56 floor-frame consensus with legacy floor any-hit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VERTICAL_ROLES = {"stairs_height_change", "mixed_height_change"}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("scene_id")), int(row.get("episode_id", -1))


def _mean(values) -> float | None:
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _metric(row: dict[str, Any], name: str) -> float | None:
    value = (row.get("free_metrics_observed_domain") or {}).get(name)
    return None if value is None else float(value)


def _summarize(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    rows = [record.get(field) or {} for record in records]
    return {
        "episode_count": len(records),
        "valid_count": sum(bool(row.get("valid")) for row in rows),
        "free_precision_mean": _mean(_metric(row, "precision") for row in rows),
        "free_recall_mean": _mean(_metric(row, "recall") for row in rows),
        "false_free_rate_mean": _mean(row.get("false_free_rate") for row in rows),
        "unknown_coverage_mean": _mean(row.get("unknown_coverage") for row in rows),
        "executed_route_free_recall_mean": _mean(
            row.get("executed_route_predicted_free_recall") for row in rows
        ),
        "historical_edge_strict_free_recall_mean": _mean(
            row.get("historical_edge_strict_free_recall") for row in rows
        ),
    }


def _delta(new: dict[str, Any], legacy: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key in (
        "free_precision_mean",
        "free_recall_mean",
        "false_free_rate_mean",
        "unknown_coverage_mean",
        "executed_route_free_recall_mean",
        "historical_edge_strict_free_recall_mean",
    ):
        result[key] = (
            None
            if new.get(key) is None or legacy.get(key) is None
            else float(new[key] - legacy[key])
        )
    return result


def analyze(run_root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = {_key(row): row for row in json.loads(manifest_path.read_text())}
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
        legacy = row.get("stage23b_navmesh_traversability_current_clearance") or {}
        consensus = row.get("stage56_navmesh_current_floor_frame_consensus") or {}
        if not legacy.get("valid"):
            errors.append(f"legacy_invalid:{key}")
        if not consensus.get("valid"):
            errors.append(f"consensus_invalid:{key}")
        if consensus.get("readout_mode") != "floor_frame_consensus":
            errors.append(f"consensus_mode_mismatch:{key}")
        if consensus.get("decision_applied") is not False:
            errors.append(f"decision_applied:{key}")
        if not consensus.get("frame_masks_available"):
            errors.append(f"frame_masks_unavailable:{key}")
        if row.get("stage23a_gt_fields_used_for_navigation"):
            errors.append(f"gt_navigation_leakage:{key}")
        records.append(
            {
                "scene_id": key[0],
                "episode_id": key[1],
                "episode_eval_seed": row.get("episode_eval_seed"),
                "audit_role": expected.get("audit_role"),
                "legacy": legacy,
                "floor_frame_consensus": consensus,
            }
        )

    flat = [row for row in records if row.get("audit_role") not in VERTICAL_ROLES]
    vertical = [row for row in records if row.get("audit_role") in VERTICAL_ROLES]
    groups = {}
    for name, values in (("all", records), ("flat", flat), ("vertical", vertical)):
        legacy = _summarize(values, "legacy")
        consensus = _summarize(values, "floor_frame_consensus")
        groups[name] = {
            "legacy": legacy,
            "floor_frame_consensus": consensus,
            "delta_consensus_minus_legacy": _delta(consensus, legacy),
        }

    flat_delta = groups["flat"]["delta_consensus_minus_legacy"]
    vertical_delta = groups["vertical"]["delta_consensus_minus_legacy"]
    no_false_free_regression = all(
        delta is not None and delta <= 1e-12
        for delta in (
            flat_delta.get("false_free_rate_mean"),
            vertical_delta.get("false_free_rate_mean"),
        )
    )
    vertical_recall_improved = bool(
        vertical_delta.get("executed_route_free_recall_mean") is not None
        and vertical_delta["executed_route_free_recall_mean"] > 0.0
    )
    integrity = len(records) == len(manifest) and not errors
    return {
        "task": "stage56_floor_relative_frame_consensus",
        "schema_version": "stage56_floor_frame_consensus_audit_v1",
        "expected_episode_count": len(manifest),
        "completed_episode_count": len(records),
        "integrity_passed": integrity,
        "shadow_only": True,
        "decision_applied": False,
        "unknown_is_free": False,
        "pixel_translation_allowed": False,
        "groups": groups,
        "gate": {
            "no_false_free_regression": no_false_free_regression,
            "vertical_executed_route_recall_improved": vertical_recall_improved,
            "stage57_shadow_ready": bool(
                integrity and no_false_free_regression and vertical_recall_improved
            ),
        },
        "errors": errors,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    report = analyze(args.run_root, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require_all and not report["integrity_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
