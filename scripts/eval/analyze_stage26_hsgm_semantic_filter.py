"""Compare raw and HSGM-inspired filtered semantic surfaces on one trajectory."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

try:
    from scripts.eval.analyze_stage24d_online_lseg_shadow import (
        _compare_ledgers, _jsonl, _ledgers, _semantic_dirs,
    )
except ModuleNotFoundError:
    from analyze_stage24d_online_lseg_shadow import (
        _compare_ledgers, _jsonl, _ledgers, _semantic_dirs,
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_key(row: Dict[str, Any]) -> tuple[str, str]:
    return str(row["scene_id"]), str(row["episode_id"])


def _distance_metrics(nodes: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    nodes = list(nodes)
    distances = np.asarray([
        float(node["gt_surface_distance_m"])
        for node in nodes if node.get("gt_surface_distance_m") is not None
    ], dtype=np.float64)
    hits = int(np.count_nonzero(distances <= 0.50))
    per_label: Dict[str, List[float]] = defaultdict(list)
    for node in nodes:
        if node.get("gt_surface_distance_m") is not None:
            per_label[str(node["label"])].append(float(node["gt_surface_distance_m"]))
    return {
        "node_count": len(nodes),
        "strong_node_count": sum(
            node.get("evidence_tier") == "strong" for node in nodes
        ),
        "compatible_node_count": int(distances.size),
        "surface_distance_le_050m_count": hits,
        "surface_distance_le_050m_rate": (
            float(hits / distances.size) if distances.size else None
        ),
        "surface_distance_m_median": (
            float(np.median(distances)) if distances.size else None
        ),
        "surface_distance_m_p95": (
            float(np.percentile(distances, 95)) if distances.size else None
        ),
        "per_label": {
            label: {
                "compatible_node_count": len(values),
                "surface_distance_le_050m_count": sum(value <= 0.50 for value in values),
                "surface_distance_le_050m_rate": float(
                    np.mean(np.asarray(values) <= 0.50)
                ),
                "surface_distance_m_p95": float(np.percentile(values, 95)),
            }
            for label, values in sorted(per_label.items())
        },
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _surface_rows(path: Path) -> Counter:
    with np.load(path, allow_pickle=False) as payload:
        return Counter(
            (
                payload["map_xyz"][index].tobytes(),
                int(payload["class_id"][index]),
                payload["confidence"][index].tobytes(),
                int(payload["occ_state"][index]),
            )
            for index in range(len(payload["map_xyz"]))
        )


def analyze(
    run_root: Path, baseline_root: Path, manifest: Path, output: Path,
) -> Dict[str, Any]:
    expected_rows = _load_json(manifest)
    expected = {_episode_key(row): row for row in expected_rows}
    ledgers = _ledgers(run_root)
    baseline_ledgers = _ledgers(baseline_root)
    semantic_dirs = _semantic_dirs(run_root)
    baseline_semantic_dirs = _semantic_dirs(baseline_root)
    errors: List[str] = []
    raw_nodes: List[Dict[str, Any]] = []
    filtered_nodes: List[Dict[str, Any]] = []
    episode_reports = []
    raw_samples = filtered_samples = 0
    small_rejected = density_rejected = 0
    raw_severe = filtered_severe = 0
    raw_strong_severe = filtered_strong_severe = 0
    edge_touch_components = total_components = 0

    for key, row in expected.items():
        label = f"{key[0]}/{key[1]}"
        ledger = ledgers.get(key)
        baseline = baseline_ledgers.get(key)
        semantic_dir = semantic_dirs.get(key)
        baseline_semantic_dir = baseline_semantic_dirs.get(key)
        if ledger is None:
            errors.append(f"{label}:missing_ledger")
            continue
        if baseline is None:
            errors.append(f"{label}:missing_baseline_ledger")
        if semantic_dir is None:
            errors.append(f"{label}:missing_semantic_dir")
            continue
        if baseline_semantic_dir is None:
            errors.append(f"{label}:missing_baseline_semantic_dir")
        meta = _load_json(semantic_dir / "episode_meta.json")
        if int(meta.get("episode_eval_seed", -1)) != int(row["episode_eval_seed"]):
            errors.append(f"{label}:episode_eval_seed_mismatch")
        summary = _load_json(semantic_dir / "summary.json")
        component = summary.get("component_filter") or {}
        if not component.get("enabled"):
            errors.append(f"{label}:component_filter_disabled")
        if summary.get("decision_status") != "audit_only_not_navigation_ready":
            errors.append(f"{label}:decision_status_violation")
        if int(summary.get("action_applied_count", -1)) != 0:
            errors.append(f"{label}:action_applied_violation")
        if int(summary.get("error_count", -1)) != 0:
            errors.append(f"{label}:lseg_error")
        mismatch = [] if baseline is None else _compare_ledgers(ledger, baseline)
        errors.extend(f"{label}:trajectory_mismatch:{item}" for item in mismatch)

        current_raw = _load_json(semantic_dir / "nodes.json")
        current_filtered = _load_json(semantic_dir / "nodes_filtered.json")
        if len(current_raw) != int(summary.get("node_count", -1)):
            errors.append(f"{label}:raw_node_count_mismatch")
        if len(current_filtered) != int(component.get("filtered_node_count", -1)):
            errors.append(f"{label}:filtered_node_count_mismatch")
        raw_surface_path = semantic_dir / "semantic_surface_memory.npz"
        filtered_surface_path = semantic_dir / "semantic_surface_memory_filtered.npz"
        raw_rows = _surface_rows(raw_surface_path) if raw_surface_path.is_file() else Counter()
        raw_semantics_exact = False
        if baseline_semantic_dir is not None:
            baseline_nodes_path = baseline_semantic_dir / "nodes.json"
            baseline_surface_path = baseline_semantic_dir / "semantic_surface_memory.npz"
            if not baseline_nodes_path.is_file() or not baseline_surface_path.is_file():
                errors.append(f"{label}:missing_baseline_raw_semantics")
            else:
                raw_semantics_exact = (
                    current_raw == _load_json(baseline_nodes_path)
                    and raw_rows == _surface_rows(baseline_surface_path)
                )
                if not raw_semantics_exact:
                    errors.append(f"{label}:raw_semantics_mismatch")
        if filtered_surface_path.is_file():
            if not raw_surface_path.is_file():
                errors.append(f"{label}:filtered_surface_without_raw_surface")
            else:
                filtered_rows = _surface_rows(filtered_surface_path)
                if any(filtered_rows[item] > raw_rows[item] for item in filtered_rows):
                    errors.append(f"{label}:filtered_surface_not_exact_raw_subset")
        raw_nodes.extend(current_raw)
        filtered_nodes.extend(current_filtered)

        raw_conflicts = summary.get("cross_label_conflict_audit") or {}
        filtered_conflicts = component.get("filtered_cross_label_conflict_audit") or {}
        raw_severe += int(raw_conflicts.get("severe_count", 0))
        filtered_severe += int(filtered_conflicts.get("severe_count", 0))
        raw_strong_severe += int(raw_conflicts.get("strong_severe_count", 0))
        filtered_strong_severe += int(filtered_conflicts.get("strong_severe_count", 0))
        frame_events = [
            event for event in _jsonl(semantic_dir / "events.jsonl")
            if event.get("valid")
        ]
        for event in frame_events:
            stats = event.get("component_filter") or {}
            raw_samples += int(stats.get("raw_sample_count", 0))
            filtered_samples += int(stats.get("retained_sample_count", 0))
            small_rejected += int(stats.get("small_component_rejected_sample_count", 0))
            density_rejected += int(stats.get("density_rejected_sample_count", 0))
            edge_touch_components += int(stats.get("edge_touch_component_count", 0))
            total_components += int(stats.get("component_count", 0))
        episode_reports.append({
            "scene_id": key[0],
            "episode_id": key[1],
            "episode_eval_seed": int(row["episode_eval_seed"]),
            "trajectory_exact_match": baseline is not None and not mismatch,
            "raw_semantics_exact_match": raw_semantics_exact,
            "valid_frame_count": summary.get("valid_frame_count"),
            "raw_node_count": len(current_raw),
            "filtered_node_count": len(current_filtered),
            "stored_surface_retention_rate": component.get(
                "stored_surface_retention_rate"
            ),
        })

    unexpected = sorted(set(ledgers) - set(expected))
    missing = sorted(set(expected) - set(ledgers))
    errors.extend(f"{scene}/{episode}:unexpected_episode" for scene, episode in unexpected)
    errors.extend(f"{scene}/{episode}:manifest_episode_missing" for scene, episode in missing)
    raw = _distance_metrics(raw_nodes)
    filtered = _distance_metrics(filtered_nodes)
    raw["severe_cross_label_conflict_count"] = raw_severe
    raw["strong_severe_cross_label_conflict_count"] = raw_strong_severe
    filtered["severe_cross_label_conflict_count"] = filtered_severe
    filtered["strong_severe_cross_label_conflict_count"] = filtered_strong_severe
    hit_retention = _ratio(
        filtered["surface_distance_le_050m_count"],
        raw["surface_distance_le_050m_count"],
    )
    strong_retention = _ratio(filtered["strong_node_count"], raw["strong_node_count"])
    p95_delta = (
        filtered["surface_distance_m_p95"] - raw["surface_distance_m_p95"]
        if filtered["surface_distance_m_p95"] is not None
        and raw["surface_distance_m_p95"] is not None else None
    )
    severe_reduction = (
        1.0 - _ratio(filtered_severe, raw_severe) if raw_severe else None
    )
    result = {
        "audit_name": "stage26_hsgm_semantic_filter",
        "integrity_passed": not errors,
        "episode_count": len(episode_reports),
        "manifest_expected_count": len(expected),
        "all_trajectories_exact_match": bool(episode_reports) and all(
            item["trajectory_exact_match"] for item in episode_reports
        ),
        "all_raw_semantics_exact_match": bool(episode_reports) and all(
            item["raw_semantics_exact_match"] for item in episode_reports
        ),
        "errors": errors,
        "frame_filter": {
            "raw_sample_count": raw_samples,
            "filtered_sample_count": filtered_samples,
            "sample_retention_rate": _ratio(filtered_samples, raw_samples),
            "small_component_rejected_sample_count": small_rejected,
            "density_rejected_sample_count": density_rejected,
            "component_count": total_components,
            "edge_touch_component_count": edge_touch_components,
        },
        "raw": raw,
        "filtered": filtered,
        "delta": {
            "gt_hit_count_retention": hit_retention,
            "strong_node_retention": strong_retention,
            "surface_distance_m_p95_delta": p95_delta,
            "severe_conflict_reduction_rate": severe_reduction,
            "node_reduction_rate": 1.0 - _ratio(
                filtered["node_count"], raw["node_count"]
            ) if raw["node_count"] else None,
        },
        "evidence_gate": {
            "coverage_preserved": hit_retention is not None and hit_retention >= 0.85,
            "strong_evidence_preserved": (
                strong_retention is not None and strong_retention >= 0.70
            ),
            "tail_or_conflict_improved": (
                (p95_delta is not None and p95_delta <= -0.10)
                or (severe_reduction is not None and severe_reduction >= 0.20)
            ),
        },
        "episodes": episode_reports,
    }
    result["evidence_gate"]["passed"] = (
        result["integrity_passed"]
        and all(result["evidence_gate"][key] for key in (
            "coverage_preserved", "strong_evidence_preserved",
            "tail_or_conflict_improved",
        ))
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run_root, args.baseline_root, args.manifest, args.output)


if __name__ == "__main__":
    main()
