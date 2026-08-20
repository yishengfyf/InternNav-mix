"""Aggregate Stage24E replay frequency comparisons and enforce the audit gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from internnav.utils.lseg_replay_frequency import evaluate_frequency_gate


def analyze(root: Path, output: Path) -> Dict[str, Any]:
    episodes = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("**/episode_frequency_comparison.json"))
    ]
    if not episodes:
        raise SystemExit("No Stage24E episode comparisons found")
    aggregate = {}
    for name in ("q", "q_plus_k", "all"):
        items = [episode["variants"][name] for episode in episodes]
        calls = sum(int(item["call_count"]) for item in items)
        nodes = sum(int(item["node_count"]) for item in items)
        conflicts = sum(int(item["conflict_count"]) for item in items)
        severe_conflicts = sum(int(item["severe_conflict_count"]) for item in items)
        strong_severe_conflicts = sum(
            int(item["strong_severe_conflict_count"]) for item in items
        )
        strong_nodes = sum(int(item["strong_node_count"]) for item in items)
        weak_nodes = sum(int(item["weak_node_count"]) for item in items)
        compatible = sum(int(item["gt_compatible_node_count"]) for item in items)
        hits = sum(int(item["gt_hit_count"]) for item in items)
        classes = set().union(*(set(item["classes"]) for item in items))
        short_lived = set().union(*(set(item["short_lived_classes"]) for item in items))
        class_episode_count = sum(int(item["class_count"]) for item in items)
        short_lived_episode_count = sum(
            int(item["short_lived_class_count"]) for item in items
        )
        aggregate[name] = {
            "call_count": calls, "node_count": nodes,
            "conflict_count": conflicts,
            "conflicts_per_100_nodes": 100.0 * conflicts / max(1, nodes),
            "severe_conflict_count": severe_conflicts,
            "severe_conflicts_per_100_nodes": (
                100.0 * severe_conflicts / max(1, nodes)
            ),
            "strong_severe_conflict_count": strong_severe_conflicts,
            "strong_severe_conflicts_per_100_strong_nodes": (
                100.0 * strong_severe_conflicts / max(1, strong_nodes)
            ),
            "strong_node_count": strong_nodes, "weak_node_count": weak_nodes,
            "gt_compatible_node_count": compatible, "gt_hit_count": hits,
            "gt_hit_rate": hits / max(1, compatible),
            # Counts are episode-label pairs so misses cannot be hidden by another scene.
            "class_count": class_episode_count,
            "unique_class_count": len(classes), "classes": sorted(classes),
            "short_lived_class_count": short_lived_episode_count,
            "unique_short_lived_class_count": len(short_lived),
            "short_lived_classes": sorted(short_lived),
        }
    for name in aggregate:
        aggregate[name]["call_fraction_of_all"] = (
            aggregate[name]["call_count"] / max(1, aggregate["all"]["call_count"])
        )
    gate = evaluate_frequency_gate(aggregate["q"], aggregate["q_plus_k"], aggregate["all"])
    consistency_passed = all(
        bool((episode.get("online_q_consistency") or {}).get("passed"))
        for episode in episodes
    )
    gate["checks"]["online_q_replay_exact_match"] = consistency_passed
    gate["passed"] = bool(gate["passed"] and consistency_passed)
    if not gate["passed"]:
        gate["decision"] = "retain_audit_only_and_tune_keyframes"
    result = {
        "audit_name": "stage24e_lseg_replay_frequency",
        "episode_count": len(episodes), "variants": aggregate, "gate": gate,
        "decision_status": (
            "shadow_downstream_approved" if gate["passed"]
            else "audit_only_not_navigation_ready"
        ),
        "episodes": episodes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.root, args.output)


if __name__ == "__main__":
    main()
