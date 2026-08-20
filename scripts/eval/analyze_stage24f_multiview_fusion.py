"""Aggregate Stage24F fusion audits and select a query-frame policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


VARIANTS = (
    "f0_q_hard", "f1_q_probability", "f2_all_probability",
    "f3_q_embedding", "f4_q_robust_probability",
)
QUERY_CANDIDATES = (
    "f1_q_probability", "f3_q_embedding", "f4_q_robust_probability",
)


def _aggregate_variant(episodes: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    items = [episode["variants"][name] for episode in episodes]
    voxel_count = sum(int(item["voxel_count"]) for item in items)
    multi_count = sum(int(item["multi_view_voxel_count"]) for item in items)
    compatible = sum(int(item["gt_audit"]["compatible_voxel_count"]) for item in items)
    hits = sum(int(item["gt_audit"]["surface_distance_le_050m_count"]) for item in items)
    class_episode_count = sum(int(item["class_count"]) for item in items)

    def weighted(key: str, denominator: int = voxel_count) -> float:
        return sum(float(item[key]) * int(item["voxel_count"]) for item in items) / max(1, denominator)

    conflict_events = sum(
        float(item["multi_view_conflict_rate"]) * int(item["multi_view_voxel_count"])
        for item in items
    )
    agreement = sum(
        float(item["multi_view_agreement_mean"] or 0.0) * int(item["multi_view_voxel_count"])
        for item in items
    ) / max(1, multi_count)
    comparisons = [item.get("comparison_to_f0") or {} for item in items]
    classes = sorted(set().union(*(set(item["class_voxel_counts"]) for item in items)))
    macro_rates = [
        float(item["gt_episode_label_macro_hit_rate"])
        for item in items if item.get("gt_episode_label_macro_hit_rate") is not None
    ]
    return {
        "episode_count": len(items),
        "frame_count": sum(int(item["frame_count"]) for item in items),
        "voxel_count": voxel_count, "multi_view_voxel_count": multi_count,
        "multi_view_voxel_rate": multi_count / max(1, voxel_count),
        "multi_view_conflict_rate": conflict_events / max(1, multi_count),
        "multi_view_agreement_mean": agreement,
        "class_episode_count": class_episode_count, "unique_classes": classes,
        "gt_compatible_voxel_count": compatible, "gt_hit_count": hits,
        "gt_hit_rate": hits / max(1, compatible),
        "gt_episode_label_macro_hit_rate": (
            sum(macro_rates) / len(macro_rates) if macro_rates else None
        ),
        "confidence_mean": weighted("confidence_mean"),
        "margin_mean": weighted("margin_mean"),
        "normalized_entropy_mean": weighted("normalized_entropy_mean"),
        "isolated_voxel_rate": weighted("isolated_voxel_rate"),
        "fusion_seconds_total": sum(float(item["fusion_seconds"]) for item in items),
        "estimated_map_bytes_total": sum(int(item["estimated_map_bytes"]) for item in items),
        "persistent_feature_dim": int(items[0]["persistent_feature_dim"]),
        "persistent_semantic_payload_bytes_fp32_total": sum(
            int(item["persistent_semantic_payload_bytes_fp32"]) for item in items
        ),
        "changed_voxel_count": sum(int(item.get("changed_voxel_count", 0)) for item in comparisons),
        "gt_corrected_count": sum(int(item.get("gt_corrected_count", 0)) for item in comparisons),
        "gt_harmed_count": sum(int(item.get("gt_harmed_count", 0)) for item in comparisons),
        "gt_net_correction": sum(int(item.get("gt_net_correction", 0)) for item in comparisons),
    }


def analyze(root: Path, output: Path) -> Dict[str, Any]:
    episodes = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("**/episode_multiview_fusion.json"))
    ]
    if not episodes:
        raise SystemExit(f"No Stage24F episode reports under {root}")
    aggregate = {name: _aggregate_variant(episodes, name) for name in VARIANTS}
    baseline = aggregate["f0_q_hard"]
    checks = {}
    eligible = []
    for name in QUERY_CANDIDATES:
        item = aggregate[name]
        item_checks = {
            "gt_rate_not_worse_by_more_than_0_5pp": item["gt_hit_rate"] >= baseline["gt_hit_rate"] - 0.005,
            "gt_macro_rate_not_worse_by_more_than_1pp": (
                item["gt_episode_label_macro_hit_rate"] is not None
                and baseline["gt_episode_label_macro_hit_rate"] is not None
                and item["gt_episode_label_macro_hit_rate"]
                >= baseline["gt_episode_label_macro_hit_rate"] - 0.01
            ),
            "episode_class_coverage_at_least_95pct_f0": item["class_episode_count"] >= 0.95 * baseline["class_episode_count"],
            "isolated_voxel_rate_not_worse": item["isolated_voxel_rate"] <= baseline["isolated_voxel_rate"] + 1e-9,
            "nonnegative_gt_net_correction": item["gt_net_correction"] >= 0,
        }
        checks[name] = item_checks
        if all(item_checks.values()):
            eligible.append(name)
    consistency = all(
        bool((episode.get("online_q_consistency") or {}).get("passed"))
        for episode in episodes
    )
    selected = None
    if consistency and eligible:
        selected = max(
            eligible,
            key=lambda name: (
                aggregate[name]["gt_net_correction"],
                aggregate[name]["gt_hit_rate"],
                -aggregate[name]["isolated_voxel_rate"],
                -aggregate[name]["persistent_semantic_payload_bytes_fp32_total"],
            ),
        )
    result = {
        "audit_name": "stage24f_multiview_fusion",
        "episode_count": len(episodes), "variants": aggregate,
        "selection": {
            "online_q_consistency_passed": consistency,
            "candidate_checks": checks, "eligible_query_variants": eligible,
            "selected_query_variant": selected,
            "all_frame_variant_is_quality_ceiling_only": True,
            "decision": (
                "adopt_selected_for_downstream_shadow" if selected
                else "retain_f0_and_do_not_enter_downstream"
            ),
        },
        "decision_status": (
            "shadow_downstream_approved" if selected
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
