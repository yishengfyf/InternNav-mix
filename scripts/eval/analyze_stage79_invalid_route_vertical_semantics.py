"""Offline audit for Stage78 invalid routes and vertically flattened semantics.

This analyzer consumes saved causal ledgers only.  Habitat height and
pathfinder labels are reported as offline attribution and never authorize an
online route, prompt, waypoint, or action.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from PIL import Image, ImageDraw


CELL_SIZE_M = 0.05
NEIGHBOR_RADII_M = (0.25, 0.50, 0.75)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _all_rows(root: Path, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for text in glob.glob(str(root / "**" / name), recursive=True):
        rows.extend(_jsonl(Path(text)))
    return rows


def _cell(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None


def _distance_m(left: tuple[int, int], right: tuple[int, int]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1]) * CELL_SIZE_M


def _nearest_distance_m(
    cell: tuple[int, int], targets: Iterable[tuple[int, int]]
) -> float | None:
    distances = [_distance_m(cell, target) for target in targets]
    return min(distances) if distances else None


def _state_at(cell: tuple[int, int] | None, channels: dict[str, Any]) -> str:
    if cell is None:
        return "missing"
    for state, key in (("free", "known_free"), ("occupied", "occupied"), ("unknown", "unknown")):
        if cell in {_cell(item) for item in channels.get(key) or []}:
            return state
    return "outside_snapshot"


def _height_summary(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    heights = []
    for node in nodes:
        centroid = node.get("centroid")
        if isinstance(centroid, (list, tuple)) and len(centroid) >= 3:
            try:
                heights.append(float(centroid[2]))
            except (TypeError, ValueError):
                pass
    if not heights:
        return {"count": 0, "min_m": None, "median_m": None, "max_m": None, "span_m": None}
    return {
        "count": len(heights),
        "min_m": min(heights),
        "median_m": median(heights),
        "max_m": max(heights),
        "span_m": max(heights) - min(heights),
        "height_bin_0p20m_count": len({round(value / 0.20) for value in heights}),
    }


def _nearby_summary(
    nodes: list[dict[str, Any]], target: tuple[int, int] | None, radius_m: float
) -> dict[str, Any]:
    selected = []
    if target is not None:
        for node in nodes:
            grid = _cell(node.get("grid"))
            if grid is not None and _distance_m(grid, target) <= radius_m + 1e-9:
                selected.append(node)
    labels = Counter(str(node.get("label") or "unknown") for node in selected)
    strong = sum(str(node.get("evidence_tier")) == "strong" for node in selected)
    return {
        "radius_m": radius_m,
        "node_count": len(selected),
        "strong_evidence_node_count": strong,
        "label_counts": dict(sorted(labels.items())),
        "floor_height": _height_summary([node for node in selected if node.get("label") == "floor"]),
        "stairs_height": _height_summary([node for node in selected if node.get("label") == "stairs"]),
    }


def _selected_anchor(native_event: dict[str, Any]) -> dict[str, Any]:
    stage59 = native_event.get("stage59_productive_onset") or {}
    for anchor in stage59.get("anchors") or []:
        if anchor.get("anchor") == "last_productive_pre_loop":
            return anchor
    return {}


def _geometry_summary(native_event: dict[str, Any]) -> dict[str, Any]:
    anchor = _selected_anchor(native_event)
    policy = anchor.get("stage58_support_policy") or {}
    arms = []
    for arm in policy.get("arms") or []:
        graph = arm.get("graph") or {}
        arms.append({
            "policy": arm.get("policy"),
            "predicted_first_primitive_safe": arm.get("predicted_first_primitive_safe"),
            "offline_truth_safe": arm.get("offline_truth_safe"),
            "eligible_corridor": graph.get("eligible_corridor"),
            "headroom_blocked_count": graph.get("headroom_blocked_count"),
            "corridor_support_coverage": graph.get("corridor_support_coverage"),
            "leading_full_footprint_safe_segment_m": graph.get(
                "leading_full_footprint_safe_segment_m"
            ),
        })
    truth = anchor.get("offline_primitive_truth") or {}
    return {
        "anchor_grid": anchor.get("grid"),
        "initial_route_state_counts": anchor.get("route_state_counts") or {},
        "initial_sparseocc_connectivity": anchor.get("current_sparseocc_connectivity"),
        "initial_bridge_reason": anchor.get("image_bridge_reason"),
        "offline_primitive_truth_valid": truth.get("valid"),
        "offline_primitive_safe": truth.get("primitive_safe"),
        "offline_primitive_reason": truth.get("reason"),
        "offline_vertical_delta_m": truth.get("vertical_delta_m"),
        "support_arms": arms,
    }


def _render(event: dict[str, Any], report: dict[str, Any], output: Path) -> None:
    spatial = event.get("stage78_recovery_bev_spatial") or {}
    channels = spatial.get("channels") or {}
    current = _cell(event.get("start_grid"))
    anchor = _cell(event.get("anchor_grid"))
    cells = []
    for key in ("known_free", "occupied", "unknown"):
        cells.extend(cell for cell in (_cell(item) for item in channels.get(key) or []) if cell)
    cells.extend(cell for cell in (_cell(item) for item in spatial.get("executed_route") or []) if cell)
    if current:
        cells.append(current)
    if anchor:
        cells.append(anchor)
    if not cells:
        return
    min_r, max_r = min(c[0] for c in cells), max(c[0] for c in cells)
    min_c, max_c = min(c[1] for c in cells), max(c[1] for c in cells)
    scale, pad, text_h = 6, 12, 92
    width = (max_c - min_c + 1) * scale + 2 * pad
    height = (max_r - min_r + 1) * scale + 2 * pad + text_h
    image = Image.new("RGB", (max(width, 520), height), (25, 29, 34))
    draw = ImageDraw.Draw(image)

    def xy(cell: tuple[int, int]) -> tuple[int, int, int, int]:
        x = pad + (cell[1] - min_c) * scale
        y = pad + (cell[0] - min_r) * scale
        return x, y, x + scale - 1, y + scale - 1

    colors = {"known_free": (46, 139, 87), "occupied": (215, 73, 79), "unknown": (170, 153, 56)}
    for key, color in colors.items():
        for raw in channels.get(key) or []:
            cell = _cell(raw)
            if cell:
                draw.rectangle(xy(cell), fill=color)
    for raw in spatial.get("executed_route") or []:
        cell = _cell(raw)
        if cell:
            draw.rectangle(xy(cell), fill=(46, 121, 202))
    for node in channels.get("semantic_nodes") or []:
        cell = _cell(node.get("grid"))
        if cell:
            box = xy(cell)
            draw.rectangle(box, outline=(245, 62, 232), width=1)
    if anchor:
        draw.ellipse(xy(anchor), outline=(255, 220, 45), width=2)
    if current:
        draw.ellipse(xy(current), outline=(255, 255, 255), width=2)
    y0 = height - text_h + 6
    lines = [
        f"{report['scene_id']}/{report['episode_id']} step={report['step_id']} reason={report['route_reason']}",
        f"anchor={report['anchor_state']} nearest_sem={report['nearest_semantic_to_anchor_m']}m vertical_omission={report['vertical_pose_omission_m']}m",
        f"attribution={report['attribution']}  magenta=semantic blue=history white=current yellow=anchor",
    ]
    for index, line in enumerate(lines):
        draw.text((pad, y0 + index * 23), line, fill=(235, 235, 235))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def analyze(*, input_root: Path, output: Path, viz_dir: Path) -> dict[str, Any]:
    rows = _all_rows(input_root, "s2_recovery_context_events.jsonl")
    route_events = [row for row in rows if row.get("event_type") == "stage75_route_guidance"]
    native_by_episode = {
        (str(row.get("scene_id")), str(row.get("episode_id"))): row
        for row in rows
        if row.get("event_type") == "stage65_native_recovery_set"
    }
    errors = []
    if not route_events:
        errors.append("missing_route_events")
    event_reports = []
    invalid_rendered = set()
    attribution_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    vertical_suspect_episodes = set()
    trustworthy_landmark_episodes = set()
    stable_stage78_episodes = set()

    stage78_path = input_root / "stage78_semantic_attachment_audit.json"
    if stage78_path.is_file():
        stage78 = json.loads(stage78_path.read_text(encoding="utf-8"))
        stable_stage78_episodes = {
            (str(key[0]), str(key[1]))
            for key in stage78.get("stable_route_landmark_episode_keys") or []
            if isinstance(key, (list, tuple)) and len(key) >= 2
        }

    for event in route_events:
        key = (str(event.get("scene_id")), str(event.get("episode_id")))
        spatial = event.get("stage78_recovery_bev_spatial") or {}
        channels = spatial.get("channels") or {}
        nodes = list(channels.get("semantic_nodes") or [])
        anchor = _cell(event.get("anchor_grid"))
        current = _cell(event.get("start_grid"))
        history = [cell for cell in (_cell(item) for item in spatial.get("executed_route") or []) if cell]
        anchor_state = _state_at(anchor, channels)
        current_pose = spatial.get("current_pose") or {}
        pose_z = float(current_pose.get("z", 0.0) or 0.0)
        gt_height = current_pose.get("gt_relative_height_m")
        gt_height = float(gt_height) if gt_height is not None else None
        omission = None if gt_height is None else gt_height - pose_z
        vertical_trustworthy = omission is not None and abs(omission) <= 0.25 + 1e-9
        geometry = _geometry_summary(native_by_episode.get(key, {}))
        offline_safe = geometry.get("offline_primitive_safe") is True
        valid = bool(event.get("valid"))
        if valid:
            attribution = "route_valid"
        elif omission is not None and abs(omission) > 0.25 and offline_safe:
            attribution = "vertical_pose_omission_false_block_suspect"
            vertical_suspect_episodes.add(key)
        elif offline_safe:
            attribution = "sparseocc_false_block_suspect"
        elif geometry.get("offline_primitive_truth_valid"):
            attribution = "offline_geometric_block"
        else:
            attribution = "unresolved"
        attribution_counts[attribution] += 1
        reason_counts[str(event.get("reason") or "unknown")] += 1

        node_cells = [(node, _cell(node.get("grid"))) for node in nodes]
        anchor_distances = [
            _distance_m(cell, anchor) for _, cell in node_cells if cell is not None and anchor is not None
        ]
        history_distances = [
            _nearest_distance_m(cell, history) for _, cell in node_cells if cell is not None
        ]
        report = {
            "scene_id": key[0],
            "episode_id": key[1],
            "step_id": event.get("current_query_step"),
            "route_valid": valid,
            "route_reason": event.get("reason"),
            "start_grid": list(current) if current else None,
            "anchor_grid": list(anchor) if anchor else None,
            "anchor_state": anchor_state,
            "anchor_distance_from_current_m": (
                _distance_m(current, anchor) if current is not None and anchor is not None else None
            ),
            "causal_local_semantic_node_count": len(nodes),
            "nearest_semantic_to_anchor_m": min(anchor_distances) if anchor_distances else None,
            "nearest_semantic_to_history_route_m": min(
                value for value in history_distances if value is not None
            ) if any(value is not None for value in history_distances) else None,
            "anchor_neighborhoods": [
                _nearby_summary(nodes, anchor, radius) for radius in NEIGHBOR_RADII_M
            ],
            "current_pose_height_m": pose_z,
            "offline_gt_relative_height_m": gt_height,
            "vertical_pose_omission_m": omission,
            "pose_height_source": current_pose.get("pose_height_source"),
            "vertical_projection_trustworthy_0p25m": vertical_trustworthy,
            "attribution": attribution,
            "geometry": geometry,
            "unknown_is_free": False,
            "semantic_can_override_safety": False,
            "gt_used_for_navigation": False,
            "action_applied": False,
        }
        if valid and key in stable_stage78_episodes and vertical_trustworthy:
            trustworthy_landmark_episodes.add(key)
        event_reports.append(report)
        if not valid and key not in invalid_rendered:
            _render(event, report, viz_dir / f"{key[0]}_{key[1]}_invalid_route.png")
            invalid_rendered.add(key)

    episodes = {(row["scene_id"], row["episode_id"]) for row in event_reports}
    invalid_episodes = {
        (row["scene_id"], row["episode_id"]) for row in event_reports if not row["route_valid"]
    }
    result = {
        "task": "stage79_invalid_route_vertical_semantics_offline_audit",
        "schema_version": "stage79_invalid_route_vertical_semantics_v1",
        "integrity_passed": not errors,
        "errors": errors,
        "episode_count": len(episodes),
        "route_event_count": len(event_reports),
        "invalid_route_event_count": sum(not row["route_valid"] for row in event_reports),
        "invalid_route_episode_count": len(invalid_episodes),
        "route_reason_counts": dict(sorted(reason_counts.items())),
        "attribution_counts": dict(sorted(attribution_counts.items())),
        "vertical_pose_omission_suspect_episode_count": len(vertical_suspect_episodes),
        "vertical_pose_omission_suspect_episode_keys": [list(key) for key in sorted(vertical_suspect_episodes)],
        "stage78_stable_route_landmark_episode_count": len(stable_stage78_episodes),
        "vertically_trustworthy_stable_route_landmark_episode_count": len(
            trustworthy_landmark_episodes
        ),
        "vertically_trustworthy_stable_route_landmark_episode_keys": [
            list(key) for key in sorted(trustworthy_landmark_episodes)
        ],
        "semantic_prompt_release_gate_min_episode_count": 4,
        "semantic_prompt_release_gate_passed": len(trustworthy_landmark_episodes) >= 4,
        "invalid_route_visualization_count": len(invalid_rendered),
        "event_reports": event_reports,
        "contract": {
            "offline_audit_only": True,
            "unknown_is_free": False,
            "semantic_can_override_safety": False,
            "gt_used_for_navigation": False,
            "prompt_injected": False,
            "action_applied": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--viz-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(input_root=args.input_root, output=args.output, viz_dir=args.viz_dir), indent=2))


if __name__ == "__main__":
    main()
