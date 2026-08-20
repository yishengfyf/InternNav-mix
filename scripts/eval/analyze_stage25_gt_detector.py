#!/usr/bin/env python3
"""Audit Stage25 replay contracts and mine causal stuck-event candidates.

The script is offline-only. It treats final navigation outcome as metadata, not
event ground truth, and never reads future observations when deciding an onset.
Future observations are used only to label self-recovery and persistence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


def jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def distance(a: Any, b: Any) -> float:
    try:
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
    except (TypeError, ValueError, IndexError):
        return float("inf")


def sha256_array(path: Path, *, rgb: bool) -> str:
    if rgb:
        array = np.ascontiguousarray(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8))
    else:
        with np.load(path) as payload:
            array = np.ascontiguousarray(payload["depth_m"], dtype=np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest()


def episode_key(row: Mapping[str, Any]) -> str:
    return f"{row.get('scene_id')}|{row.get('episode_id')}"


def discover_episodes(run_root: Path) -> List[Path]:
    return sorted({path.parent for path in run_root.glob("**/replay_ledger/*/observations.jsonl")})


def progress_by_episode(run_root: Path) -> Dict[str, Dict[str, Any]]:
    output = {}
    for path in run_root.glob("**/progress.json"):
        for row in jsonl(path):
            output[episode_key(row)] = row
    return output


def resolve_episode_eval_seed(
    meta: Mapping[str, Any], progress: Mapping[str, Any],
) -> Tuple[Optional[Any], str]:
    """Resolve a causal run seed while supporting ledgers predating meta storage."""
    if meta.get("episode_eval_seed") is not None:
        return meta["episode_eval_seed"], "episode_meta"
    if progress.get("episode_eval_seed") is not None:
        return progress["episode_eval_seed"], "progress_fallback"
    return None, "missing"


def loop_events_by_episode(run_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    output: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for path in run_root.glob("**/s2_action_loop_events.jsonl"):
        for row in jsonl(path):
            if row.get("transition") == "start":
                output[episode_key(row)].append(row)
    return output


def lseg_events(episode_dir: Path) -> List[Dict[str, Any]]:
    run_dir = episode_dir.parent.parent
    prefix = episode_dir.name.rsplit("_r", 1)[0]
    candidates = sorted((run_dir / "online_lseg_shadow").glob(f"{prefix}_r*/events.jsonl"))
    if not candidates:
        return []
    semantic_dir = candidates[-1].parent
    events = jsonl(candidates[-1])
    for event in events:
        relative = event.get("surface_path")
        path = semantic_dir / str(relative) if relative else None
        if path is None or not path.is_file():
            event["spatial_semantic_cells"] = []
            continue
        with np.load(path) as payload:
            points = np.asarray(payload["map_xyz"], dtype=np.float32)
            class_ids = np.asarray(payload["class_id"], dtype=np.int16)
            confidence = np.asarray(payload["confidence"], dtype=np.float32)
        event["spatial_semantic_cells"] = semantic_cells(
            points, class_ids, confidence
        )
    return events


def semantic_cells(
    points: np.ndarray, class_ids: np.ndarray, confidence: np.ndarray,
    *, cell_size_m: float = 0.50, min_points: int = 8,
    min_confidence: float = 0.40,
) -> List[str]:
    """Build compact strong spatial identities from one causal LSeg frame."""
    if not len(points):
        return []
    cells: Dict[Tuple[int, int, int, int], List[float]] = defaultdict(list)
    quantized = np.floor(points / float(cell_size_m)).astype(np.int32)
    for cell, class_id, score in zip(quantized, class_ids, confidence):
        cells[(int(class_id), int(cell[0]), int(cell[1]), int(cell[2]))].append(
            float(score)
        )
    return [
        "%d:%d:%d:%d" % key
        for key, scores in sorted(cells.items())
        if len(scores) >= int(min_points)
        and float(np.mean(scores)) >= float(min_confidence)
    ]


def compact_observation(row: Mapping[str, Any]) -> Dict[str, Any]:
    pose = row.get("pose") or {}
    occ = row.get("occ_summary") or {}
    audit = row.get("audit_metrics") or {}
    return {
        "record_index": int(row.get("record_index", 0)),
        "step_id": int(row.get("step_id", 0)),
        "observation_key": row.get("observation_key"),
        "gps": pose.get("gps"),
        "compass": pose.get("compass"),
        "previous_action": row.get("previous_action"),
        "previous_action_applied": row.get("previous_action_applied"),
        "collision_count": audit.get("collision_count"),
        "collision_delta": audit.get("collision_delta"),
        "distance_to_goal": audit.get("distance_to_goal"),
        "success_so_far": audit.get("success"),
        "occupied_added": occ.get("occupied_added"),
        "free_added": occ.get("free_added"),
        "occupied_voxel_count": occ.get("occupied_voxel_count"),
        "free_voxel_count": occ.get("free_voxel_count"),
        "frontier_count": occ.get("frontier_count"),
        "rgb_path": row.get("rgb_path"),
        "depth_path": row.get("depth_path"),
    }


def window_displacement(rows: Sequence[Mapping[str, Any]]) -> float:
    finite = [row.get("gps") for row in rows if isinstance(row.get("gps"), (list, tuple))]
    return 0.0 if len(finite) < 2 else distance(finite[0], finite[-1])


def path_length(rows: Sequence[Mapping[str, Any]]) -> float:
    return sum(distance(a.get("gps"), b.get("gps")) for a, b in zip(rows, rows[1:]))


def cumulative_path_length(rows: Sequence[Mapping[str, Any]]) -> List[float]:
    cumulative = [0.0]
    for previous, current in zip(rows, rows[1:]):
        cumulative.append(
            cumulative[-1] + distance(previous.get("gps"), current.get("gps"))
        )
    return cumulative


def unique_occ_growth(rows: Sequence[Mapping[str, Any]]) -> int:
    if len(rows) < 2:
        return 0
    start, end = rows[0], rows[-1]
    return max(0, int(end.get("occupied_voxel_count") or 0) - int(start.get("occupied_voxel_count") or 0)) + max(
        0, int(end.get("free_voxel_count") or 0) - int(start.get("free_voxel_count") or 0)
    )


def recovery_label(rows: Sequence[Mapping[str, Any]], index: int) -> Tuple[str, Optional[int], Optional[float]]:
    start = rows[index].get("gps")
    if not isinstance(start, (list, tuple)):
        return "unknown", None, None
    for later in range(index + 8, len(rows)):
        moved = distance(start, rows[later].get("gps"))
        if moved >= 0.60:
            latency = later - index
            label = "self_recovered_quick" if latency <= 32 else "self_recovered_delayed"
            return label, int(rows[later]["step_id"]), moved
    return "persistent_episode", None, None


def route_revisit(
    rows: Sequence[Mapping[str, Any]], index: int, *, radius_m: float = 0.35,
    min_path_m: float = 0.75, min_gap: int = 12,
    cumulative_path_m: Optional[Sequence[float]] = None,
) -> Optional[Dict[str, Any]]:
    current = rows[index]
    for prior in range(index - int(min_gap), -1, -1):
        route_m = (
            float(cumulative_path_m[index]) - float(cumulative_path_m[prior])
            if cumulative_path_m is not None
            else path_length(rows[prior:index + 1])
        )
        revisit_m = distance(current.get("gps"), rows[prior].get("gps"))
        if route_m >= float(min_path_m) and revisit_m <= float(radius_m):
            return {
                "prior_step": int(rows[prior]["step_id"]),
                "route_path_m": route_m,
                "revisit_distance_m": revisit_m,
            }
    return None


def semantic_context(events: Sequence[Mapping[str, Any]], step: int) -> Dict[str, Any]:
    past = [event for event in events if event.get("valid") and int(event.get("step_id", -1)) <= step]
    recent = past[-4:]
    labels = [set((event.get("class_surface_counts") or {}).keys()) for event in recent]
    cells = [set(event.get("spatial_semantic_cells") or []) for event in recent]
    union = sorted(set().union(*labels)) if labels else []
    previous_cells = set().union(*cells[:-1]) if len(cells) >= 2 else set()
    latest_cells = cells[-1] if cells else set()
    overlap = latest_cells & previous_cells
    novelty = (len(latest_cells - previous_cells) / max(1, len(latest_cells)))
    recurrence = (len(overlap) / max(1, len(latest_cells)))
    spatial_stagnation = bool(
        len(cells) >= 3 and len(latest_cells) >= 2
        and recurrence >= 0.60 and novelty <= 0.25
    )
    return {
        "available": bool(past),
        "recent_query_count": len(recent),
        "recent_labels": union,
        "strong_spatial_cell_count": len(latest_cells),
        "repeated_spatial_cell_count": len(overlap),
        "spatial_recurrence": recurrence,
        "semantic_novelty": novelty,
        "spatial_stagnation": spatial_stagnation,
        "last_query_step": int(past[-1]["step_id"]) if past else None,
    }


def canonical_observations(
    observations: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep the latest ledger observation for each evaluator step."""
    by_step: Dict[int, Dict[str, Any]] = {}
    for observation in observations:
        row = compact_observation(observation)
        by_step[int(row["step_id"])] = row
    return [by_step[step] for step in sorted(by_step)]


def _fit_panel(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    fitted = image.convert("RGB").copy()
    fitted.thumbnail(size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, (245, 245, 245))
    panel.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return panel


def render_event_evidence(
    episode_dir: Path, observations: Sequence[Mapping[str, Any]],
    semantic: Sequence[Mapping[str, Any]], event: Mapping[str, Any],
    output_path: Path, *, status: str,
) -> None:
    rows = canonical_observations(observations)
    if not rows:
        return
    step = int(event.get("step_id", rows[len(rows) // 2]["step_id"]))
    signal = int(event.get("signal_step", step))
    frame_steps = [max(0, signal - 8), signal, step, min(int(rows[-1]["step_id"]), step + 8)]
    by_step = {int(row["step_id"]): row for row in rows}
    selected = [min(rows, key=lambda row: abs(int(row["step_id"]) - target)) for target in frame_steps]
    panel_size = (320, 240)
    canvas = Image.new("RGB", (1280, 760), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title = (
        f"{episode_dir.name} | {status} | {event.get('event_family', 'hard_negative')} "
        f"signal={signal} detect={step} delay={step - signal}"
    )
    draw.text((12, 10), title, fill=(0, 0, 0))
    draw.text((12, 30), "evidence=" + "+".join(event.get("evidence") or []), fill=(0, 0, 0))
    for column, (target, row) in enumerate(zip(frame_steps, selected)):
        relative = row.get("rgb_path")
        path = episode_dir / str(relative) if relative else None
        if path is not None and path.is_file():
            canvas.paste(_fit_panel(Image.open(path), panel_size), (column * 320, 58))
        draw.text(
            (column * 320 + 8, 302),
            f"target={target} actual={row['step_id']} coll={row.get('collision_count')} dgoal={row.get('distance_to_goal')}",
            fill=(0, 0, 0),
        )

    chart_top, chart_height = 350, 360
    chart_width = 410
    chart_left = 20
    gps = [row.get("gps") for row in rows if isinstance(row.get("gps"), (list, tuple))]
    if gps:
        xs = [float(point[0]) for point in gps]
        ys = [float(point[1]) for point in gps]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        scale = min((chart_width - 30) / max(0.1, x_max - x_min), (chart_height - 50) / max(0.1, y_max - y_min))
        trajectory = [
            (
                chart_left + 15 + int((float(point[0]) - x_min) * scale),
                chart_top + chart_height - 25 - int((float(point[1]) - y_min) * scale),
            )
            for point in gps
        ]
        if len(trajectory) >= 2:
            draw.line(trajectory, fill=(70, 110, 180), width=3)
        for target, color in ((signal, (230, 145, 30)), (step, (210, 45, 45))):
            row = min(rows, key=lambda item: abs(int(item["step_id"]) - target))
            point = row.get("gps")
            if isinstance(point, (list, tuple)):
                px = chart_left + 15 + int((float(point[0]) - x_min) * scale)
                py = chart_top + chart_height - 25 - int((float(point[1]) - y_min) * scale)
                draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=color)
        draw.text((chart_left, chart_top), "Authoritative executed trajectory", fill=(0, 0, 0))

    plot_left, plot_width = 455, 390
    window = [row for row in rows if signal - 24 <= int(row["step_id"]) <= step + 32]
    if window:
        max_collision = max(float(row.get("collision_count") or 0) for row in window)
        distances = [float(row.get("distance_to_goal") or 0) for row in window]
        max_distance = max(distances) if distances else 1.0
        for offset, row in enumerate(window):
            x = plot_left + int(offset * (plot_width - 1) / max(1, len(window) - 1))
            collision_y = chart_top + 170 - int(150 * float(row.get("collision_count") or 0) / max(1.0, max_collision))
            distance_y = chart_top + 350 - int(150 * float(row.get("distance_to_goal") or 0) / max(0.1, max_distance))
            if offset:
                draw.line((prior_x, prior_collision_y, x, collision_y), fill=(190, 40, 40), width=2)
                draw.line((prior_x, prior_distance_y, x, distance_y), fill=(30, 130, 70), width=2)
            prior_x, prior_collision_y, prior_distance_y = x, collision_y, distance_y
        draw.text((plot_left, chart_top), "collision count (red)", fill=(190, 40, 40))
        draw.text((plot_left, chart_top + 180), "distance to goal (green)", fill=(30, 130, 70))

    recent_semantic = [item for item in semantic if item.get("valid") and int(item.get("step_id", -1)) <= step]
    semantic_root = None
    run_dir = episode_dir.parent.parent
    prefix = episode_dir.name.rsplit("_r", 1)[0]
    semantic_candidates = sorted((run_dir / "online_lseg_shadow").glob(f"{prefix}_r*"))
    if semantic_candidates:
        semantic_root = semantic_candidates[-1]
    if recent_semantic and semantic_root is not None:
        sem = recent_semantic[-1]
        overlay = sem.get("overlay_path")
        overlay_path = semantic_root / str(overlay) if overlay else None
        if overlay_path is not None and overlay_path.is_file():
            canvas.paste(_fit_panel(Image.open(overlay_path), (390, 292)), (875, 370))
            draw.text((875, 350), f"latest causal Q-frame LSeg step={sem.get('step_id')}", fill=(0, 0, 0))
    draw.text((875, 680), "semantic is confirmation-only; unknown remains unknown", fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _event(
    rows: Sequence[Mapping[str, Any]], index: int, *, family: str,
    evidence: Sequence[str], semantic: Sequence[Mapping[str, Any]],
    signal_step: Optional[int] = None, extra_window: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    row = rows[index]
    recent8 = rows[max(0, index - 7):index + 1]
    recent12 = rows[max(0, index - 11):index + 1]
    collision_burst = sum(float(item.get("collision_delta") or 0.0) for item in recent8)
    forward_count = sum(
        int(item.get("previous_action") == 1 and item.get("previous_action_applied") is not False)
        for item in recent8
    )
    recovery, recovery_step, recovery_m = recovery_label(rows, index)
    sem = semantic_context(semantic, int(row["step_id"]))
    onset = int(row["step_id"] if signal_step is None else signal_step)
    return {
        "step_id": int(row["step_id"]),
        "signal_step": onset,
        "confirmation_delay_steps": int(row["step_id"]) - onset,
        "observation_index": int(row["record_index"]),
        "event_family": family,
        "evidence": list(evidence),
        "window": {
            "collision_delta_8": collision_burst,
            "forward_count_8": forward_count,
            "displacement_8_m": window_displacement(recent8),
            "path_length_8_m": path_length(recent8),
            "occ_new_voxels_12": sum(
                int(item.get("occupied_added") or 0) + int(item.get("free_added") or 0)
                for item in recent12
            ),
            "occ_unique_growth_12": unique_occ_growth(recent12),
            **dict(extra_window or {}),
        },
        "semantic_confirmation": {
            **sem,
            "supports_existing_suspicion": bool(sem.get("spatial_stagnation")),
        },
        "recoverability_proxy": recovery,
        "recovery_step": recovery_step,
        "recovery_displacement_m": recovery_m,
        "rgb_path": row.get("rgb_path"),
        "position": row.get("gps"),
    }


def merge_geometry_intervals(
    events: Sequence[Mapping[str, Any]], *, max_gap_steps: int = 4,
    max_region_distance_m: float = 0.50,
) -> List[Dict[str, Any]]:
    """Merge continuous G1 evidence into causal event intervals."""
    merged: List[Dict[str, Any]] = []
    for source in sorted(events, key=lambda item: int(item["step_id"])):
        event = dict(source)
        event["end_step"] = int(event["step_id"])
        event["support_count"] = 1
        event["semantic_confirmation_step"] = (
            int(event["step_id"])
            if event["semantic_confirmation"].get("supports_existing_suspicion")
            else None
        )
        if not merged:
            merged.append(event)
            continue
        current = merged[-1]
        gap = int(event["step_id"]) - int(current["end_step"])
        same_region = distance(event.get("position"), current.get("position")) <= float(
            max_region_distance_m
        )
        if gap > int(max_gap_steps) or not same_region:
            merged.append(event)
            continue
        current["end_step"] = int(event["step_id"])
        current["support_count"] = int(current["support_count"]) + 1
        current["evidence"] = sorted(set(current["evidence"]) | set(event["evidence"]))
        if (
            current.get("semantic_confirmation_step") is None
            and event["semantic_confirmation"].get("supports_existing_suspicion")
        ):
            current["semantic_confirmation_step"] = int(event["step_id"])
            current["semantic_confirmation"] = event["semantic_confirmation"]
    for event in merged:
        event["duration_steps"] = int(event["end_step"]) - int(event["step_id"]) + 1
        event["semantic_confirmation"]["supports_existing_suspicion"] = bool(
            event.get("semantic_confirmation_step") is not None
        )
    return merged


def mine_events(
    observations: Sequence[Mapping[str, Any]],
    loops: Sequence[Mapping[str, Any]],
    semantic: Sequence[Mapping[str, Any]],
    *, route_radius_m: float = 0.35, route_min_path_m: float = 0.75,
    route_confirm_min_steps: int = 8, route_confirm_max_steps: int = 16,
    route_max_displacement_m: float = 0.25, route_max_unique_occ_growth: int = 512,
) -> Dict[str, List[Dict[str, Any]]]:
    rows = canonical_observations(observations)
    policy_loops: List[Dict[str, Any]] = []
    geometry_samples: List[Dict[str, Any]] = []
    raw_revisits: List[Dict[str, Any]] = []
    confirmed_revisits: List[Dict[str, Any]] = []
    loop_steps = {int(row.get("step_id", -1)): row for row in loops}
    cumulative_path_m = cumulative_path_length(rows)
    last_raw_revisit_step = -1000
    pending_revisit: Optional[Dict[str, Any]] = None
    active_revisit_region: Optional[Sequence[float]] = None
    for index, row in enumerate(rows):
        step = int(row["step_id"] or 0)
        if active_revisit_region is not None and distance(
            row.get("gps"), active_revisit_region
        ) > 0.60:
            active_revisit_region = None
        recent8 = rows[max(0, index - 7):index + 1]
        recent12 = rows[max(0, index - 11):index + 1]
        collision_burst = sum(float(item.get("collision_delta") or 0.0) for item in recent8)
        forward_count = sum(
            int(item.get("previous_action") == 1 and item.get("previous_action_applied") is not False)
            for item in recent8
        )
        displacement = window_displacement(recent8)
        occ_growth = sum(
            int(item.get("occupied_added") or 0) + int(item.get("free_added") or 0)
            for item in recent12
        )
        geometry_evidence: List[str] = []
        if step in loop_steps:
            policy_loops.append(_event(
                rows, index, family="G2_policy_loop",
                evidence=["strict_s2_turn_loop"], semantic=semantic,
            ))
        if collision_burst >= 2 and displacement <= 0.25:
            geometry_evidence.append("collision_burst_low_displacement")
        if forward_count >= 3 and displacement <= 0.15:
            geometry_evidence.append("commanded_forward_not_realized")
        if geometry_evidence:
            geometry_samples.append(_event(
                rows, index, family="G1_geometry_execution",
                evidence=geometry_evidence, semantic=semantic,
            ))
        revisit = route_revisit(
            rows, index, radius_m=route_radius_m, min_path_m=route_min_path_m,
            cumulative_path_m=cumulative_path_m,
        )
        if revisit is not None:
            if step - last_raw_revisit_step >= 8:
                raw_revisits.append(_event(
                    rows, index, family="G3_route_topology",
                    evidence=["route_revisit_signal"], semantic=semantic,
                    extra_window=revisit,
                ))
                last_raw_revisit_step = step
            if pending_revisit is None and active_revisit_region is None:
                pending_revisit = {"index": index, "step": step, "revisit": revisit}

        if pending_revisit is not None:
            onset_index = int(pending_revisit["index"])
            age = index - onset_index
            confirm_rows = rows[onset_index:index + 1]
            confirm_displacement = window_displacement(confirm_rows)
            confirm_growth = unique_occ_growth(confirm_rows)
            left_region = max(
                (distance(confirm_rows[0].get("gps"), item.get("gps")) for item in confirm_rows),
                default=0.0,
            ) > 0.60
            if age >= int(route_confirm_min_steps):
                sem = semantic_context(semantic, step)
                low_motion = confirm_displacement <= float(route_max_displacement_m)
                low_growth = confirm_growth <= int(route_max_unique_occ_growth)
                if low_motion and low_growth:
                    confirmed_revisits.append(_event(
                        rows, index, family="G3_route_topology",
                        evidence=["route_revisit_confirmed_low_progress"],
                        semantic=semantic, signal_step=int(pending_revisit["step"]),
                        extra_window={
                            **dict(pending_revisit["revisit"]),
                            "confirmation_window_steps": age,
                            "confirmation_displacement_m": confirm_displacement,
                            "confirmation_unique_occ_growth": confirm_growth,
                            "semantic_stagnation_at_confirmation": bool(
                                sem.get("spatial_stagnation")
                            ),
                        },
                    ))
                    active_revisit_region = row.get("gps")
                    pending_revisit = None
                elif age >= int(route_confirm_max_steps) or left_region:
                    pending_revisit = None
            elif left_region:
                pending_revisit = None

    geometry_intervals = merge_geometry_intervals(geometry_samples)
    d0 = policy_loops
    d1 = policy_loops + geometry_intervals
    d2 = d1 + confirmed_revisits
    return {
        "D0": d0,
        "D1": d1,
        "D2_raw_revisit": d1 + raw_revisits,
        "D2": d2,
        "D3Q_confirmed": [
            event for event in d2
            if event["semantic_confirmation"]["supports_existing_suspicion"]
        ],
    }


def audit_episode(episode_dir: Path, loops: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
    errors: List[str] = []
    meta = json.loads((episode_dir / "episode_meta.json").read_text(encoding="utf-8"))
    summary = json.loads((episode_dir / "summary.json").read_text(encoding="utf-8"))
    observations = jsonl(episode_dir / "observations.jsonl")
    queries = jsonl(episode_dir / "queries.jsonl")
    actions = jsonl(episode_dir / "actions.jsonl")
    keys = {row.get("observation_key") for row in observations}
    prior_collision = 0.0
    for row in observations:
        key = row.get("observation_key")
        pose = row.get("pose") or {}
        audit = row.get("audit_metrics") or {}
        for field in ("distance_to_goal", "collision_count", "collision_delta"):
            if audit.get(field) is None:
                errors.append(f"missing_{field}:{key}")
        current_collision = float(audit.get("collision_count", prior_collision) or 0.0)
        expected_delta = max(0.0, current_collision - prior_collision)
        if abs(float(audit.get("collision_delta", expected_delta) or 0.0) - expected_delta) > 1e-6:
            errors.append(f"collision_delta_mismatch:{key}")
        if current_collision < prior_collision:
            errors.append(f"collision_not_monotonic:{key}")
        prior_collision = current_collision
        if pose.get("gps") is None or pose.get("stage23_gt_camera_pose_map") is None:
            errors.append(f"pose_missing:{key}")
        for field, rgb in (("rgb_path", True), ("depth_path", False)):
            relative = row.get(field)
            path = episode_dir / str(relative) if relative else None
            if path is None or not path.is_file():
                errors.append(f"{field}_missing:{key}")
            elif sha256_array(path, rgb=rgb) != row.get("rgb_sha256" if rgb else "depth_sha256"):
                errors.append(f"{field}_hash_mismatch:{key}")
    if any(row.get("observation_key") not in keys for row in queries + actions):
        errors.append("invalid_observation_reference")
    prior_action_collision = 0.0
    for row in actions:
        audit = row.get("audit_metrics") or {}
        action_index = row.get("action_index")
        for field in ("distance_to_goal", "collision_count", "collision_delta"):
            if audit.get(field) is None:
                errors.append(f"action_missing_{field}:{action_index}")
        current_collision = float(
            audit.get("collision_count", prior_action_collision) or 0.0
        )
        expected_delta = max(0.0, current_collision - prior_action_collision)
        if abs(float(audit.get("collision_delta", expected_delta) or 0.0) - expected_delta) > 1e-6:
            errors.append(f"action_collision_delta_mismatch:{action_index}")
        if current_collision < prior_action_collision:
            errors.append(f"action_collision_not_monotonic:{action_index}")
        prior_action_collision = current_collision
    if not (meta.get("semantic_scene_gt") or {}).get("available"):
        errors.append("semantic_scene_gt_missing")
    final_collision = (summary.get("final_metrics") or {}).get("collision_count")
    if final_collision is None or abs(float(final_collision) - prior_action_collision) > 1e-6:
        errors.append("final_collision_mismatch")
    return {
        "scene_id": meta.get("scene_id"),
        "episode_id": meta.get("episode_id"),
        "episode_eval_seed": meta.get("episode_eval_seed"),
        "observation_count": len(observations),
        "query_count": len(queries),
        "action_count": len(actions),
        "loop_count": len(loops),
        "final_metrics": summary.get("final_metrics") or {},
        "ledger_dir": str(episode_dir),
    }, errors


def analyze(
    run_root: Path, output: Path, require_all: bool,
    *, detector_options: Optional[Mapping[str, Any]] = None,
    episode_manifest: Optional[Path] = None,
) -> Dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    evidence_dir = output / "event_evidence"
    options = dict(detector_options or {})
    progress = progress_by_episode(run_root)
    loops = loop_events_by_episode(run_root)
    contract_errors = []
    episode_reports = []
    all_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    annotation = []
    expected_manifest: Dict[str, Dict[str, Any]] = {}
    if episode_manifest is not None:
        for row in json.loads(episode_manifest.read_text(encoding="utf-8")):
            expected_manifest[episode_key(row)] = row
    observed_keys = set()
    for episode_dir in discover_episodes(run_root):
        meta = json.loads((episode_dir / "episode_meta.json").read_text(encoding="utf-8"))
        key = episode_key(meta)
        observed_keys.add(key)
        report, errors = audit_episode(episode_dir, loops.get(key, []))
        expected = expected_manifest.get(key)
        actual_seed, seed_source = resolve_episode_eval_seed(
            meta, progress.get(key, {})
        )
        report["episode_eval_seed"] = actual_seed
        report["episode_eval_seed_source"] = seed_source
        report["expected_episode_eval_seed"] = (
            None if expected is None else expected.get("episode_eval_seed")
        )
        if expected_manifest and expected is None:
            errors.append("episode_not_in_manifest")
        if expected is not None:
            try:
                seed_matches = (
                    actual_seed is not None
                    and int(actual_seed) == int(expected["episode_eval_seed"])
                )
            except (TypeError, ValueError):
                seed_matches = False
            if not seed_matches:
                errors.append("episode_eval_seed_mismatch")
        report["audit_role"] = None if expected is None else expected.get("audit_role")
        episode_reports.append(report)
        contract_errors.extend(f"{key}:{error}" for error in errors)
        observations = jsonl(episode_dir / "observations.jsonl")
        semantic = lseg_events(episode_dir)
        variants = mine_events(observations, loops.get(key, []), semantic, **options)
        outcome = progress.get(key, {})
        for variant, events in variants.items():
            for event in events:
                event.update({
                    "scene_id": meta.get("scene_id"),
                    "episode_id": meta.get("episode_id"),
                    "outcome": {
                        "success": outcome.get("success"),
                        "spl": outcome.get("spl"),
                        "steps": outcome.get("steps"),
                    },
                })
                all_events[variant].append(event)
                if variant == "D2":
                    evidence_name = (
                        f"{meta.get('scene_id')}_{meta.get('episode_id')}_"
                        f"step{int(event['step_id']):04d}_{event['event_family']}.png"
                    )
                    render_event_evidence(
                        episode_dir, observations, semantic, event,
                        evidence_dir / evidence_name, status="D2_candidate",
                    )
                    event["evidence_image"] = str(
                        (Path("event_evidence") / evidence_name)
                    )
                    annotation.append({
                        **event,
                        "annotation": {
                            "state": None,
                            "type": None,
                            "onset_step": None,
                            "end_step": None,
                            "recoverability": None,
                            "failure_link": None,
                            "intervention_likely_needed": None,
                            "confidence": None,
                            "notes": "",
                        },
                    })
        if not variants["D2"]:
            negative = {
                "step_id": len(canonical_observations(observations)) // 2,
                "signal_step": len(canonical_observations(observations)) // 2,
                "event_family": "hard_negative_no_D2", "evidence": [],
            }
            evidence_name = f"{meta.get('scene_id')}_{meta.get('episode_id')}_hard_negative.png"
            render_event_evidence(
                episode_dir, observations, semantic, negative,
                evidence_dir / evidence_name, status="hard_negative_no_D2",
            )
    missing_manifest = sorted(set(expected_manifest) - observed_keys)
    contract_errors.extend(f"{key}:manifest_episode_missing" for key in missing_manifest)
    detector_summary = {}
    for variant in ("D0", "D1", "D2_raw_revisit", "D2", "D3Q_confirmed"):
        events = all_events.get(variant, [])
        detector_summary[variant] = {
            "event_count": len(events),
            "episode_count": len({episode_key(event) for event in events}),
            "event_family_counts": dict(Counter(event["event_family"] for event in events)),
            "self_recovered_count": sum(event["recoverability_proxy"].startswith("self_recovered") for event in events),
            "quick_self_recovered_count": sum(event["recoverability_proxy"] == "self_recovered_quick" for event in events),
            "delayed_self_recovered_count": sum(event["recoverability_proxy"] == "self_recovered_delayed" for event in events),
            "persistent_proxy_count": sum(event["recoverability_proxy"].startswith("persistent") for event in events),
        }
    report = {
        "task": "stage25_gt_detector_contract_and_candidate_mining",
        "contract_passed": not contract_errors and bool(episode_reports),
        "event_gt_status": "objective_proxy_pending_manual_interval_annotation",
        "outcome_is_event_gt": False,
        "future_used_by_detector": False,
        "future_used_for_recoverability_label_only": True,
        "detector_options": options,
        "episode_manifest": None if episode_manifest is None else str(episode_manifest),
        "manifest_expected_count": len(expected_manifest),
        "manifest_verified_count": len(expected_manifest) - len(missing_manifest),
        "episode_count": len(episode_reports),
        "episodes": episode_reports,
        "detectors": detector_summary,
        "errors": contract_errors,
    }
    (output / "stage25_contract_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "stage25_event_candidates.json").write_text(json.dumps(all_events, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "stage25_annotation_manifest.json").write_text(json.dumps(annotation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if require_all and not report["contract_passed"]:
        raise SystemExit(1)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--episode-manifest", type=Path)
    parser.add_argument("--route-radius-m", type=float, default=0.35)
    parser.add_argument("--route-min-path-m", type=float, default=0.75)
    parser.add_argument("--route-confirm-min-steps", type=int, default=8)
    parser.add_argument("--route-confirm-max-steps", type=int, default=16)
    parser.add_argument("--route-max-displacement-m", type=float, default=0.25)
    parser.add_argument("--route-max-unique-occ-growth", type=int, default=512)
    args = parser.parse_args()
    analyze(
        args.run_root, args.output, args.require_all, episode_manifest=args.episode_manifest,
        detector_options={
            "route_radius_m": args.route_radius_m,
            "route_min_path_m": args.route_min_path_m,
            "route_confirm_min_steps": args.route_confirm_min_steps,
            "route_confirm_max_steps": args.route_confirm_max_steps,
            "route_max_displacement_m": args.route_max_displacement_m,
            "route_max_unique_occ_growth": args.route_max_unique_occ_growth,
        },
    )


if __name__ == "__main__":
    main()
