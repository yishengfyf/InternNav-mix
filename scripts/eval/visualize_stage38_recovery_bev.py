#!/usr/bin/env python3
"""Render a read-only Stage38 RecoveryBEV audit report.

The renderer is deliberately independent of navigation state.  SparseOcc
channels are painted first; route/pose/candidate overlays are drawn on top,
and semantic evidence is an outline only.  In particular, unknown cells are
never folded into known-free cells.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image, ImageDraw


COLORS = {
    "background": (26, 29, 33),
    "free": (52, 150, 91),
    "occupied": (210, 72, 62),
    "unknown": (184, 170, 72),
    "route": (73, 142, 226),
    "pose": (245, 245, 245),
    "candidate": (255, 155, 54),
    "semantic": (221, 90, 210),
    "depth_surface": (255, 92, 205),
    "depth_lookahead": (0, 220, 220),
    "hfov": (245, 205, 65),
}


def _xy(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        value = value.get("grid", value.get("point", value.get("xy")))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _points(value: Any) -> list[tuple[float, float]]:
    if isinstance(value, Mapping):
        value = value.get("cells", value.get("points", value.get("path_cells", [])))
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        return []
    result = []
    for item in value:
        point = _xy(item)
        if point is not None:
            result.append(point)
    return result


def _channel_points(channels: Mapping[str, Any], names: tuple[str, ...]) -> list[tuple[float, float]]:
    for name in names:
        if name in channels:
            points = _points(channels[name])
            if points:
                return points
    return []


def render_recovery_bev(
    anchor: Mapping[str, Any], digest: Mapping[str, Any], output: Path, *, scale: int = 12
) -> dict[str, Any]:
    """Render one anchor/digest pair and return auditable rendering metadata."""
    capture = dict(anchor.get("capture") or {})
    channels = dict(digest.get("channels") or {})
    required = ("known_free", "occupied", "unknown")
    if not all(isinstance(channels.get(name), list) for name in required):
        raise ValueError("RecoveryBEV requires explicit known_free/occupied/unknown cell lists")
    free = _channel_points(channels, ("known_free", "free", "known_free_cells"))
    occupied = _channel_points(channels, ("occupied", "occ", "occupied_cells"))
    unknown = _channel_points(channels, ("unknown", "unknown_cells", "unobserved"))
    route = _points(capture.get("path_cells"))
    candidate = _xy(capture.get("candidate_grid"))
    semantic = _channel_points(channels, ("semantic", "semantic_cells", "semantic_evidence"))
    current_pose = dict(digest.get("current_pose") or {})
    pose = _xy(capture.get("pose")) or _xy(current_pose.get("grid"))
    center_value = digest.get("center_grid") or current_pose.get("grid") or [0, 0]
    center = (float(center_value[0]), float(center_value[1]))
    yaw = float(current_pose.get("yaw", digest.get("pose_yaw_rad", 0.0)) or 0.0)
    depth_records = [dict(item) for item in digest.get("depth_endpoints") or [] if isinstance(item, Mapping)]
    candidate_path = _points(digest.get("candidate_path") or [])
    footprint = _points(digest.get("footprint_corridor") or [])
    semantic_nodes = [dict(item) for item in channels.get("semantic_nodes") or [] if isinstance(item, Mapping)]
    semantic_grid_nodes = [node.get("grid") for node in semantic_nodes if _xy(node.get("grid")) is not None]
    all_points = free + occupied + unknown + route + semantic + candidate_path + footprint
    all_points += [p for p in (_xy(item.get("surface_grid")) for item in depth_records) if p is not None]
    all_points += [p for p in (_xy(item.get("lookahead_grid")) for item in depth_records) if p is not None]
    all_points += [p for p in (_xy(node) for node in semantic_grid_nodes) if p is not None]
    if pose is not None:
        all_points.append(pose)
    if candidate is not None:
        all_points.append(candidate)
    if not all_points:
        all_points = [(0.0, 0.0)]
    def local(point: tuple[float, float]) -> tuple[float, float]:
        """Map grid row/col into robot-forward-up coordinates."""
        row, col = float(point[0]) - center[0], float(point[1]) - center[1]
        world_x, world_y = -row, -col
        forward = world_x * math.cos(yaw) + world_y * math.sin(yaw)
        lateral = -world_x * math.sin(yaw) + world_y * math.cos(yaw)
        return forward, lateral

    local_points = [local(point) for point in all_points]
    min_f = min(p[0] for p in local_points); max_f = max(p[0] for p in local_points)
    min_l = min(p[1] for p in local_points); max_l = max(p[1] for p in local_points)
    margin = 3
    width = max(96, int(math.ceil(max_l - min_l + 1 + 2 * margin)) * max(1, int(scale)))
    height = max(96, int(math.ceil(max_f - min_f + 1 + 2 * margin)) * max(1, int(scale)))
    image = Image.new("RGB", (width, height), COLORS["background"])
    draw = ImageDraw.Draw(image)
    def pixel(point: tuple[float, float]) -> tuple[int, int]:
        forward, lateral = local(point)
        return ((int(round(lateral - min_l + margin)) * scale + scale // 2,
                 int(round(max_f - forward + margin)) * scale + scale // 2))

    def paint(points: list[tuple[float, float]], color: tuple[int, int, int], radius: int = 0) -> None:
        for point in points:
            x, y = pixel(point)
            if radius:
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
            else:
                draw.rectangle((x - scale // 2, y - scale // 2, x + scale // 2, y + scale // 2), fill=color)

    # Safety channels are painted in a fixed order.  Unknown is explicit and
    # never receives the free color, even when a point occurs in both lists.
    paint(free, COLORS["free"])
    paint(occupied, COLORS["occupied"])
    paint(unknown, COLORS["unknown"])
    if len(route) >= 2:
        draw.line([pixel(p) for p in route], fill=COLORS["route"], width=max(1, scale // 3))
    if len(candidate_path) >= 2:
        draw.line([pixel(p) for p in candidate_path], fill=COLORS["candidate"], width=max(2, scale // 2))
    if len(footprint) >= 2:
        draw.line([pixel(p) for p in footprint], fill=COLORS["hfov"], width=max(1, scale // 3))
    # Draw the camera frustum in the normalized robot-forward frame.  It is a
    # diagnostic field of view, not a safety mask.
    hfov = digest.get("hfov_deg")
    if hfov is not None:
        radius = max(3.0, float(max(max_f - min_f, max_l - min_l)) * 0.45)
        half = math.radians(float(hfov)) / 2.0
        rays = []
        for angle in (-half, half):
            forward, lateral = radius * math.cos(angle), radius * math.sin(angle)
            # Convert local forward/lateral back to a grid point for pixel().
            wx = forward * math.cos(yaw) - lateral * math.sin(yaw)
            wy = forward * math.sin(yaw) + lateral * math.cos(yaw)
            rays.append((center[0] - wx, center[1] - wy))
        draw.line([pixel(center), pixel(rays[0])], fill=COLORS["hfov"], width=max(1, scale // 4))
        draw.line([pixel(center), pixel(rays[1])], fill=COLORS["hfov"], width=max(1, scale // 4))
    if pose is not None:
        x, y = pixel(pose)
        draw.ellipse((x - scale, y - scale, x + scale, y + scale), fill=COLORS["pose"])
        draw.line([(x, y), (x, y - 2 * scale)], fill=COLORS["pose"], width=max(2, scale // 3))
    if candidate is not None:
        x, y = pixel(candidate)
        draw.ellipse((x - scale, y - scale, x + scale, y + scale), outline=COLORS["candidate"], width=max(2, scale // 3))
    # Semantic evidence is diagnostic only: outline, never a filled safety cell.
    for point in semantic:
        x, y = pixel(point)
        draw.rectangle((x - scale // 2, y - scale // 2, x + scale // 2, y + scale // 2), outline=COLORS["semantic"], width=max(1, scale // 4))
    for node in semantic_nodes:
        grid = _xy(node.get("grid"))
        if grid is None:
            continue
        x, y = pixel(grid)
        draw.rectangle((x - scale, y - scale, x + scale, y + scale), outline=COLORS["semantic"], width=max(1, scale // 3))
        draw.text((x + scale + 1, y - scale), str(node.get("label") or "other"), fill=COLORS["semantic"])
    for item in depth_records:
        for key, color in (("surface_grid", COLORS["depth_surface"]), ("lookahead_grid", COLORS["depth_lookahead"])):
            point = _xy(item.get(key))
            if point is None:
                continue
            x, y = pixel(point)
            draw.ellipse((x - scale // 2, y - scale // 2, x + scale // 2, y + scale // 2), outline=color, width=max(1, scale // 3))
    # Keep a compact legend in the image itself so RGB/BEV comparisons do not
    # depend on a separate README.  Labels describe diagnostics only.
    legend_items = [
        ("free", COLORS["free"]), ("occupied", COLORS["occupied"]),
        ("unknown", COLORS["unknown"]), ("route", COLORS["route"]),
        ("semantic", COLORS["semantic"]), ("depth surface", COLORS["depth_surface"]),
        ("lookahead", COLORS["depth_lookahead"]), ("HFOV/path", COLORS["hfov"]),
    ]
    legend_x, legend_y = 6, 6
    for index, (label, color) in enumerate(legend_items):
        x0 = legend_x + (index % 2) * 108
        y0 = legend_y + (index // 2) * 14
        draw.rectangle((x0, y0, x0 + 8, y0 + 8), fill=color)
        draw.text((x0 + 12, y0 - 2), label, fill=(235, 235, 235))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")
    return {
        "png": str(output),
        "anchor_id": anchor.get("anchor_id"),
        "unknown_cells_drawn": len(unknown),
        "known_free_cells_drawn": len(free),
        "occupied_cells_drawn": len(occupied),
        "semantic_cells_overlayed": len(semantic),
        "semantic_overlay_mode": "outline_diagnostic_only",
        "coordinate_frame": "robot_forward_up",
        "pose_yaw_rad": float(yaw),
        "hfov_deg": None if hfov is None else float(hfov),
        "depth_endpoint_count": len(depth_records),
        "semantic_node_count": len(semantic_nodes),
        "unknown_is_free": False,
        "semantic_can_override_safety": False,
        "shadow_only": True,
        "action_applied": False,
    }


def render_stage27_event(event: Mapping[str, Any], output: Path, *, scale: int = 6) -> dict[str, Any]:
    spatial = dict(event.get("recovery_bev_spatial") or {})
    channels = dict(spatial.get("channels") or {})
    if not all(isinstance(channels.get(name), list) for name in ("known_free", "occupied", "unknown")):
        raise ValueError("Stage27 event lacks explicit RecoveryBEV spatial channels")
    final_pool = list(event.get("ablation", {}).get(
        "route_occ_clearance_frontier_semantic_filtered", {}
    ).get("candidates") or [])
    candidate = final_pool[0] if final_pool else {}
    event_id = "_".join(str(event.get(key)) for key in ("scene_id", "episode_id", "step_id"))
    anchor = {
        "anchor_id": event_id,
        "capture": {
            "path_cells": candidate.get("path_cells") or spatial.get("executed_route") or [],
            "pose": spatial.get("center_grid"),
            "candidate_grid": candidate.get("grid"),
        },
    }
    meta = render_recovery_bev(anchor, spatial, output, scale=scale)
    meta["event_key"] = {key: event.get(key) for key in ("scene_id", "episode_id", "step_id")}
    meta["candidate_count"] = len(final_pool)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--report", type=Path)
    source.add_argument("--events-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rendered = []
    if args.report is not None:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        anchors = {str(row.get("anchor_id")): row for row in report.get("anchors", [])}
        for digest in report.get("digests", []):
            anchor = anchors.get(str(digest.get("anchor_id")))
            if anchor is None:
                continue
            safe_name = str(digest.get("anchor_id") or "anchor").replace("/", "_").replace(":", "_")
            rendered.append(render_recovery_bev(anchor, digest, args.output_dir / f"{safe_name}.png"))
        source_report = str(args.report)
    else:
        for path in sorted(args.events_root.glob("**/stage27_m3_candidate_events.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                name = "_".join(str(event.get(key)) for key in ("scene_id", "episode_id", "step_id"))
                rendered.append(render_stage27_event(event, args.output_dir / f"{name}.png"))
        source_report = str(args.events_root)
    metadata = {
        "task": "stage38_recovery_bev_visualization",
        "schema_version": "stage38_recovery_bev_visualization_v1",
        "source_report": source_report,
        "rendered_count": len(rendered),
        "rendered": rendered,
        "unknown_is_free": False,
        "semantic_can_override_safety": False,
        "shadow_only": True,
        "action_applied": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
