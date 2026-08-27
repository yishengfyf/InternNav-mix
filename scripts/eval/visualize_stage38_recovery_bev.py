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
    free = _channel_points(channels, ("known_free", "free", "known_free_cells"))
    occupied = _channel_points(channels, ("occupied", "occ", "occupied_cells"))
    unknown = _channel_points(channels, ("unknown", "unknown_cells", "unobserved"))
    route = _points(capture.get("path_cells"))
    pose = _xy(capture.get("pose"))
    candidate = _xy(capture.get("candidate_grid"))
    semantic = _channel_points(channels, ("semantic", "semantic_cells", "semantic_evidence"))
    all_points = free + occupied + unknown + route + semantic
    if pose is not None:
        all_points.append(pose)
    if candidate is not None:
        all_points.append(candidate)
    if not all_points:
        all_points = [(0.0, 0.0)]
    min_r = int(min(p[0] for p in all_points))
    max_r = int(max(p[0] for p in all_points))
    min_c = int(min(p[1] for p in all_points))
    max_c = int(max(p[1] for p in all_points))
    margin = 3
    width = max(96, (max_c - min_c + 1 + 2 * margin) * max(1, int(scale)))
    height = max(96, (max_r - min_r + 1 + 2 * margin) * max(1, int(scale)))
    image = Image.new("RGB", (width, height), COLORS["background"])
    draw = ImageDraw.Draw(image)

    def pixel(point: tuple[float, float]) -> tuple[int, int]:
        row, col = point
        return ((int(round(col)) - min_c + margin) * scale + scale // 2,
                (int(round(row)) - min_r + margin) * scale + scale // 2)

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
    if pose is not None:
        x, y = pixel(pose)
        draw.ellipse((x - scale, y - scale, x + scale, y + scale), fill=COLORS["pose"])
    if candidate is not None:
        x, y = pixel(candidate)
        draw.ellipse((x - scale, y - scale, x + scale, y + scale), outline=COLORS["candidate"], width=max(2, scale // 3))
    # Semantic evidence is diagnostic only: outline, never a filled safety cell.
    for point in semantic:
        x, y = pixel(point)
        draw.rectangle((x - scale // 2, y - scale // 2, x + scale // 2, y + scale // 2), outline=COLORS["semantic"], width=max(1, scale // 4))
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
        "unknown_is_free": False,
        "semantic_can_override_safety": False,
        "shadow_only": True,
        "action_applied": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    anchors = {str(row.get("anchor_id")): row for row in report.get("anchors", [])}
    rendered = []
    for digest in report.get("digests", []):
        anchor = anchors.get(str(digest.get("anchor_id")))
        if anchor is None:
            continue
        safe_name = str(digest.get("anchor_id") or "anchor").replace("/", "_").replace(":", "_")
        rendered.append(render_recovery_bev(anchor, digest, args.output_dir / f"{safe_name}.png"))
    metadata = {
        "task": "stage38_recovery_bev_visualization",
        "schema_version": "stage38_recovery_bev_visualization_v1",
        "source_report": str(args.report),
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
