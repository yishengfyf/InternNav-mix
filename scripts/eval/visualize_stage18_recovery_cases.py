"""Export lightweight visualizations for Stage18 semantic recovery cases.

The visualizer is intentionally offline and conservative:

* it reads Stage18f rows and, when available, Stage18e ``memory_events.jsonl``;
* it copies the original OccMem BEV candidate snapshot if the run still has it;
* it draws a compact top-down recovery schematic;
* it writes a tiny PLY point cloud with pose/current/backtrack anchors.

This is meant for debugging and presentation, not as a replacement for Habitat
rollouts. If ``occ_memory.validation_enable`` was enabled during collection,
the copied BEV / validation files can be placed next to these summaries.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{lineno}") from exc
            if isinstance(row, dict):
                yield row


def _episode_key(row: Mapping[str, Any]) -> str:
    return f"{row.get('scene_id')}|{row.get('episode_id')}"


def _event_key(row: Mapping[str, Any]) -> str:
    return str(row.get("event_key") or f"{_episode_key(row)}|step={row.get('step_id')}")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _find_memory_event_files(paths: Sequence[Path]) -> List[Path]:
    found: List[Path] = []
    seen = set()
    for path in paths:
        candidates: List[Path] = []
        if path.is_file() and path.name == "memory_events.jsonl":
            candidates.append(path)
        if path.is_dir():
            candidates.extend(
                [
                    path / "occ_memory" / "memory_events.jsonl",
                    path / "memory_events.jsonl",
                    path / "vlmap_safety_debug" / "run_001" / "occ_memory" / "memory_events.jsonl",
                ]
            )
            candidates.extend(path.rglob("memory_events.jsonl"))
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen or not candidate.exists():
                continue
            seen.add(resolved)
            found.append(candidate)
    return sorted(found)


def _load_events_for_keys(paths: Sequence[Path], wanted: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    wanted_set = set(wanted)
    if not wanted_set or not paths:
        return {}
    events: Dict[str, Dict[str, Any]] = {}
    for memory_file in _find_memory_event_files(paths):
        for event in _read_jsonl(memory_file):
            if event.get("event_type") != "occ_memory_query_candidates":
                continue
            key = _event_key(event)
            if key not in wanted_set or key in events:
                continue
            item = dict(event)
            item["_memory_file"] = str(memory_file)
            events[key] = item
            if len(events) >= len(wanted_set):
                return events
    return events


def _select_cases(rows: Sequence[Dict[str, Any]], *, count: int, prefer: str, distinct_episodes: bool) -> List[Dict[str, Any]]:
    def _sort_key(row: Mapping[str, Any]) -> Tuple[int, float, float]:
        success = row.get("success")
        if prefer == "failed":
            success_rank = 0 if success is False else 1
        elif prefer == "success":
            success_rank = 0 if success is True else 1
        else:
            success_rank = 0
        margin = _safe_float(row.get("advantage_margin_proxy"), -999.0)
        utility = _safe_float(row.get("backtrack_utility_proxy"), -999.0)
        return (success_rank, -margin, -utility)

    candidates = [row for row in rows if row.get("strong_recovery_proxy")]
    candidates.sort(key=_sort_key)
    selected: List[Dict[str, Any]] = []
    used_episodes = set()
    for row in candidates:
        key = _episode_key(row)
        if distinct_episodes and key in used_episodes:
            continue
        selected.append(row)
        used_episodes.add(key)
        if len(selected) >= count:
            return selected
    return selected


def _grid_to_xy(grid: Any, *, grid_size: int, cell_size: float) -> Optional[Tuple[float, float]]:
    if not isinstance(grid, (list, tuple)) or len(grid) < 2:
        return None
    row = _safe_float(grid[0])
    col = _safe_float(grid[1])
    return ((grid_size / 2.0 - row) * cell_size, (grid_size / 2.0 - col) * cell_size)


def _xy_from_candidate(
    candidate: Mapping[str, Any],
    *,
    grid_size: int,
    cell_size: float,
) -> Optional[Tuple[float, float]]:
    xy = candidate.get("xy")
    if isinstance(xy, (list, tuple)) and len(xy) >= 2:
        return (_safe_float(xy[0]), _safe_float(xy[1]))
    return _grid_to_xy(candidate.get("grid"), grid_size=grid_size, cell_size=cell_size)


def _current_candidate_with_event(row: Mapping[str, Any], event: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    current = dict(row.get("current_policy_candidate") or {})
    if event and not current.get("grid"):
        current["grid"] = event.get("current_waypoint_goal_grid")
    if event and not current.get("direction_bucket"):
        current["direction_bucket"] = event.get("current_waypoint_direction_bucket")
    if event and not current.get("goal_state"):
        current["goal_state"] = event.get("current_waypoint_goal_state")
    return current


def _draw_schematic(
    path: Path,
    *,
    row: Mapping[str, Any],
    event: Optional[Mapping[str, Any]],
    grid_size: int,
    cell_size: float,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        path.with_suffix(".txt").write_text(
            f"PIL is unavailable; schematic was not drawn: {exc}\n",
            encoding="utf-8",
        )
        return

    current = _current_candidate_with_event(row, event)
    backtrack = row.get("best_backtrack_candidate") or {}
    pose_grid = row.get("start_grid") or (event or {}).get("start_grid")
    pose_xy = _grid_to_xy(pose_grid, grid_size=grid_size, cell_size=cell_size)
    current_xy = _xy_from_candidate(current, grid_size=grid_size, cell_size=cell_size)
    backtrack_xy = _xy_from_candidate(backtrack, grid_size=grid_size, cell_size=cell_size)
    raw_candidates = (event or {}).get("candidates") or []

    points: List[Tuple[float, float]] = []
    for item in [pose_xy, current_xy, backtrack_xy]:
        if item is not None:
            points.append(item)
    for candidate in raw_candidates:
        xy = _xy_from_candidate(candidate, grid_size=grid_size, cell_size=cell_size)
        if xy is not None:
            points.append(xy)

    if not points:
        points = [(0.0, 0.0)]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    margin = max(2.5, 0.20 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0))
    xmin, xmax = min(xs) - margin, max(xs) + margin
    ymin, ymax = min(ys) - margin, max(ys) + margin
    width = height = 720

    def to_px(xy: Tuple[float, float]) -> Tuple[int, int]:
        x, y = xy
        px = int((x - xmin) / max(1e-6, xmax - xmin) * (width - 80) + 40)
        py = int((ymax - y) / max(1e-6, ymax - ymin) * (height - 100) + 60)
        return px, py

    image = Image.new("RGB", (width, height), (28, 28, 31))
    draw = ImageDraw.Draw(image)

    for gx in range(40, width - 39, 80):
        draw.line((gx, 60, gx, height - 40), fill=(46, 46, 50))
    for gy in range(60, height - 39, 80):
        draw.line((40, gy, width - 40, gy), fill=(46, 46, 50))

    color_by_type = {
        "frontier": (255, 225, 80),
        "semantic_frontier": (255, 160, 80),
        "semantic_keyframe": (240, 105, 255),
        "open_floor": (90, 220, 225),
        "resilience_backtrack": (115, 255, 125),
    }
    for candidate in raw_candidates:
        xy = _xy_from_candidate(candidate, grid_size=grid_size, cell_size=cell_size)
        if xy is None:
            continue
        x, y = to_px(xy)
        color = color_by_type.get(str(candidate.get("candidate_type")), (210, 210, 210))
        radius = 9
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(0, 0, 0), width=2)
        draw.text((x + 10, y - 8), str(candidate.get("candidate_id") or "?"), fill=(235, 235, 235))

    if pose_xy is not None:
        x, y = to_px(pose_xy)
        draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        draw.text((x + 12, y - 10), "pose", fill=(255, 255, 255))
    if current_xy is not None:
        x, y = to_px(current_xy)
        draw.rectangle((x - 11, y - 11, x + 11, y + 11), fill=(245, 95, 80), outline=(0, 0, 0), width=2)
        draw.text((x + 12, y - 10), "S2 current", fill=(255, 150, 140))
        if pose_xy is not None:
            draw.line((to_px(pose_xy), (x, y)), fill=(245, 95, 80), width=3)
    if backtrack_xy is not None:
        x, y = to_px(backtrack_xy)
        draw.ellipse((x - 13, y - 13, x + 13, y + 13), fill=(100, 245, 140), outline=(0, 0, 0), width=2)
        draw.text((x + 14, y - 10), "backtrack", fill=(150, 255, 170))
        if pose_xy is not None:
            draw.line((to_px(pose_xy), (x, y)), fill=(100, 245, 140), width=4)

    title = (
        f"{row.get('scene_id')} ep={row.get('episode_id')} step={row.get('step_id')} "
        f"success={row.get('success')} margin={_safe_float(row.get('advantage_margin_proxy')):.3f}"
    )
    draw.rectangle((0, 0, width, 40), fill=(18, 18, 20))
    draw.text((12, 12), title, fill=(245, 245, 245))
    legend = "white=pose red=S2/current green=recovery; copied BEV contains OccMem free/occ/frontier when available"
    draw.text((12, height - 28), legend, fill=(220, 220, 220))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _write_ply(
    path: Path,
    *,
    row: Mapping[str, Any],
    event: Optional[Mapping[str, Any]],
    grid_size: int,
    cell_size: float,
) -> None:
    current = _current_candidate_with_event(row, event)
    backtrack = row.get("best_backtrack_candidate") or {}
    pose_xy = _grid_to_xy(row.get("start_grid") or (event or {}).get("start_grid"), grid_size=grid_size, cell_size=cell_size)
    current_xy = _xy_from_candidate(current, grid_size=grid_size, cell_size=cell_size)
    backtrack_xy = _xy_from_candidate(backtrack, grid_size=grid_size, cell_size=cell_size)

    vertices: List[Tuple[float, float, float, int, int, int]] = []

    def add_point(xy: Optional[Tuple[float, float]], z: float, color: Tuple[int, int, int]) -> None:
        if xy is None:
            return
        vertices.append((float(xy[0]), float(xy[1]), float(z), int(color[0]), int(color[1]), int(color[2])))

    def add_segment(a: Optional[Tuple[float, float]], b: Optional[Tuple[float, float]], z: float, color: Tuple[int, int, int]) -> None:
        if a is None or b is None:
            return
        for idx in range(21):
            t = idx / 20.0
            xy = (a[0] * (1.0 - t) + b[0] * t, a[1] * (1.0 - t) + b[1] * t)
            add_point(xy, z, color)

    add_point(pose_xy, 0.10, (255, 255, 255))
    add_point(current_xy, 0.35, (245, 95, 80))
    add_point(backtrack_xy, 0.55, (100, 245, 140))
    add_segment(pose_xy, current_xy, 0.22, (245, 95, 80))
    add_segment(pose_xy, backtrack_xy, 0.42, (100, 245, 140))

    for candidate in (event or {}).get("candidates") or []:
        xy = _xy_from_candidate(candidate, grid_size=grid_size, cell_size=cell_size)
        ctype = str(candidate.get("candidate_type") or "")
        color = (255, 225, 80)
        if ctype == "resilience_backtrack":
            color = (100, 245, 140)
        elif "semantic" in ctype:
            color = (240, 105, 255)
        add_point(xy, 0.25, color)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for x, y, z, r, g, b in vertices:
            handle.write(f"{x:.5f} {y:.5f} {z:.5f} {r} {g} {b}\n")


def _copy_bev(event: Optional[Mapping[str, Any]], out_dir: Path, case_name: str) -> Optional[str]:
    if not event:
        return None
    candidate_path = event.get("candidate_bev_path")
    if not candidate_path:
        return None
    src = Path(str(candidate_path))
    if not src.exists():
        return None
    dst = out_dir / f"{case_name}_occmem_bev.png"
    shutil.copy2(src, dst)
    return dst.name


def export_cases(
    *,
    rows_path: Path,
    memory_roots: Sequence[Path],
    output_dir: Path,
    case_count: int,
    prefer: str,
    distinct_episodes: bool,
    grid_size: int,
    cell_size: float,
) -> Dict[str, Any]:
    rows = list(_read_jsonl(rows_path))
    cases = _select_cases(rows, count=case_count, prefer=prefer, distinct_episodes=distinct_episodes)
    events = _load_events_for_keys(memory_roots, [_event_key(row) for row in cases])
    output_dir.mkdir(parents=True, exist_ok=True)

    exported: List[Dict[str, Any]] = []
    for idx, row in enumerate(cases, start=1):
        event = events.get(_event_key(row))
        safe_scene = str(row.get("scene_id") or "scene").replace("/", "_")
        case_name = f"case{idx:02d}_{safe_scene}_ep{row.get('episode_id')}_step{row.get('step_id')}"
        schematic_name = f"{case_name}_schematic.png"
        ply_name = f"{case_name}_anchors.ply"
        _draw_schematic(
            output_dir / schematic_name,
            row=row,
            event=event,
            grid_size=grid_size,
            cell_size=cell_size,
        )
        _write_ply(
            output_dir / ply_name,
            row=row,
            event=event,
            grid_size=grid_size,
            cell_size=cell_size,
        )
        bev_name = _copy_bev(event, output_dir, case_name)
        item = {
            "case_name": case_name,
            "event_key": _event_key(row),
            "scene_id": row.get("scene_id"),
            "episode_id": row.get("episode_id"),
            "step_id": row.get("step_id"),
            "success": row.get("success"),
            "spl": row.get("spl"),
            "ne": row.get("ne"),
            "trigger_reasons": row.get("trigger_reasons"),
            "recovery_context_tags": row.get("recovery_context_tags"),
            "advantage_margin_proxy": row.get("advantage_margin_proxy"),
            "current_risk_proxy": row.get("current_risk_proxy"),
            "backtrack_utility_proxy": row.get("backtrack_utility_proxy"),
            "schematic_png": schematic_name,
            "anchors_ply": ply_name,
            "copied_occmem_bev_png": bev_name,
            "memory_event_found": event is not None,
            "source_candidate_bev_path": None if event is None else event.get("candidate_bev_path"),
            "best_backtrack_candidate": row.get("best_backtrack_candidate"),
            "current_policy_candidate": row.get("current_policy_candidate"),
        }
        (output_dir / f"{case_name}.json").write_text(
            json.dumps(item, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        exported.append(item)

    html_cards = []
    for item in exported:
        bev_html = ""
        if item.get("copied_occmem_bev_png"):
            bev_html = f'<img src="{html.escape(item["copied_occmem_bev_png"])}" class="case-image" />'
        else:
            bev_html = '<p class="muted">OccMem BEV was not copied. Pass the Stage18e run root on the server to copy candidate_bev_path images.</p>'
        html_cards.append(
            f"""
            <section class="card">
              <h2>{html.escape(item["case_name"])}</h2>
              <p><code>{html.escape(item["event_key"])}</code></p>
              <p>success={item["success"]} SPL={item["spl"]} NE={item["ne"]}</p>
              <p>margin={item["advantage_margin_proxy"]} current_risk={item["current_risk_proxy"]} backtrack_utility={item["backtrack_utility_proxy"]}</p>
              <p>reasons={html.escape(str(item["trigger_reasons"]))}</p>
              <p>context={html.escape(str(item["recovery_context_tags"]))}</p>
              <div class="images">
                <img src="{html.escape(item["schematic_png"])}" class="case-image" />
                {bev_html}
              </div>
              <p><a href="{html.escape(item["anchors_ply"])}">anchors PLY</a> · <a href="{html.escape(item["case_name"])}.json">case JSON</a></p>
            </section>
            """
        )

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Stage18 Recovery Visual Cases</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #151518; color: #ececf0; margin: 24px; }}
    .card {{ background: #222228; border: 1px solid #3a3a42; border-radius: 12px; padding: 16px; margin-bottom: 20px; }}
    .images {{ display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; }}
    .case-image {{ max-width: 680px; width: 48%; min-width: 360px; border-radius: 8px; border: 1px solid #444; }}
    code {{ color: #a8d8ff; }}
    a {{ color: #8fd1ff; }}
    .muted {{ color: #a6a6ad; }}
  </style>
</head>
<body>
  <h1>Stage18 Recovery Visual Cases</h1>
  <p class="muted">Green marks the proposed backtrack/re-observation anchor; red marks the frozen S2 current waypoint. Copied OccMem BEV images show the original free/occupied/frontier map when available.</p>
  {''.join(html_cards)}
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")
    summary = {
        "rows": str(rows_path),
        "memory_roots": [str(path) for path in memory_roots],
        "exported_case_count": len(exported),
        "cases": exported,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--memory-root", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-count", type=int, default=8)
    parser.add_argument("--prefer", choices=["failed", "success", "mixed"], default="failed")
    parser.add_argument("--allow-same-episode", action="store_true")
    parser.add_argument("--grid-size", type=int, default=1000)
    parser.add_argument("--cell-size", type=float, default=0.05)
    args = parser.parse_args()

    summary = export_cases(
        rows_path=args.rows,
        memory_roots=args.memory_root,
        output_dir=args.output_dir,
        case_count=max(1, int(args.case_count)),
        prefer=args.prefer,
        distinct_episodes=not bool(args.allow_same_episode),
        grid_size=max(1, int(args.grid_size)),
        cell_size=float(args.cell_size),
    )
    print(json.dumps({"exported_case_count": summary["exported_case_count"], "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output_dir / 'index.html'}")
    print(f"Wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
