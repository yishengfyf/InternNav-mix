"""Causal semantic-window helpers for Stage25 stuck confirmation."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence


def select_causal_window(
    observations: Sequence[Mapping[str, Any]], event_step: int, window_steps: int,
    *, max_frames: int | None = None,
) -> List[Mapping[str, Any]]:
    start = int(event_step) - int(window_steps)
    selected = [
        observation for observation in observations
        if start <= int(observation["step_id"]) <= int(event_step)
    ]
    if max_frames is not None:
        selected = selected[-int(max_frames):]
    return selected


def summarize_semantic_window(
    frames: Sequence[Mapping[str, Any]], *, recent_frame_count: int = 4,
) -> Dict[str, Any]:
    valid = [frame for frame in frames if frame.get("valid")]
    recent = valid[-int(recent_frame_count):]
    cell_sets = [set(frame.get("spatial_semantic_cells") or []) for frame in recent]
    previous_cells = set().union(*cell_sets[:-1]) if len(cell_sets) >= 2 else set()
    latest_cells = cell_sets[-1] if cell_sets else set()
    overlap = latest_cells & previous_cells
    recurrence = len(overlap) / max(1, len(latest_cells))
    novelty = len(latest_cells - previous_cells) / max(1, len(latest_cells))
    class_support = Counter(
        label
        for frame in valid
        for label, count in (frame.get("class_surface_counts") or {}).items()
        if int(count) > 0
    )
    recurrent_cells = Counter(
        cell for frame in valid
        for cell in set(frame.get("spatial_semantic_cells") or [])
    )
    return {
        "valid_frame_count": len(valid),
        "recent_frame_count": len(recent),
        "latest_strong_cell_count": len(latest_cells),
        "repeated_recent_cell_count": len(overlap),
        "recent_spatial_recurrence": recurrence,
        "recent_semantic_novelty": novelty,
        "spatial_stagnation": bool(
            len(recent) >= 3 and len(latest_cells) >= 2
            and recurrence >= 0.60 and novelty <= 0.25
        ),
        "classes": sorted(class_support),
        "classes_with_multiframe_support": sorted(
            label for label, count in class_support.items() if count >= 2
        ),
        "recurrent_cell_count": sum(
            count >= 2 for count in recurrent_cells.values()
        ),
    }
