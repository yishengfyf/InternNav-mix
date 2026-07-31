"""Merge Stage17 per-rank VLMap debug runs into one label-collection run.

The Stage17 candidate collection eval can be launched with torchrun. Each rank
gets a different subset of episodes and writes its own VLMap debug run directory
such as ``rank0_run_001``. The existing label collector expects one run
directory, so this utility concatenates the JSONL files required by
``collect_gt_candidate_labels.py`` into a synthetic merged run.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


REQUIRED_FILES = (
    Path("progress.json"),
    Path("trajectory_events.jsonl"),
    Path("occ_memory") / "memory_events.jsonl",
)

OPTIONAL_FILES = (
    Path("semantic_events.jsonl"),
    Path("semantic_episode_summary.jsonl"),
    Path("events.jsonl"),
    Path("waypoint_events.jsonl"),
    Path("log.jsonl"),
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{lineno}") from exc
    return records


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _sort_key(record: Dict[str, Any]) -> tuple:
    scene_id = str(record.get("scene_id", ""))
    episode_id = record.get("episode_id", -1)
    try:
        episode_id = int(episode_id)
    except (TypeError, ValueError):
        episode_id = -1
    eval_step = record.get("eval_step", record.get("step", record.get("step_id", -1)))
    try:
        eval_step = int(eval_step)
    except (TypeError, ValueError):
        eval_step = -1
    time_value = str(record.get("time", ""))
    return scene_id, episode_id, eval_step, time_value


def _discover_run_dirs(run_root: Path, output_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    resolved_output = output_dir.resolve()
    for child in sorted(run_root.iterdir()):
        if not child.is_dir():
            continue
        if child.resolve() == resolved_output:
            continue
        if all((child / rel_path).exists() for rel_path in REQUIRED_FILES):
            candidates.append(child)
    return candidates


def _merge_file(run_dirs: Sequence[Path], rel_path: Path, output_dir: Path, sort_records: bool) -> Dict[str, Any]:
    merged: List[Dict[str, Any]] = []
    source_counts: Dict[str, int] = {}
    for run_dir in run_dirs:
        path = run_dir / rel_path
        if not path.exists():
            continue
        records = _read_jsonl(path)
        source_counts[str(run_dir)] = len(records)
        for record in records:
            record.setdefault("_stage17_source_run", run_dir.name)
        merged.extend(records)

    if sort_records:
        merged.sort(key=_sort_key)
    output_count = _write_jsonl(output_dir / rel_path, merged)
    return {
        "file": str(rel_path),
        "output_records": output_count,
        "source_counts": source_counts,
    }


def _copy_optional_directories(run_dirs: Sequence[Path], output_dir: Path) -> Dict[str, Any]:
    copied: Dict[str, Any] = {}
    for run_dir in run_dirs:
        candidates_dir = run_dir / "occ_memory" / "candidates"
        if not candidates_dir.exists():
            continue
        target_dir = output_dir / "occ_memory" / f"candidates_{run_dir.name}"
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(candidates_dir, target_dir)
        copied[str(candidates_dir)] = str(target_dir)
    return copied


def merge_rank_runs(run_root: Path, output_dir: Path, run_dirs: Optional[Sequence[Path]] = None) -> Dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if run_dirs is None:
        run_dirs = _discover_run_dirs(run_root, output_dir)
    else:
        run_dirs = [path.expanduser().resolve() for path in run_dirs]

    if not run_dirs:
        raise FileNotFoundError(f"No Stage17 rank run directories found under {run_root}")
    for run_dir in run_dirs:
        missing = [str(rel_path) for rel_path in REQUIRED_FILES if not (run_dir / rel_path).exists()]
        if missing:
            raise FileNotFoundError(f"Missing required files in {run_dir}: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_files: List[Dict[str, Any]] = []
    for rel_path in REQUIRED_FILES:
        merged_files.append(_merge_file(run_dirs, rel_path, output_dir, sort_records=True))
    for rel_path in OPTIONAL_FILES:
        if any((run_dir / rel_path).exists() for run_dir in run_dirs):
            merged_files.append(_merge_file(run_dirs, rel_path, output_dir, sort_records=False))

    summary = {
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "source_run_dirs": [str(run_dir) for run_dir in run_dirs],
        "merged_files": merged_files,
        "copied_optional_directories": _copy_optional_directories(run_dirs, output_dir),
    }
    (output_dir / "merge_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Stage17 per-rank debug runs for label collection.")
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Directory containing rank debug runs, e.g. logs/.../vlmap_safety_debug.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Synthetic merged run directory to create.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        help="Explicit source run directory. Repeat to bypass auto-discovery.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = merge_rank_runs(args.run_root, args.output_dir, args.run_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
