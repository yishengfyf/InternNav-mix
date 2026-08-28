from __future__ import annotations

import importlib.util
import json
from pathlib import Path


_path = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "eval"
    / "analyze_stage55_post_turn_guard.py"
)
_spec = importlib.util.spec_from_file_location(
    "analyze_stage55_post_turn_guard", _path
)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_analyzer_accepts_complete_paired_guard_fixture(tmp_path):
    manifest_rows = [
        {
            "scene_id": f"scene{index}",
            "episode_id": index,
            "episode_eval_seed": 100 + index,
        }
        for index in range(4)
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(manifest_rows), encoding="utf-8")
    baseline_root = tmp_path / "baseline"
    guard_root = tmp_path / "guard"
    baseline_progress = []
    guard_progress = []
    for row in manifest_rows:
        common = {
            "scene_id": row["scene_id"],
            "episode_id": row["episode_id"],
            "episode_eval_seed": row["episode_eval_seed"],
            "success": 0.0,
            "spl": 0.0,
            "ne": 5.0,
            "steps": 100,
            "collision_count": 4,
        }
        baseline_progress.append(common)
        guard_progress.append(
            {
                **common,
                "collision_count": 2 if row["episode_id"] == 0 else 4,
            }
        )
    _write_jsonl(baseline_root / "progress.json", baseline_progress)
    _write_jsonl(guard_root / "progress.json", guard_progress)
    _write_jsonl(
        guard_root
        / "vlmap_safety_debug"
        / "rank0_run_001"
        / "stage55_post_turn_collision_guard_events.jsonl",
        [
            {
                "scene_id": "scene0",
                "episode_id": 0,
                "previous_action": 1,
                "previous_action_source": "system2_action_queue",
                "collision_delta": 1.0,
                "environment_action_applied": False,
                "pixel_translation_applied": False,
                "gt_fields_used": [],
            }
        ],
    )

    report = _module.analyze(baseline_root, guard_root, manifest)

    assert report["integrity_passed"] is True
    assert report["guard_event_count"] == 1
    assert report["guard_episode_count_with_event"] == 1
    assert report["collision_delta_sum"] == -2.0
