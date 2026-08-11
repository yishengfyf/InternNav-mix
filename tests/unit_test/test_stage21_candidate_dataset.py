import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "eval"
    / "build_stage21_candidate_recoverability_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("stage21_dataset", SCRIPT_PATH)
stage21_dataset = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage21_dataset)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _make_run(root, name, *, duplicate=False):
    run_dir = root / "vlmap_safety_debug" / name
    event = {
        "event_type": "occ_memory_query_candidates",
        "scene_id": "scene_a",
        "episode_id": 1,
        "step_id": 5,
        "candidate_count": 2,
        "candidates": [{"candidate_id": "A"}, {"candidate_id": "B"}],
    }
    _write_jsonl(run_dir / "occ_memory" / "memory_events.jsonl", [event])
    _write_jsonl(run_dir / "trajectory_events.jsonl", [])
    _write_jsonl(
        run_dir / "progress.json",
        [{"scene_id": "scene_a", "episode_id": 1, "success": 1.0, "spl": 0.5}],
    )
    _write_jsonl(
        run_dir / "stage19_semantic_resilience_active_events.jsonl",
        [{
            "scene_id": "scene_a",
            "episode_id": 1,
            "step_id": 5,
            "v2_evidence_tier": "adapter_candidate",
            "applied": False,
        }],
    )
    return run_dir


def test_discover_and_deduplicate_without_reference_path(tmp_path):
    _make_run(tmp_path, "rank0_run_001")
    _make_run(tmp_path, "rank1_run_001", duplicate=True)
    output_dir = tmp_path / "dataset"

    summary = stage21_dataset.build_stage21_dataset(
        run_root=tmp_path,
        episodes_file=None,
        output_dir=output_dir,
    )

    assert summary["run_dir_count"] == 2
    assert summary["counts"]["collected_rows_before_dedup"] == 2
    assert summary["counts"]["duplicate_rows_removed"] == 1
    assert summary["counts"]["label_rows"] == 1
    assert summary["label_status_counts"] == {"no_reference_path": 1}
    assert summary["candidate_labels_are_route_progress_not_episode_success"] is True
    assert summary["active_safety_check"] == {
        "applied_count": 0,
        "expected_for_shadow": 0,
        "passed": True,
    }
    row = json.loads((output_dir / "labels.jsonl").read_text(encoding="utf-8"))
    assert row["episode_outcome_context_only"]["success"] == 1.0
    assert row["triage_context"]["tier"] == "adapter_candidate"
    assert (output_dir / "rows.jsonl").read_text(encoding="utf-8") == ""


def test_scene_split_has_no_overlap():
    rows = [
        {"scene_id": "scene_a", "episode_id": 1},
        {"scene_id": "scene_a", "episode_id": 2},
        {"scene_id": "scene_b", "episode_id": 3},
    ]
    split_by_scene = {}
    for row in rows:
        split = stage21_dataset.build_dataset.__globals__["_split_for_row"](
            row, 0.5, 21, "scene"
        )
        split_by_scene.setdefault(row["scene_id"], set()).add(split)
    assert all(len(splits) == 1 for splits in split_by_scene.values())


def test_candidate_rows_use_relative_s2_value_and_keep_outcome_as_context(tmp_path):
    episodes_file = tmp_path / "train.json"
    episodes_file.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "scene_id": "scene_a",
                        "episode_id": 1,
                        "start_position": [0.0, 0.0, 0.0],
                        "start_rotation": [0.0, 0.0, 0.0, 1.0],
                        "reference_path": [
                            [0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [2.0, 0.0, 0.0],
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    labels = [
        {
            "scene_id": "scene_a",
            "episode_id": 1,
            "step_id": 5,
            "label_status": "ok",
            "current_xy": [0.0, 0.0],
            "current_policy_candidate": {
                "candidate_id": "S2",
                "xy": [0.0, -0.5],
                "geometry_safe": True,
                "active_gate_safe": True,
            },
            "candidates": [
                {
                    "candidate_id": "A",
                    "candidate_type": "semantic_frontier",
                    "direction_bucket": "front",
                    "xy": [0.0, -1.5],
                    "geometry_safe": True,
                    "active_gate_safe": True,
                },
                {
                    "candidate_id": "B",
                    "candidate_type": "semantic_keyframe",
                    "direction_bucket": "left",
                    "xy": [1.5, 0.0],
                    "geometry_safe": True,
                    "active_gate_safe": True,
                },
            ],
            "triage_context": {"tier": "adapter_candidate"},
            "episode_outcome_context_only": {"success": 0.0},
        }
    ]

    rows, audit = stage21_dataset._build_candidate_rows(
        labels,
        episodes_file=episodes_file,
        reference_frame="episodic_gps",
        quaternion_order="xyzw",
        coordinate_mode="x_neg_y",
    )

    assert len(rows) == 2
    assert audit["counts"]["positive"] == 1
    assert audit["counts"]["negative"] == 1
    assert audit["counts"]["positive_vs_negative_pairs"] == 1
    assert rows[0]["offline_labels"]["advantage_vs_s2_m"] > 0.0
    assert rows[1]["offline_labels"]["advantage_vs_s2_m"] < 0.0
    assert rows[0]["episode_outcome_context_only"] == {"success": 0.0}
    assert "success" not in rows[0]["online_inputs"]
