#!/usr/bin/env python3
"""Replay causal pre-onset RGB-D windows for Stage25 D3R confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from internnav.utils.lseg_online_shadow import DEFAULT_LABELS, OnlineLSegSemanticShadow
from internnav.utils.stage25_semantic_confirmation import (
    select_causal_window, summarize_semantic_window,
)
from scripts.eval.analyze_stage25_gt_detector import semantic_cells


def jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def episode_key(row: Mapping[str, Any]) -> str:
    return f"{row.get('scene_id')}|{row.get('episode_id')}"


def load_rgb_depth(ledger: Path, observation: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if observation.get("rgb_storage_format") != "png":
        raise RuntimeError(f"D3R requires lossless PNG RGB: {ledger}")
    rgb = np.ascontiguousarray(
        np.asarray(Image.open(ledger / str(observation["rgb_path"])).convert("RGB"), dtype=np.uint8)
    )
    with np.load(ledger / str(observation["depth_path"])) as payload:
        depth = np.ascontiguousarray(payload["depth_m"], dtype=np.float32)
    if hashlib.sha256(rgb.tobytes()).hexdigest() != observation.get("rgb_sha256"):
        raise RuntimeError(f"RGB hash mismatch: {ledger}/{observation['rgb_path']}")
    if hashlib.sha256(depth.tobytes()).hexdigest() != observation.get("depth_sha256"):
        raise RuntimeError(f"depth hash mismatch: {ledger}/{observation['depth_path']}")
    return rgb, depth


def replay_episode(
    ledger: Path, candidates: List[Dict[str, Any]], output: Path, args: argparse.Namespace,
) -> Dict[str, Any]:
    meta = json.loads((ledger / "episode_meta.json").read_text(encoding="utf-8"))
    observations = jsonl(ledger / "observations.jsonl")
    selected_by_event = {
        int(event["step_id"]): select_causal_window(
            observations, int(event["step_id"]), args.window_steps,
            max_frames=args.max_frames,
        )
        for event in candidates
    }
    selected_indices = sorted({
        int(observation["record_index"])
        for selected in selected_by_event.values() for observation in selected
    })
    observation_by_index = {
        int(observation["record_index"]): observation for observation in observations
    }
    config = {
        "lseg_online_shadow_enable": True,
        "lseg_online_shadow_repo": str(args.vlmaps_repo),
        "lseg_online_shadow_checkpoint": str(args.checkpoint),
        "lseg_online_shadow_device": args.device,
        "lseg_online_shadow_labels": DEFAULT_LABELS,
        "lseg_online_shadow_confidence_threshold": args.confidence,
        "lseg_online_shadow_sample_stride": args.stride,
        "lseg_online_shadow_merge_radius_m": 0.50,
        "lseg_online_shadow_max_surface_samples": args.max_surface_samples,
        "lseg_online_shadow_save_overlay": False,
        "lseg_online_shadow_save_surface": False,
        "lseg_online_shadow_save_visualizations": False,
    }
    source = OnlineLSegSemanticShadow(
        config, np.asarray(meta["camera_model"]["intrinsic"], dtype=np.float32), args.device
    )
    source.set_root(str(output / "inference"))
    source.reset_episode(**meta, replay_frequency="d3r_causal_window")
    frame_by_index: Dict[int, Dict[str, Any]] = {}
    for offset, record_index in enumerate(selected_indices, start=1):
        observation = observation_by_index[record_index]
        rgb, depth = load_rgb_depth(ledger, observation)
        surface_count_before = len(source.surface_frames)
        record = source.process_query_frame(
            rgb=rgb, depth_m=depth,
            camera_pose_map=(observation.get("pose") or {})["stage23_gt_camera_pose_map"],
            step_id=int(observation["step_id"]), query_id=record_index,
            observation_index=int(observation["observation_index"]), occ_memory=None,
        )
        if not record.get("valid"):
            raise RuntimeError(f"LSeg D3R failed: {record}")
        surface = (
            source.surface_frames[-1]
            if len(source.surface_frames) > surface_count_before else None
        )
        record["spatial_semantic_cells"] = [] if surface is None else semantic_cells(
            surface["map_xyz"], surface["class_id"], surface["confidence"]
        )
        frame_by_index[record_index] = record
        print(
            f"STAGE25_D3R_FRAME scene={meta['scene_id']} episode={meta['episode_id']} "
            f"frame={offset}/{len(selected_indices)}", flush=True,
        )
    event_results = []
    for event in candidates:
        selected = selected_by_event[int(event["step_id"])]
        frames = [frame_by_index[int(item["record_index"])] for item in selected]
        summary = summarize_semantic_window(frames)
        summary.update({
            "supports_existing_suspicion": summary["spatial_stagnation"],
            "node_count": None,
            "gt_hit_rate": None,
        })
        event_frames = [
            frame for frame in source.surface_frames
            if int(frame["observation_index"][0]) in {
                int(item["observation_index"]) for item in selected
            }
        ]
        nodes = source._merge_nodes(event_frames)
        gt_audit = source._audit_nodes_with_gt(nodes)
        conflicts = source._audit_conflicts(nodes)
        summary.update({
            "node_count": len(nodes),
            "multi_view_node_count": sum(
                len(node["source_observations"]) >= 2 for node in nodes
            ),
            "gt_compatible_node_count": int(gt_audit.get("compatible_node_count") or 0),
            "gt_hit_count": int(gt_audit.get("surface_distance_le_050m_count") or 0),
            "gt_hit_rate": gt_audit.get("surface_distance_le_050m_rate"),
            "cross_label_conflict_count": int(conflicts.get("raw_count") or 0),
            "severe_cross_label_conflict_count": int(conflicts.get("severe_count") or 0),
        })
        event_results.append({
            "scene_id": event["scene_id"], "episode_id": event["episode_id"],
            "step_id": event["step_id"], "event_family": event["event_family"],
            "recoverability_proxy": event["recoverability_proxy"],
            "outcome": event["outcome"],
            "d3q_supported": bool(
                (event.get("semantic_confirmation") or {}).get("supports_existing_suspicion")
            ),
            "d3r": summary,
            "window_start_step": min(int(item["step_id"]) for item in selected),
            "window_end_step": max(int(item["step_id"]) for item in selected),
            "selected_frame_count": len(selected),
        })
    result = {
        "scene_id": meta["scene_id"], "episode_id": meta["episode_id"],
        "ledger": str(ledger), "event_count": len(candidates),
        "unique_lseg_call_count": len(selected_indices),
        "inference_seconds_total": float(sum(source.inference_seconds)),
        "window_steps": args.window_steps,
        "max_frames": args.max_frames,
        "future_used_by_detector": False,
        "semantic_can_create_event": False,
        "events": event_results,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "d3r_episode.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vlmaps-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--window-steps", type=int, default=24)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-surface-samples", type=int, default=10000000)
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.device == "distributed":
        args.device = f"cuda:{local_rank}"
    payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for event in payload["D2"]:
        grouped[episode_key(event)].append(event)
    ledgers = {}
    for path in args.run_root.glob("**/replay_ledger/*/episode_meta.json"):
        meta = json.loads(path.read_text(encoding="utf-8"))
        ledgers[episode_key(meta)] = path.parent
    missing = sorted(set(grouped) - set(ledgers))
    if missing:
        raise SystemExit(f"missing ledgers: {missing}")
    selected_keys = sorted(grouped)
    for index, key in enumerate(selected_keys):
        if index % world != rank:
            continue
        scene_id, episode_id = key.split("|", 1)
        output = args.output_root / f"rank{rank}" / f"{scene_id}_{episode_id}"
        result = replay_episode(ledgers[key], grouped[key], output, args)
        print("STAGE25_D3R_EPISODE_COMPLETE " + json.dumps({
            "scene_id": result["scene_id"], "episode_id": result["episode_id"],
            "event_count": result["event_count"],
            "call_count": result["unique_lseg_call_count"],
        }), flush=True)


if __name__ == "__main__":
    main()
