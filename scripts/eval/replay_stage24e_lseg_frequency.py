"""Replay frozen RGB-D-pose ledgers for Q/Q+K/ALL LSeg frequency audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from internnav.utils.lseg_online_shadow import (
    DEFAULT_LABELS, OnlineLSegSemanticShadow, _jsonable,
)
from internnav.utils.lseg_replay_frequency import (
    select_causal_keyframes, short_lived_labels,
)


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _descriptor(image: np.ndarray) -> np.ndarray:
    resized = Image.fromarray(image).resize((32, 24), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _variant(
    source: OnlineLSegSemanticShadow, selected: List[int], output: Path,
    name: str, meta: Dict[str, Any], steps: int, frame_lookup: Dict[int, Dict[str, np.ndarray]],
    record_to_observation: Dict[int, int], record_lookup: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    cfg = {
        "lseg_online_shadow_enable": True,
        "lseg_online_shadow_labels": source.labels,
        "lseg_online_shadow_confidence_threshold": source.confidence_threshold,
        "lseg_online_shadow_sample_stride": source.sample_stride,
        "lseg_online_shadow_merge_radius_m": source.merge_radius_m,
        "lseg_online_shadow_max_surface_samples": source.max_surface_samples,
        "lseg_online_shadow_save_overlay": False,
        "lseg_online_shadow_save_surface": False,
        "lseg_online_shadow_save_visualizations": True,
    }
    shadow = OnlineLSegSemanticShadow(cfg, source.camera_intrinsic, source.device)
    shadow.set_root(str(output / name))
    shadow.reset_episode(**meta, replay_frequency=name)
    selected_observations = [
        record_to_observation[index] for index in selected if index in record_to_observation
    ]
    shadow.surface_frames = [
        frame_lookup[index] for index in selected_observations if index in frame_lookup
    ]
    shadow._stored_surface_count = sum(len(frame["map_xyz"]) for frame in shadow.surface_frames)
    shadow.records = [record_lookup[index] for index in selected if index in record_lookup]
    shadow.inference_seconds = [
        float(record_lookup[index]["inference_seconds"])
        for index in selected
        if index in record_lookup and record_lookup[index].get("valid")
    ]
    return shadow.finish_episode(
        metrics={}, steps=steps, occ_memory=None, frequency=name
    )


def replay_episode(ledger: Path, output: Path, args: argparse.Namespace) -> Dict[str, Any]:
    meta = json.loads((ledger / "episode_meta.json").read_text(encoding="utf-8"))
    observations = _jsonl(ledger / "observations.jsonl")
    queries = _jsonl(ledger / "queries.jsonl")
    camera_intrinsic = np.asarray(meta["camera_model"]["intrinsic"], dtype=np.float32)
    cfg = {
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
    source = OnlineLSegSemanticShadow(cfg, camera_intrinsic, args.device)
    source.set_root(str(output / "inference"))
    source.reset_episode(**meta, replay_frequency="all_inference_cache")
    descriptors: Dict[int, np.ndarray] = {}
    for observation in observations:
        index = int(observation["record_index"])
        rgb = np.ascontiguousarray(
            np.asarray(Image.open(ledger / observation["rgb_path"]).convert("RGB")).copy()
        )
        loaded_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
        expected_hash = observation.get("rgb_sha256")
        if observation.get("rgb_storage_format") != "png":
            raise RuntimeError(
                f"Stage24E requires lossless PNG replay RGB: {ledger}"
            )
        if not expected_hash or loaded_hash != expected_hash:
            raise RuntimeError(
                f"Lossless RGB hash mismatch for {ledger}/{observation['rgb_path']}: "
                f"{loaded_hash} != {expected_hash}"
            )
        descriptors[index] = _descriptor(rgb)
        with np.load(ledger / observation["depth_path"]) as payload:
            depth = np.ascontiguousarray(payload["depth_m"], dtype=np.float32)
        loaded_depth_hash = hashlib.sha256(depth.tobytes()).hexdigest()
        if loaded_depth_hash != observation.get("depth_sha256"):
            raise RuntimeError(
                f"Depth hash mismatch for {ledger}/{observation['depth_path']}"
            )
        source.process_query_frame(
            rgb=rgb, depth_m=depth,
            camera_pose_map=(observation.get("pose") or {})["stage23_gt_camera_pose_map"],
            step_id=int(observation["step_id"]), query_id=index,
            observation_index=int(observation["observation_index"]), occ_memory=None,
        )
        print(
            f"STAGE24E_FRAME scene={meta['scene_id']} episode={meta['episode_id']} "
            f"frame={index + 1}/{len(observations)}", flush=True
        )
    if len(source.records) != len(observations) or source.errors:
        raise RuntimeError(f"LSeg replay failed for {ledger}: {source.errors}")
    frame_lookup = {
        int(frame["observation_index"][0]): frame for frame in source.surface_frames
        if len(frame["observation_index"])
    }
    query_keys = {str(query["observation_key"]) for query in queries}
    q_selected = [
        int(item["record_index"]) for item in observations
        if str(item["observation_key"]) in query_keys
    ]
    qk_reasons = select_causal_keyframes(
        observations, query_keys, descriptors,
        translation_m=args.translation_m, rotation_deg=args.rotation_deg,
        height_m=args.height_m, pitch_deg=args.pitch_deg,
        visual_change=args.visual_change, min_gap=args.min_gap, max_gap=args.max_gap,
    )
    selections = {
        "q": q_selected,
        "q_plus_k": sorted(qk_reasons),
        "all": [int(item["record_index"]) for item in observations],
    }
    record_to_observation = {
        int(item["record_index"]): int(item["observation_index"])
        for item in observations
    }
    record_lookup = {
        int(observation["record_index"]): record
        for observation, record in zip(observations, source.records)
    }
    summaries = {
        name: _variant(
            source, selected, output, name, meta, len(observations), frame_lookup,
            record_to_observation, record_lookup,
        )
        for name, selected in selections.items()
    }
    all_frame_counts = [record.get("class_surface_counts") or {} for record in source.records]
    transient = short_lived_labels(all_frame_counts)
    metrics = {}
    for name, selected in selections.items():
        summary = summaries[name]
        labels = set(summary.get("class_surface_counts") or {})
        gt = summary.get("gt_audit") or {}
        nodes = int(summary.get("node_count") or 0)
        selected_labels = {
            label for index in selected
            for label, count in (record_lookup[index].get("class_surface_counts") or {}).items()
            if int(count) > 0
        }
        metrics[name] = {
            "call_count": len(selected),
            "call_fraction_of_all": len(selected) / max(1, len(observations)),
            "class_count": len(labels),
            "classes": sorted(labels),
            "short_lived_class_count": len(transient.intersection(selected_labels)),
            "short_lived_classes": sorted(transient.intersection(selected_labels)),
            "gt_compatible_node_count": int(gt.get("compatible_node_count") or 0),
            "gt_hit_count": int(gt.get("surface_distance_le_050m_count") or 0),
            "gt_hit_rate": gt.get("surface_distance_le_050m_rate"),
            "node_count": nodes,
            "multi_view_node_rate": summary.get("multi_view_node_rate"),
            "conflict_count": int(summary.get("cross_label_conflict_count") or 0),
            "conflicts_per_100_nodes": 100.0 * int(
                summary.get("cross_label_conflict_count") or 0
            ) / max(1, nodes),
        }
    result = {
        "scene_id": str(meta["scene_id"]), "episode_id": str(meta["episode_id"]),
        "ledger": str(ledger), "observation_count": len(observations),
        "query_count": len(q_selected), "short_lived_classes_in_all": sorted(transient),
        "keyframe_reasons": {str(key): value for key, value in qk_reasons.items()},
        "variants": metrics,
        "selector": {
            "translation_m": args.translation_m, "rotation_deg": args.rotation_deg,
            "height_m": args.height_m, "pitch_deg": args.pitch_deg,
            "visual_change": args.visual_change, "min_gap": args.min_gap,
            "max_gap": args.max_gap,
        },
        "decision_status": "audit_only_not_navigation_ready",
    }
    online_dirs = [
        path.parent for path in args.ledger_root.glob("**/online_lseg_shadow/*/episode_meta.json")
        if (lambda value: str(value.get("scene_id")) == str(meta["scene_id"])
            and str(value.get("episode_id")) == str(meta["episode_id"]))
        (json.loads(path.read_text(encoding="utf-8")))
    ]
    consistency_errors = []
    if len(online_dirs) != 1:
        consistency_errors.append(f"expected_one_online_dir_got_{len(online_dirs)}")
    else:
        online_events = _jsonl(online_dirs[0] / "events.jsonl")
        observation_by_index = {
            int(item["observation_index"]): item for item in observations
        }
        replay_by_observation = {
            int(record["observation_index"]): record for record in source.records
        }
        if len(online_events) != len(q_selected):
            consistency_errors.append(
                f"query_count:{len(online_events)}!={len(q_selected)}"
            )
        for online in online_events:
            observation_index = int(online["observation_index"])
            replayed = replay_by_observation.get(observation_index)
            if replayed is None:
                consistency_errors.append(f"missing_replay_observation:{observation_index}")
                continue
            observation = observation_by_index.get(observation_index)
            if observation is None:
                consistency_errors.append(
                    f"missing_ledger_observation:{observation_index}"
                )
                continue
            pose = np.ascontiguousarray(
                (observation.get("pose") or {})[
                    "stage23_gt_camera_pose_map"
                ], dtype=np.float32,
            )
            expected_pose_hash = hashlib.sha256(pose.tobytes()).hexdigest()
            if online.get("camera_pose_sha256") != expected_pose_hash:
                consistency_errors.append(
                    f"observation_{observation_index}:camera_pose_sha256"
                )
            for key in (
                "rgb_sha256", "depth_sha256", "class_surface_counts",
                "surface_sample_count",
            ):
                if online.get(key) != replayed.get(key):
                    consistency_errors.append(f"observation_{observation_index}:{key}")
    result["online_q_consistency"] = {
        "passed": not consistency_errors, "errors": consistency_errors,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "episode_frequency_comparison.json").write_text(
        json.dumps(_jsonable(result), indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vlmaps-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-surface-samples", type=int, default=1000000)
    parser.add_argument("--translation-m", type=float, default=0.50)
    parser.add_argument("--rotation-deg", type=float, default=30.0)
    parser.add_argument("--height-m", type=float, default=0.20)
    parser.add_argument("--pitch-deg", type=float, default=15.0)
    parser.add_argument("--visual-change", type=float, default=0.12)
    parser.add_argument("--min-gap", type=int, default=2)
    parser.add_argument("--max-gap", type=int, default=4)
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.device == "distributed":
        args.device = f"cuda:{local_rank}"
    ledgers = sorted(path.parent for path in args.ledger_root.glob(
        "**/replay_ledger/*/episode_meta.json"
    ))
    if not ledgers:
        raise SystemExit(f"No replay ledgers found under {args.ledger_root}")
    for index, ledger in enumerate(ledgers):
        if index % world != rank:
            continue
        meta = json.loads((ledger / "episode_meta.json").read_text(encoding="utf-8"))
        episode_output = (
            args.output_root / f"rank{rank}"
            / f"{str(meta['scene_id']).replace('/', '_')}_{meta['episode_id']}"
        )
        result = replay_episode(ledger, episode_output, args)
        print("STAGE24E_EPISODE_COMPLETE " + json.dumps({
            "scene_id": result["scene_id"], "episode_id": result["episode_id"],
            "variants": result["variants"],
        }), flush=True)


if __name__ == "__main__":
    main()
