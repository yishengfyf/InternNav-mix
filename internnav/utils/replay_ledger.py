"""Episode replay ledger for audit-only RGB-D, S2, action and map alignment.

The ledger is deliberately side-effect-only: it records observations and decisions
but never participates in action selection.  Large arrays are stored separately so
the JSONL index remains inspectable and query/action events stay aligned by step.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image


class ReplayLedger:
    """Persist an audit-only per-episode observation/query/action ledger."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("replay_ledger_enable", False))
        self.save_rgb = bool(cfg.get("replay_ledger_save_rgb", True))
        self.rgb_format = str(cfg.get("replay_ledger_rgb_format", "jpg")).lower()
        if self.rgb_format not in {"jpg", "jpeg", "png"}:
            raise ValueError(f"Unsupported replay RGB format: {self.rgb_format}")
        self.save_depth = bool(cfg.get("replay_ledger_save_depth", True))
        self.max_observations = max(
            0, int(cfg.get("replay_ledger_max_observations", 0) or 0)
        )
        self.max_queries = max(0, int(cfg.get("replay_ledger_max_queries", 0) or 0))
        self.max_actions = max(0, int(cfg.get("replay_ledger_max_actions", 0) or 0))
        self.repeat_episode_meta = bool(
            cfg.get("replay_ledger_repeat_episode_meta", True)
        )
        self.root: Optional[Path] = None
        self.episode_dir: Optional[Path] = None
        self._obs_file = None
        self._query_file = None
        self._action_file = None
        self._records = 0
        self._queries = 0
        self._actions = 0
        self._observation_keys = set()
        self._last_observation_key = None
        self._episode_meta: Dict[str, Any] = {}
        self._record_meta: Dict[str, Any] = {}

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (Path,)):
            return str(value)
        if isinstance(value, dict):
            return {str(k): ReplayLedger._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [ReplayLedger._jsonable(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def set_root(self, root: Optional[str]) -> None:
        self.root = None if not root else Path(root)

    def reset_episode(self, **meta: Any) -> None:
        self.close()
        self._episode_meta = self._jsonable(meta)
        self._record_meta = (
            dict(self._episode_meta)
            if self.repeat_episode_meta
            else {
                key: self._episode_meta.get(key)
                for key in (
                    "scene_id",
                    "episode_id",
                    "episode_index",
                    "episode_count",
                    "rank",
                    "world_size",
                )
            }
        )
        self._records = 0
        self._queries = 0
        self._actions = 0
        self._observation_keys = set()
        self._last_observation_key = None
        if not self.enabled or self.root is None:
            return
        scene_id = str(meta.get("scene_id", "unknown_scene")).replace("/", "_")
        episode_id = str(meta.get("episode_id", "unknown_episode"))
        rank = str(meta.get("rank", "0"))
        self.episode_dir = self.root / "replay_ledger" / f"{scene_id}_{episode_id}_r{rank}"
        (self.episode_dir / "rgb").mkdir(parents=True, exist_ok=True)
        (self.episode_dir / "depth").mkdir(parents=True, exist_ok=True)
        self._obs_file = open(self.episode_dir / "observations.jsonl", "a", encoding="utf-8")
        self._query_file = open(self.episode_dir / "queries.jsonl", "a", encoding="utf-8")
        self._action_file = open(self.episode_dir / "actions.jsonl", "a", encoding="utf-8")
        self._write_json(self.episode_dir / "episode_meta.json", self._episode_meta)

    def _write_json(self, path: Path, value: Any) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self._jsonable(value), handle, ensure_ascii=False, indent=2)

    def _write_line(self, handle, value: Dict[str, Any]) -> None:
        if handle is None:
            return
        handle.write(json.dumps(self._jsonable(value), ensure_ascii=False) + "\n")
        handle.flush()

    @staticmethod
    def _step_key(step_id: Any, observation_index: Any) -> str:
        return f"{int(step_id) if step_id is not None else -1}:{int(observation_index)}"

    def record_observation(
        self,
        *,
        step_id: Any,
        observation_index: int,
        rgb: Any,
        depth: Any,
        pose: Optional[Dict[str, Any]] = None,
        camera_pitch_deg: Any = None,
        previous_action: Any = None,
        previous_action_source: Any = None,
        previous_pre_safety_action: Any = None,
        previous_action_applied: Any = None,
        route_node: Any = None,
        occ_summary: Optional[Dict[str, Any]] = None,
        semantic_state: Optional[Dict[str, Any]] = None,
        audit_metrics: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled or self.episode_dir is None:
            return None
        if self.max_observations > 0 and self._records >= self.max_observations:
            return None
        key = self._step_key(step_id, observation_index)
        if key in self._observation_keys:
            return None
        self._observation_keys.add(key)
        self._last_observation_key = key
        prefix = f"obs_{self._records:05d}_step_{int(step_id) if step_id is not None else -1}_{int(observation_index)}"
        record: Dict[str, Any] = {
            "event_type": "replay_observation",
            **self._record_meta,
            "record_index": int(self._records),
            "observation_key": key,
            "step_id": None if step_id is None else int(step_id),
            "observation_index": int(observation_index),
            "camera_pitch_deg": None if camera_pitch_deg is None else float(camera_pitch_deg),
            "pose": pose,
            "previous_action": previous_action,
            "previous_action_source": previous_action_source,
            "previous_pre_safety_action": previous_pre_safety_action,
            "previous_action_applied": previous_action_applied,
            "route_node": route_node,
            "occ_summary": occ_summary or {},
            "semantic_state": semantic_state or {},
            "audit_metrics": audit_metrics or {},
        }
        if rgb is not None:
            rgb_arr = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
            record["rgb_shape"] = list(rgb_arr.shape)
            record["rgb_dtype"] = str(rgb_arr.dtype)
            record["rgb_sha256"] = hashlib.sha256(rgb_arr.tobytes()).hexdigest()
            record["rgb_storage_format"] = self.rgb_format
            if self.save_rgb:
                suffix = "png" if self.rgb_format == "png" else "jpg"
                rgb_path = self.episode_dir / "rgb" / f"{prefix}.{suffix}"
                if self.rgb_format == "png":
                    Image.fromarray(rgb_arr).save(
                        rgb_path, format="PNG", compress_level=3
                    )
                else:
                    Image.fromarray(rgb_arr).save(rgb_path, quality=90)
                record["rgb_path"] = str(rgb_path.relative_to(self.episode_dir))
        if depth is not None:
            depth_arr = np.ascontiguousarray(np.asarray(depth, dtype=np.float32))
            record["depth_shape"] = list(depth_arr.shape)
            record["depth_sha256"] = hashlib.sha256(depth_arr.tobytes()).hexdigest()
            finite = depth_arr[np.isfinite(depth_arr)]
            record["depth_valid_count"] = int(finite.size)
            if finite.size:
                record["depth_min_m"] = float(np.min(finite))
                record["depth_max_m"] = float(np.max(finite))
            if self.save_depth:
                depth_path = self.episode_dir / "depth" / f"{prefix}.npz"
                np.savez_compressed(depth_path, depth_m=depth_arr)
                record["depth_path"] = str(depth_path.relative_to(self.episode_dir))
        self._write_line(self._obs_file, record)
        self._records += 1
        return record

    def record_query(self, *, step_id: Any, query_id: Any, output: Any, pixel_goal: Any = None,
                     input_steps: Any = None, semantic_state: Optional[Dict[str, Any]] = None,
                     prompt_fingerprint: Any = None) -> None:
        if not self.enabled or self._query_file is None:
            return
        if self.max_queries > 0 and self._queries >= self.max_queries:
            return
        self._write_line(self._query_file, {
            "event_type": "replay_query",
            **self._record_meta,
            "query_index": int(self._queries),
            "query_id": query_id,
            "step_id": None if step_id is None else int(step_id),
            "output": output,
            "pixel_goal": pixel_goal,
            "input_steps": input_steps,
            "semantic_state": semantic_state or {},
            "prompt_fingerprint": prompt_fingerprint,
            "observation_key": self._last_observation_key,
        })
        self._queries += 1

    def record_action(self, *, step_id: Any, action: Any, action_source: Any,
                      pre_safety_action: Any, action_applied: bool,
                      safety_decision: Optional[Dict[str, Any]] = None,
                      audit_metrics: Optional[Dict[str, Any]] = None,
                      next_observation_step_id: Any = None) -> None:
        if not self.enabled or self._action_file is None:
            return
        if self.max_actions > 0 and self._actions >= self.max_actions:
            return
        self._write_line(self._action_file, {
            "event_type": "replay_action",
            **self._record_meta,
            "action_index": int(self._actions),
            "step_id": None if step_id is None else int(step_id),
            "action": action,
            "action_source": action_source,
            "pre_safety_action": pre_safety_action,
            "action_applied": bool(action_applied),
            "next_observation_step_id": next_observation_step_id,
            "safety_decision": safety_decision or {},
            "audit_metrics": audit_metrics or {},
            "observation_key": self._last_observation_key,
        })
        self._actions += 1

    def finish_episode(self, **summary: Any) -> Dict[str, Any]:
        result = {
            "event_type": "replay_episode_summary",
            **self._episode_meta,
            "observation_count": int(self._records),
            "query_count": int(self._queries),
            "action_count": int(self._actions),
            **summary,
        }
        if self.enabled and self.episode_dir is not None:
            self._write_json(self.episode_dir / "summary.json", result)
        self.close()
        return result

    def close(self) -> None:
        for handle_name in ("_obs_file", "_query_file", "_action_file"):
            handle = getattr(self, handle_name)
            if handle is not None:
                handle.close()
                setattr(self, handle_name, None)
        self.episode_dir = None
