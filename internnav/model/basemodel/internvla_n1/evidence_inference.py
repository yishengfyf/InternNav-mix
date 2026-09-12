from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from .evidence_history import build_causal_history_metadata
from .evidence_sequence import prepare_conditioned_sequences
from .internvla_n1 import EVIDENCE_TOKEN_INDEX, TRAJ_TOKEN_INDEX


EVIDENCE_ADAPTER_PREFIXES = (
    "model.task_state_estimator.",
    "model.evidence_memory.",
    "model.cond_projector.",
    "model.latent_queries",
)

TASK_SPATIAL_INFERENCE_CONFIG = {
    "use_evidence_memory": True,
    "evidence_task_dim": 512,
    "evidence_bottleneck_dim": 512,
    "num_evidence_tokens": 4,
    "evidence_num_heads": 8,
    "evidence_num_stages": 4,
    "evidence_dropout": 0.0,
    "evidence_gradient_bypass": True,
    "evidence_gradient_bypass_mode": "cross_attention",
    "evidence_gradient_bypass_scale": 0.5,
    "evidence_ablation": "task_spatial",
    "evidence_task_state_scale": 1.0,
    "evidence_normalize_task_state": False,
}


def configure_task_spatial_inference(config):
    """Apply the exact architecture used by the selected N2 M2-lr01 adapter."""
    for key, value in TASK_SPATIAL_INFERENCE_CONFIG.items():
        setattr(config, key, value)
    return config


def load_evidence_adapter(model, adapter_path: str | Path) -> dict:
    """Load a training-produced adapter only when every expected key matches."""
    adapter_path = Path(adapter_path)
    if not adapter_path.is_file():
        raise FileNotFoundError(f"evidence adapter does not exist: {adapter_path}")
    try:
        adapter_state = torch.load(adapter_path, map_location="cpu", weights_only=True)
    except TypeError:
        adapter_state = torch.load(adapter_path, map_location="cpu")
    if not isinstance(adapter_state, Mapping) or not adapter_state:
        raise ValueError("evidence adapter must be a non-empty state dictionary")

    model_state = model.state_dict()
    expected_keys = {
        key for key in model_state if any(key == prefix or key.startswith(prefix) for prefix in EVIDENCE_ADAPTER_PREFIXES)
    }
    adapter_keys = set(adapter_state)
    missing = sorted(expected_keys - adapter_keys)
    unexpected = sorted(adapter_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(
            "adapter state does not exactly match the inference modules: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    required_groups = {
        "task_state_estimator": "model.task_state_estimator.",
        "evidence_memory": "model.evidence_memory.",
        "cond_projector": "model.cond_projector.",
        "latent_queries": "model.latent_queries",
    }
    absent_groups = [name for name, prefix in required_groups.items() if not any(k.startswith(prefix) for k in adapter_keys)]
    if absent_groups:
        raise ValueError(f"adapter is missing required parameter groups: {absent_groups}")

    incompatible = model.load_state_dict(adapter_state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"unexpected adapter keys after loading: {incompatible.unexpected_keys}")
    return {
        "adapter_path": str(adapter_path),
        "parameter_tensors": len(adapter_keys),
        "parameter_elements": sum(int(tensor.numel()) for tensor in adapter_state.values()),
        "loaded_keys": sorted(adapter_keys),
    }


def planar_pose_from_habitat_observation(observation: Mapping) -> np.ndarray:
    """Read the causal ideal GPS/compass observation as planar x, y, yaw."""
    if "gps" not in observation or "compass" not in observation:
        raise ValueError("evidence inference requires Habitat gps and compass observations")
    gps = np.asarray(observation["gps"], dtype=np.float32).reshape(-1)
    compass = np.asarray(observation["compass"], dtype=np.float32).reshape(-1)
    if gps.size != 2 or compass.size != 1 or not np.isfinite(gps).all() or not np.isfinite(compass).all():
        raise ValueError("Habitat gps/compass must be finite with shapes [2] and [1]")
    return np.asarray([gps[0], gps[1], compass[0]], dtype=np.float32)


def prepare_evidence_inputs(
    model,
    inputs,
    history_ids: Sequence[int],
    planar_poses: Sequence[np.ndarray],
    image_count: int,
    device: torch.device | str,
):
    """Insert evidence placeholders and construct causal online metadata for one sample."""
    if not getattr(model.config, "use_evidence_memory", False):
        return inputs, {}
    if inputs.input_ids.shape[0] != 1:
        raise ValueError("online evidence preparation currently supports batch size one")
    if image_count < len(history_ids) + 1:
        raise ValueError("input images must contain selected history followed by a current image")
    if len(planar_poses) == 0:
        raise ValueError("current causal pose is missing")

    input_ids = inputs.input_ids[0]
    image_mask = input_ids.eq(model.config.image_token_id)
    previous_is_image = torch.roll(image_mask, 1)
    previous_is_image[0] = False
    image_block_starts = torch.nonzero(image_mask & ~previous_is_image, as_tuple=False).flatten()
    if len(image_block_starts) <= len(history_ids):
        raise ValueError("cannot locate the current observation image block during evidence prefill")
    insert_position = max(0, int(image_block_starts[len(history_ids)].item()) - 1)
    conditioned = prepare_conditioned_sequences(
        input_ids=(input_ids,),
        labels=(torch.full_like(input_ids, -100),),
        evidence_insert_positions=(insert_position,),
        evidence_token_id=EVIDENCE_TOKEN_INDEX,
        trajectory_token_id=TRAJ_TOKEN_INDEX,
        num_evidence_tokens=model.config.num_evidence_tokens,
        num_trajectory_tokens=0,
    )
    inputs["input_ids"] = conditioned.input_ids[0].unsqueeze(0)
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    poses = np.asarray(planar_poses, dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] != 3:
        raise ValueError("planar pose history must have shape [frames, 3]")
    metadata = build_causal_history_metadata(
        history_ids,
        poses[:, :2],
        poses[:, 2],
        current_frame_id=len(poses) - 1,
    )
    target_device = torch.device(device)
    evidence_kwargs = {
        "evidence_relative_poses": torch.from_numpy(metadata.relative_poses).unsqueeze(0).to(target_device),
        "evidence_ages": torch.from_numpy(metadata.ages).unsqueeze(0).to(target_device),
        "evidence_qualities": torch.from_numpy(metadata.qualities).unsqueeze(0).to(target_device),
        "evidence_valid_mask": torch.ones((1, len(history_ids)), dtype=torch.bool, device=target_device),
        "evidence_history_counts": torch.tensor([len(history_ids)], device=target_device),
        "evidence_image_counts": torch.tensor([image_count], device=target_device),
        "evidence_prompt_lengths": torch.tensor([inputs["input_ids"].shape[1]], device=target_device),
    }
    return inputs, evidence_kwargs
