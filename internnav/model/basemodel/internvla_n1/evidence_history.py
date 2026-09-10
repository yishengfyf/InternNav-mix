from dataclasses import dataclass
from typing import Sequence

import numpy as np

from internnav.utils.geometry_utils import quat_to_euler_angles


@dataclass(frozen=True)
class CausalHistoryMetadata:
    frame_ids: np.ndarray
    relative_poses: np.ndarray
    ages: np.ndarray
    qualities: np.ndarray


def select_causal_history_ids(current_frame_id: int, max_history: int) -> list[int]:
    """Select a deterministic, uniformly spaced prefix without the current/future frame."""
    if current_frame_id < 0:
        raise ValueError("current_frame_id must be non-negative")
    if max_history < 0:
        raise ValueError("max_history must be non-negative")
    if current_frame_id == 0 or max_history == 0:
        return []
    count = min(current_frame_id, max_history)
    return np.unique(np.linspace(0, current_frame_id - 1, count, dtype=np.int64)).tolist()


def _wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def relative_pose_from_planar(
    history_positions: np.ndarray,
    history_yaws: np.ndarray,
    current_position: np.ndarray,
    current_yaw: float,
) -> np.ndarray:
    """Encode history poses in the current robot frame as dx, dy, sin(dyaw), cos(dyaw)."""
    history_positions = np.asarray(history_positions, dtype=np.float32)
    history_yaws = np.asarray(history_yaws, dtype=np.float32)
    current_position = np.asarray(current_position, dtype=np.float32)
    if history_positions.ndim != 2 or history_positions.shape[1] != 2:
        raise ValueError("history_positions must have shape [history, 2]")
    if history_yaws.shape != (history_positions.shape[0],):
        raise ValueError("history_yaws must have shape [history]")
    if current_position.shape != (2,):
        raise ValueError("current_position must have shape [2]")

    delta = history_positions - current_position[None, :]
    cosine = np.cos(current_yaw)
    sine = np.sin(current_yaw)
    world_to_current = np.asarray([[cosine, sine], [-sine, cosine]], dtype=np.float32)
    local_delta = delta @ world_to_current.T
    relative_yaw = _wrap_angle(history_yaws - current_yaw)
    return np.concatenate(
        (local_delta, np.sin(relative_yaw)[:, None], np.cos(relative_yaw)[:, None]), axis=1
    ).astype(np.float32)


def planar_pose_from_sim_observation(position: Sequence[float], quaternion_wxyz: Sequence[float]) -> tuple[np.ndarray, float]:
    """Convert causal simulator GPS/quaternion to a planar ideal-odometry pose."""
    position = np.asarray(position, dtype=np.float32)
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float32)
    if position.shape[0] < 2:
        raise ValueError("global position must contain at least x and y")
    if quaternion.shape != (4,):
        raise ValueError("global rotation must be a wxyz quaternion")
    yaw = float(quat_to_euler_angles(quaternion)[2])
    return position[:2], yaw


def planar_poses_from_transforms(transforms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract planar translation/yaw from homogeneous local-to-world transforms."""
    transforms = np.asarray(transforms, dtype=np.float32)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError("transforms must have shape [frames, 4, 4]")
    positions = transforms[:, :2, 3]
    yaws = np.arctan2(transforms[:, 1, 0], transforms[:, 0, 0])
    return positions.astype(np.float32), yaws.astype(np.float32)


def build_causal_history_metadata(
    frame_ids: Sequence[int],
    positions: np.ndarray,
    yaws: np.ndarray,
    current_frame_id: int,
    pose_quality: float = 1.0,
    observation_quality: float = 1.0,
) -> CausalHistoryMetadata:
    """Build observable metadata for selected prefix frames."""
    ids = np.asarray(frame_ids, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.float32)
    yaws = np.asarray(yaws, dtype=np.float32)
    if np.any(ids < 0) or np.any(ids >= current_frame_id):
        raise ValueError("history frame ids must belong to the strict causal prefix")
    if positions.shape != (len(yaws), 2):
        raise ValueError("positions and yaws must describe the same planar trajectory")
    if current_frame_id >= len(positions):
        raise ValueError("current_frame_id is outside the provided trajectory")
    if not (0.0 <= pose_quality <= 1.0 and 0.0 <= observation_quality <= 1.0):
        raise ValueError("qualities must be in [0, 1]")

    relative = relative_pose_from_planar(positions[ids], yaws[ids], positions[current_frame_id], yaws[current_frame_id])
    ages = (current_frame_id - ids).astype(np.float32)[:, None]
    qualities = np.tile(np.asarray([pose_quality, observation_quality], dtype=np.float32), (len(ids), 1))
    return CausalHistoryMetadata(ids, relative, ages, qualities)
