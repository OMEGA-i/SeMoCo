"""Recover SOMA77 joints from physical UMR499 features."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..tokenizer.geometry import SomaFK

_TEMPLATE = Path(__file__).resolve().parents[1] / "assets" / "soma77_template.npz"

NUM_JOINTS = 77
FEATURE_DIM = 499
_EYE6 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def canonical_anchor() -> dict[str, np.ndarray]:
    """Neutral text-to-motion seed: origin root and identity rotations."""
    return {
        "init_root_pos": np.zeros(3, dtype=np.float32),
        "init_root_rot6d": _EYE6.copy(),
        "init_joints76_rot6d": np.tile(_EYE6, (76, 1)),
    }


def _rot6d_to_matrix(value: np.ndarray) -> np.ndarray:
    first, second = value[..., :3], value[..., 3:]
    column_1 = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
    column_2 = second - np.sum(column_1 * second, axis=-1, keepdims=True) * column_1
    column_2 = column_2 / np.maximum(np.linalg.norm(column_2, axis=-1, keepdims=True), 1e-8)
    column_3 = np.cross(column_1, column_2, axis=-1)
    return np.stack((column_1, column_2, column_3), axis=-1).astype(np.float32)


def _root_translation(features: np.ndarray, root_rotation: np.ndarray, origin: np.ndarray) -> np.ndarray:
    forward = root_rotation[:, :, 2]
    yaw = np.arctan2(forward[:, 0], forward[:, 2])
    cosine, sine = np.cos(yaw), np.sin(yaw)
    local = np.zeros((len(features), 3), dtype=np.float32)
    local[:, 0] = features[:, 0]
    local[:, 2] = features[:, 1]
    from_local = np.stack((
        cosine * local[:, 0] + sine * local[:, 2],
        np.zeros(len(features), dtype=np.float32),
        -sine * local[:, 0] + cosine * local[:, 2],
    ), axis=-1)
    direct = np.zeros_like(from_local)
    direct[:, 0] = features[:, 2]
    direct[:, 2] = features[:, 3]
    delta = 0.5 * from_local + 0.5 * direct
    translation = np.empty((len(features) + 1, 3), dtype=np.float32)
    translation[0] = origin
    cumulative = np.cumsum(delta[:, [0, 2]], axis=0)
    translation[1:, 0] = origin[0] + cumulative[:, 0]
    translation[1:, 2] = origin[2] + cumulative[:, 1]
    translation[1:, 1] = features[:, 4]
    return translation


def recover_joints(
    features: np.ndarray,
    template: str | Path | None = None,
    anchor: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Convert physical UMR499 records ``[T, 499]`` to world joints ``[T, 77, 3]``.

    ``template`` defaults to the packaged SOMA77 parents and offsets. The
    returned frames are the transition records; the anchor pose is the pose
    before the first record.
    """
    template = _TEMPLATE if template is None else template
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM or not len(features):
        raise ValueError("expected nonempty physical UMR499 features [T, 499]")
    seed = canonical_anchor() if anchor is None else anchor
    rotations = np.empty((len(features) + 1, NUM_JOINTS, 6), dtype=np.float32)
    rotations[0, 0] = np.asarray(seed["init_root_rot6d"], dtype=np.float32).reshape(6)
    rotations[0, 1:] = np.asarray(seed["init_joints76_rot6d"], dtype=np.float32).reshape(76, 6)
    rotations[1:, 0] = features[:, 9:15]
    rotations[1:, 1:] = features[:, 15:471].reshape(len(features), 76, 6)
    root_rotation = _rot6d_to_matrix(rotations[1:, 0])
    origin = np.asarray(seed["init_root_pos"], dtype=np.float32).reshape(3)
    translation = _root_translation(features, root_rotation, origin)
    local = SomaFK(template)(torch.from_numpy(rotations)).detach().cpu().numpy()
    return (local + translation[:, None, :]).astype(np.float32)[1:]
