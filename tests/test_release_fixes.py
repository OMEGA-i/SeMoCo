"""Checks for the released configuration and Omega joint recovery."""
from pathlib import Path

import numpy as np
import torch
import yaml

from semoco.data.omega import canonical_anchor, recover_joints
from semoco.generator.ar import MotionGPT, MotionGPTConfig


ROOT = Path(__file__).parents[1]


def test_released_ar_configs_load_a_numeric_norm_epsilon():
    for name in ("humanml3d_ar.yaml", "omega_ar.yaml"):
        raw = yaml.safe_load((ROOT / "configs" / name).read_text())["model"]
        config = MotionGPTConfig(**raw)
        assert isinstance(config.norm_eps, float)
        MotionGPT(config)


def test_omega_recovery_places_the_child_on_the_template_offset(tmp_path):
    parents = np.full(77, 0, dtype=np.int64)
    parents[0] = -1
    offsets = np.zeros((77, 3), dtype=np.float32)
    offsets[1] = (0.0, 1.0, 0.0)
    template = tmp_path / "soma77.npz"
    np.savez(template, parents=parents, offsets=offsets)
    features = np.zeros((1, 499), dtype=np.float32)
    eye = canonical_anchor()["init_root_rot6d"]
    features[0, 9:15] = eye
    features[0, 15:471] = np.tile(eye, 76)
    features[0, 4] = 0.5
    joints = recover_joints(features, template)
    assert joints.shape == (1, 77, 3)
    np.testing.assert_allclose(joints[0, 0], (0.0, 0.5, 0.0), atol=1e-5)
    np.testing.assert_allclose(joints[0, 1], (0.0, 1.5, 0.0), atol=1e-5)
    torch.testing.assert_close(torch.tensor(joints[0, 2]), torch.tensor([0.0, 0.5, 0.0]), atol=1e-5, rtol=0)
