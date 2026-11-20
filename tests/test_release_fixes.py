"""Checks for the released configuration and Omega joint recovery."""
from pathlib import Path

import numpy as np
import torch
import yaml

from semoco.data.omega import canonical_anchor, recover_joints
from semoco.generator.ar import MotionGPT, MotionGPTConfig
from semoco.generator.flow import PartFlow
from semoco.generator.pipeline import generate
from semoco.tokenizer.codec import CodecConfig, MotionTokenizer


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
    packaged = recover_joints(features)
    assert packaged.shape == (1, 77, 3)


def test_later_batches_keep_the_seeded_ar_stream():
    codec = MotionTokenizer(CodecConfig(
        latent_dim=16, depth=1, dropout=0., codebook_size=16,
        semantic_codebook_size=16, semantic_dim=8,
    )).eval()
    flow = PartFlow(width=16, depth=1, heads=4, ydim=16, text_dim=7, q0_vocab=16, max_t=8).eval()
    ar = MotionGPT(MotionGPTConfig(
        d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=64, dropout=0.,
        codebook_size=16, clip_dim=7, max_seq_len=32, len_max=8,
    )).eval()
    text = torch.randn(4, 3, 7)
    mask = torch.ones(4, 3, dtype=torch.bool)
    frames = torch.tensor([8, 8, 12, 12])
    mean, std = torch.zeros(16), torch.ones(16)

    def run():
        torch.manual_seed(7)
        rng = torch.Generator().manual_seed(7)
        codes = []
        for start in (0, 2):
            _, q0, _ = generate(
                ar=ar, flow=flow, codec=codec, text=text[start:start + 2],
                text_valid=mask[start:start + 2], frames=frames[start:start + 2],
                latent_mean=mean, latent_std=std, seed=7, steps=2,
                generator=rng, reseed=False,
            )
            codes.append(q0.clone())
        return codes

    first, second = run(), run()
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    assert not torch.equal(first[0], first[1])
