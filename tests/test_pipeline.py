"""Semantic motion generation contracts."""

import json
from dataclasses import asdict

import numpy as np
import pytest
import torch
from torch import nn

from semoco.tokenizer.codec import CodecConfig, MotionTokenizer
from semoco.generator.flow import PartFlow
from semoco.generator.parts import PART_DIMS, pack_part_targets
from semoco.generator.pipeline import generate, load_semantic_ar, motion_lengths, sample_q0
from semoco.evaluation.score import score_embeddings
from semoco.training.flow import main as train_flow
from semoco.generator.ar import MotionGPT, MotionGPTConfig


@pytest.mark.parametrize("dataset,joints,anchor,dims", [
    ("humanml3d", 22, 0, (36, 60, 48, 48, 36, 36)),
    ("omega", 77, 1, (36, 120, 336, 336, 48, 48)),
])
def test_auxiliary_target_uses_framewise_hips_and_the_correct_trajectory_anchor(dataset, joints, anchor, dims):
    reference = torch.arange(joints).view(1, joints, 1) * torch.ones(1, 1, 3)
    poses = reference + torch.arange(8 + anchor).view(-1, 1, 1)
    packed = pack_part_targets(poses, dataset)
    assert PART_DIMS[dataset] == dims
    assert packed.shape == (2, sum(dims))
    root = packed[:, :36].reshape(8, 3, 3)
    torch.testing.assert_close(root[:, 0], (67 if anchor else 1) * torch.ones(8, 3))
    torch.testing.assert_close(root[:, 2], torch.arange(anchor, 8 + anchor).float()[:, None].expand(-1, 3))


def test_factorized_flow_ignores_motion_padding_and_uses_shared_weights():
    torch.manual_seed(17)
    model = PartFlow(width=16, heads=4, depth=2, ydim=5, text_dim=7, max_t=8).eval()
    for block in model.blocks:
        nn.init.normal_(block.ada[-1].weight, std=.15)
    nn.init.normal_(model.out.weight)
    valid = torch.tensor([[True, True, False, False], [True] * 4])
    words = torch.randn(2, 3, 7)
    text, pad = model.encode_text(words, torch.ones(2, 3, dtype=torch.bool))
    x = torch.randn(2, 4, 5)
    q0 = torch.randint(0, 1024, (2, 4))
    time = torch.tensor([.2, .7])
    first = model(x, time, q0, valid, text, pad)
    x[0, 2:] = 419
    q0[0, 2:] = 512
    second = model(x, time, q0, valid, text, pad)
    torch.testing.assert_close(first[valid], second[valid])
    assert not any("spatial" in key or "temporal" in key for key in model.state_dict())


def tiny_ar():
    config = MotionGPTConfig(
        num_codebooks=1, codebook_size=16, use_text=True, clip_dim=7,
        d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=64,
        context_length=16, max_seq_len=16, len_cond=1, len_max=8,
    )
    return MotionGPT(config).eval()


def test_q0_checkpoint_and_reference_duration_rollout(tmp_path):
    source = tiny_ar()
    cfg = dict(asdict(source.cfg), depth_inject=0)
    path = tmp_path / "ar.pt"
    torch.save(dict(model=source.state_dict(), model_config=cfg), path)
    net = load_semantic_ar(path, "cpu")
    words = torch.randn(2, 3, 7)
    mask = torch.tensor([[False, True, True], [True, True, True]])
    lengths = motion_lengths(torch.tensor([11, 16]), "humanml3d")
    assert lengths.tolist() == [2, 4]
    assert motion_lengths(torch.tensor([11, 16]), "omega").tolist() == [3, 4]
    result = sample_q0(net, words, mask, lengths, temperature=0.)
    assert result.shape == (2, 4) and result.min() >= 0 and result.max() < 16
    scores, _ = net.t2m_train_step(words, mask, result[..., None], torch.arange(4)[None] < lengths[:, None])
    assert scores[0].shape == (2, 4, 16)


class _SemanticEmbedding(nn.Embedding):
    def dequantize(self, indices):
        return self(indices)


class _RepeatDecoder(nn.Module):
    def forward(self, x):
        return {"features": x.repeat_interleave(4, dim=-1)}


def test_generated_latent_reaches_codec_without_kinematic_requantization():
    flow = PartFlow(width=16, heads=4, depth=1, ydim=5, text_dim=7, max_t=8, dataset="omega").eval()
    codec = nn.Module()
    codec.quantizer = nn.Module()
    quantizer = codec.quantizer
    quantizer.num_quantizers = 8
    quantizer.in_proj_kin = nn.Identity()
    quantizer.out_proj_kin = nn.Identity()
    quantizer.out_proj_sem = nn.Identity()
    quantizer.semantic_vq = _SemanticEmbedding(1024, 5)
    codec.decoder = _RepeatDecoder()
    codec.eval()
    q0 = torch.tensor([[0, 1], [2, 3]])
    frames = torch.tensor([5, 8])
    clips, plan, raw = generate(
        ar=None, flow=flow, codec=codec, text=torch.rand(2, 3, 7),
        text_valid=torch.ones(2, 3, dtype=torch.bool), q0=q0, frames=frames,
        latent_mean=torch.ones(5), latent_std=torch.full((5,), 2.), steps=2,
    )
    assert torch.equal(plan, q0)
    expected_noise = torch.randn((2, 2, 5), generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(raw, expected_noise * 2 + 1)
    for i, clip in enumerate(clips):
        torch.testing.assert_close(clip, (raw[i] + quantizer.semantic_vq(q0[i])).repeat_interleave(4, 0)[:frames[i]])


def test_train_resume_matches_uninterrupted_step_and_loads_ema(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    rng = np.random.default_rng(1)
    rows = [{"code_start": 2*i, "code_len": 2} for i in range(4)]
    text = [{"text_start": 3*i, "text_len": 3} for i in range(4)]
    (store / "train.index.json").write_text(json.dumps(rows))
    (store / "train.text_index.flantmr.json").write_text(json.dumps(text))
    np.save(store / "train.codes.npy", rng.integers(0, 1024, (8, 1), dtype=np.int16))
    np.save(store / "train.text_emb.flantmr.npy", rng.normal(size=(12, 7)).astype("float32"))
    np.save(store / "train.kin_latent.npy", rng.normal(size=(8, 5)).astype("float32"))
    np.save(store / "targets.npy", rng.normal(size=(8, 264)).astype("float32"))
    np.savez(store / "kin_stats.npz", mean=np.zeros(5), std=np.ones(5))
    np.savez(store / "targets_norm.npz", mean=np.zeros(264), std=np.ones(264))
    base = ["--recipe", "hml-base", "--device", "cpu", "--store", str(store), "--targets", str(store),
            "--width", "16", "--depth", "1", "--heads", "4", "--global-batch", "4", "--microbatch", "2"]
    short, full = (tmp_path / name for name in ("resumed", "full"))
    train_flow(base + ["--output", str(short), "--steps", "1"])
    train_flow(base + ["--output", str(short), "--steps", "2", "--resume"])
    train_flow(base + ["--output", str(full), "--steps", "2"])
    get = lambda path: PartFlow.load_ema(path / "ema_000002.pt", dataset="humanml3d", device="cpu")
    for a, b in zip(get(short).parameters(), get(full).parameters(), strict=True):
        assert torch.equal(a, b)


def test_scoring_uses_full_panel_for_fid_and_discards_only_retrieval_remainder():
    rng = np.random.default_rng(7)
    paired = rng.normal(size=(34, 5)).astype("float32")
    paired /= np.linalg.norm(paired, axis=1, keepdims=True)
    result = score_embeddings(paired, paired, paired, dataset="omega")
    assert abs(result["FID"]) < 1e-9 and result["MM-Dist"] == 0.
    assert result["R@1"] == 1. and result["samples"] == 34 and result["retrieval_samples"] == 32
