"""Check physical recovery and the motion/text cache preparation entry points."""
from dataclasses import asdict
import json

import numpy as np
import torch

from semoco.cli.export import main as export
from semoco.cli import encode_text
from semoco.data.humanml3d import recover_joints
from semoco.generator.parts import pack_part_targets
from semoco.tokenizer.codec import CodecConfig, MotionTokenizer, encode_flow_targets, decode_continuous


def test_humanml_recovery_matches_quaternion_ric_convention():
    data = torch.randn(2, 12, 263, generator=torch.Generator().manual_seed(4))
    angle = torch.zeros_like(data[..., 0])
    angle[..., 1:] = data[..., :-1, 0]
    angle = angle.cumsum(-1)
    q = torch.zeros(*data.shape[:-1], 4)
    q[..., 0], q[..., 2] = angle.cos(), -angle.sin()
    def rotate(q, v):
        uv = torch.cross(q[..., 1:], v, dim=-1)
        uuv = torch.cross(q[..., 1:], uv, dim=-1)
        return v + 2 * (q[..., :1] * uv + uuv)
    root = torch.zeros(*data.shape[:-1], 3)
    root[..., 1:, 0] = data[..., :-1, 1]
    root[..., 1:, 2] = data[..., :-1, 2]
    root = rotate(q, root).cumsum(-2)
    root[..., 1] = data[..., 3]
    local = data[..., 4:67].reshape(2, 12, 21, 3)
    local = rotate(q[..., None, :].expand(2, 12, 21, 4), local)
    local[..., 0] += root[..., 0:1]
    local[..., 2] += root[..., 2:3]
    expected = torch.cat((root[..., None, :], local), -2)
    torch.testing.assert_close(recover_joints(data), expected, atol=2e-6, rtol=2e-6)


def test_humanml_export_recovers_auxiliary_targets_without_a_joint_cache(tmp_path):
    cfg = CodecConfig(latent_dim=16, depth=1, dropout=0., codebook_size=16,
                      semantic_codebook_size=16, semantic_dim=8)
    codec = MotionTokenizer(cfg).eval()
    torch.save(dict(format="semoco.tokenizer.v1", config=asdict(cfg), model=codec.state_dict(),
                    norm=dict(mean=torch.zeros(263), std=torch.ones(263))), tmp_path / "codec.pt")
    motion = np.random.default_rng(5).normal(size=(8, 263)).astype(np.float32)
    np.save(tmp_path / "motion.npy", motion)
    np.savez(tmp_path / "text.npz", text=np.ones((3, 2048), dtype=np.float32))
    manifest = tmp_path / "train.json"
    manifest.write_text(json.dumps([dict(motion="motion.npy", text="text.npz")]))
    export(["--dataset", "humanml3d", "--codec", str(tmp_path / "codec.pt"),
            "--manifest", str(manifest), "--output", str(tmp_path / "store"), "--device", "cpu"])
    with torch.inference_mode():
        q0, latent = encode_flow_targets(codec, torch.tensor(motion)[None])
        rec = decode_continuous(codec, q0, latent, torch.tensor([2]))[0]
        expected = pack_part_targets(recover_joints(rec), "humanml3d").numpy()
    np.testing.assert_array_equal(np.load(tmp_path / "store/targets.npy"), expected)
    np.testing.assert_array_equal(np.load(tmp_path / "store/train.codes.npy")[:, 0], q0[0])
    assert np.load(tmp_path / "store/train.kin_latent.npy").shape == (2, 16)


def test_text_cli_preserves_order_and_left_padding_across_batches(tmp_path, monkeypatch):
    class Encoder:
        def __init__(self, *args, **kwargs):
            pass
        def encode(self, captions, retrieval):
            width = max(len(s) for s in captions)
            text = torch.zeros(len(captions), width, 2048)
            valid = torch.zeros(len(captions), width, dtype=torch.bool)
            for i, s in enumerate(captions):
                text[i, -len(s):] = ord(s[0])
                valid[i, -len(s):] = True
            return text, valid
    monkeypatch.setattr(encode_text, "TextConditioner", Encoder)
    prompts = [dict(caption=s, frames=8+4*i) for i, s in enumerate(("a", "bbb", "cc"))]
    (tmp_path / "prompts.json").write_text(json.dumps(prompts))
    retrieval = np.zeros((3, 256), dtype=np.float32)
    retrieval[:, 0] = 1
    np.save(tmp_path / "tmr.npy", retrieval)
    encode_text.main(["--prompts", str(tmp_path / "prompts.json"), "--retrieval", str(tmp_path / "tmr.npy"),
                      "--output", str(tmp_path / "inputs.npz"), "--batch-size", "2", "--device", "cpu"])
    with np.load(tmp_path / "inputs.npz") as result:
        assert result["text_valid"].tolist() == [[False, False, True], [True]*3, [False, True, True]]
        assert result["frames"].tolist() == [8, 12, 16]
        for row, mask, prompt in zip(result["text"], result["text_valid"], prompts):
            assert (row[mask] == ord(prompt["caption"][0])).all()
