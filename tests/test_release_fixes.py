"""Checks for the released configuration and Omega joint recovery."""
from pathlib import Path

import numpy as np
import torch
import yaml

from semoco.data.omega import canonical_anchor, recover_joints
from semoco.generator.sentence import SentenceEncoder
from semoco.generator.ar import MotionGPT, MotionGPTConfig
from semoco.generator.flow import PartFlow
from semoco.generator.pipeline import generate
from semoco.tokenizer.codec import CodecConfig, MotionTokenizer


ROOT = Path(__file__).parents[1]


def test_sentence_encoder_reloads_its_mean_token(tmp_path):
    model = SentenceEncoder(32, layers=1, width=16, heads=4, feedforward=32).eval()
    state = {key.replace("sequence.", "seqTransEncoder."): value
             for key, value in model.state_dict().items() if "encoding" not in key}
    path = tmp_path / "text_encoder.pt"
    torch.save(state, path)
    loaded = SentenceEncoder.load(path)
    tokens = torch.randn(2, 4, 32)
    valid = torch.tensor([[True, True, False, False], [True, True, True, True]])
    torch.testing.assert_close(model.encode(tokens, valid), loaded.encode(tokens, valid))


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


def test_encode_text_supports_both_768d_humanml3d_and_2048d_omega_encoders(tmp_path, monkeypatch):
    from semoco.cli import encode_text
    from semoco.generator.pipeline import append_retrieval_token

    class FakeFlan:
        def __init__(self, *args, **kwargs):
            pass

        def token_states(self, captions):
            width = max(len(c) for c in captions)
            out = torch.ones(len(captions), width, 2048)
            valid = torch.zeros(len(captions), width, dtype=torch.bool)
            for i, c in enumerate(captions):
                valid[i, :len(c)] = True
            return out, valid

        def pack(self, hidden, valid, sentence):
            rows = [append_retrieval_token(hidden[i:i + 1, valid[i]], sentence[i:i + 1])[0]
                    for i in range(hidden.shape[0])]
            width = max(map(len, rows))
            text = hidden.new_zeros((len(rows), width, 2048))
            mask = torch.zeros(len(rows), width, dtype=torch.bool)
            for i, row in enumerate(rows):
                text[i, -len(row):] = row
                mask[i, -len(row):] = True
            return text, mask

    class FakeBert:
        def __init__(self, *args, **kwargs):
            pass

        def token_states(self, captions):
            width = max(len(c) for c in captions)
            out = torch.full((len(captions), width, 768), 0.5)
            valid = torch.ones(len(captions), width, dtype=torch.bool)
            return out, valid

    monkeypatch.setattr(encode_text, "TextConditioner", FakeFlan)
    monkeypatch.setattr(encode_text, "_BackboneTextEncoder", FakeBert)

    import json
    prompts = [{"caption": "walk", "frames": 40}, {"caption": "jump up", "frames": 80}]
    prompt_file = tmp_path / "prompts.json"
    prompt_file.write_text(json.dumps(prompts))

    for name, feat_dim in (("humanml3d", 768), ("omega", 2048)):
        enc = SentenceEncoder(feat_dim, layers=1, width=256, heads=4, feedforward=64).eval()
        ckpt = tmp_path / f"{name}_text_encoder.pt"
        torch.save(enc.state_dict(), ckpt)
        out_npz = tmp_path / f"{name}_inputs.npz"
        encode_text.main([
            "--dataset", name, "--text-encoder", str(ckpt),
            "--prompts", str(prompt_file), "--output", str(out_npz), "--device", "cpu",
        ])
        with np.load(out_npz, allow_pickle=False) as loaded:
            assert loaded["text"].shape == (2, 8, 2048)
            assert loaded["text_valid"].shape == (2, 8)
            assert loaded["frames"].tolist() == [40, 80]
            last_token = loaded["text"][:, -1, :]
            assert np.allclose(np.linalg.norm(last_token[:, :256], axis=-1), 8.0, atol=1e-4)
            assert np.allclose(last_token[:, 256:], 0.0)


def test_omega_export_recovers_joints_automatically_and_tmr186_canonicalizes(tmp_path):
    from dataclasses import asdict
    import json
    from semoco.cli.export import main as export_cli
    from semoco.evaluation.omega_v2 import soma77_to_tmr186

    cfg = CodecConfig(input_dim=499, latent_dim=16, depth=1, dropout=0., codebook_size=16,
                      semantic_codebook_size=16, semantic_dim=8)
    codec = MotionTokenizer(cfg).eval()
    torch.save(dict(format="semoco.tokenizer.v1", config=asdict(cfg), model=codec.state_dict(),
                    norm=dict(mean=torch.zeros(499), std=torch.ones(499))), tmp_path / "omega_codec.pt")
    motion = np.zeros((16, 499), dtype=np.float32)
    eye = canonical_anchor()["init_root_rot6d"]
    motion[:, 9:15] = eye
    motion[:, 15:471] = np.tile(eye, 76)
    motion[:, 4] = 0.9
    np.save(tmp_path / "omega_motion.npy", motion)
    np.savez(tmp_path / "omega_text.npz", text=np.ones((3, 2048), dtype=np.float32))
    manifest = tmp_path / "omega_train.json"
    manifest.write_text(json.dumps([dict(motion="omega_motion.npy", text="omega_text.npz")]))
    export_cli(["--dataset", "omega", "--codec", str(tmp_path / "omega_codec.pt"),
                "--manifest", str(manifest), "--output", str(tmp_path / "omega_store"), "--device", "cpu"])
    targets = np.load(tmp_path / "omega_store" / "targets.npy")
    assert targets.shape == (4, 924) and np.isfinite(targets).all()

    joints = recover_joints(motion)
    features_186 = soma77_to_tmr186(joints)
    assert features_186.ndim == 2 and features_186.shape[1] == 186 and torch.isfinite(features_186).all()
    torch.testing.assert_close(features_186[0, [0, 2]], torch.zeros(2), atol=1e-5, rtol=0)
    torch.testing.assert_close(features_186[0, 3:5], torch.tensor([1.0, 0.0]), atol=1e-5, rtol=0)


def test_built_wheel_and_sdist_include_soma77_template(tmp_path):
    import os
    from pathlib import Path
    import shutil
    import tarfile
    import zipfile
    import setuptools.build_meta as build_meta

    root = Path(__file__).resolve().parents[1]
    old_cwd = Path.cwd()
    os.chdir(root)
    try:
        wheel_name = build_meta.build_wheel(str(tmp_path))
        sdist_name = build_meta.build_sdist(str(tmp_path))
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(root / "build", ignore_errors=True)
    with zipfile.ZipFile(tmp_path / wheel_name) as zf:
        assert "semoco/assets/soma77_template.npz" in zf.namelist()
    with tarfile.open(tmp_path / sdist_name) as tf:
        assert any(name.endswith("src/semoco/assets/soma77_template.npz") for name in tf.getnames())

