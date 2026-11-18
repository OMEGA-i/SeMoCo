"""Verify model-only exports and strict active-parameter compatibility."""
from dataclasses import asdict
import json
import numpy as np
import pytest
import torch

from semoco.tokenizer.codec import CodecConfig, MotionTokenizer, load_codec
from semoco.generator.ar import MotionGPT, MotionGPTConfig
from semoco.generator.ar.checkpoint import read_semantic_ar, load_semantic_ar
from semoco.generator.flow import PartFlow
from semoco.cli.export_checkpoint import main as export_checkpoint
from semoco.cli.generate import main as generate_cli
from semoco.cli.tokenize import main as tokenize_cli


def tiny_codec():
    return MotionTokenizer(CodecConfig(latent_dim=16, depth=1, dropout=0., codebook_size=16,
                                       semantic_codebook_size=16, semantic_dim=8)).eval()


def tiny_ar():
    return MotionGPT(MotionGPTConfig(d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, ffn_hidden=64,
                                     dropout=0., codebook_size=16, clip_dim=7, max_seq_len=32, len_max=8)).eval()


def test_semantic_export_removes_only_unused_predictor_and_private_metadata(tmp_path):
    model = tiny_ar()
    state = model.state_dict()
    state["decoder.code_blocks.0.unused.weight"] = torch.randn(5, 5)
    config = {**asdict(model.cfg), "code_pred_layers": 5, "depth_inject": 0, "loss_weights": [1.]}
    source, clean = tmp_path / "training.pt", tmp_path / "clean.pt"
    private_path = str(tmp_path / "private_training")
    torch.save(dict(model=state, model_config=config, source=private_path, optimizer={"private": private_path}), source)
    export_checkpoint(["ar", "--dataset", "humanml3d", "--input", str(source), "--output", str(clean)])
    result = torch.load(clean, weights_only=True)
    assert set(result) == {"format", "model_config", "model"}
    assert not any("code_blocks" in key for key in result["model"])
    assert private_path not in json.dumps(result["model_config"])
    loaded = load_semantic_ar(clean)
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, loaded.state_dict()[key], atol=0, rtol=0)
    result["model"]["unexpected.weight"] = torch.ones(1)
    with pytest.raises(RuntimeError):
        read_semantic_ar(result)


def test_exported_tokenizer_preserves_codes_and_decoder(tmp_path):
    model = tiny_codec()
    path, output = tmp_path / "tokenizer.pt", tmp_path / "exported.pt"
    torch.save(dict(format="semoco.tokenizer.v1", config=asdict(model.cfg), model=model.state_dict(),
                    norm=dict(mean=torch.zeros(263), std=torch.ones(263)), private_source=str(tmp_path)), path)
    export_checkpoint(["tokenizer", "--dataset", "humanml3d", "--input", str(path), "--output", str(output)])
    loaded, norms = load_codec(output, dataset="humanml3d", device="cpu")
    features = torch.randn(1, 16, 263)
    torch.testing.assert_close(model.encode(features), loaded.encode(features), atol=0, rtol=0)
    codes = model.encode(features)
    torch.testing.assert_close(model.decode(codes), loaded.decode(codes), atol=0, rtol=0)
    assert norms[0].shape == (263,)


def test_tokenizer_encode_and_decode_entry_points(tmp_path):
    model = tiny_codec()
    path = tmp_path / "tokenizer.pt"
    torch.save(dict(format="semoco.tokenizer.v1", config=asdict(model.cfg), model=model.state_dict(),
                    norm=dict(mean=torch.zeros(263), std=torch.ones(263))), path)
    x = np.random.default_rng(4).normal(size=(8, 263)).astype("float32")
    np.save(tmp_path / "motion.npy", x)
    shared = ["--dataset", "humanml3d", "--codec", str(path), "--device", "cpu"]
    tokenize_cli(["encode", *shared, "--input", str(tmp_path / "motion.npy"), "--output", str(tmp_path / "codes.npz")])
    tokenize_cli(["decode", *shared, "--input", str(tmp_path / "codes.npz"), "--output", str(tmp_path / "motion.npz")])
    with np.load(tmp_path / "codes.npz") as codes, np.load(tmp_path / "motion.npz") as motion:
        assert codes["codes"].shape == (2, 8)
        expected = model.decode(torch.tensor(codes["codes"])[None])[0].detach().numpy()
        np.testing.assert_array_equal(expected, motion["features"])
        np.testing.assert_array_equal(motion["features"], motion["physical_features"])


def test_full_generation_cli_runs_from_a_standalone_checkpoint_bundle(tmp_path):
    codec = tiny_codec()
    flow = PartFlow(width=16, depth=1, heads=4, ydim=16, text_dim=7, q0_vocab=16, max_t=8).eval()
    ar = tiny_ar()
    torch.save(dict(format="semoco.tokenizer.v1", config=asdict(codec.cfg), model=codec.state_dict()), tmp_path / "codec.pt")
    torch.save(dict(model_config=asdict(ar.cfg), model=ar.state_dict()), tmp_path / "ar.pt")
    torch.save(dict(ema=flow.state_dict(), config={"model": {"heads": 4}}), tmp_path / "flow.pt")
    np.savez(tmp_path / "stats.npz", mean=np.zeros(16), std=np.ones(16))
    np.savez(tmp_path / "text.npz", text=np.ones((2, 3, 7)),
             text_valid=np.ones((2, 3), dtype=bool), frames=np.array([8, 12], dtype=np.int64))
    args = ["--dataset", "humanml3d", "--device", "cpu", "--steps", "2"]
    for key, value in {"codec": "codec.pt", "ar": "ar.pt", "flow": "flow.pt", "kin-stats": "stats.npz",
                       "inputs": "text.npz", "output": "generated.npz"}.items():
        args += ["--" + key, str(tmp_path / value)]
    generate_cli(args)
    preset = {"dataset": "humanml3d", "codec": str(tmp_path / "codec.pt"), "ar": str(tmp_path / "ar.pt"),
              "flow": str(tmp_path / "flow.pt"), "kin_stats": str(tmp_path / "stats.npz"), "steps": 2}
    config = tmp_path / "inference.yaml"
    config.write_text(json.dumps(preset))
    generate_cli(["--config", str(config), "--device", "cpu", "--inputs", str(tmp_path / "text.npz"),
                  "--output", str(tmp_path / "from_config.npz")])
    with np.load(tmp_path / "generated.npz", allow_pickle=False) as data:
        assert data["features"].shape == (20, 263)
        assert data["offsets"].tolist() == [0, 8, 20]
        assert data["q0"].shape == (2, 3)
        assert np.isfinite(data["features"]).all()
        with np.load(tmp_path / "from_config.npz", allow_pickle=False) as configured:
            np.testing.assert_array_equal(data["features"], configured["features"])


def test_kv_cached_semantic_predictions_match_teacher_forcing():
    model = tiny_ar()
    text = torch.randn(2, 3, 7)
    mask = torch.tensor([[False, True, True], [True, True, True]])
    codes = torch.randint(0, 16, (2, 4, 1))
    valid = torch.ones(2, 4, dtype=torch.bool)
    expected, _ = model.t2m_train_step(text, mask, codes, valid)
    words = model.project_text(text)
    lengths = valid.sum(1)
    bos = model.motion_bos.expand(2, 1, -1) + model.length_embedding(lengths, 1)
    roles = torch.zeros(2, 4, dtype=torch.long); roles[:, -1] = 1
    keys = torch.cat([mask, torch.ones(2, 1, dtype=torch.bool)], 1)
    cache, hidden = model.forward_embeds(torch.cat([words, bos], 1),
        attn_mask=model.mixed_attn_mask(roles, keys), caches=model.init_caches(2, "cpu", words.dtype), run_decoder=False)
    for position in range(4):
        torch.testing.assert_close(model.decode_packet(hidden[:, -1]), expected[0][:, position], atol=1e-6, rtol=1e-5)
        if position == 3:
            break
        keys = torch.cat([keys, torch.ones(2, 1, dtype=torch.bool)], 1)
        embedding = model.embed(codes[:, position:position+1]) + model.length_embedding(lengths, 1, position+1)
        cache, hidden = model.forward_embeds(embedding, attn_mask=keys[:, None, None], caches=cache,
                                              start_pos=4+position, run_decoder=False)
