"""Small CPU integration checks for the tokenizer and semantic training stages."""
import json
import numpy as np
import torch
import pytest
from semoco.training.tokenizer import main as train_tokenizer
from semoco.training.ar import main as train_ar
from semoco.tokenizer.codec import load_codec
from semoco.generator.ar.checkpoint import load_semantic_ar
from semoco.training.ar import learning_rate


def test_tokenizer_training_restores_optimizer_and_random_state(tmp_path, monkeypatch):
    rng = np.random.default_rng(12)
    np.save(tmp_path / "windows.npy", rng.normal(size=(4, 64, 263)).astype("float32"))
    np.save(tmp_path / "teacher.npy", rng.normal(size=(4, 8)).astype("float32"))
    np.savez(tmp_path / "stats.npz", mean=np.zeros(263), std=np.ones(263))
    cfg = dict(dataset="humanml3d", model=dict(latent_dim=16, depth=1, dropout=0., codebook_size=16,
            semantic_codebook_size=16, semantic_dim=8), training=dict(seed=19, steps=2, batch_size=2, save_every=1,
            lr=.0002, betas=[.9,.99], weight_decay=0., warmup_steps=2, semantic_weight=.15, semantic_warmup=2))
    path = tmp_path / "config.yaml"; path.write_text(json.dumps(cfg))
    args = ["--config", str(path), "--windows", str(tmp_path / "windows.npy"), "--teacher", str(tmp_path / "teacher.npy"),
            "--feature-stats", str(tmp_path / "stats.npz"), "--device", "cpu"]
    from semoco.training import tokenizer as trainer
    original = trainer.save
    def interrupted_save(path, state):
        original(path, state)
        if path.name == "last.pt" and state["step"] == 1:
            raise InterruptedError("simulated interruption after atomic checkpoint")
    monkeypatch.setattr(trainer, "save", interrupted_save)
    with pytest.raises(InterruptedError):
        train_tokenizer(args + ["--output", str(tmp_path / "run")])
    monkeypatch.setattr(trainer, "save", original)
    train_tokenizer(args + ["--output", str(tmp_path / "run"), "--resume"])
    train_tokenizer(args + ["--output", str(tmp_path / "continuous")])
    model, _ = load_codec(tmp_path / "run/model_000002.pt", dataset="humanml3d", device="cpu")
    reference, _ = load_codec(tmp_path / "continuous/model_000002.pt", dataset="humanml3d", device="cpu")
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, reference.state_dict()[key], atol=0, rtol=0)
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())


def test_semantic_training_can_load_its_inference_checkpoint(tmp_path):
    rng = np.random.default_rng(9)
    np.save(tmp_path / "train.codes.npy", rng.integers(0, 16, (8, 1), dtype=np.int16))
    np.save(tmp_path / "train.text_emb.flantmr.npy", rng.normal(size=(12, 7)).astype("float32"))
    (tmp_path / "train.index.json").write_text(json.dumps([dict(code_start=2*i, code_len=2) for i in range(4)]))
    (tmp_path / "train.text_index.flantmr.json").write_text(json.dumps([dict(text_start=3*i,text_len=3) for i in range(4)]))
    cfg = dict(model=dict(d_model=32,n_layers=1,n_heads=4,n_kv_heads=2,ffn_hidden=64,dropout=0.,
                codebook_size=16,clip_dim=7,max_seq_len=32,len_max=8),
               training=dict(seed=9,steps=1,batch_size=2,lr=.0003,warmup_steps=4))
    path=tmp_path / "config.yaml"; path.write_text(json.dumps(cfg))
    train_ar(["--config",str(path),"--store",str(tmp_path),"--output",str(tmp_path / "run"),"--device","cpu"])
    model=load_semantic_ar(tmp_path / "run/model_000001.pt")
    assert model.cfg.num_codebooks == 1
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())


def test_dataset_ar_learning_rate_matches_the_original_schedules():
    import math
    settings = dict(lr=3e-4, warmup_steps=4000, schedule_steps=40000, steps=10000,
                    min_lr_ratio=.1, schedule_offset=1)
    assert learning_rate(settings, 1) == pytest.approx(3e-4 / 4000)
    assert learning_rate(settings, 4001) == pytest.approx(3e-4)
    assert learning_rate(settings, 40001) == pytest.approx(3e-5)
    expected = 3e-4 * (.1 + .9 * .5 * (1 + math.cos(math.pi * (9999-4000)/36000)))
    assert learning_rate(settings, 10000) == pytest.approx(expected)
    settings.update(min_lr_ratio=0., schedule_offset=0)
    assert learning_rate(settings, 1) == pytest.approx(3e-4 / 4000)
    assert learning_rate(settings, 40000) == pytest.approx(0.)


def test_ar_proposal_continuation_updates_flow_and_keeps_ar_frozen(tmp_path, monkeypatch):
    from dataclasses import asdict
    from semoco.generator.ar import MotionGPT, MotionGPTConfig
    from semoco.generator.flow import PartFlow
    from semoco.training import flow as trainer
    rng = np.random.default_rng(3)
    np.save(tmp_path / "train.codes.npy", rng.integers(0,1024,(8,1),dtype=np.int16))
    np.save(tmp_path / "train.kin_latent.npy", rng.normal(size=(8,5)).astype("float32"))
    np.save(tmp_path / "train.text_emb.flantmr.npy", rng.normal(size=(12,7)).astype("float32"))
    np.save(tmp_path / "targets.npy", rng.normal(size=(8,264)).astype("float32"))
    np.savez(tmp_path / "kin_stats.npz",mean=np.zeros(5),std=np.ones(5))
    np.savez(tmp_path / "targets_norm.npz",mean=np.zeros(264),std=np.ones(264))
    (tmp_path / "train.index.json").write_text(json.dumps([dict(code_start=2*i,code_len=2) for i in range(4)]))
    (tmp_path / "train.text_index.flantmr.json").write_text(json.dumps([dict(text_start=3*i,text_len=3) for i in range(4)]))
    flow=PartFlow(width=16,depth=1,heads=4,ydim=5,text_dim=7)
    torch.save(dict(step=40000,ema=flow.state_dict()),tmp_path / "base.pt")
    ar=MotionGPT(MotionGPTConfig(d_model=32,n_layers=1,n_heads=4,n_kv_heads=2,ffn_hidden=64,
                                 clip_dim=7,max_seq_len=32,len_max=64,dropout=0.))
    torch.save(dict(model_config=asdict(ar.cfg),model=ar.state_dict()),tmp_path / "ar.pt")
    loaded=[]
    original=trainer.load_semantic_ar
    def track(path,device):
        model=original(path,device);loaded.append(model);return model
    monkeypatch.setattr(trainer,"load_semantic_ar",track)
    trainer.main(["--recipe","hml-continue","--store",str(tmp_path),"--targets",str(tmp_path),
        "--output",str(tmp_path / "adapt"),"--init",str(tmp_path / "base.pt"),"--ar",str(tmp_path / "ar.pt"),
        "--device","cpu","--steps","1","--global-batch","4","--microbatch","2", "--width","16","--depth","1","--heads","4"])
    result=PartFlow.load_ema(tmp_path / "adapt/ema_000001.pt",dataset="humanml3d",device="cpu")
    assert result.out.weight.abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for p in loaded[0].parameters())
