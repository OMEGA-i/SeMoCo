"""Train the parallel 1+7 tokenizer on aligned physical feature windows."""
import argparse
import math
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from ..tokenizer.codec import CodecConfig, MotionTokenizer
from ..tokenizer.losses import humanml_loss, OmegaReconstructionLoss, semantic_loss
from .common import seed_all, save, random_state, restore_random, codebook_runtime, restore_codebook_runtime


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "windows", "teacher", "feature-stats", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--template", type=Path, help="SOMA77 parents/offsets NPZ for Omega geometry loss")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    settings, dataset = cfg["training"], cfg["dataset"]
    device = torch.device(args.device)
    seed_all(settings["seed"])
    model = MotionTokenizer(CodecConfig(**cfg["model"])).to(device).train()
    windows = np.load(args.windows, mmap_mode="r", allow_pickle=False)
    teacher = np.load(args.teacher, mmap_mode="r", allow_pickle=False)
    if windows.ndim != 3 or windows.shape[1:] != (64, model.cfg.input_dim):
        parser.error("physical windows must be [N,64,feature_dim]")
    if teacher.shape != (len(windows), model.cfg.semantic_dim):
        parser.error("frozen TMR motion targets must align one-to-one with the exact 64-frame windows")
    if len(windows) < 1:
        parser.error("empty window cache")
    with np.load(args.feature_stats, allow_pickle=False) as stats:
        mean, std = (torch.tensor(stats[k], dtype=torch.float32, device=device) for k in ("mean", "std"))
    if mean.shape != (model.cfg.input_dim,) or not torch.isfinite(mean).all() or not (torch.isfinite(std) & (std > 0)).all():
        parser.error("invalid feature statistics")
    geometry = None
    if dataset == "omega":
        if args.template is None:
            parser.error("Omega requires --template for the FK loss")
        geometry = OmegaReconstructionLoss(args.template, mean, std).to(device)
    elif dataset != "humanml3d":
        parser.error("unsupported dataset")
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"],
                                  betas=tuple(settings["betas"]), weight_decay=settings["weight_decay"])
    last = args.output / "last.pt"
    start = 0
    if args.resume:
        record = torch.load(last, map_location="cpu", weights_only=False)
        if record["training_config"] != cfg:
            raise ValueError("resume configuration changed")
        model.load_state_dict(record["model"], strict=True)
        restore_codebook_runtime(model, record["codebook_runtime"])
        optimizer.load_state_dict(record["optimizer"])
        restore_random(record["rng"], device)
        start = record["step"]
    elif last.exists():
        parser.error("output already contains last.pt; use --resume or a new directory")
    for step in range(start + 1, settings["steps"] + 1):
        ids = np.random.randint(0, len(windows), settings["batch_size"])
        physical = torch.tensor(np.asarray(windows[ids]), device=device)
        target = (physical - mean) / std
        tmr = torch.tensor(np.asarray(teacher[ids]), device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            rec, codes, losses, semantic = model(target)
        reconstruction = (humanml_loss(rec, target, losses["commit_loss"]) if geometry is None
                          else geometry(rec, target, losses["commit_loss"]))
        warm = min(1., step / settings["semantic_warmup"])
        sem_weight = settings.get("semantic_start_weight", 0.)
        sem_weight += (settings["semantic_weight"] - sem_weight) * warm
        distillation = semantic_loss(semantic, tmr)
        loss = reconstruction + sem_weight * distillation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        rate = min(1., step / settings["warmup_steps"])
        if dataset == "omega" and step >= settings["warmup_steps"]:
            horizon = settings["schedule_steps"]
            progress = min(1., (step - settings["warmup_steps"]) / (horizon - settings["warmup_steps"]))
            rate = .5 * (1 + math.cos(math.pi * progress))
        if step > settings.get("lr_decay_at", settings["steps"]):
            rate *= .1
        for group in optimizer.param_groups:
            group["lr"] = settings["lr"] * rate
        optimizer.step()
        if step == start + 1 or step % 25 == 0:
            print(json.dumps(dict(step=step, reconstruction=float(reconstruction.detach()),
                                  semantic=float(distillation.detach()))), flush=True)
        if step % settings.get("save_every", 1000) == 0 or step == settings["steps"]:
            state = dict(format="semoco.tokenizer.v1", model=model.state_dict(), config=asdict(model.cfg),
                         norm=dict(mean=mean.cpu(), std=std.cpu()), step=step)
            save(last, dict(state, optimizer=optimizer.state_dict(), training_config=cfg, rng=random_state(device),
                            codebook_runtime=codebook_runtime(model)))
            save(args.output / f"model_{step:06}.pt", state)


if __name__ == "__main__":
    main()
