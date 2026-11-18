"""Train the main semantic prior from frozen codec and text caches."""
import argparse
import math
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import yaml
from ..generator.ar import MotionGPT, MotionGPTConfig
from ..data.flow_store import collate_flow
from .common import seed_all, save, random_state, restore_random


def learning_rate(settings, step):
    """Dataset-specific learning-rate schedule."""
    warmup = settings.get("warmup_steps", 4000)
    horizon = settings.get("schedule_steps", settings["steps"])
    index = step - settings.get("schedule_offset", 0)
    floor = settings.get("min_lr_ratio", 0.)
    if floor:
        if index < warmup:
            return settings["lr"] * (index + 1) / max(1, warmup)
        progress = min(1., max(0., (index - warmup) / max(1, horizon - warmup)))
        return settings["lr"] * (floor + (1 - floor) * .5 * (1 + math.cos(math.pi * progress)))
    progress = min(1., max(0., (step - warmup) / max(1, horizon - warmup)))
    return settings["lr"] * min(1., step / warmup) * .5 * (1 + math.cos(math.pi * progress))


class SemanticStore:
    def __init__(self, root, max_tokens):
        self.rows = json.loads((root / "train.index.json").read_text())
        self.captions = json.loads((root / "train.text_index.flantmr.json").read_text())
        self.codes = np.load(root / "train.codes.npy", mmap_mode="r", allow_pickle=False)
        self.text = np.load(root / "train.text_emb.flantmr.npy", mmap_mode="r", allow_pickle=False)
        self.max_tokens = max_tokens
        if len(self.rows) != len(self.captions):
            raise ValueError("motion and caption indices differ")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row, cap = self.rows[i], self.captions[i]
        n, start = min(row["code_len"], self.max_tokens), row["code_start"]
        words, begin = cap["text_len"], cap["text_start"]
        if n < 1 or words < 1:
            raise ValueError("empty motion or caption")
        return dict(q0=torch.tensor(self.codes[start:start+n, 0], dtype=torch.long),
                    y=torch.zeros(n, 1), target=torch.zeros(n, 1),
                    text=torch.tensor(self.text[begin:begin+words], dtype=torch.float32))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "store", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    settings = cfg["training"]
    device = torch.device(args.device)
    seed_all(settings["seed"])
    model = MotionGPT(MotionGPTConfig(**cfg["model"])).to(device).train()
    data = SemanticStore(args.store, settings.get("max_tokens", cfg["model"]["len_max"]))
    batch_size = settings["batch_size"]
    if len(data) < batch_size:
        parser.error("training store must contain one complete batch")
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], betas=(.9, .95),
                                  weight_decay=settings.get("weight_decay", .01))
    last = args.output / "last.pt"
    start = 0
    if args.resume:
        record = torch.load(last, map_location="cpu", weights_only=False)
        if record["training_config"] != cfg:
            raise ValueError("resume configuration changed")
        model.load_state_dict(record["model"], strict=True)
        optimizer.load_state_dict(record["optimizer"])
        restore_random(record["rng"], device)
        start = record["step"]
    elif last.exists():
        parser.error("output already exists; use --resume or a new directory")
    for step in range(start + 1, settings["steps"] + 1):
        epoch, offset = divmod(step - 1, len(data) // batch_size)
        if step == start + 1 or offset == 0:
            order = np.random.default_rng(settings["seed"] + epoch).permutation(len(data))
        ids = order[offset * batch_size:(offset + 1) * batch_size]
        rows = [data[int(i)] for i in ids]
        token_count = sum(len(r["q0"]) for r in rows)
        eos_count = token_count + len(rows)
        microbatch = settings.get("microbatch", batch_size)
        if microbatch < 1:
            raise ValueError("microbatch must be positive")
        optimizer.zero_grad(set_to_none=True)
        ce_total = stop_total = 0.
        for begin in range(0, len(rows), microbatch):
            b = {k: v.to(device) for k, v in collate_flow(rows[begin:begin+microbatch]).items()}
            drop = torch.rand(len(b["q0"]), device=device) < model.cfg.text_cond_prob
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                scores, eos = model.t2m_train_step(b["text"], b["text_valid"], b["q0"][..., None], b["valid"], drop_text=drop)
                mask = b["valid"]
                ce = F.cross_entropy(scores[0].float().flatten(0, 1), b["q0"].flatten(), reduction="none")
                local_mean = settings.get("loss_reduction", "logical_tokens") == "microbatch_mean"
                factor = len(b["q0"]) / len(rows) if local_mean else 1.
                ce = (ce * mask.flatten()).sum() / (mask.sum() if local_mean else token_count) * factor
                positions = torch.arange(eos.shape[1], device=device)[None]
                lengths = mask.sum(1, keepdim=True)
                eos_mask = positions <= lengths
                eos_target = (positions == lengths).float()
                stop = F.binary_cross_entropy_with_logits(eos.float(), eos_target, reduction="none")
                stop = (stop * eos_mask).sum() / (eos_mask.sum() if local_mean else eos_count) * factor
                loss = ce + model.cfg.eos_loss_scale * stop
            loss.backward()
            ce_total += float(ce.detach()); stop_total += float(stop.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(settings, step)
        optimizer.step()
        if step == start + 1 or step % 25 == 0:
            print(json.dumps(dict(step=step, ce=ce_total, eos=stop_total)), flush=True)
        if step % settings.get("save_every", 1000) == 0 or step == settings["steps"]:
            state = dict(model=model.state_dict(), model_config=asdict(model.cfg), step=step)
            save(last, dict(state, optimizer=optimizer.state_dict(), training_config=cfg, rng=random_state(device)))
            save(args.output / f"model_{step:06}.pt", state)


if __name__ == "__main__":
    main()
