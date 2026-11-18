"""Train six-part conditional flow matching."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from ..generator.flow import PartFlow
from ..generator.pipeline import load_semantic_ar
from ..data.flow_store import FlowStore, collate_flow, load_statistics


# dataset, total new updates, global batch, microbatch, learning rate, EMA update
RECIPES = {
    "hml-base": ("humanml3d", 40000, 256, 32, 2e-4, .001),
    "hml-continue": ("humanml3d", 2000, 128, 16, 3e-5, .002),
    "omega-warm": ("omega", 20000, 256, 32, 5e-5, .001),
}


@torch.no_grad()
def ar_corruption(q0, valid, text, text_valid, ar, rng, replacement: float = .30):
    """Replace random positions with samples from the frozen AR given clean history."""
    rate = torch.rand(len(q0), 1, device=q0.device) * replacement
    mask = torch.rand(q0.shape, device=q0.device) < rate
    # Preserve the random stream before drawing AR proposals.
    torch.randint(ar.cfg.codebook_size, q0.shape, device=q0.device)
    scores, _ = ar.t2m_train_step(text, text_valid, q0[..., None], valid)
    probs = scores[0].float().softmax(-1)
    proposed = torch.multinomial(probs.flatten(0, 1), 1, generator=rng).reshape_as(q0)
    return torch.where(mask, proposed, q0)


def learning_rate(recipe: str, base: float, step: int, schedule_steps: int) -> float:
    if recipe == "hml-base":
        s = step - 1
        return base * (min(1., step / 2000) * .5 * (1 + math.cos(math.pi * min(1., s / schedule_steps))) * .95 + .05)
    if recipe == "hml-continue":
        return base * min(1., step / 200)
    return base


def _save(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".pending")
    torch.save(data, temporary)
    temporary.replace(path)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=RECIPES, required=True)
    for name in ("store", "targets", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    starting = parser.add_mutually_exclusive_group()
    starting.add_argument("--init", type=Path, help="Load a matching PartFlow EMA and start a fresh optimizer.")
    starting.add_argument("--ordinary-parent", type=Path, help="Warm-start the shared backbone from an ordinary Flow EMA.")
    starting.add_argument("--resume", action="store_true", help="Resume this trainer's last.pt in the output directory.")
    parser.add_argument("--ar", type=Path, help="Frozen semantic AR, required for hml-continue.")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--global-batch", type=int)
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, help="Training crop cap; defaults to 50 (HML) or 125 (Omega).")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--heads", type=int, default=16)
    args = parser.parse_args(argv)

    dataset, schedule_steps, global_batch, microbatch, base_lr, ema_rate = RECIPES[args.recipe]
    if args.seed is None:
        args.seed = 3432 if args.recipe == "hml-continue" else 3407
    steps = schedule_steps if args.steps is None else args.steps
    global_batch = global_batch if args.global_batch is None else args.global_batch
    microbatch = microbatch if args.microbatch is None else args.microbatch
    max_t = 64 if dataset == "humanml3d" else 125
    if args.max_tokens is None:
        args.max_tokens = 50 if dataset == "humanml3d" else max_t
    if min(steps, global_batch, microbatch, args.save_every, args.max_tokens) < 1 or args.max_tokens > max_t:
        parser.error(f"positive steps/batches/save interval and 1–{max_t} motion tokens required")
    if args.recipe == "hml-continue" and (args.ar is None or (args.init is None and not args.resume)):
        parser.error("hml-continue requires --ar and --init (or --resume)")
    if args.recipe == "omega-warm" and not (args.ordinary_parent or args.init or args.resume):
        parser.error("omega-warm requires --ordinary-parent, --init, or --resume")
    if args.recipe != "omega-warm" and args.ordinary_parent:
        parser.error("--ordinary-parent is for the Omega warm start")

    data = FlowStore(args.store, args.targets, dataset, max_tokens=args.max_tokens)
    blocks = len(data) // global_batch
    if blocks < 1:
        parser.error("the store must contain at least one complete logical batch")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    model_config = dict(dataset=dataset, width=args.width, depth=args.depth, heads=args.heads, max_t=max_t,
                        ydim=data.latents.shape[-1], text_dim=data.text.shape[-1])
    model = PartFlow(**model_config)
    if args.init or args.ordinary_parent:
        parent = torch.load(args.init or args.ordinary_parent, map_location="cpu", mmap=True, weights_only=True)
        if args.init:
            model.load_state_dict(parent["ema"], strict=True)
        else:
            model.load_ordinary_parent(parent["ema"])
        del parent
    model.to(device).train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(.9, .99), weight_decay=.01,
                            fused=device.type == "cuda")
    ymean, ystd = load_statistics(args.store / "kin_stats.npz", model.ydim, device)
    partmean, partstd = load_statistics(args.targets / "targets_norm.npz", sum(model.aux_dims), device)
    ar = load_semantic_ar(args.ar, device) if args.recipe == "hml-continue" else None
    rng = torch.Generator(device=device).manual_seed(args.seed + 1) if ar is not None else None
    if args.recipe == "hml-continue":
        # The production adaptation resets the training stream after loading
        # frozen modules; constructors must not consume the adaptation stream.
        torch.manual_seed(args.seed)

    args.output.mkdir(parents=True, exist_ok=True)
    last = args.output / "last.pt"
    identity = dict(recipe=args.recipe, model=model_config, seed=args.seed, global_batch=global_batch,
                    microbatch=microbatch, max_tokens=args.max_tokens, train_rows=len(data))
    start = 0
    if args.resume:
        state = torch.load(last, map_location="cpu", mmap=True, weights_only=True)
        if state["identity"] != identity:
            raise ValueError("resume requires the original recipe, model, seed, and batch layout")
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        opt.load_state_dict(state["optimizer"])
        torch.random.set_rng_state(state["cpu_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], device)
        if rng is not None:
            rng.set_state(state["proposal_rng"])
        start = state["step"]
        del state
    elif last.exists():
        raise ValueError("output already contains last.pt; use --resume or a new directory")
    if steps <= start:
        parser.error("--steps must exceed the last completed step")

    print(json.dumps(dict(identity, start=start, stop=steps)), flush=True)
    for step in range(start + 1, steps + 1):
        epoch, offset = divmod(step - 1, blocks)
        if step == start + 1 or offset == 0:
            order = np.random.default_rng(args.seed + epoch).permutation(len(data))
        rows = [data[int(i)] for i in order[offset * global_batch:(offset + 1) * global_batch]]
        total = torch.tensor(sum(len(row["q0"]) for row in rows), device=device)
        opt.zero_grad(set_to_none=True)
        loss_total = 0.
        for i in range(0, len(rows), microbatch):
            b = {k: v.to(device) for k, v in collate_flow(rows[i:i + microbatch]).items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                q0 = b["q0"] if ar is None else ar_corruption(
                    b["q0"], b["valid"], b["text"], b["text_valid"], ar, rng)
                loss = model.loss((b["y"] - ymean) / ystd, q0, b["valid"], b["text"], b["text_valid"],
                                  q0_corrupt=.30 if ar is None else 0.,
                                  target=(b["target"] - partmean) / partstd, denominator=total)
            loss.backward()
            loss_total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        lr = learning_rate(args.recipe, base_lr, step, schedule_steps)
        for group in opt.param_groups:
            group["lr"] = lr
        opt.step()
        with torch.no_grad():
            for average, live in zip(ema.parameters(), model.parameters(), strict=True):
                average.lerp_(live, ema_rate)
        if step == start + 1 or step % 25 == 0:
            print(json.dumps({"step": step, "loss": loss_total, "lr": lr}), flush=True)
        if step % args.save_every == 0 or step == steps:
            common = dict(step=step, ema=ema.state_dict(), config={"model": model_config})
            _save(last, dict(common, identity=identity, model=model.state_dict(), optimizer=opt.state_dict(),
                             cpu_rng=torch.random.get_rng_state(),
                             cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                             proposal_rng=rng.get_state() if rng is not None else None))
            _save(args.output / f"ema_{step:06}.pt", common)


if __name__ == "__main__":
    main()
