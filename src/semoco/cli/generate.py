"""Generate SeMoCo features from cached text and a local checkpoint bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from ..tokenizer.codec import load_codec
from ..generator.flow import PartFlow
from ..generator.pipeline import RECIPES, generate, load_semantic_ar
from ..data.flow_store import load_statistics


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Dataset preset; explicit arguments override its values.")
    parser.add_argument("--dataset", choices=RECIPES)
    for name in ("codec", "flow", "kin-stats", "inputs", "output"):
        parser.add_argument("--" + name, required=name in ("inputs", "output"), type=Path)
    parser.add_argument("--ar", type=Path, help="Omit only if the input NPZ supplies a sampled q0 plan.")
    parser.add_argument("--feature-stats", type=Path, help="Override codec feature stats with this mean/std NPZ.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--ar-guidance", type=float)
    parser.add_argument("--flow-guidance", type=float)
    args = parser.parse_args(argv)
    if args.config:
        preset = yaml.safe_load(args.config.read_text())
        for key in ("dataset", "codec", "ar", "flow", "kin_stats", "feature_stats", "steps", "ar_guidance", "flow_guidance"):
            if getattr(args, key) is None and key in preset:
                value = preset[key]
                setattr(args, key, Path(value) if key in ("codec", "ar", "flow", "kin_stats", "feature_stats") else value)
    for key in ("dataset", "codec", "flow", "kin_stats"):
        if getattr(args, key) is None:
            parser.error(f"provide --{key.replace('_', '-')} or include it in --config")
    if args.steps is None:
        args.steps = 32
    if args.dataset not in RECIPES:
        parser.error("unknown dataset in generation configuration")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    with np.load(args.inputs, allow_pickle=False) as raw:
        text = torch.tensor(raw["text"], device=device, dtype=torch.float32)
        mask = torch.tensor(raw["text_valid"], device=device, dtype=torch.bool)
        frames = torch.tensor(raw["frames"], device=device)
        q0 = torch.tensor(raw["q0"], device=device, dtype=torch.long) if "q0" in raw else None
        noise = torch.tensor(raw["noise"], device=device, dtype=torch.float32) if "noise" in raw else None
    if text.ndim != 3 or text.shape[:2] != mask.shape or frames.shape != (len(text),):
        parser.error("input NPZ needs text [B,L,D], text_valid [B,L], and frames [B]")
    if len(text) == 0 or not bool(mask.any(1).all()):
        parser.error("provide at least one valid caption token per item")
    if q0 is None and args.ar is None:
        parser.error("provide --ar or q0 in --inputs")
    ar = load_semantic_ar(args.ar, device) if q0 is None else None
    flow = PartFlow.load_ema(args.flow, dataset=args.dataset, device=device)
    if text.shape[-1] != flow.txt.in_features:
        parser.error("text width does not match the Flow checkpoint")
    codec, feature_norm = load_codec(args.codec, dataset=args.dataset, device=device)
    mean, std = load_statistics(args.kin_stats, flow.ydim, device)
    motion, q0, latent = generate(
        ar=ar, flow=flow, codec=codec, text=text, text_valid=mask, frames=frames, q0=q0,
        latent_mean=mean, latent_std=std, seed=args.seed, ar_guidance=args.ar_guidance,
        flow_guidance=args.flow_guidance, steps=args.steps, initial_noise=noise,
    )
    if args.feature_stats:
        feature_norm = load_statistics(args.feature_stats, motion[0].shape[-1], device)
    report = dict(features=torch.cat(motion).cpu().numpy(),
                  offsets=np.cumsum([0, *(len(item) for item in motion)], dtype=np.int64),
                  q0=q0.cpu().numpy(), latent=latent.cpu().numpy(),
                  requested_frames=frames.cpu().numpy(), fps=RECIPES[args.dataset].fps)
    if feature_norm is not None:
        report["physical_features"] = torch.cat([x * feature_norm[1] + feature_norm[0] for x in motion]).cpu().numpy()
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **report)


if __name__ == "__main__":
    main()
