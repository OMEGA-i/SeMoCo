"""Aggregate aligned embeddings produced by the HumanML3D Guo or Omega V2 evaluator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

from .metrics import fid


def score_embeddings(
    generated: np.ndarray, reference: np.ndarray, text: np.ndarray, *, dataset: str,
    seed: int = 7, group_size: int = 32,
) -> dict[str, float | int]:
    """Score one repeat. HML input order must already follow the Guo loader."""
    if generated.ndim != 2 or generated.shape != reference.shape or generated.shape != text.shape:
        raise ValueError("generated, reference and text arrays must align as [samples, dim]")
    if not all(np.isfinite(values).all() for values in (generated, reference, text)):
        raise ValueError("invalid scorer embeddings")
    if dataset == "humanml3d":
        if group_size != 32 or len(generated) % 32:
            raise ValueError("HumanML3D uses a whole number of Guo batches of 32")
        order = np.arange(len(generated))
    elif dataset == "omega":
        if group_size not in (32, 256):
            raise ValueError("Omega uses groups of 32 (main) or 256")
        order = np.random.default_rng(seed).permutation(len(generated))
    else:
        raise ValueError(f"unknown evaluation dataset {dataset!r}")
    count = len(order) // group_size * group_size
    if count == 0:
        raise ValueError("evaluation needs at least one complete retrieval group")
    targets = np.arange(group_size)[None, :, None]
    hits = np.zeros(3, dtype=np.int64)
    groups = order[:count].reshape(-1, group_size)
    for block in np.array_split(groups, max(1, (len(groups) + 63) // 64)):
        m, t = generated[block], text[block]
        sim = t @ m.transpose(0, 2, 1)
        if dataset == "humanml3d":
            squared = (t ** 2).sum(-1, keepdims=True) + (m ** 2).sum(-1)[:, None, :] - 2 * sim
            top = np.argsort(squared.clip(min=0), axis=-1)[..., :3]
        else:
            top = torch.from_numpy(sim).topk(3, dim=-1).indices.numpy()
        correct = targets == top
        hits += [correct[..., :k].any(axis=-1).sum() for k in (1, 2, 3)]
    return {
        "samples": len(generated),
        "retrieval_samples": count,
        "FID": fid(generated, reference),
        "MM-Dist": float(np.linalg.norm(generated - text, axis=1).mean()),
        **{f"R@{k}": int(hits[k - 1]) / count for k in (1, 2, 3)},
    }


def summarize_repeats(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Compute repeat-weighted means and the 1.96 × standard-error half-width."""
    if not rows:
        raise ValueError("no evaluation repeats")
    metrics = {}
    for key in ("FID", "R@1", "R@2", "R@3", "MM-Dist"):
        values = [row[key] for row in rows]
        stat = {"mean": mean(values)}
        if len(values) >= 2:
            stat["ci95_half_width"] = 1.96 * stdev(values) / math.sqrt(len(values))
        metrics[key] = stat
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("humanml3d", "omega"), required=True)
    parser.add_argument("--embeddings", type=Path, nargs="+", required=True,
                        help="One NPZ per complete repeat, with generated, reference and text arrays.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--output", type=Path, help="Optional JSON output file.")
    args = parser.parse_args()
    rows = []
    for path in args.embeddings:
        with np.load(path, allow_pickle=False) as data:
            rows.append(score_embeddings(data["generated"], data["reference"], data["text"],
                                         dataset=args.dataset, seed=args.seed, group_size=args.group_size))
    payload = json.dumps({"repeats": rows, "summary": summarize_repeats(rows)}, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
