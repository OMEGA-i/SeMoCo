"""Read an aligned, memory-mapped tensor store for six-part Flow training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..generator.parts import PART_DIMS


def load_statistics(path: Path, channels: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path, allow_pickle=False) as raw:
        mean, std = (torch.tensor(raw[key], dtype=torch.float32, device=device) for key in ("mean", "std"))
    if (mean.shape != (channels,) or std.shape != mean.shape or not bool(mean.isfinite().all())
            or not bool((std.isfinite() & (std > 0)).all())):
        raise ValueError(f"invalid mean/std in {path}; expected {channels} finite channels and positive std")
    return mean, std


class FlowStore:
    def __init__(self, store: Path, parts: Path, dataset: str, *, max_tokens: int = 50):
        self.rows = json.loads((store / "train.index.json").read_text())
        self.text_index = json.loads((store / "train.text_index.flantmr.json").read_text())
        self.codes = np.load(store / "train.codes.npy", mmap_mode="r", allow_pickle=False)
        self.latents = np.load(store / "train.kin_latent.npy", mmap_mode="r", allow_pickle=False)
        self.text = np.load(store / "train.text_emb.flantmr.npy", mmap_mode="r", allow_pickle=False)
        self.targets = np.load(parts / "targets.npy", mmap_mode="r", allow_pickle=False)
        if (len(self.rows) != len(self.text_index) or self.codes.ndim != 2 or self.codes.shape[1] < 1
                or self.latents.ndim != 2 or self.text.ndim != 2 or len(self.latents) != len(self.codes)
                or self.targets.shape != (len(self.codes), sum(PART_DIMS[dataset]))):
            raise ValueError("motion, text or auxiliary arrays do not match the provided indices")
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        caption = self.text_index[index]
        start = row["code_start"]
        length = min(row["code_len"], self.max_tokens)
        begin, words = caption["text_start"], caption["text_len"]
        if (length < 1 or start < 0 or start + length > len(self.codes)
                or words < 1 or begin < 0 or begin + words > len(self.text)):
            raise ValueError(f"invalid motion/text offset for training row {index}")
        span = slice(start, start + length)
        return {
            "q0": torch.tensor(self.codes[span, 0], dtype=torch.long),
            "y": torch.tensor(self.latents[span], dtype=torch.float32),
            "target": torch.tensor(self.targets[span], dtype=torch.float32),
            "text": torch.tensor(self.text[begin:begin + words], dtype=torch.float32),
        }


def collate_flow(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Motion is right padded; text is left padded as in the cached-text recipe."""
    max_motion = max(len(row["q0"]) for row in items)
    max_words = max(len(row["text"]) for row in items)
    batch = len(items)
    output = {
        "q0": torch.zeros(batch, max_motion, dtype=torch.long),
        "y": torch.zeros(batch, max_motion, items[0]["y"].shape[-1]),
        "target": torch.zeros(batch, max_motion, items[0]["target"].shape[-1]),
        "valid": torch.zeros(batch, max_motion, dtype=torch.bool),
        "text": torch.zeros(batch, max_words, items[0]["text"].shape[-1]),
        "text_valid": torch.zeros(batch, max_words, dtype=torch.bool),
    }
    for i, row in enumerate(items):
        n = len(row["q0"])
        l = len(row["text"])
        for key in ("q0", "y", "target"):
            output[key][i, :n] = row[key]
        output["valid"][i, :n] = True
        output["text"][i, -l:] = row["text"]
        output["text_valid"][i, -l:] = True
    return output
