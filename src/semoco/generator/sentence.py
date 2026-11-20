"""Frozen sentence encoder used to build the retrieval token."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class _Positions(nn.Module):
    def __init__(self, width: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len, dtype=torch.float32)[:, None]
        scale = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / width))
        angles = position * scale
        encoding = torch.zeros(max_len, width)
        encoding[:, 0::2] = angles.sin()
        encoding[:, 1::2] = angles.cos()
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.dropout(value + self.encoding[:value.shape[1]])


class SentenceEncoder(nn.Module):
    """ACTOR-style encoder. The first token is the normalized latent mean."""

    def __init__(
        self,
        features: int,
        layers: int = 6,
        width: int = 256,
        heads: int = 4,
        feedforward: int = 1024,
        num_tokens: int = 2,
    ):
        super().__init__()
        self.features = features
        self.tokens = nn.Parameter(torch.randn(num_tokens, width))
        self.projection = nn.Linear(features, width)
        self.positions = _Positions(width, dropout=0.1)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=feedforward,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.sequence = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)

    @classmethod
    def from_state_dict(
        cls, state: dict[str, torch.Tensor], device: str | torch.device = "cpu"
    ) -> "SentenceEncoder":
        renamed = {key.replace("seqTransEncoder.", "sequence."): value for key, value in state.items()}
        renamed.pop("sequence_pos_encoding.pe", None)
        width, features = (int(size) for size in renamed["projection.weight"].shape)
        num_tokens = int(renamed["tokens"].shape[0])
        feedforward = int(renamed["sequence.layers.0.linear1.weight"].shape[0])
        depth = max(int(key.split(".")[2]) for key in renamed if key.startswith("sequence.layers.")) + 1
        heads = 4 if width % 4 == 0 else 1
        model = cls(features, layers=depth, width=width, heads=heads,
                    feedforward=feedforward, num_tokens=num_tokens)
        model.load_state_dict(renamed, strict=False)
        missing = [key for key in model.state_dict() if key not in renamed and "positions" not in key]
        if missing:
            raise ValueError(f"sentence encoder is missing {missing[:4]}")
        return model.to(device).eval().requires_grad_(False)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "SentenceEncoder":
        record = torch.load(path, map_location="cpu", weights_only=True)
        state = record
        for key in ("state_dict", "model", "text_encoder"):
            if isinstance(record, dict) and key in record and isinstance(record[key], dict):
                state = record[key]
                break
        return cls.from_state_dict(state, device=device)

    def encode(self, tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or valid.shape != tokens.shape[:2]:
            raise ValueError("tokens must be [batch, length, features] with a matching mask")
        batch = tokens.shape[0]
        projected = self.projection(tokens)
        prefix = self.tokens.expand(batch, -1, -1)
        sequence = torch.cat((prefix, projected), dim=1)
        prefix_valid = torch.ones(batch, self.tokens.shape[0], dtype=torch.bool, device=tokens.device)
        mask = torch.cat((prefix_valid, valid), dim=1)
        sequence = self.positions(sequence)
        encoded = self.sequence(sequence, src_key_padding_mask=~mask)
        return F.normalize(encoded[:, 0].float(), dim=-1)
