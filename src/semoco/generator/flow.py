"""The factorized PartFlow used with a frozen split codec and semantic AR."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..generator.parts import PART_DIMS, part_loss


def timestep_embedding(t: torch.Tensor, dim: int = 256) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    phase = t.float()[:, None] * 1000 * freq[None]
    return torch.cat((phase.cos(), phase.sin()), dim=-1)


def factorized_attention(
    attention: nn.MultiheadAttention, hidden: torch.Tensor, padding: torch.Tensor
) -> torch.Tensor:
    """Apply one shared MHA within a time step, then along each part's timeline."""
    batch, tokens, width = hidden.shape
    length = tokens // 6
    joints = hidden.reshape(batch, length, 6, width)
    rows = joints.reshape(batch * length, 6, width)
    spatial = attention(rows, rows, rows, need_weights=False)[0].reshape_as(joints)
    spatial = spatial.masked_fill(padding.reshape(batch, length, 6, 1), 0)
    tracks = (joints + spatial).transpose(1, 2).reshape(batch * 6, length, width)
    track_padding = padding.reshape(batch, length, 6).transpose(1, 2).reshape(batch * 6, length)
    temporal = attention(tracks, tracks, tracks, key_padding_mask=track_padding, need_weights=False)[0]
    return spatial.reshape_as(hidden) + temporal.reshape(batch, 6, length, width).transpose(1, 2).reshape_as(hidden)


class PartBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.n1 = nn.LayerNorm(width, elementwise_affine=False)
        self.sa = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(width, elementwise_affine=False)
        self.ca = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.n3 = nn.LayerNorm(width, elementwise_affine=False)
        mlp = [nn.Linear(width, 4 * width), nn.GELU(approximate="tanh"), nn.Linear(4 * width, width)]
        self.mlp = nn.Sequential(*(mlp + ([nn.Dropout(dropout)] if dropout else [])))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(width, 9 * width))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, cond, text, text_pad, padding):
        s1, b1, g1, s2, b2, g2, s3, b3, g3 = self.ada(cond)[:, None].chunk(9, -1)
        h = self.n1(x) * (1 + s1) + b1
        x = x + g1 * factorized_attention(self.sa, h, padding)
        h = self.n2(x) * (1 + s2) + b2
        x = x + g2 * self.ca(h, text, text, key_padding_mask=text_pad, need_weights=False)[0]
        h = self.n3(x) * (1 + s3) + b3
        return x + g3 * self.mlp(h)


class PartFlow(nn.Module):
    """Flow over all 512 kinematic channels with six internal queries per token.

    Parameter names and input math match the factorized training checkpoints.
    Dataset choice changes the training readout widths, rather than the denoiser.
    """

    def __init__(
        self,
        *,
        dataset: str = "humanml3d",
        width: int = 1024,
        depth: int = 16,
        heads: int = 16,
        ydim: int = 512,
        text_dim: int = 2048,
        q0_vocab: int = 1024,
        max_t: int = 64,
        dropout: float = 0.,
        checkpoint_blocks: bool = True,
    ):
        super().__init__()
        self.dataset = dataset
        self.aux_dims = PART_DIMS[dataset]
        self.ydim = ydim
        self.checkpoint_blocks = checkpoint_blocks
        self.y_in = nn.Linear(ydim, width)
        self.q0 = nn.Embedding(q0_vocab, width)
        self.pos = nn.Parameter(torch.zeros(1, max_t, width))
        nn.init.normal_(self.pos, std=.02)
        self.txt = nn.Linear(text_dim, width)
        self.null_txt = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.normal_(self.null_txt, std=.02)
        self.t_mlp = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))
        self.len_emb = nn.Embedding(max_t + 1, width)
        self.blocks = nn.ModuleList(PartBlock(width, heads, dropout) for _ in range(depth))
        self.out_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.out_ada = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.out = nn.Linear(width, ydim)
        for layer in (self.out, self.out_ada[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        with torch.random.fork_rng(devices=[]):
            self.readouts = nn.ModuleList(nn.Linear(width, size) for size in self.aux_dims)
            self.part_query = nn.Parameter(torch.empty(6, width))
            self.part_pos = nn.Parameter(torch.empty(6, width))
            self.fusion = nn.Linear(width, 1)
            nn.init.normal_(self.part_query, std=.02)
            nn.init.normal_(self.part_pos, std=.02)
            nn.init.zeros_(self.fusion.weight)
            nn.init.zeros_(self.fusion.bias)

    def encode_text(self, text, valid, drop=None):
        projected = self.txt(F.normalize(text, dim=-1))
        if drop is not None:
            projected = torch.where(drop[:, None, None], self.null_txt.expand_as(projected), projected)
        return projected, ~valid

    def forward(self, xt, time, q0, valid, text, text_pad, *, return_aux: bool = False):
        batch, length, _ = xt.shape
        x = self.y_in(xt) + self.q0(q0) + self.pos[:, :length]
        cond = self.t_mlp(timestep_embedding(time)) + self.len_emb(
            valid.sum(1).clamp(max=self.len_emb.num_embeddings - 1)
        )
        x = (x[:, :, None] + self.part_query + self.part_pos).flatten(1, 2)
        padding = (~valid).repeat_interleave(6, dim=1)
        for block in self.blocks:
            if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, cond, text, text_pad, padding, use_reentrant=False)
            else:
                x = block(x, cond, text, text_pad, padding)
        parts = x.reshape(batch, length, 6, -1)
        fused = (parts * self.fusion(parts).softmax(dim=2)).sum(dim=2)
        scale, bias = self.out_ada(cond)[:, None].chunk(2, -1)
        velocity = self.out(self.out_norm(fused) * (1 + scale) + bias)
        if return_aux:
            return velocity, [head(parts[:, :, i]) for i, head in enumerate(self.readouts)]
        return velocity

    def loss(
        self, y, q0, valid, text, text_valid, *, target, cfg_drop: float = .1,
        q0_corrupt: float = 0., denominator: torch.Tensor | None = None,
    ):
        """Masked velocity MSE + 0.1 × part loss, for normalized training tensors."""
        batch = y.shape[0]
        if q0_corrupt > 0:
            rate = torch.rand(batch, 1, device=y.device) * q0_corrupt
            q0 = torch.where(torch.rand(q0.shape, device=y.device) < rate,
                             torch.randint_like(q0, self.q0.num_embeddings), q0)
        drop = torch.rand(batch, device=y.device) < cfg_drop
        cond, padding = self.encode_text(text, text_valid, drop)
        time = torch.rand(batch, device=y.device)
        noise = torch.randn_like(y)
        xt = (1 - time[:, None, None]) * noise + time[:, None, None] * y
        velocity, predictions = self(xt, time, q0, valid, cond, padding, return_aux=True)
        error = (velocity.float() - (y - noise).float()).square().mean(-1)
        denom = valid.sum() if denominator is None else denominator
        return ((error * valid).sum() / denom.clamp_min(1)
                + .1 * part_loss(predictions, target, valid, time, denom, self.aux_dims))

    @torch.no_grad()
    def sample(
        self, q0, valid, text, text_valid, *, cfg: float = 3., steps: int = 32,
        generator: torch.Generator | None = None, initial_noise: torch.Tensor | None = None,
    ):
        """Explicit Euler from noise at t=0 to a normalized latent at t=1."""
        if steps < 1:
            raise ValueError("Euler sampling needs at least one step")
        batch, length = q0.shape
        shape = (batch, length, self.ydim)
        if initial_noise is None:
            source = q0.device if generator is None else generator.device
            x = torch.randn(shape, dtype=torch.float32, device=source, generator=generator).to(q0.device)
        else:
            if (initial_noise.shape != shape or initial_noise.dtype != torch.float32
                    or initial_noise.device != q0.device):
                raise ValueError("initial noise must be float32 [batch, tokens, latent_dim] on the motion device")
            x = initial_noise
        cond, padding = self.encode_text(text, text_valid)
        condition = torch.cat((cond, self.null_txt.expand_as(cond)))
        packet = torch.cat((q0, q0))
        motion_valid = torch.cat((valid, valid))
        text_pad = torch.cat((padding, padding))
        times = torch.linspace(0, 1, steps + 1, device=q0.device)
        for i in range(steps):
            t = times[i].expand(batch)
            vc, vu = self(torch.cat((x, x)), torch.cat((t, t)), packet,
                          motion_valid, condition, text_pad).float().chunk(2)
            x = x + (times[i + 1] - times[i]) * (vu + cfg * (vc - vu))
        return x

    @classmethod
    def load_ema(cls, path: str | Path, *, dataset: str, device: str | torch.device, heads: int | None = None):
        """Load a trusted EMA checkpoint (initial training or continuation)."""
        recorded = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        state = recorded["ema"]
        width = state["y_in.weight"].shape[0]
        depth = len({key.split(".")[1] for key in state if key.startswith("blocks.")})
        # Head count is not encoded in QKV matrix shapes.
        root = recorded.get("config")
        config = root.get("model", {}) if isinstance(root, dict) else {}
        if heads is None:
            heads = config.get("heads", 16)
        if config.get("heads", heads) != heads:
            raise ValueError(f"checkpoint specifies {config['heads']} heads, got {heads}")
        with torch.device("meta"):
            model = cls(dataset=dataset, width=width, depth=depth, heads=heads,
                        ydim=state["y_in.weight"].shape[1], text_dim=state["txt.weight"].shape[1],
                        q0_vocab=state["q0.weight"].shape[0], max_t=state["pos"].shape[1],
                        dropout=config.get("dropout", 0.))
        model.load_state_dict(state, strict=True, assign=True)
        return model.to(device).eval()

    def load_ordinary_parent(self, weights) -> None:
        """Warm-start only the shared backbone; initialize part-specific weights locally."""
        missing = {"part_query", "part_pos", "fusion.weight", "fusion.bias"}
        missing.update(f"readouts.{i}.{name}" for i in range(6) for name in ("weight", "bias"))
        if set(weights) != set(self.state_dict()) - missing:
            raise ValueError("expected an ordinary Flow with the same backbone and no part parameters")
        self.load_state_dict(weights, strict=False)
