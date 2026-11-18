"""Shared 1D-conv encoder/decoder backbone for the UMR structured codec.

Progressive dilated-conv encoder/decoder primitives consumed by
the split tokenizer (the native, config-driven codec). These blocks
are quantizer-agnostic; quantization is handled separately by
the quantizer module.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

DIM_FEATURES = 499

from .layers import (
    CausalConv1d,
    DownsampleBlock1d,
)


def _norm_layer(channels: int, mode: str) -> nn.Module:
    if mode != "none":
        raise ValueError("the main codec uses no residual normalization")
    return nn.Identity()


def _factor_stride(stride: int, layers: int) -> list[int]:
    if stride < 1:
        raise ValueError(f"temporal_stride must be >= 1; got {stride}")
    if layers < 1:
        raise ValueError(f"num_downsample_layers must be >= 1; got {layers}")
    if stride == 1:
        return [1]
    factors: list[int] = []
    remaining = stride
    for _ in range(layers - 1):
        if remaining % 2 != 0:
            raise ValueError(f"temporal_stride={stride} cannot be split into {layers} stride-2 stages")
        factors.append(2)
        remaining //= 2
    factors.append(remaining)
    if any(f < 1 for f in factors):
        raise ValueError(f"invalid stride factorization for temporal_stride={stride}, layers={layers}")
    return factors


class DilatedResBlock1d(nn.Module):
    """Residual block with dilation and dropout."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilation: int,
        causal: bool,
        dropout: float,
        norm: str,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            _norm_layer(channels, norm),
            nn.ReLU(),
            CausalConv1d(channels, channels, kernel_size, dilation=dilation, causal=causal),
            _norm_layer(channels, norm),
            nn.ReLU(),
            CausalConv1d(channels, channels, kernel_size=1, causal=causal),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


def _resnet_stage(
    channels: int,
    *,
    depth: int,
    kernel_size: int,
    dilation_growth_rate: int,
    causal: bool,
    dropout: float,
    norm: str,
    reverse_dilation: bool = False,
) -> nn.Module:
    dilations = [dilation_growth_rate ** i for i in range(depth)]
    if reverse_dilation:
        dilations = list(reversed(dilations))
    return nn.Sequential(
        *[
            DilatedResBlock1d(
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
                causal=causal,
                dropout=dropout,
                norm=norm,
            )
            for dilation in dilations
        ]
    )


class NearestUpsampleBlock1d(nn.Module):
    def __init__(self, channels: int, stride: int, *, causal: bool) -> None:
        super().__init__()
        self.stride = stride
        self.conv = CausalConv1d(channels, channels, kernel_size=3, causal=causal)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(torch.repeat_interleave(x, self.stride, dim=-1))


@dataclass
class StructuredEncoderConfig:
    input_dim: int = DIM_FEATURES                # 499
    latent_dim: int = 512
    temporal_stride: int = 4
    num_residual_blocks: int = 6
    kernel_size: int = 3
    causal: bool = False
    num_downsample_layers: int = 1
    residual_blocks_per_stage: int | None = None
    dilation_growth_rate: int = 3
    residual_dropout: float = 0.0
    residual_norm: str = "none"               # channel | none


class StructuredEncoder(nn.Module):
    """Motion features → ``latent_dim`` token-rate latent."""

    def __init__(self, cfg: StructuredEncoderConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or StructuredEncoderConfig()
        self.cfg = cfg

        strides = _factor_stride(cfg.temporal_stride, cfg.num_downsample_layers)
        depth = cfg.residual_blocks_per_stage or cfg.num_residual_blocks
        self.input_proj = nn.Sequential(
            CausalConv1d(cfg.input_dim, cfg.latent_dim, kernel_size=3, causal=cfg.causal),
            nn.ReLU(),
        )
        stages: list[nn.Module] = []
        for stride in strides:
            if stride > 1:
                stages.append(DownsampleBlock1d(cfg.latent_dim, stride=stride, causal=cfg.causal))
            stages.append(
                _resnet_stage(
                    cfg.latent_dim,
                    depth=depth,
                    kernel_size=cfg.kernel_size,
                    dilation_growth_rate=cfg.dilation_growth_rate,
                    causal=cfg.causal,
                    dropout=cfg.residual_dropout,
                    norm=cfg.residual_norm,
                )
            )
        self.stages = nn.Sequential(*stages)
        self.out_proj = nn.Sequential(
            _norm_layer(cfg.latent_dim, cfg.residual_norm),
            nn.ReLU(),
            CausalConv1d(cfg.latent_dim, cfg.latent_dim, kernel_size=3, causal=cfg.causal),
        )

    @property
    def latent_dim(self) -> int:
        return self.cfg.latent_dim

    @property
    def temporal_stride(self) -> int:
        return self.cfg.temporal_stride

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 3 or x.shape[1] != self.cfg.input_dim:
            raise ValueError(
                f"StructuredEncoder expects [B, {self.cfg.input_dim}, T]; got {tuple(x.shape)}"
            )
        if x.shape[-1] % self.cfg.temporal_stride != 0:
            raise ValueError(
                f"T={x.shape[-1]} is not divisible by temporal_stride={self.cfg.temporal_stride}"
            )
        return self.out_proj(self.stages(self.input_proj(x)))


@dataclass
class StructuredDecoderConfig:
    output_dim: int = DIM_FEATURES               # 499
    latent_dim: int = 512
    temporal_stride: int = 4
    num_residual_blocks: int = 6
    kernel_size: int = 3
    causal: bool = False
    num_upsample_layers: int = 1
    residual_blocks_per_stage: int | None = None
    dilation_growth_rate: int = 3
    residual_dropout: float = 0.0
    upsample_mode: str = "nearest"
    residual_norm: str = "none"               # channel | none
    decoder_head_mode: str = "single"


class StructuredDecoder(nn.Module):
    """``latent_dim`` token-rate latent → one packed motion-feature head."""

    def __init__(self, cfg: StructuredDecoderConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or StructuredDecoderConfig()
        self.cfg = cfg

        strides = _factor_stride(cfg.temporal_stride, cfg.num_upsample_layers)
        depth = cfg.residual_blocks_per_stage or cfg.num_residual_blocks
        self.input_proj = nn.Sequential(
            CausalConv1d(cfg.latent_dim, cfg.latent_dim, kernel_size=3, causal=cfg.causal),
            nn.ReLU(),
        )
        stages: list[nn.Module] = []
        for stride in reversed(strides):
            stages.append(
                _resnet_stage(
                    cfg.latent_dim,
                    depth=depth,
                    kernel_size=cfg.kernel_size,
                    dilation_growth_rate=cfg.dilation_growth_rate,
                    causal=cfg.causal,
                    dropout=cfg.residual_dropout,
                    norm=cfg.residual_norm,
                    reverse_dilation=True,
                )
            )
            if stride > 1:
                if cfg.upsample_mode != "nearest":
                    raise ValueError("the main decoder uses nearest upsampling")
                stages.append(NearestUpsampleBlock1d(cfg.latent_dim, stride=stride, causal=cfg.causal))
        self.stages = nn.Sequential(*stages)
        self.refine = nn.Sequential(
            CausalConv1d(cfg.latent_dim, cfg.latent_dim, kernel_size=3, causal=cfg.causal),
            nn.ReLU(),
        )
        if cfg.decoder_head_mode != "single" or cfg.upsample_mode != "nearest":
            raise ValueError("the main codec uses a single feature head and nearest upsampling")
        self.head_features = CausalConv1d(cfg.latent_dim, cfg.output_dim, kernel_size=3, causal=cfg.causal)

    def forward(self, z_q: Tensor) -> dict[str, Tensor]:
        if z_q.dim() != 3 or z_q.shape[1] != self.cfg.latent_dim:
            raise ValueError(
                f"StructuredDecoder expects [B, {self.cfg.latent_dim}, T_token]; got {tuple(z_q.shape)}"
            )
        h = self.refine(self.stages(self.input_proj(z_q)))
        return {"features": self.head_features(h)}



__all__ = [
    "DilatedResBlock1d",
    "NearestUpsampleBlock1d",
    "StructuredDecoder",
    "StructuredDecoderConfig",
    "StructuredEncoder",
    "StructuredEncoderConfig",
]
