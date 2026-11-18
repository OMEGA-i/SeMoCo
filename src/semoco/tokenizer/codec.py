"""Parallel semantic VQ and seven-layer motion RVQ with a shared decoder."""
from __future__ import annotations
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import StructuredEncoder, StructuredDecoder, StructuredEncoderConfig, StructuredDecoderConfig
from .quantizers.split_semantic_rvq import SplitSemanticRVQ


@dataclass
class CodecConfig:
    input_dim: int = 263
    latent_dim: int = 512
    stride: int = 4
    num_layers: int = 2
    depth: int = 3
    dropout: float = .2
    causal: bool = False
    codebook_size: int = 1024
    semantic_codebook_size: int = 1024
    mu: float = .99
    quantize_dropout: bool = True
    quantize_dropout_cutoff_index: int = 0
    quantize_dropout_prob: float = .2
    sample_codebook_temp: float = .5
    semantic_dim: int = 256
    semantic_supervision: bool = True

    def __post_init__(self):
        if self.input_dim not in (263, 499) or self.stride != 4:
            raise ValueError("main codecs use 263 or 499 features at stride four")
        if not self.semantic_supervision:
            raise ValueError("main tokenizer includes semantic supervision")


class SemanticHead(nn.Module):
    def __init__(self, width, output):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, output))

    def forward(self, semantic):
        return F.normalize(self.proj(semantic).mean(1), dim=-1)


class MotionTokenizer(nn.Module):
    """Input/output normalized features [B,F,D]; codes [B,F/4,8]."""
    def __init__(self, cfg: CodecConfig):
        super().__init__()
        self.cfg = cfg
        common = dict(latent_dim=cfg.latent_dim, temporal_stride=cfg.stride, kernel_size=3,
                      causal=cfg.causal, residual_blocks_per_stage=cfg.depth, dilation_growth_rate=3,
                      residual_dropout=cfg.dropout, residual_norm="none")
        self.encoder = StructuredEncoder(StructuredEncoderConfig(input_dim=cfg.input_dim,
                                          num_downsample_layers=cfg.num_layers, **common))
        self.decoder = StructuredDecoder(StructuredDecoderConfig(output_dim=cfg.input_dim,
                                          num_upsample_layers=cfg.num_layers, upsample_mode="nearest",
                                          decoder_head_mode="single", **common))
        self.quantizer = SplitSemanticRVQ(cfg.latent_dim, 7, cfg.codebook_size,
                           semantic_codebook_size=cfg.semantic_codebook_size, mu=cfg.mu,
                           quantize_dropout=cfg.quantize_dropout,
                           quantize_dropout_cutoff_index=cfg.quantize_dropout_cutoff_index,
                           quantize_dropout_prob=cfg.quantize_dropout_prob,
                           sample_codebook_temp=cfg.sample_codebook_temp)
        self.sem_head = SemanticHead(cfg.latent_dim, cfg.semantic_dim)

    def forward(self, features):
        z = self.encoder(features.transpose(1, 2)).transpose(1, 2).contiguous()
        self.quantizer.return_layer_codes = True
        zq, codes, losses = self.quantizer(z)
        semantic = losses.pop("z_sem")
        losses.pop("layer_quantized", None)
        reconstructed = self.decoder(zq.transpose(1, 2).contiguous())["features"].transpose(1, 2)
        return reconstructed, codes, losses, self.sem_head(semantic)

    @torch.no_grad()
    def encode(self, features):
        z = self.encoder(features.transpose(1, 2)).transpose(1, 2).contiguous()
        return self.quantizer(z)[1]

    @torch.no_grad()
    def decode(self, codes):
        if codes.ndim != 3 or codes.shape[-1] != 8:
            raise ValueError("discrete reconstruction requires all eight codes")
        fused = self.quantizer.get_layer_embeddings(codes).sum(2)
        return self.decoder(fused.transpose(1, 2).contiguous())["features"].transpose(1, 2)


def read_codec_config(record):
    if record.get("format") == "semoco.tokenizer.v1":
        return CodecConfig(**record["config"])
    if "codec_config" in record:
        c = record["codec_config"]
        if c.get("groups"):
            raise ValueError("part-wise tokenizers are outside the main method")
        b, q = c["backbone"], dict(c["quantizer"])
        if b.get("family", "conv") != "conv" or b.get("decoder_head_mode", "single") != "single":
            raise ValueError("expected the main convolutional codec")
        if b.get("upsample_mode", "nearest") != "nearest" or b.get("residual_norm", "none") != "none":
            raise ValueError("incompatible convolutional decoder configuration")
        raw = dict(input_dim=c.get("input_dim", 499), latent_dim=b["latent_dim"],
                   stride=b["temporal_stride"], num_layers=b["num_layers"],
                   depth=b.get("residual_blocks_per_stage", 3), dropout=b.get("residual_dropout", .2),
                   causal=b.get("causal", False))
        state = record["net"]
        if "semantic_head.proj.2.weight" in state:
            raw["semantic_dim"] = state["semantic_head.proj.2.weight"].shape[0]
    else:
        raw = dict(record["cfg"]["codec"])
        q = raw.pop("quantizer")
        sem = raw.pop("semantic", {})
        if not sem.get("enabled", False):
            raise ValueError("expected semantic-supervised main tokenizer")
        raw.update(semantic_dim=sem.get("dsem", 256))
    if q.get("kind") != "split_rvq" or q.get("num_quantizers") != 7:
        raise ValueError("expected one semantic and seven parallel motion codebooks")
    raw.update({k: v for k, v in q.items() if k not in ("kind", "num_quantizers")})
    allowed = {f.name for f in fields(CodecConfig)}
    unknown = raw.keys() - allowed
    if unknown:
        raise ValueError(f"unsupported codec settings: {sorted(unknown)}")
    return CodecConfig(**raw)


def read_codec(record):
    cfg = read_codec_config(record)
    model = MotionTokenizer(cfg)
    state = record.get("model", record.get("net"))
    state = {k.replace("semantic_head.", "sem_head.", 1) if k.startswith("semantic_head.") else k: v
             for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    norms = record.get("norm")
    if norms is not None:
        norms = tuple(torch.as_tensor(norms[k], dtype=torch.float32).flatten() for k in ("mean", "std"))
        if any(x.shape != (cfg.input_dim,) or not torch.isfinite(x).all() for x in norms) or not (norms[1] > 0).all():
            raise ValueError("invalid embedded feature normalization")
    return model, norms


def load_codec(path: str | Path, *, dataset: str, device="cpu"):
    record = torch.load(path, map_location="cpu", weights_only=False)
    model, norm = read_codec(record)
    expected = {"humanml3d": 263, "omega": 499}[dataset]
    if model.cfg.input_dim != expected:
        raise ValueError("codec dataset and input width differ")
    if norm is not None:
        norm = tuple(x.to(device) for x in norm)
    return model.to(device).eval().requires_grad_(False), norm


@torch.no_grad()
def encode_flow_targets(codec, features):
    if codec.training or features.ndim != 3 or not features.shape[1] or features.shape[1] % 4:
        raise ValueError("use frozen eval-mode codec and complete four-frame groups")
    z = codec.encoder(features.transpose(1, 2)).transpose(1, 2).contiguous()
    q0 = codec.quantizer(z)[1][..., 0]
    return q0, codec.quantizer.in_proj_kin(z)


def decode_continuous(codec, q0, y, lengths):
    q = codec.quantizer
    fused = q.out_proj_sem(q.semantic_vq.dequantize(q0)) + q.out_proj_kin(y)
    fused = fused.transpose(1, 2).contiguous()
    return [codec.decoder(fused[i:i+1, :, :n])["features"][0].T
            for i, n in enumerate(lengths.tolist())]
