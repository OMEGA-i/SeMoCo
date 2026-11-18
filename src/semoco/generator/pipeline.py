"""Known-duration semantic sampling followed by whole-clip kinematic generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .sampling import SamplingConfig, _sample_packet_cfg
from .ar import MotionGPT, MotionGPTConfig
from ..tokenizer.codec import decode_continuous
from ..generator.flow import PartFlow


@dataclass(frozen=True)
class SamplingRecipe:
    fps: int
    ar_guidance: float
    flow_guidance: float
    round_up: bool


RECIPES = {
    "humanml3d": SamplingRecipe(20, 3.5, 3., False),
    "omega": SamplingRecipe(50, 3., 1.5, True),
}


def append_retrieval_token(words: torch.Tensor, sentence: torch.Tensor) -> torch.Tensor:
    """Append the Flan/TMR cache's extra token to already aligned Flan tokens."""
    if words.ndim != 3 or words.shape[-1] != 2048 or sentence.shape != (words.shape[0], 256):
        raise ValueError("expected Flan [B,L,2048] and TMR [B,256]")
    extra = words.new_zeros((words.shape[0], 1, 2048))
    extra[:, 0, :256] = sentence.to(words) * 8.
    return torch.cat((words, extra), dim=1)


def motion_lengths(frames: torch.Tensor, dataset: str) -> torch.Tensor:
    """Translate reference record counts to the dataset's four-frame token convention."""
    if frames.ndim != 1 or frames.is_floating_point() or frames.dtype == torch.bool:
        raise ValueError("frames must be a vector of integer record counts")
    result = (frames.long() + (3 if RECIPES[dataset].round_up else 0)) // 4
    if not bool((result >= 1).all()):
        raise ValueError("each duration must produce at least one token")
    return result


from .ar.checkpoint import load_semantic_ar


@torch.no_grad()
def sample_q0(
    model: MotionGPT, text: torch.Tensor, text_valid: torch.Tensor, lengths: torch.Tensor,
    *, guidance: float = 3.5, top_p: float = .95, temperature: float = 1.,
) -> torch.Tensor:
    """A KV-cached CFG rollout over q0. EOS does not change the supplied lengths."""
    if model.cfg.num_codebooks != 1 or not model.cfg.use_text:
        raise ValueError("semantic sampling requires a text-conditioned one-codebook AR")
    if model.training:
        raise ValueError("put the semantic AR in eval mode before sampling")
    batch, width = text.shape[:2]
    if lengths.shape != (batch,) or bool((lengths < 1).any()):
        raise ValueError("need a positive token count for each clip")
    limit = int(lengths.max())
    if width + 1 + limit > model.cfg.max_seq_len:
        raise ValueError("text prefix and requested motion exceed the AR position table")

    words = model.project_text(text)
    words = torch.cat((words, model.null_text.to(words.dtype).expand_as(words)))
    bos = model.motion_bos.to(words.dtype).expand(2 * batch, 1, -1)
    lengths2 = lengths.repeat(2)
    if model.cfg.len_cond:
        bos = bos + model.length_embedding(lengths2, 1, 0)
    roles = torch.zeros(2 * batch, width + 1, dtype=torch.long, device=text.device)
    roles[:, -1] = 1
    keys = torch.cat((text_valid.repeat(2, 1), torch.ones_like(roles[:, :1], dtype=torch.bool)), dim=1)
    cache = model.init_caches(2 * batch, text.device, words.dtype)
    cache, hidden = model.forward_embeds(
        torch.cat((words, bos), dim=1), attn_mask=model.mixed_attn_mask(roles, keys),
        caches=cache, start_pos=0, run_decoder=False,
    )
    settings = SamplingConfig(temperature=temperature, top_p=top_p, top_k=0)
    output = []
    for index in range(limit):
        packet = _sample_packet_cfg(model, hidden, settings, cfg_scale=guidance, n_cond=batch)
        output.append(packet[:, 0])
        if index + 1 == limit:
            break
        keys = torch.cat((keys, torch.ones_like(keys[:, :1])), dim=1)
        embedded = model.embed(packet.repeat(2, 1)[:, None])
        if model.cfg.len_cond:
            embedded = embedded + model.length_embedding(lengths2, 1, index + 1)
        cache, hidden = model.forward_embeds(
            embedded, attn_mask=keys[:, None, None, :], caches=cache,
            start_pos=width + index + 1, run_decoder=False,
        )
    return torch.stack(output, dim=1)


@torch.inference_mode()
def generate(
    *, ar: MotionGPT | None, flow: PartFlow, codec, text: torch.Tensor, text_valid: torch.Tensor,
    frames: torch.Tensor, latent_mean: torch.Tensor, latent_std: torch.Tensor,
    q0: torch.Tensor | None = None, seed: int = 7, ar_guidance: float | None = None,
    flow_guidance: float | None = None, steps: int = 32, initial_noise: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Return unpadded codec features, semantic codes, and raw continuous latents."""
    recipe = RECIPES[flow.dataset]
    lengths = motion_lengths(frames, flow.dataset)
    max_length = int(lengths.max())
    if max_length > flow.pos.shape[1]:
        raise ValueError("duration exceeds the Flow checkpoint's learned position table")
    valid = torch.arange(max_length, device=text.device)[None] < lengths[:, None]
    if flow.training or codec.training:
        raise ValueError("put the Flow and codec in eval mode before generating")
    # CPU noise and a separate AR stream allow controlled comparisons between checkpoints.
    rng = torch.Generator().manual_seed(seed)
    with torch.autocast(text.device.type, dtype=torch.bfloat16, enabled=text.device.type == "cuda"):
        if q0 is None:
            if ar is None:
                raise ValueError("supply a semantic AR or an already sampled q0 plan")
            devices = [text.device.index or 0] if text.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(seed)
                if devices:
                    with torch.cuda.device(text.device):
                        torch.cuda.manual_seed(seed)
                q0 = sample_q0(ar, text, text_valid, lengths,
                               guidance=recipe.ar_guidance if ar_guidance is None else ar_guidance)
        if q0.shape != valid.shape:
            raise ValueError(f"q0 shape must match the duration mask {tuple(valid.shape)}")
        if q0.dtype != torch.long or not bool(((q0 >= 0) & (q0 < flow.q0.num_embeddings)).all()):
            raise ValueError("q0 must contain in-vocabulary long indices")
        latent = flow.sample(
            q0, valid, text, text_valid, generator=rng, steps=steps,
            cfg=recipe.flow_guidance if flow_guidance is None else flow_guidance,
            initial_noise=initial_noise,
        )
        raw = latent * latent_std + latent_mean
    # Decode each valid span independently. Padded tokens must not affect clip edges.
    with torch.autocast(text.device.type, dtype=torch.bfloat16,
                        enabled=text.device.type == "cuda" and flow.dataset == "omega"):
        motion = decode_continuous(codec, q0, raw, lengths)
    motion = [features[:n].float() for features, n in zip(motion, frames.tolist(), strict=True)]
    return motion, q0, raw
