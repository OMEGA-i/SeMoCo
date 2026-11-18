"""Nucleus sampling of the next semantic code."""
from dataclasses import dataclass
import torch
from torch.nn import functional as F

@dataclass
class SamplingConfig:
    temperature: float = 1.
    top_p: float = .95
    top_k: int = 0

def _sample_one(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int = 0,
    *,
    generators: list[torch.Generator] | None = None,
) -> torch.Tensor:
    """``logits [B, V]`` -> sampled indices ``[B]`` (long).

    When ``generators`` is provided, sampling is performed row-by-row so each
    sample can own an independent RNG stream. This lets batched eval generation
    match per-prompt sampling semantics while still sharing the expensive model
    forward pass across prompts.
    """
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits.float() / temperature
    if top_k > 0 and top_k < logits.shape[-1]:
        kth = torch.topk(logits, k=top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    if generators is not None:
        if len(generators) != probs.shape[0]:
            raise ValueError(f"expected {probs.shape[0]} generators, got {len(generators)}")
        out: list[torch.Tensor] = []
        if 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cdf = torch.cumsum(sorted_probs, dim=-1)
            keep = cdf - sorted_probs <= top_p
            sorted_probs = sorted_probs * keep
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            for i, gen in enumerate(generators):
                choice = torch.multinomial(sorted_probs[i], num_samples=1, generator=gen)
                out.append(sorted_idx[i].gather(0, choice))
        else:
            for i, gen in enumerate(generators):
                out.append(torch.multinomial(probs[i], num_samples=1, generator=gen))
        return torch.cat(out, dim=0)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cdf = torch.cumsum(sorted_probs, dim=-1)
        keep = cdf - sorted_probs <= top_p           # always keep top-1
        sorted_probs = sorted_probs * keep
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        choice = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_idx.gather(-1, choice).squeeze(-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _sample_packet_cfg(model, hidden, sampling, *, cfg_scale, n_cond):
    logits = model.decode_packet(hidden[:, -1])
    conditional, unconditional = logits[:n_cond], logits[n_cond:]
    guided = model.cfg_guide(conditional, unconditional, cfg_scale)
    return _sample_one(guided, sampling.temperature, sampling.top_p, sampling.top_k)[:, None]
