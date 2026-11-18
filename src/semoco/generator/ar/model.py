"""Text-conditioned causal semantic-code Transformer with KV-cache inference."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint
from .config import MotionGPTConfig
from .transformer import Block, KVCache, RMSNorm, build_rope_cache


class SemanticEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tables = nn.ModuleList([nn.Embedding(cfg.codebook_size, cfg.d_model)])
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, codes):
        if codes.shape[-1] != 1:
            raise ValueError("semantic AR accepts only q0")
        return self.dropout(self.tables[0](codes[..., 0]))


class SemanticHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.q0_head = nn.Linear(cfg.d_model, cfg.codebook_size, bias=False)

    def forward(self, hidden, target_codes=None):
        return [self.q0_head(hidden)]


class MotionGPT(nn.Module):
    def __init__(self, cfg: MotionGPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = SemanticEmbedding(cfg)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.decoder = SemanticHead(cfg)
        self.text_proj = nn.Linear(cfg.clip_dim, cfg.d_model, bias=True)
        self.null_text = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.motion_bos = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.eos_head = nn.Linear(cfg.d_model, 1, bias=True)
        if cfg.len_cond:
            self.len_emb = nn.Embedding(cfg.len_max + 1, cfg.d_model)
        nn.init.normal_(self.null_text, std=.02)
        nn.init.normal_(self.motion_bos, std=.02)
        cos, sin = build_rope_cache(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta,
                                   device=torch.device("cpu"), dtype=torch.float32)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if name.endswith(("wo.weight", "w2.weight")):
                    parameter.mul_(1 / math.sqrt(cfg.n_layers))

    def decode_packet(self, hidden):
        return self.decoder.q0_head(hidden)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def _rope_slice(self, start: int, length: int, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        end = start + length
        if end > self.cfg.max_seq_len:
            raise ValueError(f"sequence position {end} exceeds max_seq_len={self.cfg.max_seq_len}")
        cos = self.rope_cos[start:end].to(dtype)
        sin = self.rope_sin[start:end].to(dtype)
        return cos, sin

    @staticmethod
    def mixed_attn_mask(role: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Prefix + mixed-attention mask -> bool ``[B, 1, S, S]``.

        ``role``  ``[B, S]``: 0 = text token, 1 = motion token (incl. motion BOS).
        ``valid`` ``[B, S]``: True where the key is a real (non-padding) token.

        Rules (the paper's mixed attention): text attends **bidirectionally** to
        text; motion attends to **all** text and **causally** to motion; text
        never attends to motion. The diagonal is always allowed so padding query
        rows are never fully masked (no NaNs).
        """
        B, S = role.shape
        dev = role.device
        is_text = role == 0                       # [B, S]
        is_mot = role == 1
        seq = torch.arange(S, device=dev)
        causal = (seq[None, :] <= seq[:, None])    # [S, S] key <= query
        qi_text = is_text[:, :, None]
        qi_mot = is_mot[:, :, None]
        kj_text = is_text[:, None, :]
        kj_mot = is_mot[:, None, :]
        kj_valid = valid[:, None, :]
        allow = kj_valid & (
            (qi_text & kj_text)
            | (qi_mot & kj_text)
            | (qi_mot & kj_mot & causal[None])
        )
        eye = torch.eye(S, dtype=torch.bool, device=dev)[None]
        allow = allow | eye
        return allow.unsqueeze(1)                   # [B, 1, S, S]

    def _run_backbone(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        caches: list[KVCache] | None = None,
        doc_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[KVCache] | None]:
        """Run all backbone blocks + final RMSNorm. Returns (hidden, new_caches)."""
        new_caches: list[KVCache] | None = [] if caches is not None else None
        use_ckpt = self.training and self.cfg.gradient_checkpointing and caches is None
        for i, block in enumerate(self.blocks):
            layer_cache = caches[i] if caches is not None else None
            if use_ckpt:
                x, nc = torch.utils.checkpoint.checkpoint(
                    lambda x, cos, sin, cache, doc_mask, block=block: block(x, cos, sin, cache=cache, doc_mask=doc_mask),
                    x, cos, sin, layer_cache, doc_mask,
                    use_reentrant=False,
                )
            else:
                x, nc = block(x, cos, sin, cache=layer_cache, doc_mask=doc_mask)
            if new_caches is not None:
                new_caches.append(nc)
        return self.norm(x), new_caches

    def forward_embeds(
        self,
        inputs_embeds: torch.Tensor,
        *,
        attn_mask: torch.Tensor,
        caches: list[KVCache] | None = None,
        start_pos: int = 0,
        run_decoder: bool = True,
        return_hidden: bool = False,
    ) -> tuple:
        """Forward from pre-computed embeddings (text+motion prefill / decode).

        ``inputs_embeds [B, S, D]`` with an explicit ``attn_mask [B, 1, S, S]``.
        ``run_decoder=False`` returns ``(new_caches, hidden)`` so the caller can
        slice the motion span before running the semantic head.
        """
        cos, sin = self._rope_slice(start_pos, inputs_embeds.shape[1], inputs_embeds.dtype)
        x, new_caches = self._run_backbone(inputs_embeds, cos, sin, caches=caches, doc_mask=attn_mask)
        if not run_decoder:
            return new_caches, x
        logits = self.decoder(x)
        if return_hidden:
            return logits, new_caches, x
        return logits, new_caches

    def project_text(self, text_emb: torch.Tensor) -> torch.Tensor:
        """L2-normalize per token, then project ``clip_dim -> d_model``.

        Raw encoder states are not scale-compatible across Flan / SigLIP / Qwen3;
        normalizing here keeps the shared per-token conditioning interface without
        re-exporting caches. Zero pad rows stay zero (``F.normalize`` eps path).
        """
        return self.text_proj(F.normalize(text_emb, dim=-1))

    def replace_text_with_null(
        self, text_x: torch.Tensor, drop_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Replace projected text with the learned null token for CFG training.

        ``text_x [B, L, d_model]``, ``drop_mask [B]`` bool (True = drop this sample).
        Returns ``text_x`` with dropped rows replaced by ``self.null_text``.
        If ``drop_mask`` is None, returns ``text_x`` unchanged.
        """
        if drop_mask is None:
            return text_x
        B = text_x.shape[0]
        null = self.null_text.to(text_x.dtype).expand(B, text_x.shape[1], -1)
        return torch.where(drop_mask.view(B, 1, 1), null, text_x)

    @staticmethod
    def cfg_guide(cond: torch.Tensor, uncond: torch.Tensor, scale: float) -> torch.Tensor:
        """Classifier-free guidance: ``uncond + scale * (cond - uncond)``."""
        if scale == 1.0:
            return cond
        return uncond + scale * (cond - uncond)

    def t2m_train_step(
        self,
        text_emb: torch.Tensor,
        text_valid: torch.Tensor,
        motion_codes: torch.Tensor,
        motion_valid: torch.Tensor,
        *,
        drop_text: torch.Tensor | None = None,
    ) -> tuple:
        """Teacher-forced text2motion pass.

        ``text_emb``    ``[B, Lt, clip_dim]`` frozen Flan-T5 word embeddings.
        ``text_valid``  ``[B, Lt]`` bool (True = real word).
        ``motion_codes````[B, Tm, Q]`` long packet codes.
        ``motion_valid````[B, Tm]`` bool (True = real motion frame).
        ``drop_text``   ``[B]`` bool (CFG: replace text with the null token).

        Returns ``(q_logits, eos_logits)`` where ``q_logits[i]`` is ``[B, Tm, V]``
        predicting ``motion_codes[..., 0]`` (BOS->m0, m0->m1, ...), and
        ``eos_logits`` is ``[B, Tm+1]`` (stop logit at each motion-block position;
        index 0 = BOS).
        """
        if not self.cfg.use_text:
            raise RuntimeError("t2m_train_step requires cfg.use_text=True")
        B, Tm, _ = motion_codes.shape
        text_x = self.project_text(text_emb)                     # [B, Lt, D]
        text_x = self.replace_text_with_null(text_x, drop_text)
        mot_emb = self.embed(motion_codes)
        bos = self.motion_bos.to(mot_emb.dtype).expand(B, 1, -1)
        mot_x = torch.cat([bos, mot_emb], dim=1)                 # [B, Tm+1, D]
        if self.cfg.len_cond:
            mot_x = mot_x + self.length_embedding(motion_valid.sum(1), Tm + 1).to(mot_x.dtype)
        x = torch.cat([text_x, mot_x], dim=1)                    # [B, S, D]

        Lt = text_x.shape[1]
        dev = x.device
        role = torch.cat(
            [
                torch.zeros(B, Lt, dtype=torch.long, device=dev),
                torch.ones(B, Tm + 1, dtype=torch.long, device=dev),
            ],
            dim=1,
        )
        valid = torch.cat(
            [
                text_valid.bool(),
                torch.ones(B, 1, dtype=torch.bool, device=dev),   # BOS
                motion_valid.bool(),
            ],
            dim=1,
        )
        mask = self.mixed_attn_mask(role, valid)
        _, hidden = self.forward_embeds(x, attn_mask=mask, run_decoder=False)
        mot_hidden = hidden[:, Lt:, :]                            # [B, Tm+1, D]
        pred_hidden = mot_hidden[:, :Tm, :]
        q_logits = self.decoder(pred_hidden, target_codes=motion_codes)
        eos_logits = self.eos_head(mot_hidden).squeeze(-1)       # [B, Tm+1]
        return q_logits, eos_logits

    def length_embedding(self, n_total: torch.Tensor, length: int, start: int = 0) -> torch.Tensor:
        """Remaining packet count at each input position (BOS is position zero)."""
        idx = torch.arange(start, start + length, device=n_total.device)
        remaining = (n_total.long()[:, None] - idx[None]).clamp(0, self.cfg.len_max)
        return self.len_emb(remaining)

    def init_caches(self, batch_size: int, device, dtype) -> list[KVCache]:
        hd = self.cfg.head_dim
        hkv = self.cfg.kv_heads
        empty = lambda: torch.zeros(batch_size, hkv, 0, hd, device=device, dtype=dtype)
        return [KVCache(k=empty(), v=empty()) for _ in range(self.cfg.n_layers)]
