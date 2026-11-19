"""Configuration of the text-conditioned, single-codebook semantic prior."""
from dataclasses import dataclass


@dataclass
class MotionGPTConfig:
    num_codebooks: int = 1
    codebook_size: int = 1024
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 8
    n_kv_heads: int = 4
    ffn_hidden: int = 2730
    rope_theta: float = 1_000_000.
    norm_eps: float = 1e-5
    dropout: float = .2
    max_seq_len: int = 4096
    context_length: int = 122
    gradient_checkpointing: bool = False
    use_text: bool = True
    clip_dim: int = 2048
    text_cond_prob: float = .1
    eos_loss_scale: float = 1.
    len_cond: int = 1
    len_max: int = 64

    def __post_init__(self):
        self.norm_eps = float(self.norm_eps)
        if self.num_codebooks != 1 or not self.use_text:
            raise ValueError("the main prior predicts one semantic code from text")
        if self.d_model % self.n_heads or self.n_heads % self.kv_heads or self.head_dim % 2:
            raise ValueError("invalid grouped-query attention dimensions")

    @property
    def head_dim(self):
        return self.d_model // self.n_heads

    @property
    def kv_heads(self):
        return self.n_kv_heads or self.n_heads
