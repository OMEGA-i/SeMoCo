"""Load the text-conditioned semantic prior."""
from dataclasses import fields
from pathlib import Path
import torch
from .config import MotionGPTConfig
from .model import MotionGPT


def read_semantic_ar(record):
    raw = record.get("model_config", record.get("config", {}).get("model"))
    if raw is None:
        raise ValueError("missing semantic model configuration")
    raw = dict(raw)
    if raw.pop("depth_inject", 0):
        raise ValueError("depth-conditioned priors are outside the main method")
    sizes = raw.pop("codebook_sizes", None)
    if sizes is not None and sizes != [raw["codebook_size"]]:
        raise ValueError("expected a single semantic vocabulary")
    raw.pop("loss_weights", None)
    # These optional settings configure a residual predictor; q0 never
    # calls that predictor. Its weights are removed during export.
    ignored = {"code_pred_hidden", "code_pred_layers", "code_pred_heads", "code_pred_kv_heads",
               "code_pred_ffn_hidden", "code_axis_loss_scale", "code_axis_start_step", "code_axis_warmup_steps"}
    active = {k: v for k, v in raw.items() if k not in ignored}
    unknown = active.keys() - {f.name for f in fields(MotionGPTConfig)}
    if unknown:
        raise ValueError(f"unknown semantic model settings: {sorted(unknown)}")
    model = MotionGPT(MotionGPTConfig(**active))
    allowed_unused = ("decoder.code_blocks.", "decoder.code_norm.", "decoder.code_proj.")
    state = {k: v for k, v in record["model"].items() if not k.startswith(allowed_unused)}
    model.load_state_dict(state, strict=True)
    return model


def load_semantic_ar(path: str | Path, device="cpu"):
    record = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    return read_semantic_ar(record).to(device).eval().requires_grad_(False)
