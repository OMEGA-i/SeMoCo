"""Checkpoint and random-state utilities for training."""
import random
from pathlib import Path
import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending")
    torch.save(state, pending)
    pending.replace(path)


def random_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def restore_random(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def codebook_runtime(model):
    """EMA accumulators are runtime state, separate from inference codebooks."""
    from ..tokenizer.quantizers.ema_rvq import QuantizeEMAReset
    return {name: dict(initialized=module.init,
                       total=None if module.code_sum is None else module.code_sum.detach().cpu().clone(),
                       count=None if module.code_count is None else module.code_count.detach().cpu().clone())
            for name, module in model.named_modules() if isinstance(module, QuantizeEMAReset)}


def restore_codebook_runtime(model, record):
    modules = dict(model.named_modules())
    for name, state in record.items():
        module = modules[name]
        module.init = state["initialized"]
        module.code_sum = None if state["total"] is None else state["total"].to(module.codebook.device)
        module.code_count = None if state["count"] is None else state["count"].to(module.codebook.device)
