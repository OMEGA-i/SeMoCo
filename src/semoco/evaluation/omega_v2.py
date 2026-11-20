"""Independent Omega-MotionVerse V2 motion/text evaluator and 30 Hz SOMA30 feature extraction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from ..generator.sentence import SentenceEncoder

SOMA30_SUBSET = (
    0, 1, 2, 3, 4, 5, 6, 8, 9, 10,
    11, 12, 13, 14, 18, 28,
    39, 40, 41, 42, 46, 56,
    67, 68, 69, 70, 72, 73, 74, 75,
)
_RIGHT_HIP, _LEFT_HIP = 26, 22
_LEFT_FOOT, _RIGHT_FOOT = (24, 25), (28, 29)
FEATURE_DIM = 186


def _rotate_y(positions: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    cos, sin = torch.cos(angle), torch.sin(angle)
    one, zero = torch.ones_like(angle), torch.zeros_like(angle)
    mat_t = torch.stack((cos, zero, -sin, zero, one, zero, sin, zero, cos), dim=-1).reshape(
        *angle.shape, 3, 3
    )
    return positions @ mat_t


def _rotate_2d(vectors: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    cos, sin = torch.cos(angle), torch.sin(angle)
    mat_t = torch.stack((cos, sin, -sin, cos), dim=-1).reshape(*angle.shape, 2, 2)
    return vectors @ mat_t


def soma77_to_tmr186(joints: np.ndarray | torch.Tensor, *, fps_out: float = 30.0) -> torch.Tensor:
    """Resample 50 Hz SOMA77 world joints ``[T, 77, 3]`` to canonical 30 Hz SOMA30 ``[L, 186]``."""
    raw = joints.detach().cpu().numpy() if torch.is_tensor(joints) else np.asarray(joints)
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[1:] != (77, 3) or len(raw) < 2:
        raise ValueError("expected 50 Hz SOMA77 joints [T >= 2, 77, 3]")
    motion = resample_poly(raw, up=3, down=5, axis=0).astype(np.float32)[:300, SOMA30_SUBSET]
    if len(motion) < 2:
        raise ValueError("resampled 30 Hz clip needs at least two frames")
    posed = torch.from_numpy(motion)[None]
    root_pos = posed[:, :, 0]
    pelvis_rel = posed - root_pos[:, :, None]

    diff = posed[:, :, _RIGHT_HIP] - posed[:, :, _LEFT_HIP]
    heading = torch.atan2(diff[..., 2], -diff[..., 0])
    global_heading = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)

    ground_offset = torch.zeros_like(root_pos)
    ground_offset[..., 1] = root_pos[..., 1]
    local_joints = pelvis_rel[:, :, 1:] + ground_offset[:, :, None]

    delta_pos = fps_out * (posed[:, 1:] - posed[:, :-1])
    velocities = torch.cat((delta_pos, delta_pos[:, -1:]), dim=1)

    local_joints = _rotate_y(local_joints, -heading)
    velocities = _rotate_y(velocities, -heading)

    feet_l = (
        (torch.linalg.norm(velocities[:, :, _LEFT_FOOT], dim=-1) < 0.15)
        & (posed[:, :, _LEFT_FOOT, 1] < 0.10)
    ).to(posed.dtype)
    feet_r = (
        (torch.linalg.norm(velocities[:, :, _RIGHT_FOOT], dim=-1) < 0.15)
        & (posed[:, :, _RIGHT_FOOT, 1] < 0.10)
    ).to(posed.dtype)
    foot_contacts = torch.cat((feet_l, feet_r), dim=-1)

    delta_heading = -torch.atan2(global_heading[:, 0, 1], global_heading[:, 0, 0])
    root_pos = _rotate_y(root_pos, delta_heading).clone()
    global_heading = _rotate_2d(global_heading, delta_heading)
    root_pos[:, :, 0] = root_pos[:, :, 0] - root_pos[:, :1, 0]
    root_pos[:, :, 2] = root_pos[:, :, 2] - root_pos[:, :1, 2]

    return torch.cat(
        (root_pos, global_heading, local_joints.flatten(-2), velocities.flatten(-2), foot_contacts),
        dim=-1,
    )[0]


class OmegaV2Evaluator(nn.Module):
    """Frozen DualEncoder (186-D SOMA30 motion + DistilBERT text) for Omega evaluation."""

    def __init__(self, motion: SentenceEncoder, distilbert: nn.Module, text: SentenceEncoder,
                 mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.motion = motion
        self.distilbert = distilbert
        self.text = text
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        norm_stats: str | Path | None = None,
        *,
        device: str | torch.device = "cuda",
    ) -> "OmegaV2Evaluator":
        from transformers import DistilBertConfig, DistilBertModel

        record = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
        state = record["model"] if isinstance(record, dict) and "model" in record else record
        motion_state = {k.removeprefix("motion."): v for k, v in state.items() if k.startswith("motion.")}
        text_state = {k.removeprefix("text."): v for k, v in state.items() if k.startswith("text.")}
        bert_state = {k.removeprefix("distilbert."): v for k, v in state.items() if k.startswith("distilbert.")}

        bert_cfg_dict = record.get("distilbert_config") if isinstance(record, dict) else None
        bert_cfg = DistilBertConfig(**bert_cfg_dict) if bert_cfg_dict else DistilBertConfig()
        distilbert = DistilBertModel(bert_cfg)
        distilbert.load_state_dict(bert_state, strict=True)

        if norm_stats is not None:
            stats = json.loads(Path(norm_stats).read_text())
        elif isinstance(record, dict) and "normalization" in record:
            stats = record["normalization"]
        else:
            raise ValueError("provide norm_stats JSON or a checkpoint containing normalization")
        mean = torch.tensor(stats["mean"], dtype=torch.float32)
        std = torch.tensor(stats["std"], dtype=torch.float32)
        if mean.shape != (FEATURE_DIM,) or std.shape != (FEATURE_DIM,):
            raise ValueError("Omega V2 normalization requires 186-D mean and std")

        motion_enc = SentenceEncoder.from_state_dict(motion_state, device="cpu")
        text_enc = SentenceEncoder.from_state_dict(text_state, device="cpu")
        evaluator = cls(motion_enc, distilbert, text_enc, mean, std)
        return evaluator.to(device).eval().requires_grad_(False)

    @torch.inference_mode()
    def encode_motion(self, features: list[torch.Tensor]) -> torch.Tensor:
        if not features:
            raise ValueError("provide at least one motion feature sequence")
        device = self.mean.device
        lengths = torch.tensor([len(item) for item in features], device=device, dtype=torch.long)
        padded = pad_sequence([item.to(device=device, dtype=torch.float32) for item in features], batch_first=True)
        normalized = (padded - self.mean) / self.std
        valid = torch.arange(padded.shape[1], device=device)[None] < lengths[:, None]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            return self.motion.encode(normalized, valid)

    @torch.inference_mode()
    def encode_token_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        device = self.mean.device
        ids = input_ids.to(device=device, dtype=torch.long)
        valid = ids != 0
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hidden = self.distilbert(input_ids=ids, attention_mask=valid).last_hidden_state
            return self.text.encode(hidden, valid)

    @torch.inference_mode()
    def encode_captions(
        self,
        captions: list[str],
        *,
        tokenizer_name: str = "distilbert/distilbert-base-uncased",
        local_files_only: bool = False,
    ) -> torch.Tensor:
        from transformers import AutoTokenizer

        if not hasattr(self, "_tokenizer") or getattr(self, "_tokenizer_name", None) != tokenizer_name:
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=local_files_only)
            self._tokenizer_name = tokenizer_name
        tokens = self._tokenizer(
            captions, padding="max_length", truncation=True, max_length=64, return_tensors="pt"
        )["input_ids"]
        return self.encode_token_ids(tokens)
