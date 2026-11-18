"""Frozen Flan-T5-XL tokens plus a cached TMR text sentence embedding."""
import torch
from .pipeline import append_retrieval_token


class TextConditioner:
    def __init__(self, model="google/flan-t5-xl", device="cuda", local_files_only=False):
        from transformers import T5EncoderModel, T5Tokenizer
        self.device = torch.device(device)
        self.tokenizer = T5Tokenizer.from_pretrained(model, local_files_only=local_files_only)
        self.model = T5EncoderModel.from_pretrained(model, local_files_only=local_files_only)
        if self.model.config.d_model != 2048:
            raise ValueError("main generators require 2048-dimensional Flan tokens")
        self.model.to(self.device).eval().requires_grad_(False)
        if self.device.type == "cuda":
            self.model.to(torch.bfloat16)

    @torch.inference_mode()
    def encode(self, captions, retrieval_embeddings):
        """TMR embeddings must correspond to the same ordered captions."""
        batch = self.tokenizer(captions, padding=True, truncation=True, max_length=64, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}
        hidden = self.model(**batch).last_hidden_state.float()
        valid = batch["attention_mask"].bool()
        sentence = torch.as_tensor(retrieval_embeddings, device=self.device, dtype=torch.float32)
        if sentence.shape != (len(captions), 256):
            raise ValueError("one cached TMR 256-D text embedding is required per caption")
        rows = [append_retrieval_token(hidden[i:i+1, valid[i]], sentence[i:i+1])[0]
                for i in range(len(captions))]
        width = max(map(len, rows))
        text = hidden.new_zeros((len(rows), width, 2048))
        mask = torch.zeros(len(rows), width, dtype=torch.bool, device=self.device)
        for i, row in enumerate(rows):
            text[i, -len(row):] = row
            mask[i, -len(row):] = True
        return text, mask
