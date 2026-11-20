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
    def token_states(self, captions):
        batch = self.tokenizer(captions, padding=True, truncation=True, max_length=64, return_tensors="pt")
        batch = {key: value.to(self.device) for key, value in batch.items()}
        hidden = self.model(**batch).last_hidden_state.float()
        return hidden, batch["attention_mask"].bool()

    def pack(self, hidden, valid, sentence):
        sentence = torch.as_tensor(sentence, device=self.device, dtype=torch.float32)
        if sentence.shape != (hidden.shape[0], 256):
            raise ValueError("one 256-D sentence embedding is required per caption")
        rows = [append_retrieval_token(hidden[i:i+1, valid[i]], sentence[i:i+1])[0]
                for i in range(hidden.shape[0])]
        width = max(map(len, rows))
        text = hidden.new_zeros((len(rows), width, 2048))
        mask = torch.zeros(len(rows), width, dtype=torch.bool, device=self.device)
        for i, row in enumerate(rows):
            text[i, -len(row):] = row
            mask[i, -len(row):] = True
        return text, mask

    @torch.inference_mode()
    def encode(self, captions, retrieval_embeddings):
        """Sentence embeddings must correspond to the same ordered captions."""
        hidden, valid = self.token_states(captions)
        return self.pack(hidden, valid, retrieval_embeddings)
