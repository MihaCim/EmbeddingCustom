"""T5-based embedding model. Encoder-only path; decoder discarded."""

import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Tokenizer


class T5EmbeddingModel(nn.Module):
    def __init__(self, base_model: str = "t5-base", pooling: str = "mean", normalize: bool = True):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(base_model)
        self.pooling = pooling
        self.normalize = normalize

    def pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        if self.pooling == "cls":
            return hidden[:, 0]
        raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        emb = self.pool(hidden, attention_mask)
        if self.normalize:
            emb = nn.functional.normalize(emb, p=2, dim=-1)
        return emb
