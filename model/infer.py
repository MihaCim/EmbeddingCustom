"""Load fine-tuned T5EmbeddingModel and encode sentences."""

import torch
from transformers import T5Tokenizer

from model import T5EmbeddingModel


def load_model(checkpoint_path: str, base_model: str = "t5-base", device: str = "cpu"):
    tokenizer = T5Tokenizer.from_pretrained(checkpoint_path)
    model = T5EmbeddingModel(base_model=base_model)
    model.load_state_dict(torch.load(f"{checkpoint_path}.pt", map_location=device))
    model.eval().to(device)
    return model, tokenizer


@torch.no_grad()
def encode(
    texts: list[str],
    model: T5EmbeddingModel,
    tokenizer: T5Tokenizer,
    max_length: int = 128,
    device: str = "cpu",
) -> torch.Tensor:
    enc = tokenizer(texts, padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    return model(**enc)   # (N, hidden_size), L2-normalized


if __name__ == "__main__":
    model, tokenizer = load_model("checkpoint")
    embs = encode(["Hello world", "Hi there"], model, tokenizer)
    sim = embs @ embs.T
    print("Cosine similarity matrix:\n", sim)
