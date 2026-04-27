"""Fine-tune T5EmbeddingModel with MultipleNegativesRankingLoss (in-batch negatives)."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, get_linear_schedule_with_warmup

from model import T5EmbeddingModel


# ---------- dataset ----------

class PairDataset(Dataset):
    """Each item: (query, positive). Negatives come from other items in the batch."""

    def __init__(self, pairs: list[tuple[str, str]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate(batch, tokenizer, max_length=128):
    queries, positives = zip(*batch)
    enc_q = tokenizer(list(queries), padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
    enc_p = tokenizer(list(positives), padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
    return enc_q, enc_p


# ---------- loss ----------

def mnrl_loss(q_emb: torch.Tensor, p_emb: torch.Tensor, scale: float = 20.0) -> torch.Tensor:
    """Multiple Negatives Ranking Loss. All other positives in batch act as negatives."""
    scores = scale * q_emb @ p_emb.T          # (B, B)
    labels = torch.arange(len(scores), device=scores.device)
    return F.cross_entropy(scores, labels)


# ---------- training loop ----------

def train(
    pairs: list[tuple[str, str]],
    base_model: str = "t5-base",
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    max_length: int = 128,
    warmup_ratio: float = 0.1,
    save_path: str = "checkpoint",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    tokenizer = T5Tokenizer.from_pretrained(base_model)
    model = T5EmbeddingModel(base_model=base_model).to(device)

    dataset = PairDataset(pairs)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer, max_length),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for step, (enc_q, enc_p) in enumerate(loader):
            enc_q = {k: v.to(device) for k, v in enc_q.items()}
            enc_p = {k: v.to(device) for k, v in enc_p.items()}

            q_emb = model(**enc_q)
            p_emb = model(**enc_p)
            loss = mnrl_loss(q_emb, p_emb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if (step + 1) % 50 == 0:
                print(f"epoch {epoch+1} step {step+1} loss {loss.item():.4f}")

        print(f"epoch {epoch+1} avg_loss {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), f"{save_path}.pt")
    tokenizer.save_pretrained(save_path)
    print(f"Saved to {save_path}")
    return model, tokenizer


if __name__ == "__main__":
    # Toy example — replace with real data
    pairs = [
        ("What is the capital of France?", "Paris is the capital of France."),
        ("Best way to learn Python?", "Practice by building small projects."),
    ]
    train(pairs, base_model="t5-base", epochs=1, batch_size=2)
