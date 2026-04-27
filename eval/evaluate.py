"""
Embedding model evaluator.

Input format (JSONL):
  {"query": "...", "positives": ["...", ...], "corpus": ["...", ...]}

Each line = one query. `corpus` is the full candidate pool for that query.
`positives` are correct retrievals (must be subsets of `corpus`).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import T5Tokenizer

from model.model import T5EmbeddingModel
from eval.metrics import mrr, recall_at_k, ndcg_at_k, map_at_k


@dataclass
class EvalResult:
    mrr_10: float
    recall_1: float
    recall_5: float
    recall_10: float
    ndcg_10: float
    map_10: float
    n_queries: int
    extra: dict = field(default_factory=dict)

    def __str__(self):
        return (
            f"Queries : {self.n_queries}\n"
            f"MRR@10  : {self.mrr_10:.4f}\n"
            f"NDCG@10 : {self.ndcg_10:.4f}\n"
            f"MAP@10  : {self.map_10:.4f}\n"
            f"R@1     : {self.recall_1:.4f}\n"
            f"R@5     : {self.recall_5:.4f}\n"
            f"R@10    : {self.recall_10:.4f}"
        )


@torch.no_grad()
def _encode_batch(
    texts: list[str],
    model: T5EmbeddingModel,
    tokenizer: T5Tokenizer,
    batch_size: int,
    max_length: int,
    device: str,
) -> torch.Tensor:
    all_embs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tokenizer(chunk, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        all_embs.append(model(**enc).cpu())
    return torch.cat(all_embs, dim=0)


def evaluate(
    model: T5EmbeddingModel,
    tokenizer: T5Tokenizer,
    data: list[dict],
    batch_size: int = 64,
    max_length: int = 128,
    device: str = "cpu",
) -> EvalResult:
    """
    Args:
        data: list of {"query": str, "positives": [str], "corpus": [str]}
    """
    model.eval().to(device)
    all_relevance: list[list[int]] = []

    for item in data:
        query = item["query"]
        positives = set(item["positives"])
        corpus = item["corpus"]

        texts = [query] + corpus
        embs = _encode_batch(texts, model, tokenizer, batch_size, max_length, device)

        q_emb = embs[0].unsqueeze(0)          # (1, D)
        c_embs = embs[1:]                     # (N, D)

        sims = (q_emb @ c_embs.T).squeeze(0)  # (N,)
        ranked_idx = sims.argsort(descending=True).tolist()
        relevance = [1 if corpus[i] in positives else 0 for i in ranked_idx]
        all_relevance.append(relevance)

    return EvalResult(
        n_queries=len(all_relevance),
        mrr_10=mrr(all_relevance, k=10),
        recall_1=recall_at_k(all_relevance, k=1),
        recall_5=recall_at_k(all_relevance, k=5),
        recall_10=recall_at_k(all_relevance, k=10),
        ndcg_10=ndcg_at_k(all_relevance, k=10),
        map_10=map_at_k(all_relevance, k=10),
    )


def evaluate_from_file(
    data_path: str,
    checkpoint_path: str,
    base_model: str = "t5-base",
    **kwargs,
) -> EvalResult:
    tokenizer = T5Tokenizer.from_pretrained(checkpoint_path)
    model = T5EmbeddingModel(base_model=base_model)
    model.load_state_dict(torch.load(f"{checkpoint_path}.pt", map_location="cpu"))

    with open(data_path) as f:
        data = [json.loads(line) for line in f]

    result = evaluate(model, tokenizer, data, **kwargs)
    print(result)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", default="t5-base")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    evaluate_from_file(
        args.data,
        args.checkpoint,
        base_model=args.base_model,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
