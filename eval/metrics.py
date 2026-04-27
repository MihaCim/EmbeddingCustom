"""Retrieval metrics: MRR, NDCG@k, Recall@k, MAP."""

import math
import numpy as np


def mrr(relevance: list[list[int]], k: int = 10) -> float:
    """Mean Reciprocal Rank. relevance[i] = ranked list of binary labels for query i."""
    scores = []
    for ranks in relevance:
        rr = 0.0
        for j, rel in enumerate(ranks[:k], start=1):
            if rel:
                rr = 1.0 / j
                break
        scores.append(rr)
    return float(np.mean(scores))


def recall_at_k(relevance: list[list[int]], k: int = 10) -> float:
    scores = []
    for ranks in relevance:
        n_relevant = sum(ranks)
        if n_relevant == 0:
            continue
        hit = sum(ranks[:k])
        scores.append(hit / n_relevant)
    return float(np.mean(scores))


def ndcg_at_k(relevance: list[list[int]], k: int = 10) -> float:
    def dcg(rels, k):
        return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))

    scores = []
    for ranks in relevance:
        ideal = sorted(ranks, reverse=True)
        d = dcg(ranks, k)
        i = dcg(ideal, k)
        scores.append(d / i if i > 0 else 0.0)
    return float(np.mean(scores))


def map_at_k(relevance: list[list[int]], k: int = 10) -> float:
    scores = []
    for ranks in relevance:
        hits, prec_sum = 0, 0.0
        for j, rel in enumerate(ranks[:k], start=1):
            if rel:
                hits += 1
                prec_sum += hits / j
        n_rel = sum(ranks)
        scores.append(prec_sum / n_rel if n_rel else 0.0)
    return float(np.mean(scores))
