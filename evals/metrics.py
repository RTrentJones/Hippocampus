"""Retrieval metrics: recall@k, MRR, nDCG@k.

Pure functions over ranked ID lists so they are unit-testable without a
database or embeddings.
"""

import math


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items appearing in the top k."""
    if not relevant:
        return 0.0
    hits = sum(1 for item in retrieved[:k] if item in relevant)
    return hits / len(relevant)


def hit_at_k(retrieved: list[str], target: str, k: int) -> float:
    """1.0 if the target appears in the top k, else 0.0."""
    return 1.0 if target in retrieved[:k] else 0.0


def mrr(retrieved: list[str], target: str) -> float:
    """Reciprocal rank of the target (0 when absent)."""
    for rank, item in enumerate(retrieved, start=1):
        if item == target:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevance: dict[str, float], k: int) -> float:
    """Normalized discounted cumulative gain over graded relevance."""
    dcg = sum(
        relevance.get(item, 0.0) / math.log2(rank + 1)
        for rank, item in enumerate(retrieved[:k], start=1)
    )
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
