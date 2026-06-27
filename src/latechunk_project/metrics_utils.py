"""Small CPU-only retrieval metric helpers for tests and notebook smoke checks."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


Qrels = Mapping[str, Mapping[str, float]]
Rankings = Mapping[str, Sequence[str]]


def _positive_relevance(doc_rels: Mapping[str, float]) -> dict[str, float]:
    return {doc_id: float(score) for doc_id, score in doc_rels.items() if float(score) > 0}


def _dcg(relevances: Sequence[float]) -> float:
    return sum((2.0**rel - 1.0) / math.log2(rank + 2) for rank, rel in enumerate(relevances))


def compute_ndcg_at_k(rankings: Rankings, qrels: Qrels, k: int = 10) -> float:
    """Compute mean nDCG@k over the union of query IDs."""
    if k <= 0:
        raise ValueError("k must be positive")
    query_ids = sorted(set(rankings) | set(qrels))
    if not query_ids:
        return 0.0

    scores: list[float] = []
    for query_id in query_ids:
        relevant = _positive_relevance(qrels.get(query_id, {}))
        if not relevant:
            scores.append(0.0)
            continue
        ranked_rels = [relevant.get(doc_id, 0.0) for doc_id in rankings.get(query_id, [])[:k]]
        dcg = _dcg(ranked_rels)
        ideal_rels = sorted(relevant.values(), reverse=True)[:k]
        ideal = _dcg(ideal_rels)
        scores.append(dcg / ideal if ideal > 0 else 0.0)
    return sum(scores) / len(scores)


def compute_recall_at_k(rankings: Rankings, qrels: Qrels, k: int = 10) -> float:
    """Compute mean Recall@k over the union of query IDs."""
    if k <= 0:
        raise ValueError("k must be positive")
    query_ids = sorted(set(rankings) | set(qrels))
    if not query_ids:
        return 0.0

    scores: list[float] = []
    for query_id in query_ids:
        relevant_doc_ids = set(_positive_relevance(qrels.get(query_id, {})))
        if not relevant_doc_ids:
            scores.append(0.0)
            continue
        retrieved = set(rankings.get(query_id, [])[:k])
        scores.append(len(relevant_doc_ids & retrieved) / len(relevant_doc_ids))
    return sum(scores) / len(scores)


def ndcg_at_k(rankings: Rankings, qrels: Qrels, k: int = 10) -> float:
    """Alias kept short for notebooks."""
    return compute_ndcg_at_k(rankings, qrels, k)


def recall_at_k(rankings: Rankings, qrels: Qrels, k: int = 10) -> float:
    """Alias kept short for notebooks."""
    return compute_recall_at_k(rankings, qrels, k)
