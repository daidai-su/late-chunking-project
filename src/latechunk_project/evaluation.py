"""Evaluation helpers for document rankings."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


Qrels = Mapping[str, Mapping[str, float]]
DocRankings = Mapping[str, Sequence[Mapping[str, Any]]]


def ranked_doc_ids(ranking: Sequence[Mapping[str, Any]]) -> list[str]:
    """Extract document IDs from ranking rows."""
    ordered = sorted(
        ranking,
        key=lambda row: (
            int(row.get("doc_rank", 10**12)),
            -float(row.get("doc_score", 0.0)),
            str(row["doc_id"]),
        ),
    )
    return [str(row["doc_id"]) for row in ordered]


def _positive_qrels(doc_scores: Mapping[str, float]) -> dict[str, float]:
    return {str(doc_id): float(score) for doc_id, score in doc_scores.items() if float(score) > 0}


def dcg(relevances: Sequence[float]) -> float:
    """Discounted cumulative gain."""
    return sum((2.0**rel - 1.0) / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def query_ndcg_at_k(doc_ids: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    """Compute nDCG@k for one query."""
    positive = _positive_qrels(relevant)
    if not positive:
        return 0.0
    gains = [positive.get(doc_id, 0.0) for doc_id in doc_ids[:k]]
    ideal = sorted(positive.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    return dcg(gains) / ideal_dcg if ideal_dcg else 0.0


def query_recall_at_k(doc_ids: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    """Compute Recall@k for one query."""
    positive = set(_positive_qrels(relevant))
    if not positive:
        return 0.0
    return len(positive & set(doc_ids[:k])) / len(positive)


def query_mrr_at_k(doc_ids: Sequence[str], relevant: Mapping[str, float], k: int) -> float:
    """Compute reciprocal rank@k for one query."""
    positive = set(_positive_qrels(relevant))
    for rank, doc_id in enumerate(doc_ids[:k], start=1):
        if doc_id in positive:
            return 1.0 / rank
    return 0.0


def per_query_metrics(
    rankings: DocRankings,
    qrels: Qrels,
    ndcg_k: int = 10,
    recall_k: int = 100,
    mrr_k: int = 10,
) -> list[dict[str, Any]]:
    """Compute per-query metrics for document rankings."""
    rows: list[dict[str, Any]] = []
    for query_id in sorted(set(rankings) | set(qrels)):
        doc_ids = ranked_doc_ids(rankings.get(query_id, []))
        relevant = qrels.get(query_id, {})
        rows.append(
            {
                "query_id": query_id,
                f"ndcg_at_{ndcg_k}": query_ndcg_at_k(doc_ids, relevant, ndcg_k),
                f"recall_at_{recall_k}": query_recall_at_k(doc_ids, relevant, recall_k),
                f"mrr_at_{mrr_k}": query_mrr_at_k(doc_ids, relevant, mrr_k),
                "num_relevant_docs": len(_positive_qrels(relevant)),
            }
        )
    return rows


def aggregate_metrics(per_query_rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Average numeric per-query metric columns."""
    if not per_query_rows:
        return {}
    metric_keys = [
        key
        for key, value in per_query_rows[0].items()
        if key != "query_id" and isinstance(value, (int, float))
    ]
    return {
        key: sum(float(row.get(key, 0.0)) for row in per_query_rows) / len(per_query_rows)
        for key in metric_keys
    }


def evaluate_rankings(
    rankings: DocRankings,
    qrels: Qrels,
    ndcg_k: int = 10,
    recall_k: int = 100,
    mrr_k: int = 10,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Return aggregate and per-query metrics."""
    rows = per_query_metrics(rankings, qrels, ndcg_k=ndcg_k, recall_k=recall_k, mrr_k=mrr_k)
    return aggregate_metrics(rows), rows

