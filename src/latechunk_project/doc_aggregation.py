"""Chunk-to-document aggregation methods for Late Chunking experiments."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any, Mapping, Sequence

from latechunk_project.chunk_ranking import ChunkHit, sort_chunk_hits


AGGREGATION_METHODS = [
    "first_occurrence",
    "max_score",
    "topk_mean",
    "softmax_topk",
    "max_plus_coverage",
    "rrf_chunk_vote",
]

METHOD_FAMILIES = {
    "first_occurrence": "official_baseline",
    "max_score": "late_chunk_aggregation",
    "topk_mean": "late_chunk_aggregation",
    "softmax_topk": "late_chunk_aggregation",
    "max_plus_coverage": "late_chunk_aggregation",
    "rrf_chunk_vote": "late_chunk_aggregation",
    "bm25_only": "lexical",
    "late_first_occurrence_plus_bm25_rrf": "hybrid",
    "late_softmax_topk_plus_bm25_rrf": "hybrid",
}


def _group_hits_by_doc(hits: Sequence[ChunkHit]) -> dict[str, list[ChunkHit]]:
    grouped: dict[str, list[ChunkHit]] = defaultdict(list)
    for hit in sort_chunk_hits(hits):
        grouped[hit.doc_id].append(hit)
    return {doc_id: sort_chunk_hits(doc_hits) for doc_id, doc_hits in grouped.items()}


def _softmax_weighted_score(scores: Sequence[float], tau: float) -> float:
    if not scores:
        return 0.0
    if tau <= 0:
        raise ValueError("tau must be positive")
    scaled = [score / tau for score in scores]
    max_scaled = max(scaled)
    exps = [math.exp(value - max_scaled) for value in scaled]
    denom = sum(exps)
    return sum(score * weight / denom for score, weight in zip(scores, exps))


def aggregate_query_chunks(
    hits: Sequence[ChunkHit],
    method: str,
    topk_mean_k: int = 3,
    softmax_topk_k: int = 3,
    softmax_tau: float = 0.05,
    coverage_beta: float = 0.01,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Aggregate a single query's chunk ranking into a document ranking."""
    if method not in AGGREGATION_METHODS:
        raise ValueError(f"Unsupported aggregation method: {method}")
    if not hits:
        return []

    grouped = _group_hits_by_doc(hits)
    rows: list[dict[str, Any]] = []
    for doc_id, doc_hits in grouped.items():
        scores = [hit.chunk_score for hit in doc_hits]
        ranks = [hit.chunk_rank for hit in doc_hits]
        top_hit = doc_hits[0]

        if method == "first_occurrence":
            doc_score = top_hit.chunk_score
        elif method == "max_score":
            doc_score = max(scores)
        elif method == "topk_mean":
            doc_score = sum(scores[:topk_mean_k]) / min(len(scores), topk_mean_k)
        elif method == "softmax_topk":
            doc_score = _softmax_weighted_score(scores[:softmax_topk_k], softmax_tau)
        elif method == "max_plus_coverage":
            doc_score = max(scores) + coverage_beta * math.log1p(len(doc_hits))
        elif method == "rrf_chunk_vote":
            doc_score = sum(1.0 / (rrf_k + rank) for rank in ranks)
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")

        rows.append(
            {
                "doc_id": doc_id,
                "doc_score": float(doc_score),
                "best_chunk_rank": min(ranks),
                "best_chunk_score": float(max(scores)),
                "num_retrieved_chunks": len(doc_hits),
                "chunk_ids": [hit.chunk_id for hit in doc_hits],
            }
        )

    rows = sorted(
        rows,
        key=lambda row: (
            -float(row["doc_score"]),
            int(row["best_chunk_rank"]),
            str(row["doc_id"]),
        ),
    )
    for rank, row in enumerate(rows, start=1):
        row["doc_rank"] = rank
    return rows


def aggregate_chunk_rankings(
    rankings: Mapping[str, Sequence[ChunkHit]],
    method: str,
    **kwargs: Any,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Aggregate all query chunk rankings and return timings/diagnostics."""
    started = time.perf_counter()
    doc_rankings = {
        query_id: aggregate_query_chunks(hits, method=method, **kwargs)
        for query_id, hits in rankings.items()
    }
    elapsed = time.perf_counter() - started
    num_queries = len(doc_rankings)
    diagnostics = {
        "method": method,
        "total_runtime_seconds": elapsed,
        "avg_latency_per_query_seconds": elapsed / num_queries if num_queries else 0.0,
        "num_queries": num_queries,
    }
    return doc_rankings, diagnostics

