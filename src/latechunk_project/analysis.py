"""Analysis helpers for aggregation experiments."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from latechunk_project.chunk_ranking import ChunkHit
from latechunk_project.evaluation import ranked_doc_ids


def chunks_per_doc_stats(rankings: Mapping[str, Sequence[ChunkHit]]) -> dict[str, float]:
    """Compute corpus-level stats from retrieved chunk rankings."""
    counts: list[int] = []
    duplicate_fractions: list[float] = []
    for hits in rankings.values():
        per_doc: dict[str, int] = {}
        for hit in hits:
            per_doc[hit.doc_id] = per_doc.get(hit.doc_id, 0) + 1
        counts.extend(per_doc.values())
        if hits:
            duplicate_fractions.append(sum(count - 1 for count in per_doc.values() if count > 1) / len(hits))
    return {
        "num_retrieved_chunks": float(sum(len(hits) for hits in rankings.values())),
        "num_unique_retrieved_docs": float(len({hit.doc_id for hits in rankings.values() for hit in hits})),
        "avg_chunks_per_retrieved_doc": sum(counts) / len(counts) if counts else 0.0,
        "avg_duplicate_chunk_fraction": sum(duplicate_fractions) / len(duplicate_fractions)
        if duplicate_fractions
        else 0.0,
    }


def per_query_chunk_diagnostics(
    chunk_rankings: Mapping[str, Sequence[ChunkHit]],
    qrels: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    """Compute per-query chunk multiplicity diagnostics."""
    rows: list[dict[str, Any]] = []
    for query_id in sorted(set(chunk_rankings) | set(qrels)):
        hits = list(chunk_rankings.get(query_id, []))
        per_doc: dict[str, int] = {}
        for hit in hits:
            per_doc[hit.doc_id] = per_doc.get(hit.doc_id, 0) + 1
        relevant_doc_ids = {doc_id for doc_id, score in qrels.get(query_id, {}).items() if float(score) > 0}
        relevant_chunk_counts = {doc_id: per_doc.get(doc_id, 0) for doc_id in sorted(relevant_doc_ids)}
        rows.append(
            {
                "query_id": query_id,
                "num_chunks_retrieved": len(hits),
                "num_unique_docs_retrieved": len(per_doc),
                "avg_chunks_per_retrieved_doc": sum(per_doc.values()) / len(per_doc) if per_doc else 0.0,
                "duplicate_doc_fraction": sum(count - 1 for count in per_doc.values() if count > 1) / len(hits)
                if hits
                else 0.0,
                "relevant_docs_with_multiple_chunks": sum(
                    1 for count in relevant_chunk_counts.values() if count > 1
                ),
                "relevant_chunk_counts": relevant_chunk_counts,
            }
        )
    return rows


def compare_per_query(
    baseline_rows: Sequence[Mapping[str, Any]],
    proposed_rows: Sequence[Mapping[str, Any]],
    baseline_rankings: Mapping[str, Sequence[Mapping[str, Any]]],
    proposed_rankings: Mapping[str, Sequence[Mapping[str, Any]]],
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, float]],
    proposed_method: str,
    metric_key: str = "ndcg_at_10",
) -> list[dict[str, Any]]:
    """Compare baseline and proposed per-query metric rows."""
    baseline_by_query = {row["query_id"]: row for row in baseline_rows}
    proposed_by_query = {row["query_id"]: row for row in proposed_rows}
    rows: list[dict[str, Any]] = []
    for query_id in sorted(set(baseline_by_query) | set(proposed_by_query)):
        baseline_score = float(baseline_by_query.get(query_id, {}).get(metric_key, 0.0))
        proposed_score = float(proposed_by_query.get(query_id, {}).get(metric_key, 0.0))
        baseline_top10 = ranked_doc_ids(baseline_rankings.get(query_id, []))[:10]
        proposed_top10 = ranked_doc_ids(proposed_rankings.get(query_id, []))[:10]
        relevant_doc_ids = [doc_id for doc_id, score in qrels.get(query_id, {}).items() if float(score) > 0]
        rows.append(
            {
                "query_id": query_id,
                "query_text": queries.get(query_id, ""),
                "method": proposed_method,
                f"baseline_{metric_key}": baseline_score,
                f"proposed_{metric_key}": proposed_score,
                "delta": proposed_score - baseline_score,
                "relevant_doc_ids": relevant_doc_ids,
                "baseline_top10": baseline_top10,
                "proposed_top10": proposed_top10,
                "relevant_docs_moved_up": [
                    doc_id
                    for doc_id in relevant_doc_ids
                    if _rank_of(doc_id, proposed_top10) < _rank_of(doc_id, baseline_top10)
                ],
                "relevant_docs_moved_down": [
                    doc_id
                    for doc_id in relevant_doc_ids
                    if _rank_of(doc_id, proposed_top10) > _rank_of(doc_id, baseline_top10)
                ],
            }
        )
    return rows


def correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation with safe empty handling."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_den = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_den = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    return numerator / (x_den * y_den) if x_den and y_den else 0.0


def _rank_of(doc_id: str, doc_ids: Sequence[str]) -> int:
    try:
        return list(doc_ids).index(doc_id) + 1
    except ValueError:
        return 10**9

