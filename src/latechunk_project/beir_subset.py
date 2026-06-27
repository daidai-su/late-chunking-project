"""Tiny BEIR-style filtering helpers that avoid network or model downloads."""

from __future__ import annotations

import random
from typing import Any, Mapping, Union


Corpus = Mapping[str, Mapping[str, Any]]
Queries = Mapping[str, str]
Qrels = Mapping[str, Mapping[str, Union[int, float]]]


def filter_qrels_by_query_ids(qrels: Qrels, query_ids: list[str] | set[str]) -> dict[str, dict[str, int | float]]:
    """Return qrels for selected query IDs only."""
    selected = set(query_ids)
    return {qid: dict(doc_scores) for qid, doc_scores in qrels.items() if qid in selected}


def filter_queries_by_ids(queries: Queries, query_ids: list[str] | set[str]) -> dict[str, str]:
    """Return query text for selected query IDs only."""
    selected = set(query_ids)
    return {qid: text for qid, text in queries.items() if qid in selected}


def relevant_doc_ids(qrels: Qrels, query_ids: list[str] | set[str] | None = None) -> set[str]:
    """Collect positively relevant document IDs."""
    selected = set(query_ids) if query_ids is not None else None
    docs: set[str] = set()
    for query_id, doc_scores in qrels.items():
        if selected is not None and query_id not in selected:
            continue
        docs.update(doc_id for doc_id, score in doc_scores.items() if float(score) > 0)
    return docs


def sample_query_ids(queries: Queries, max_queries: int | None, seed: int) -> list[str]:
    """Sample query IDs deterministically."""
    query_ids = sorted(queries)
    if max_queries is None or max_queries >= len(query_ids):
        return query_ids
    if max_queries < 0:
        raise ValueError("max_queries must be non-negative")
    rng = random.Random(seed)
    rng.shuffle(query_ids)
    return sorted(query_ids[:max_queries])


def sample_corpus_preserving_relevance(
    corpus: Corpus,
    qrels: Qrels,
    max_docs: int | None,
    seed: int,
    query_ids: list[str] | set[str] | None = None,
) -> dict[str, Mapping[str, Any]]:
    """Sample corpus docs while always retaining positively relevant docs.

    If the number of relevant docs exceeds max_docs, this function returns all
    relevant docs rather than silently creating invalid qrels.
    """
    keep = relevant_doc_ids(qrels, query_ids=query_ids) & set(corpus)
    if max_docs is None or max_docs >= len(corpus):
        keep.update(corpus)
    elif len(keep) < max_docs:
        candidates = sorted(set(corpus) - keep)
        rng = random.Random(seed)
        rng.shuffle(candidates)
        keep.update(candidates[: max_docs - len(keep)])
    return {doc_id: corpus[doc_id] for doc_id in sorted(keep)}


def make_beir_smoke_subset(
    corpus: Corpus,
    queries: Queries,
    qrels: Qrels,
    max_queries: int,
    max_corpus_docs: int,
    seed: int,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, str], dict[str, dict[str, int | float]]]:
    """Create a deterministic, internally consistent BEIR-style subset."""
    selected_query_ids = sample_query_ids(queries, max_queries=max_queries, seed=seed)
    subset_queries = filter_queries_by_ids(queries, selected_query_ids)
    subset_qrels = filter_qrels_by_query_ids(qrels, selected_query_ids)
    subset_corpus = sample_corpus_preserving_relevance(
        corpus,
        subset_qrels,
        max_docs=max_corpus_docs,
        seed=seed,
        query_ids=selected_query_ids,
    )
    return subset_corpus, subset_queries, subset_qrels
