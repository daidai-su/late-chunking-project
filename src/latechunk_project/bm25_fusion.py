"""Document-level BM25 and RRF fusion helpers."""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from typing import Any, Mapping, Sequence


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def simple_tokenize(text: str) -> list[str]:
    """Small deterministic tokenizer for BM25 fallback."""
    return TOKEN_PATTERN.findall(text.lower())


def construct_document_text(doc: str | Mapping[str, Any]) -> str:
    """Construct a document string from BEIR-style corpus rows."""
    if isinstance(doc, str):
        return doc
    title = str(doc.get("title", "")).strip()
    text = str(doc.get("text", "")).strip()
    return f"{title} {text}".strip()


class SimpleBM25:
    """Minimal BM25 implementation for Colab-safe lexical retrieval."""

    def __init__(
        self,
        corpus: Mapping[str, str | Mapping[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids = sorted(corpus)
        self.doc_tokens = [simple_tokenize(construct_document_text(corpus[doc_id])) for doc_id in self.doc_ids]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_freq: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            self.doc_freq.update(set(tokens))
        self.num_docs = len(self.doc_ids)

    def idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        return math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5)) if self.num_docs else 0.0

    def score(self, query: str, doc_index: int) -> float:
        query_terms = simple_tokenize(query)
        if not query_terms or not self.doc_tokens:
            return 0.0
        doc_len = self.doc_lengths[doc_index] or 1
        tf = self.term_freqs[doc_index]
        denom_norm = self.k1 * (1.0 - self.b + self.b * doc_len / (self.avg_doc_length or 1.0))
        score = 0.0
        for term in query_terms:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            score += self.idf(term) * (freq * (self.k1 + 1.0)) / (freq + denom_norm)
        return score

    def rank(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        rows = [
            {"doc_id": doc_id, "doc_score": float(self.score(query, idx))}
            for idx, doc_id in enumerate(self.doc_ids)
        ]
        rows = sorted(rows, key=lambda row: (-float(row["doc_score"]), str(row["doc_id"])))
        if top_k is not None:
            rows = rows[:top_k]
        for rank, row in enumerate(rows, start=1):
            row["doc_rank"] = rank
        return rows


def bm25_rankings(
    corpus: Mapping[str, str | Mapping[str, Any]],
    queries: Mapping[str, str],
    top_k: int | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Rank documents with BM25 for each query."""
    started = time.perf_counter()
    index = SimpleBM25(corpus)
    rankings = {query_id: index.rank(query, top_k=top_k) for query_id, query in queries.items()}
    elapsed = time.perf_counter() - started
    diagnostics = {
        "method": "bm25_only",
        "total_runtime_seconds": elapsed,
        "avg_latency_per_query_seconds": elapsed / len(queries) if queries else 0.0,
        "num_queries": len(queries),
        "num_corpus_docs": len(corpus),
    }
    return rankings, diagnostics


def rrf_fuse_rankings(
    dense_rankings: Mapping[str, Sequence[Mapping[str, Any]]],
    bm25_rankings_by_query: Mapping[str, Sequence[Mapping[str, Any]]],
    rrf_k: int = 60,
) -> dict[str, list[dict[str, Any]]]:
    """Fuse two document rankings with reciprocal rank fusion."""
    fused: dict[str, list[dict[str, Any]]] = {}
    query_ids = sorted(set(dense_rankings) | set(bm25_rankings_by_query))
    for query_id in query_ids:
        scores: dict[str, float] = {}
        ranks: dict[str, dict[str, int]] = {}
        for source_name, ranking in [
            ("dense", dense_rankings.get(query_id, [])),
            ("bm25", bm25_rankings_by_query.get(query_id, [])),
        ]:
            for fallback_rank, row in enumerate(ranking, start=1):
                doc_id = str(row["doc_id"])
                rank = int(row.get("doc_rank", fallback_rank))
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
                ranks.setdefault(doc_id, {})[f"{source_name}_rank"] = rank
        rows = [
            {
                "doc_id": doc_id,
                "doc_score": score,
                **ranks.get(doc_id, {}),
            }
            for doc_id, score in scores.items()
        ]
        rows = sorted(rows, key=lambda row: (-float(row["doc_score"]), str(row["doc_id"])))
        for rank, row in enumerate(rows, start=1):
            row["doc_rank"] = rank
        fused[query_id] = rows
    return fused

