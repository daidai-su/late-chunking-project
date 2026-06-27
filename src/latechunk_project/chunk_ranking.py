"""Chunk-level ranking helpers for Late Chunking aggregation experiments."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


@dataclass(frozen=True)
class ChunkHit:
    """One chunk hit for one query."""

    query_id: str
    chunk_id: str
    doc_id: str
    chunk_rank: int
    chunk_score: float
    chunk_text: str | None = None


def extract_doc_id_from_chunk_id(chunk_id: str) -> str:
    """Extract document ID from the official ``doc_id~chunk_index`` format."""
    if "~" not in chunk_id:
        return chunk_id
    return "~".join(chunk_id.split("~")[:-1])


def sort_chunk_hits(hits: Sequence[ChunkHit]) -> list[ChunkHit]:
    """Sort hits by score descending with deterministic tie-breaking."""
    return sorted(hits, key=lambda hit: (-hit.chunk_score, hit.chunk_rank, hit.chunk_id))


def chunk_results_to_hits(
    results: Mapping[str, Mapping[str, float]],
    top_k: int | None = None,
    include_chunk_text: Mapping[str, str] | None = None,
) -> dict[str, list[ChunkHit]]:
    """Convert official ``query -> chunk_id -> score`` results into hit lists.

    Adapted from official jina-ai/late-chunking, commit
    ``1d3bb02bf091becd0771455e4e7959463935e26c``. The official evaluator
    builds a chunk score dictionary in ``AbsTaskChunkedRetrieval.get_results``;
    this helper preserves that ranking as JSONL-ready records.
    """
    converted: dict[str, list[ChunkHit]] = {}
    for query_id, chunk_scores in results.items():
        ordered = sorted(chunk_scores.items(), key=lambda item: (-float(item[1]), item[0]))
        if top_k is not None:
            ordered = ordered[:top_k]
        hits: list[ChunkHit] = []
        for rank, (chunk_id, score) in enumerate(ordered, start=1):
            chunk_text = include_chunk_text.get(chunk_id) if include_chunk_text else None
            hits.append(
                ChunkHit(
                    query_id=str(query_id),
                    chunk_id=str(chunk_id),
                    doc_id=extract_doc_id_from_chunk_id(str(chunk_id)),
                    chunk_rank=rank,
                    chunk_score=float(score),
                    chunk_text=chunk_text,
                )
            )
        converted[str(query_id)] = hits
    return converted


def save_chunk_rankings_jsonl(rankings: Mapping[str, Sequence[ChunkHit]], path: str | Path) -> Path:
    """Save query-grouped chunk rankings as JSONL."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for query_id in sorted(rankings):
            for hit in rankings[query_id]:
                handle.write(json.dumps(asdict(hit), ensure_ascii=False) + "\n")
    return output_path


def load_chunk_rankings_jsonl(path: str | Path) -> dict[str, list[ChunkHit]]:
    """Load chunk ranking JSONL into query-grouped ``ChunkHit`` lists."""
    rankings: dict[str, list[ChunkHit]] = {}
    input_path = Path(path)
    if not input_path.exists():
        return rankings
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            hit = ChunkHit(
                query_id=str(row["query_id"]),
                chunk_id=str(row["chunk_id"]),
                doc_id=str(row.get("doc_id") or extract_doc_id_from_chunk_id(str(row["chunk_id"]))),
                chunk_rank=int(row["chunk_rank"]),
                chunk_score=float(row["chunk_score"]),
                chunk_text=row.get("chunk_text"),
            )
            rankings.setdefault(hit.query_id, []).append(hit)
    return {query_id: sort_chunk_hits(hits) for query_id, hits in rankings.items()}


def save_doc_rankings_jsonl(rankings: Mapping[str, Sequence[Mapping[str, Any]]], path: str | Path) -> Path:
    """Save document rankings as JSONL."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for query_id in sorted(rankings):
            for row in rankings[query_id]:
                payload = {"query_id": query_id, **dict(row)}
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return output_path


@contextmanager
def patch_official_chunk_ranking_recorder(
    output_path: str | Path,
    top_k: int | None = None,
) -> Iterator[Path]:
    """Temporarily wrap the official evaluator to save chunk-level rankings.

    This wrapper does not modify official repository files. It monkey-patches
    ``AbsTaskChunkedRetrieval.get_results`` only inside the context manager and
    restores the original method afterwards.
    """
    from chunked_pooling.mteb_chunked_eval import AbsTaskChunkedRetrieval

    original_get_results = AbsTaskChunkedRetrieval.get_results
    output = Path(output_path)

    def recording_get_results(self, chunk_id_list, k_values, query_ids, similarity_matrix):
        results = original_get_results(self, chunk_id_list, k_values, query_ids, similarity_matrix)
        hits_by_query = chunk_results_to_hits(results, top_k=top_k)
        save_chunk_rankings_jsonl(hits_by_query, output)
        return results

    AbsTaskChunkedRetrieval.get_results = recording_get_results
    try:
        yield output
    finally:
        AbsTaskChunkedRetrieval.get_results = original_get_results

