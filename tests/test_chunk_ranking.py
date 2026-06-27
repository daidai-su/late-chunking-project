from pathlib import Path
from tempfile import TemporaryDirectory

from latechunk_project.chunk_ranking import (
    ChunkHit,
    chunk_results_to_hits,
    extract_doc_id_from_chunk_id,
    load_chunk_rankings_jsonl,
    save_chunk_rankings_jsonl,
)


def test_chunk_ranking_jsonl_serialization():
    rankings = {
        "q1": [
            ChunkHit("q1", "d1~0", "d1", 1, 0.9),
            ChunkHit("q1", "d2~0", "d2", 2, 0.8),
        ]
    }

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "chunk_rankings.jsonl"
        save_chunk_rankings_jsonl(rankings, path)
        loaded = load_chunk_rankings_jsonl(path)

        assert loaded["q1"][0].chunk_id == "d1~0"
        assert loaded["q1"][1].chunk_score == 0.8


def test_doc_id_extraction_from_chunk_id():
    assert extract_doc_id_from_chunk_id("doc-123~4") == "doc-123"
    assert extract_doc_id_from_chunk_id("doc~with~sep~2") == "doc~with~sep"
    assert extract_doc_id_from_chunk_id("doc_without_chunk") == "doc_without_chunk"


def test_score_sorting_descending_from_official_results():
    hits = chunk_results_to_hits({"q1": {"d2~0": 0.2, "d1~0": 0.9, "d3~0": 0.2}})["q1"]

    assert [hit.chunk_id for hit in hits] == ["d1~0", "d2~0", "d3~0"]
    assert [hit.chunk_rank for hit in hits] == [1, 2, 3]
