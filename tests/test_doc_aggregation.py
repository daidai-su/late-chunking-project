import math

from latechunk_project.chunk_ranking import ChunkHit
from latechunk_project.doc_aggregation import aggregate_query_chunks


def _toy_hits():
    return [
        ChunkHit("q1", "d1~0", "d1", 1, 0.90),
        ChunkHit("q1", "d2~0", "d2", 2, 0.89),
        ChunkHit("q1", "d2~1", "d2", 3, 0.88),
        ChunkHit("q1", "d3~0", "d3", 4, 0.50),
        ChunkHit("q1", "d2~2", "d2", 5, 0.40),
    ]


def test_first_occurrence_works_on_toy_chunk_ranking():
    ranking = aggregate_query_chunks(_toy_hits(), "first_occurrence")

    assert [row["doc_id"] for row in ranking[:3]] == ["d1", "d2", "d3"]
    assert ranking[1]["doc_score"] == 0.89


def test_max_score_works():
    ranking = aggregate_query_chunks(_toy_hits(), "max_score")

    assert ranking[0]["doc_id"] == "d1"
    assert ranking[1]["doc_score"] == 0.89


def test_topk_mean_works():
    ranking = aggregate_query_chunks(_toy_hits(), "topk_mean", topk_mean_k=2)
    by_doc = {row["doc_id"]: row for row in ranking}

    assert math.isclose(by_doc["d2"]["doc_score"], (0.89 + 0.88) / 2)


def test_softmax_topk_works():
    ranking = aggregate_query_chunks(_toy_hits(), "softmax_topk", softmax_topk_k=2, softmax_tau=0.05)
    by_doc = {row["doc_id"]: row for row in ranking}

    assert 0.88 < by_doc["d2"]["doc_score"] < 0.89


def test_rrf_chunk_vote_works():
    ranking = aggregate_query_chunks(_toy_hits(), "rrf_chunk_vote", rrf_k=60)

    assert ranking[0]["doc_id"] == "d2"


def test_deterministic_tie_breaking():
    hits = [
        ChunkHit("q1", "b~0", "b", 1, 0.5),
        ChunkHit("q1", "a~0", "a", 2, 0.5),
    ]

    ranking = aggregate_query_chunks(hits, "max_score")

    assert [row["doc_id"] for row in ranking] == ["b", "a"]


def test_handles_empty_chunk_list():
    assert aggregate_query_chunks([], "first_occurrence") == []

